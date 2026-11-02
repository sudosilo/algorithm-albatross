"""Algorithm Albatross v1 fetcher.

Pulls instrumentals from ccMixter into the library, plus instrumental
remixes of each one so the library holds related pairs. Every file is
checked against the whole library before it is kept: same ccMixter
upload, same bytes, or same audio content (Chromaprint) are all skipped.
Parent and remix links are stored for later ground truth.

Run: algorithm_albatross_v1_fetch.py COUNT
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import hashlib
import http.client
from pathlib import Path

import numpy as np
import requests

http.client._MAXLINE = 1 << 24
http.client._MAXHEADERS = 1000

HOME = Path.home()
LIBRARY = Path(os.environ.get("ALBATROSS_LIBRARY", HOME / "algorithm-albatross-library"))
DATA = Path(os.environ.get("ALBATROSS_DATA", HOME / "algorithm-albatross-data"))
DEST = LIBRARY / "ccmixter"
API = "https:" + "//ccmixter.org/api/query"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0",
      "Referer": "https:" + "//ccmixter.org/"}
AUDIO_EXT = (".mp3", ".flac", ".wav", ".ogg")
LIB_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".wma", ".webm"}
MAX_BYTES = 60 * 1024 * 1024
SAME_AUDIO = 0.85

DEST.mkdir(parents=True, exist_ok=True)
DB = sqlite3.connect(str(DATA / "algorithm-albatross-v1-fetch.sqlite"))
DB.executescript(
    """
    CREATE TABLE IF NOT EXISTS seen (upload_id TEXT PRIMARY KEY, outcome TEXT);
    CREATE TABLE IF NOT EXISTS prints (path TEXT PRIMARY KEY, sha TEXT, fp TEXT);
    CREATE TABLE IF NOT EXISTS lineage (child TEXT, parent TEXT, PRIMARY KEY (child, parent));
    """
)


def query(params):
    url = API + "?" + "&".join("%s=%s" % (k, v) for k, v in params.items())
    res = requests.get(url, headers=UA, timeout=60)
    res.raise_for_status()
    text = re.sub(r'(?<=[:\[,])\s*0+(\d)', r'\1', res.text)
    data = json.loads(text, strict=False)
    return data if isinstance(data, list) else []


def tags_of(u):
    return set(t for t in str(u.get("upload_tags", "")).replace(" ", ",").split(",") if t)


def audio_url(u):
    for f in u.get("files") or []:
        link = f.get("download_url") or ""
        if link.lower().endswith(AUDIO_EXT):
            return link
    return ""


def is_instrumental(u):
    t = tags_of(u)
    return "instrumental" in t and "acappella" not in t and bool(audio_url(u))


def sha_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fp_of(path):
    try:
        out = subprocess.run(["fpcalc", "-raw", "-json", "-length", "120", str(path)],
                             capture_output=True, text=True, timeout=120)
        return json.loads(out.stdout).get("fingerprint") or []
    except Exception:
        return []


def similarity(a, b):
    if len(a) < 50 or len(b) < 50:
        return 0.0
    a = np.asarray(a, dtype=np.int64).astype(np.uint32)
    b = np.asarray(b, dtype=np.int64).astype(np.uint32)
    best = 0.0
    for shift in range(-60, 61):
        pa = a[max(0, shift):]
        pb = b[max(0, -shift):]
        n = min(len(pa), len(pb))
        if n < 50:
            continue
        bits = np.unpackbits((pa[:n] ^ pb[:n]).view(np.uint8)).sum()
        best = max(best, 1.0 - bits / (32.0 * n))
    return float(best)


def remember(path):
    path = str(Path(path).resolve())
    if DB.execute("SELECT 1 FROM prints WHERE path=?", (path,)).fetchone():
        return
    DB.execute("INSERT INTO prints VALUES (?,?,?)", (path, sha_of(path), json.dumps(fp_of(path))))
    DB.commit()


def duplicate_of(path):
    sha = sha_of(path)
    row = DB.execute("SELECT path FROM prints WHERE sha=?", (sha,)).fetchone()
    if row:
        return row[0]
    fp = fp_of(path)
    for other, raw in DB.execute("SELECT path, fp FROM prints").fetchall():
        if similarity(fp, json.loads(raw)) >= SAME_AUDIO:
            return other
    return ""


def index_library():
    for p in LIBRARY.rglob("*"):
        if p.is_file() and p.suffix.lower() in LIB_EXT:
            try:
                remember(p)
            except OSError:
                pass


def safe(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "track"


def take(u):
    """Download one upload. Returns 'kept', 'duplicate', 'skipped' or 'failed'."""
    uid = str(u.get("upload_id"))
    if DB.execute("SELECT 1 FROM seen WHERE upload_id=?", (uid,)).fetchone():
        return "skipped"
    link = audio_url(u)
    ext = os.path.splitext(link)[1].lower()
    final = DEST / ("%s_%s%s" % (uid, safe(u.get("upload_name", "")), ext))
    part = final.with_suffix(ext + ".part")
    outcome = "failed"
    try:
        with requests.get(link, headers=UA, stream=True, timeout=120) as res:
            res.raise_for_status()
            size = 0
            with open(part, "wb") as fh:
                for chunk in res.iter_content(1 << 16):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise ValueError("too large")
                    fh.write(chunk)
        dup = duplicate_of(part)
        if dup:
            part.unlink()
            outcome = "duplicate"
            print("  duplicate of", os.path.basename(dup))
        else:
            part.rename(final)
            remember(final)
            outcome = "kept"
    except Exception as e:
        print("  failed:", str(e)[:100])
        if part.exists():
            part.unlink()
    DB.execute("INSERT OR REPLACE INTO seen VALUES (?,?)", (uid, outcome))
    DB.commit()
    return outcome


def main(count):
    print("indexing the current library for duplicate checks")
    index_library()
    kept = dupes = 0
    offset = 0
    while kept < count:
        rows = query({"f": "json", "tags": "instrumental", "type": "all", "sort": "rank",
                      "limit": "40", "offset": str(offset)})
        if not rows:
            break
        offset += 40
        for u in rows:
            if kept >= count:
                break
            if not is_instrumental(u):
                continue
            print("%d/%d  %s by %s" % (kept + 1, count, u.get("upload_name", "?"), u.get("user_name", "?")))
            got = take(u)
            kept += got == "kept"
            dupes += got == "duplicate"
            try:
                kids = [m for m in query({"f": "json", "remixes": str(u["upload_id"]), "limit": "10"})
                        if is_instrumental(m)]
            except Exception:
                kids = []
            for m in kids:
                if kept >= count:
                    break
                DB.execute("INSERT OR IGNORE INTO lineage VALUES (?,?)", (str(m["upload_id"]), str(u["upload_id"])))
                DB.commit()
                print("%d/%d    remix: %s by %s" % (kept + 1, count, m.get("upload_name", "?"), m.get("user_name", "?")))
                got = take(m)
                kept += got == "kept"
                dupes += got == "duplicate"
    print("FETCH DONE kept %d, duplicates skipped %d" % (kept, dupes))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
