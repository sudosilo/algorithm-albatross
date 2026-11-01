"""Algorithm Albatross v1 server.

Runs on the Frankfurt machine. Holds the local instrumental library,
analyzes songs into skeletons, compares them, and stores confirmed
cover pairs that act as ground truth for all later research.
"""

import json
import os
import queue
import re
import sqlite3
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import algorithm_albatross_v1_analyze as analyze
import algorithm_albatross_v1_compare as cmp

HOME = Path.home()
LIBRARY = Path(os.environ.get("ALBATROSS_LIBRARY", HOME / "algorithm-albatross-library"))
DATA = Path(os.environ.get("ALBATROSS_DATA", HOME / "algorithm-albatross-data"))
TOKEN = os.environ.get("ALBATROSS_TOKEN", "")
AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".aif", ".wma", ".webm"}

LIBRARY.mkdir(parents=True, exist_ok=True)
(LIBRARY / "uploads").mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

DB_LOCK = threading.Lock()
DB = sqlite3.connect(str(DATA / "algorithm-albatross-v1.sqlite"), check_same_thread=False)
DB.row_factory = sqlite3.Row
DB.executescript(
    """
    CREATE TABLE IF NOT EXISTS songs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      path TEXT UNIQUE,
      title TEXT,
      status TEXT,
      error TEXT,
      tempo REAL,
      key_name TEXT,
      duration REAL,
      n_bars INTEGER,
      n_sections INTEGER,
      skeleton TEXT,
      vector TEXT,
      added REAL
    );
    CREATE TABLE IF NOT EXISTS pairs (
      a INTEGER,
      b INTEGER,
      verdict TEXT,
      formula REAL,
      result TEXT,
      confirmed INTEGER DEFAULT 0,
      note TEXT DEFAULT '',
      updated REAL,
      PRIMARY KEY (a, b)
    );
    CREATE TABLE IF NOT EXISTS jobs (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      kind TEXT,
      status TEXT,
      detail TEXT,
      progress REAL DEFAULT 0,
      created REAL,
      finished REAL
    );
    """
)
DB.commit()
with DB_LOCK:
    DB.execute("UPDATE songs SET status='queued' WHERE status='analyzing'")
    DB.execute("UPDATE jobs SET status='stopped' WHERE status IN ('queued','running')")
    DB.commit()


def q(sql, args=(), one=False):
    with DB_LOCK:
        cur = DB.execute(sql, args)
        rows = cur.fetchall()
        DB.commit()
    return (rows[0] if rows else None) if one else rows


def ex(sql, args=()):
    with DB_LOCK:
        cur = DB.execute(sql, args)
        DB.commit()
        return cur.lastrowid


WORK = queue.Queue()
STATS_CACHE = {"stamp": None, "stats": None}


def corpus():
    stamp = q("SELECT COUNT(*) AS n, MAX(id) AS m FROM songs WHERE status='ready'", one=True)
    key = (stamp["n"], stamp["m"])
    if STATS_CACHE["stamp"] != key:
        sks = [json.loads(r["skeleton"]) for r in q("SELECT skeleton FROM songs WHERE status='ready'")]
        STATS_CACHE["stats"] = cmp.corpus_stats(sks)
        STATS_CACHE["stamp"] = key
    return STATS_CACHE["stats"]


def load_skeleton(song_id):
    row = q("SELECT skeleton, status FROM songs WHERE id=?", (song_id,), one=True)
    if row is None:
        raise HTTPException(404, "song not found")
    if row["status"] != "ready":
        raise HTTPException(409, "song is not analyzed yet")
    return json.loads(row["skeleton"])


def store_pair(a, b, result):
    if a > b:
        a, b = b, a
    slim = dict(result)
    ex(
        """INSERT INTO pairs (a, b, verdict, formula, result, updated) VALUES (?,?,?,?,?,?)
           ON CONFLICT(a,b) DO UPDATE SET verdict=excluded.verdict, formula=excluded.formula,
           result=excluded.result, updated=excluded.updated""",
        (a, b, result["verdict"], result["formula_score"], json.dumps(slim), time.time()),
    )


def run_compare(a, b, sweep=True):
    ska, skb = load_skeleton(a), load_skeleton(b)
    result = cmp.compare(ska, skb, corpus(), sweep=sweep)
    result["a"], result["b"] = a, b
    store_pair(a, b, result)
    return result


def analyze_song(song_id):
    row = q("SELECT path FROM songs WHERE id=?", (song_id,), one=True)
    if row is None:
        return
    ex("UPDATE songs SET status='analyzing', error=NULL WHERE id=?", (song_id,))
    try:
        sk = analyze.analyze_file(row["path"])
        ex(
            """UPDATE songs SET status='ready', tempo=?, key_name=?, duration=?, n_bars=?, n_sections=?,
               skeleton=?, vector=? WHERE id=?""",
            (sk["tempo"], sk["key_name"], sk["duration"], sk["n_bars"], len(sk["sections"]),
             json.dumps(sk), json.dumps(sk["vector"]), song_id),
        )
    except Exception as e:
        ex("UPDATE songs SET status='error', error=? WHERE id=?", (str(e)[:500], song_id))


def candidates_for(song_id, limit):
    row = q("SELECT vector FROM songs WHERE id=? AND status='ready'", (song_id,), one=True)
    if row is None:
        return []
    va = json.loads(row["vector"])
    scored = []
    for r in q("SELECT id, vector FROM songs WHERE status='ready' AND id!=?", (song_id,)):
        scored.append((cmp.vector_similarity(va, json.loads(r["vector"])), r["id"]))
    scored.sort(reverse=True)
    return scored[:limit]


def sweep_job(job_id, per_song):
    ids = [r["id"] for r in q("SELECT id FROM songs WHERE status='ready' ORDER BY id")]
    todo = set()
    for sid in ids:
        for _, other in candidates_for(sid, per_song):
            todo.add((min(sid, other), max(sid, other)))
    todo = sorted(todo)
    for n, (a, b) in enumerate(todo):
        try:
            run_compare(a, b, sweep=False)
        except Exception:
            pass
        ex("UPDATE jobs SET progress=?, detail=? WHERE id=?", ((n + 1) / max(1, len(todo)), f"{n + 1} of {len(todo)} pairs", job_id))


def worker():
    while True:
        kind, arg = WORK.get()
        try:
            if kind == "analyze":
                analyze_song(arg)
            elif kind == "sweep":
                job_id, per_song = arg
                ex("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
                sweep_job(job_id, per_song)
                ex("UPDATE jobs SET status='done', progress=1, finished=? WHERE id=?", (time.time(), job_id))
        except Exception:
            traceback.print_exc()
            if kind == "sweep":
                ex("UPDATE jobs SET status='error', finished=? WHERE id=?", (time.time(), arg[0]))
        finally:
            WORK.task_done()


threading.Thread(target=worker, daemon=True).start()
for r in q("SELECT id FROM songs WHERE status='queued' ORDER BY id"):
    WORK.put(("analyze", r["id"]))


def add_path(path):
    path = Path(path).resolve()
    title = re.sub(r"[_]+", " ", path.stem).strip()
    existing = q("SELECT id FROM songs WHERE path=?", (str(path),), one=True)
    if existing:
        return existing["id"], False
    sid = ex("INSERT INTO songs (path, title, status, added) VALUES (?,?,?,?)", (str(path), title, "queued", time.time()))
    WORK.put(("analyze", sid))
    return sid, True


app = FastAPI(title="algorithm-albatross-backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def auth(token):
    if TOKEN and token != TOKEN:
        raise HTTPException(401, "bad token")


class PairBody(BaseModel):
    a: int
    b: int
    sweep: bool = True


class ConfirmBody(BaseModel):
    a: int
    b: int
    confirmed: int
    note: str = ""


class TitleBody(BaseModel):
    title: str


class SweepBody(BaseModel):
    per_song: int = 5


@app.get("/api/health")
def health(x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    counts = {r["status"]: r["n"] for r in q("SELECT status, COUNT(*) AS n FROM songs GROUP BY status")}
    return {"ok": True, "library": str(LIBRARY), "counts": counts, "queue": WORK.qsize()}


@app.get("/api/songs")
def songs(x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    rows = q("SELECT id, path, title, status, error, tempo, key_name, duration, n_bars, n_sections FROM songs ORDER BY title COLLATE NOCASE")
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["path"] = str(Path(r["path"]).relative_to(LIBRARY))
        except ValueError:
            pass
        out.append(d)
    return out


@app.get("/api/songs/{song_id}")
def song(song_id: int, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    row = q("SELECT * FROM songs WHERE id=?", (song_id,), one=True)
    if row is None:
        raise HTTPException(404, "song not found")
    d = dict(row)
    sk = json.loads(d.pop("skeleton") or "null")
    d.pop("vector", None)
    if sk:
        for k in ("sweep_chroma", "bar_chroma", "vector"):
            sk.pop(k, None)
    d["skeleton"] = sk
    return d


@app.post("/api/songs/{song_id}/title")
def set_title(song_id: int, body: TitleBody, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    ex("UPDATE songs SET title=? WHERE id=?", (body.title.strip()[:200], song_id))
    return {"ok": True}


@app.post("/api/songs/{song_id}/reanalyze")
def reanalyze(song_id: int, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    ex("UPDATE songs SET status='queued' WHERE id=?", (song_id,))
    WORK.put(("analyze", song_id))
    return {"ok": True}


@app.post("/api/scan")
def scan(x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    added = 0
    for p in LIBRARY.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            _, new = add_path(p)
            added += int(new)
    return {"added": added}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    name = re.sub(r"[^A-Za-z0-9 ._()\-]+", "_", Path(file.filename or "upload").name)
    if Path(name).suffix.lower() not in AUDIO_EXT:
        raise HTTPException(400, "not an audio file")
    dest = LIBRARY / "uploads" / name
    stem, n = dest.stem, 1
    while dest.exists():
        dest = dest.with_name(f"{stem} ({n}){dest.suffix}")
        n += 1
    with open(dest, "wb") as fh:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    sid, _ = add_path(dest)
    return {"id": sid}


@app.post("/api/compare")
def compare(body: PairBody, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    if body.a == body.b:
        raise HTTPException(400, "pick two different songs")
    return run_compare(body.a, body.b, sweep=body.sweep)


@app.get("/api/match/{song_id}")
def match(song_id: int, limit: int = 8, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    load_skeleton(song_id)
    out = []
    for sim, other in candidates_for(song_id, limit):
        res = run_compare(song_id, other, sweep=False)
        out.append({
            "id": other, "vector_similarity": round(sim, 4), "verdict": res["verdict"],
            "formula_score": res["formula_score"], "lockstep_coverage": res["lockstep_coverage"],
            "section_score": res["section_score"], "cue_score": res["cue_score"], "bpm_ratio": res["bpm_ratio"],
        })
    out.sort(key=lambda r: r["formula_score"], reverse=True)
    return out


@app.post("/api/sweep")
def sweep(body: SweepBody, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    job_id = ex("INSERT INTO jobs (kind, status, detail, created) VALUES ('sweep','queued','waiting',?)", (time.time(),))
    WORK.put(("sweep", (job_id, max(1, min(20, body.per_song)))))
    return {"job": job_id}


@app.get("/api/jobs")
def jobs(x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    return [dict(r) for r in q("SELECT * FROM jobs ORDER BY id DESC LIMIT 20")]


@app.get("/api/pairs")
def pairs(confirmed_only: bool = False, limit: int = 100, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    where = "WHERE p.confirmed=1" if confirmed_only else ""
    rows = q(
        f"""SELECT p.a, p.b, p.verdict, p.formula, p.confirmed, p.note, sa.title AS title_a, sb.title AS title_b
            FROM pairs p JOIN songs sa ON sa.id=p.a JOIN songs sb ON sb.id=p.b {where}
            ORDER BY p.confirmed DESC, p.formula DESC LIMIT ?""",
        (limit,),
    )
    return [dict(r) for r in rows]


@app.get("/api/pairs/{a}/{b}")
def pair(a: int, b: int, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    if a > b:
        a, b = b, a
    row = q("SELECT * FROM pairs WHERE a=? AND b=?", (a, b), one=True)
    if row is None:
        raise HTTPException(404, "pair not compared yet")
    d = dict(row)
    d["result"] = json.loads(d["result"])
    return d


@app.post("/api/confirm")
def confirm(body: ConfirmBody, x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    a, b = min(body.a, body.b), max(body.a, body.b)
    if q("SELECT 1 FROM pairs WHERE a=? AND b=?", (a, b), one=True) is None:
        run_compare(a, b, sweep=False)
    ex("UPDATE pairs SET confirmed=?, note=?, updated=? WHERE a=? AND b=?", (int(body.confirmed), body.note[:500], time.time(), a, b))
    return {"ok": True}


@app.get("/api/calibration")
def calibration(x_albatross_token: str = Header(default="")):
    auth(x_albatross_token)
    keys = ["formula_score", "lockstep_coverage", "rarity_weighted", "diagonal", "section_score", "cue_score"]
    groups = {}
    for label, flag in (("confirmed", 1), ("rejected", -1)):
        rows = q("SELECT result FROM pairs WHERE confirmed=?", (flag,))
        vals = {k: [] for k in keys}
        layers = {l: [] for l in cmp.LAYERS}
        for r in rows:
            res = json.loads(r["result"])
            for k in keys:
                vals[k].append(res[k])
            for l in cmp.LAYERS:
                layers[l].append(res["layer_scores"][l])
        def summ(v):
            if not v:
                return None
            v = sorted(v)
            return {"min": round(v[0], 4), "median": round(v[len(v) // 2], 4), "max": round(v[-1], 4)}
        groups[label] = {"count": len(rows), "metrics": {k: summ(v) for k, v in vals.items()}, "layers": {l: summ(v) for l, v in layers.items()}}
    return groups


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("ALBATROSS_PORT", "8417")))
