"""Algorithm Albatross v1 skeleton comparison.

Aligns two song skeletons bar by bar across every layer, weights matches
by how rare each pattern is across the whole library, scores section maps
and cue points, and sweeps tempo ratios to catch sped up or slowed copies.
"""

import math
from collections import Counter

import numpy as np
import librosa

LAYER_WEIGHTS = {"harmony": 0.3, "chords": 0.2, "drums": 0.2, "melody": 0.2, "energy": 0.1}
LAYER_THRESH = {"harmony": 0.85, "chords": 0.75, "drums": 0.8, "melody": 0.5, "energy": 0.85}
LAYERS = list(LAYER_WEIGHTS.keys())


def _unit(m, axis=1):
    n = np.linalg.norm(m, axis=axis, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def _shift_chord(c, k):
    if c < 0:
        return -1
    return ((c % 12 + k) % 12) + (12 if c >= 12 else 0)


def _shift_pc(p, k):
    return -1 if p < 0 else (p + k) % 12


def best_transposition(a, b):
    ca = np.array(a["bar_chroma"]).mean(axis=0)
    cb = np.array(b["bar_chroma"]).mean(axis=0)
    best_k, best_r = 0, -2.0
    for k in range(12):
        r = float(np.corrcoef(ca, np.roll(cb, k))[0, 1])
        if r > best_r:
            best_k, best_r = k, r
    return best_k


def _regrid(sk, factor):
    """Merge pairs of bars (factor 0.5) or split bars (factor 2) to fix half or double time grids."""
    if factor == 1:
        return sk
    out = dict(sk)
    if factor == 0.5:
        n = sk["n_bars"] // 2
        idx = [(2 * i, 2 * i + 1) for i in range(n)]
        out["bar_chroma"] = [list(_unit(np.array([np.add(sk["bar_chroma"][i], sk["bar_chroma"][j])]))[0]) for i, j in idx]
        out["bar_chords"] = [sk["bar_chords"][i][::2] + sk["bar_chords"][j][::2] for i, j in idx]
        out["bar_main_chord"] = [sk["bar_main_chord"][i] for i, _ in idx]
        out["bar_melody"] = [sk["bar_melody"][i][::2] + sk["bar_melody"][j][::2] for i, j in idx]
        out["bar_energy"] = [(sk["bar_energy"][i] + sk["bar_energy"][j]) / 2 for i, j in idx]
        out["bar_drum"] = [sk["bar_drum"][i][::2] + sk["bar_drum"][j][::2] for i, j in idx]
        out["bar_times"] = [sk["bar_times"][2 * i] for i in range(n)] + [sk["bar_times"][2 * n]]
        out["sections"] = [dict(s, start=s["start"] // 2, end=s["end"] // 2, bars=max(1, (s["end"] - s["start"]) // 2)) for s in sk["sections"]]
        out["cues"] = [dict(c, bar=c["bar"] // 2) for c in sk["cues"]]
        out["n_bars"] = n
    else:
        n = sk["n_bars"] * 2
        out["bar_chroma"] = [c for c in sk["bar_chroma"] for _ in (0, 1)]
        out["bar_chords"] = [h for c in sk["bar_chords"] for h in ([c[0], c[0], c[1], c[1]], [c[2], c[2], c[3], c[3]])]
        out["bar_main_chord"] = [c for c in sk["bar_main_chord"] for _ in (0, 1)]
        out["bar_melody"] = [h for c in sk["bar_melody"] for h in ([c[0], c[0], c[1], c[1]], [c[2], c[2], c[3], c[3]])]
        out["bar_energy"] = [e for e in sk["bar_energy"] for _ in (0, 1)]
        out["bar_drum"] = [h for d in sk["bar_drum"] for h in ([x for x in d[:8] for _ in (0, 1)], [x for x in d[8:] for _ in (0, 1)])]
        bt = sk["bar_times"]
        out["bar_times"] = [t for i in range(len(bt) - 1) for t in (bt[i], (bt[i] + bt[i + 1]) / 2)] + [bt[-1]]
        out["sections"] = [dict(s, start=s["start"] * 2, end=s["end"] * 2, bars=s["bars"] * 2) for s in sk["sections"]]
        out["cues"] = [dict(c, bar=c["bar"] * 2) for c in sk["cues"]]
        out["n_bars"] = n
    return out


def layer_matrices(a, b, k):
    ca = _unit(np.array(a["bar_chroma"]))
    cb = _unit(np.roll(np.array(b["bar_chroma"]), k, axis=1))
    harmony = np.clip(ca @ cb.T, 0, 1)

    cha = np.array(a["bar_chords"])
    chb = np.array([[_shift_chord(c, k) for c in row] for row in b["bar_chords"]])
    eq = (cha[:, None, :] == chb[None, :, :]) & (cha[:, None, :] >= 0)
    chords = eq.mean(axis=2)

    da = _unit(np.array(a["bar_drum"]))
    db = _unit(np.array(b["bar_drum"]))
    drums = np.clip(da @ db.T, 0, 1)

    ma = np.array(a["bar_melody"])
    mb = np.array([[_shift_pc(p, k) for p in row] for row in b["bar_melody"]])
    valid = (ma[:, None, :] >= 0) & (mb[None, :, :] >= 0)
    meq = (ma[:, None, :] == mb[None, :, :]) & valid
    cnt = valid.sum(axis=2)
    melody = np.where(cnt > 0, meq.sum(axis=2) / np.maximum(cnt, 1), 0.0)

    ea = np.array(a["bar_energy"])
    eb = np.array(b["bar_energy"])
    energy = 1.0 - np.abs(ea[:, None] - eb[None, :])

    return {"harmony": harmony, "chords": chords, "drums": drums, "melody": melody, "energy": energy}


def _bar_tokens(sk):
    """Transposition free tokens used for rarity weighting."""
    mains = sk["bar_main_chord"]
    tokens = []
    n = len(mains)
    for i in range(n):
        gram = mains[i:i + 4]
        if len(gram) < 4 or gram[0] < 0:
            ch = "c:none"
        else:
            r0 = gram[0] % 12
            ch = "c:" + ",".join("n" if c < 0 else str((c % 12 - r0) % 12) + ("m" if c >= 12 else "") for c in gram)
        d = sk["bar_drum"][i]
        dr = "d:" + "".join("1" if x > 0.35 else "0" for x in d)
        tokens.append((ch, dr))
    return tokens


def corpus_stats(skeletons):
    df_c, df_d = Counter(), Counter()
    for sk in skeletons:
        toks = _bar_tokens(sk)
        df_c.update(set(t[0] for t in toks))
        df_d.update(set(t[1] for t in toks))
    return {"n": max(1, len(skeletons)), "chord": df_c, "drum": df_d}


def _rarity(sk, stats):
    n = stats["n"]
    out = []
    for ch, dr in _bar_tokens(sk):
        ic = math.log((n + 1) / (1 + stats["chord"].get(ch, 0))) + 0.15
        idr = math.log((n + 1) / (1 + stats["drum"].get(dr, 0))) + 0.15
        out.append(ic + idr)
    return np.array(out)


def _section_score(a, b):
    la = [s["bars"] for s in a["sections"]]
    lb = [s["bars"] for s in b["sections"]]
    n, m = len(la), len(lb)
    d = np.zeros((n + 1, m + 1))
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if abs(la[i - 1] - lb[j - 1]) <= 1 else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + sub)
    length_score = 1.0 - d[n, m] / max(n, m, 1)

    def canon(labels):
        mp = {}
        return "".join(mp.setdefault(x, chr(65 + len(mp))) for x in labels)

    pa = canon([s["label"] for s in a["sections"]])
    pb = canon([s["label"] for s in b["sections"]])
    pattern_same = pa == pb
    return float(max(0.0, length_score)), pattern_same, pa, pb


def _cue_score(a, b, a_to_b):
    ca = [c["bar"] for c in a["cues"]]
    cb = [c["bar"] for c in b["cues"]]
    if not ca and not cb:
        return 1.0, []
    if not ca or not cb:
        return 0.0, []
    hits = []
    for cbar in ca:
        mapped = a_to_b.get(cbar)
        if mapped is None:
            continue
        near = [x for x in cb if abs(x - mapped) <= 1]
        if near:
            hits.append({"a_bar": cbar, "b_bar": near[0]})
    return len(hits) / max(len(ca), len(cb)), hits


def tempo_sweep(a, b, k, lo=0.8, hi=1.25, step=0.01, max_offset_s=40.0):
    """Stretch song b in time across a ratio range and find where it lines up best with song a."""
    fps = a["sweep_fps"]
    A = _unit(np.array(a["sweep_chroma"]))
    B0 = np.roll(np.array(b["sweep_chroma"]), k, axis=1)
    nb0 = B0.shape[0]
    curve = []
    max_off = int(max_offset_s * fps)
    for r in np.arange(lo, hi + 1e-9, step):
        nb = max(8, int(round(nb0 * r)))
        src = np.linspace(0, nb0 - 1, nb)
        B = np.vstack([np.interp(src, np.arange(nb0), B0[:, c]) for c in range(12)]).T
        B = _unit(B)
        S = A @ B.T
        ia, ib = np.indices(S.shape)
        diag = (ib - ia).ravel()
        sums = np.bincount(diag - diag.min(), weights=S.ravel())
        counts = np.bincount(diag - diag.min())
        offsets = np.arange(diag.min(), diag.max() + 1)
        min_overlap = 0.5 * min(A.shape[0], nb)
        ok = (counts >= min_overlap) & (np.abs(offsets) <= max_off)
        if not ok.any():
            curve.append({"ratio": round(float(r), 3), "score": 0.0, "offset_s": 0.0})
            continue
        means = np.where(ok, sums / np.maximum(counts, 1), -1)
        j = int(np.argmax(means))
        curve.append({"ratio": round(float(r), 3), "score": round(float(means[j]), 4), "offset_s": round(float(offsets[j] / fps), 2)})
    best = max(curve, key=lambda c: c["score"])
    return {"best_ratio": best["ratio"], "best_score": best["score"], "best_offset_s": best["offset_s"], "curve": curve}


def _fold_tempo(t):
    while t > 0 and t < 80:
        t *= 2
    while t > 160:
        t /= 2
    return t


def compare(a, b, stats=None, sweep=True):
    k = best_transposition(a, b)
    best = None
    for factor in (1, 0.5, 2):
        bb = _regrid(b, factor)
        if bb["n_bars"] < 8 or bb["n_bars"] > 4 * a["n_bars"] or a["n_bars"] > 4 * bb["n_bars"]:
            continue
        mats = layer_matrices(a, bb, k)
        combined = sum(LAYER_WEIGHTS[l] * mats[l] for l in LAYERS)
        D, wp = librosa.sequence.dtw(C=1.0 - combined)
        cost = D[-1, -1] / len(wp)
        if best is None or cost < best[0]:
            best = (cost, factor, bb, mats, combined, wp[::-1])
    cost, factor, bb, mats, combined, path = best

    a_to_b = {}
    steps = []
    for i, j in path:
        i, j = int(i), int(j)
        if i not in a_to_b:
            a_to_b[i] = j
            flags = {l: bool(mats[l][i, j] >= LAYER_THRESH[l]) for l in LAYERS}
            steps.append((i, j, flags))

    diag_steps = sum(1 for p, q in zip(path[:-1], path[1:]) if q[0] - p[0] == 1 and q[1] - p[1] == 1)
    diagonal = diag_steps / max(1, len(path) - 1)

    layer_scores = {l: round(float(np.mean([s[2][l] for s in steps])), 4) for l in LAYERS}
    lock = np.array([all(s[2].values()) for s in steps])
    strong = np.array([sum(s[2].values()) >= 4 for s in steps])

    longest, run, prev = 0, 0, None
    run_start, best_start = 0, 0
    for idx, (i, j, flags) in enumerate(steps):
        ok = sum(flags.values()) >= 4
        cont = prev is not None and j == prev + 1
        if ok:
            run = run + 1 if cont and run > 0 else 1
            if run == 1:
                run_start = i
            if run > longest:
                longest, best_start = run, run_start
        else:
            run = 0
        prev = j

    if stats is not None:
        wa = _rarity(a, stats)
        weighted = float((wa * strong).sum() / (wa.sum() or 1))
    else:
        weighted = float(strong.mean())

    section_score, pattern_same, pa, pb = _section_score(a, bb)
    cue_score, cue_hits = _cue_score(a, bb, a_to_b)

    ta, tb = _fold_tempo(a["tempo"]), _fold_tempo(b["tempo"])
    bpm_ratio = tb / ta if ta else 1.0

    sw = tempo_sweep(a, b, k) if sweep else None

    mean_layer = float(np.mean([layer_scores[l] for l in LAYERS]))
    formula = 0.35 * weighted + 0.15 * diagonal + 0.2 * section_score + 0.15 * cue_score + 0.15 * mean_layer
    coverage = float(strong.mean())

    if coverage >= 0.7 and section_score >= 0.75 and cue_score >= 0.6 and diagonal >= 0.8:
        verdict = "ABSOLUTE MATCH"
    elif formula >= 0.55:
        verdict = "TEMPLATE MATCH"
    elif formula >= 0.35:
        verdict = "SHARED PARTS"
    else:
        verdict = "DIFFERENT"

    return {
        "verdict": verdict,
        "formula_score": round(formula, 4),
        "lockstep_coverage": round(coverage, 4),
        "all_layer_coverage": round(float(lock.mean()), 4),
        "rarity_weighted": round(weighted, 4),
        "diagonal": round(diagonal, 4),
        "longest_run_bars": int(longest),
        "longest_run_start_bar": int(best_start),
        "layer_scores": layer_scores,
        "section_score": round(section_score, 4),
        "section_pattern_same": pattern_same,
        "section_pattern_a": pa,
        "section_pattern_b": pb,
        "cue_score": round(cue_score, 4),
        "cue_hits": cue_hits,
        "transpose_semitones": int((-k) % 12 if (-k) % 12 <= 6 else (-k) % 12 - 12),
        "grid_factor": factor,
        "tempo_a": a["tempo"],
        "tempo_b": b["tempo"],
        "bpm_ratio": round(bpm_ratio, 4),
        "sweep": sw,
        "alignment": [[i, j, sum(1 << n for n, l in enumerate(LAYERS) if f[l])] for i, j, f in steps],
        "layers": LAYERS,
        "sections_a": a["sections"],
        "sections_b": bb["sections"],
        "cues_a": a["cues"],
        "cues_b": bb["cues"],
        "n_bars_a": a["n_bars"],
        "n_bars_b": bb["n_bars"],
    }


def vector_similarity(va, vb):
    return float(np.dot(va, vb))
