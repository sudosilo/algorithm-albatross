"""Algorithm Albatross v1 skeleton extraction.

Turns an instrumental audio file into a song skeleton measured in bars:
beat grid, key, per bar chroma, chords, drum pattern, top line melody,
energy, sections, and cue points at every transition.
"""

import json
import subprocess

import numpy as np
import librosa

SR = 22050
HOP = 512
SWEEP_FPS = 4.0
SKELETON_VERSION = 1

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def load_audio(path):
    """Decode any audio file to mono float32 at SR using ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-ac", "1", "-ar", str(SR), "-f", "f32le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError("ffmpeg could not decode file: " + proc.stderr.decode("utf8", "ignore")[-300:])
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if y.size < SR * 10:
        raise RuntimeError("audio is shorter than ten seconds")
    return y


def _unit(v, axis=0):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def _chord_templates():
    temps = []
    for quality in (0, 1):
        third = 4 if quality == 0 else 3
        for root in range(12):
            t = np.zeros(12)
            t[root] = 1.0
            t[(root + third) % 12] = 1.0
            t[(root + 7) % 12] = 1.0
            temps.append(t / np.linalg.norm(t))
    return np.array(temps)


CHORD_TEMPLATES = _chord_templates()


def chord_name(idx):
    if idx < 0:
        return "N"
    root = idx % 12
    return NOTE_NAMES[root] + ("" if idx < 12 else "m")


def estimate_key(chroma_mean):
    best = (-2.0, 0, "major")
    for tonic in range(12):
        for mode, prof in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(chroma_mean, np.roll(prof, tonic))[0, 1]
            if r > best[0]:
                best = (float(r), tonic, mode)
    return best[1], best[2], best[0]


def _segment_mean(mat, bounds):
    out = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        e = max(e, s + 1)
        out.append(mat[:, s:e].mean(axis=1))
    return np.array(out).T


def _foote_novelty(ssm, half):
    n = ssm.shape[0]
    if n < 2 * half + 1:
        return np.zeros(n)
    k = np.ones((2 * half, 2 * half))
    k[:half, half:] = -1
    k[half:, :half] = -1
    g = np.exp(-0.5 * (np.linspace(-1.5, 1.5, 2 * half) ** 2))
    k *= np.outer(g, g)
    pad = np.pad(ssm, half, mode="edge")
    nov = np.zeros(n)
    for i in range(n):
        nov[i] = np.sum(pad[i:i + 2 * half, i:i + 2 * half] * k)
    nov[nov < 0] = 0
    if nov.max() > 0:
        nov /= nov.max()
    return nov


def analyze_file(path):
    y = load_audio(path)
    duration = float(len(y) / SR)
    y_h, y_p = librosa.effects.hpss(y)

    onset_env = librosa.onset.onset_strength(y=y_p, sr=SR, hop_length=HOP)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP, units="frames")
    tempo = float(np.atleast_1d(tempo)[0])
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 24:
        raise RuntimeError("not enough beats found to build a bar grid")

    chroma = librosa.feature.chroma_cqt(y=y_h, sr=SR, hop_length=HOP)
    n_frames = chroma.shape[1]
    beats = beats[beats < n_frames - 1]
    beat_bounds = list(beats) + [min(n_frames, beats[-1] + int(np.median(np.diff(beats))))]
    beat_chroma = _unit(_segment_mean(chroma, beat_bounds))
    n_beats = beat_chroma.shape[1]

    scores = CHORD_TEMPLATES @ beat_chroma
    beat_chords = scores.argmax(axis=0)
    weak = scores.max(axis=0) < 0.55
    beat_chords[weak] = -1

    # top line melody: strongest upper register pitch per beat from the harmonic part
    # harmonics of lower notes get subtracted so the line follows the played note, not its overtones
    cqt = np.abs(librosa.cqt(y_h, sr=SR, hop_length=HOP, fmin=librosa.note_to_hz("C3"), n_bins=60, bins_per_octave=12))
    cqt = cqt[:, :n_frames]
    beat_cqt = _segment_mean(cqt, beat_bounds)
    clean = beat_cqt.copy()
    for semis, amt in ((12, 0.9), (19, 0.8), (24, 0.7), (28, 0.6), (31, 0.5)):
        clean[semis:] -= amt * beat_cqt[:-semis]
    clean = np.clip(clean, 0, None)
    upper = clean[12:]
    peak = upper.max(axis=0)
    strong = upper >= 0.55 * peak[None, :]
    top_idx = upper.shape[0] - 1 - np.argmax(strong[::-1], axis=0)
    mel_bins = top_idx + 12
    mel_strength = upper[top_idx, np.arange(upper.shape[1])]
    beat_melody = (mel_bins % 12).astype(int)
    beat_melody[mel_strength < np.percentile(mel_strength, 20)] = -1

    rms = librosa.feature.rms(y=y, hop_length=HOP)[0][:n_frames]
    beat_rms = np.array([rms[s:max(e, s + 1)].mean() for s, e in zip(beat_bounds[:-1], beat_bounds[1:])])

    beat_onset = np.array([onset_env[s:max(e, s + 1)].max() for s, e in zip(beat_bounds[:-1], beat_bounds[1:])])

    # downbeat phase in four four time
    chord_change = np.r_[0, (np.diff(beat_chords) != 0).astype(float)]
    chroma_change = np.r_[0, 1 - np.sum(beat_chroma[:, 1:] * beat_chroma[:, :-1], axis=0)]
    on_norm = beat_onset / (beat_onset.max() or 1)
    phase_score = []
    for p in range(4):
        idx = np.arange(p, n_beats, 4)
        phase_score.append(chord_change[idx].mean() + chroma_change[idx].mean() * 2 + on_norm[idx].mean())
    phase = int(np.argmax(phase_score))

    bar_starts = list(range(phase, n_beats - 3, 4))
    n_bars = len(bar_starts)
    if n_bars < 8:
        raise RuntimeError("song too short for structure analysis")

    beat_times = librosa.frames_to_time(np.array(beat_bounds), sr=SR, hop_length=HOP)
    bar_times = [float(beat_times[b]) for b in bar_starts] + [float(beat_times[bar_starts[-1] + 4])]

    bar_chroma, bar_chords, bar_main, bar_melody, bar_energy, bar_drum, bar_mfcc_idx = [], [], [], [], [], [], []
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)[:, :n_frames]
    mfcc_beats = _segment_mean(mfcc, beat_bounds)
    bar_mfcc = []
    for b in bar_starts:
        sl = slice(b, b + 4)
        bar_chroma.append(_unit(beat_chroma[:, sl].mean(axis=1)))
        ch = beat_chords[sl]
        bar_chords.append([int(c) for c in ch])
        valid = ch[ch >= 0]
        bar_main.append(int(np.bincount(valid).argmax()) if valid.size else -1)
        bar_melody.append([int(m) for m in beat_melody[sl]])
        bar_energy.append(float(beat_rms[sl].mean()))
        bar_mfcc.append(mfcc_beats[:, sl].mean(axis=1))
        # sixteen step drum pattern from the percussive onset envelope
        f0 = beat_bounds[b]
        f1 = beat_bounds[b + 4]
        steps = np.linspace(f0, f1, 17)
        pattern = []
        for s0, s1 in zip(steps[:-1], steps[1:]):
            a0, a1 = int(round(s0)), max(int(round(s1)), int(round(s0)) + 1)
            pattern.append(float(onset_env[a0:a1].max()) if a1 <= len(onset_env) else 0.0)
        bar_drum.append(pattern)

    bar_chroma = np.array(bar_chroma)
    bar_energy = np.array(bar_energy)
    bar_energy = bar_energy / (bar_energy.max() or 1)
    bar_drum = np.array(bar_drum)
    drum_peak = np.percentile(bar_drum, 98) or 1
    bar_drum = np.clip(bar_drum / drum_peak, 0, 1)
    bar_mfcc = np.array(bar_mfcc)
    bar_mfcc = (bar_mfcc - bar_mfcc.mean(axis=0)) / (bar_mfcc.std(axis=0) + 1e-6)

    # sections from bar level self similarity
    feat = np.hstack([bar_chroma * 2.0, _unit(bar_mfcc, axis=1), bar_energy[:, None], _unit(bar_drum, axis=1)])
    feat = _unit(feat, axis=1)
    ssm = feat @ feat.T
    novelty = _foote_novelty(ssm, 4)
    thresh = novelty.mean() + 0.5 * novelty.std()
    boundaries = [0]
    for i in range(2, n_bars - 2):
        if novelty[i] >= thresh and novelty[i] == novelty[max(0, i - 3):i + 4].max():
            if i - boundaries[-1] >= 4:
                boundaries.append(i)
    if n_bars - boundaries[-1] < 4 and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(n_bars)

    sections = []
    label_feats = []
    for s, e in zip(boundaries[:-1], boundaries[1:]):
        f = _unit(feat[s:e].mean(axis=0))
        label = None
        for lab, lf in label_feats:
            if float(f @ lf) > 0.93:
                label = lab
                break
        if label is None:
            label = chr(ord("A") + len(label_feats))
            label_feats.append((label, f))
        sections.append({
            "start": int(s), "end": int(e), "bars": int(e - s), "label": label,
            "start_time": round(bar_times[s], 3), "end_time": round(bar_times[e], 3),
        })

    density = (bar_drum > 0.35).sum(axis=1).astype(float)
    med_density = float(np.median(density)) or 1.0
    med_energy = float(np.median(bar_energy)) or 1.0
    downbeat_onset = np.array([beat_onset[b] for b in bar_starts])
    med_down = float(np.median(downbeat_onset)) or 1.0
    cues = []
    for b in boundaries[1:-1]:
        prev = b - 1
        late = bar_drum[prev, 8:]
        cues.append({
            "bar": int(b),
            "time": round(bar_times[b], 3),
            "fill": round(float(density[prev] / med_density), 3),
            "late_hits": round(float((late > 0.35).sum() / 8.0), 3),
            "dropout": round(float(1.0 - bar_energy[prev] / med_energy), 3),
            "crash": round(float(downbeat_onset[b] / med_down), 3),
            "novelty": round(float(novelty[b]), 3),
        })

    key_tonic, key_mode, key_conf = estimate_key(chroma.mean(axis=1))

    # fixed frame rate chroma for the tempo sweep
    hop_sweep = max(1, int(round(SR / HOP / SWEEP_FPS)))
    sweep_chroma = _unit(_segment_mean(chroma, list(range(0, n_frames, hop_sweep)) + [n_frames]))

    skeleton = {
        "version": SKELETON_VERSION,
        "duration": round(duration, 3),
        "tempo": round(tempo, 2),
        "key_tonic": int(key_tonic),
        "key_mode": key_mode,
        "key_name": NOTE_NAMES[key_tonic] + (" minor" if key_mode == "minor" else " major"),
        "key_confidence": round(key_conf, 3),
        "phase": phase,
        "n_bars": n_bars,
        "bar_times": [round(t, 3) for t in bar_times],
        "bar_chroma": np.round(bar_chroma, 4).tolist(),
        "bar_chords": bar_chords,
        "bar_main_chord": bar_main,
        "bar_chord_names": [chord_name(c) for c in bar_main],
        "bar_melody": bar_melody,
        "bar_energy": np.round(bar_energy, 4).tolist(),
        "bar_drum": np.round(bar_drum, 3).tolist(),
        "novelty": np.round(novelty, 3).tolist(),
        "sections": sections,
        "cues": cues,
        "sweep_fps": SR / HOP / hop_sweep,
        "sweep_chroma": np.round(sweep_chroma.T, 3).tolist(),
    }
    skeleton["vector"] = summary_vector(skeleton)
    return skeleton


def summary_vector(sk):
    """Transposition invariant fingerprint for fast candidate retrieval."""
    chroma = np.array(sk["bar_chroma"]).mean(axis=0)
    chroma = np.roll(chroma, -sk["key_tonic"])
    drum = np.array(sk["bar_drum"]).mean(axis=0)
    motion = np.zeros(24)
    mains = [c for c in sk["bar_main_chord"] if c >= 0]
    for a, b in zip(mains[:-1], mains[1:]):
        if a != b:
            motion[((b % 12) - (a % 12)) % 12 + (12 if b >= 12 else 0)] += 1
    if motion.sum():
        motion /= motion.sum()
    lens = np.zeros(6)
    for s in sk["sections"]:
        lens[min(5, (s["bars"] - 1) // 4)] += 1
    if lens.sum():
        lens /= lens.sum()
    tempo_band = np.zeros(8)
    t = sk["tempo"]
    while t > 0 and t < 80:
        t *= 2
    while t > 160:
        t /= 2
    tempo_band[min(7, int((t - 80) / 10))] = 1
    v = np.concatenate([_unit(chroma) * 1.2, _unit(drum), motion, lens, tempo_band * 0.3])
    return np.round(_unit(v), 5).tolist()


if __name__ == "__main__":
    import sys
    print(json.dumps({k: v for k, v in analyze_file(sys.argv[1]).items() if k not in ("sweep_chroma", "bar_chroma", "bar_drum")}, indent=1))
