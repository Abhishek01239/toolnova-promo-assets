"""Fit a re-voiced narration (one wav per line) into the original line slots, so the
already-rendered video (cuts, captions, effects) stays in sync without re-rendering.

Each new line starts exactly where the old line started and is time-stretched (formant-
preserving rubberband) toward the old line's spoken length: gently (±15%) for timing, more
only if the line would otherwise run into the next one.
"""
import json
import os
import subprocess
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 24000


def stretch(a, sr, tempo):
    if abs(tempo - 1) < 0.01:
        return a
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                        "-af", f"rubberband=tempo={tempo:.4f}:formant=preserved:pitchq=quality",
                        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-"],
                       input=np.ascontiguousarray(a, np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32)


def resample(a, sr_in, sr_out):
    if sr_in == sr_out:
        return a.astype(np.float32)
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(sr_in, sr_out)
    return resample_poly(a, sr_out // g, sr_in // g).astype(np.float32)


def main(lines_dir, out_wav):
    d = json.load(open(os.path.join(HERE, "assets", "words.json")))
    words, total = d["words"], d["duration"]
    n = 1 + max(w["line"] for w in words)
    ls = [min(w["start"] for w in words if w["line"] == i) for i in range(n)]
    le = [max(w["end"] for w in words if w["line"] == i) for i in range(n)]
    out = np.zeros(int((total + 1.0) * SR), np.float32)
    report = []
    for i in range(n):
        a, sr = sf.read(os.path.join(lines_dir, f"line_{i:03d}.wav"), dtype="float32")
        if a.ndim > 1:
            a = a.mean(1)
        a = resample(a, sr, SR)
        d_new, d_old = len(a) / SR, le[i] - ls[i]
        room = (ls[i + 1] if i + 1 < n else total) - ls[i] - 0.08
        tempo = float(np.clip(d_new / d_old, 0.85, 1.15))     # gentle match to the old pacing
        if d_new / tempo > room:                               # must not overlap the next line
            tempo = min(1.3, d_new / room)
        a = stretch(a, SR, tempo)
        s = int(ls[i] * SR)
        k = min(len(a), len(out) - s)
        out[s:s + k] += a[:k]
        report.append((i, round(d_old, 2), round(d_new, 2), round(tempo, 3), round(len(a) / SR, 2), round(room, 2)))
    out = out[: int(total * SR)]
    out = out / (np.abs(out).max() + 1e-9) * 0.9
    sf.write(out_wav, out, SR)
    for r in report:
        flag = "  <-- squeezed" if r[3] > 1.15 else ""
        print("line %2d old %5.2fs new %5.2fs tempo %.3f -> %5.2fs (room %5.2fs)%s" % (*r, flag))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
