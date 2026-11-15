"""Free, local narration with Kokoro (Apache-2.0) — no API key or credits.

Model files (~350 MB) are downloaded once by scripts/get_models.sh into models/.
Each script line is synthesized separately, so line timings are exact; word
timings inside a line are estimated from word length (good enough for captions).
Audio tags like "[excited]" are used only to pick speed/pause per line.
"""
import os
import re
import numpy as np

from .voice import strip_tags

MODEL_DIR = os.environ.get("KOKORO_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

# delivery hints from ElevenLabs-style tags -> (speed, pause after line in s)
STYLE = {
    "excited": (1.15, 0.25), "shouting": (1.18, 0.15), "panicked": (1.2, 0.15), "intense": (1.12, 0.15),
    "whispering": (0.95, 0.35), "suspenseful": (1.0, 0.35), "curious": (1.08, 0.25),
    "determined": (1.08, 0.2), "relieved": (1.05, 0.25),
}


def _style(line, default_speed):
    tags = [t.strip().lower() for t in re.findall(r"\[([^\]]*)\]", line) for t in t.split(",")]
    for t in tags:
        for k, v in STYLE.items():
            if k in t:
                return v
    return default_speed, 0.22


def synthesize(lines, voice="am_puck", speed=1.12, lang="en-us", lead_in=0.12):
    """-> (audio float32 mono, sample_rate, words[{word,line,start,end}])"""
    from kokoro_onnx import Kokoro
    k = Kokoro(os.path.join(MODEL_DIR, "kokoro-v1.0.onnx"), os.path.join(MODEL_DIR, "voices-v1.0.bin"))
    parts, words, t = [], [], lead_in
    sr = 24000
    parts.append(np.zeros(int(sr * lead_in), np.float32))
    for li, line in enumerate(lines):
        text = strip_tags(line)
        sp, pause = _style(line, speed)
        a, sr = k.create(text, voice=voice, speed=sp, lang=lang)
        a = _trim(a, sr)
        dur = len(a) / sr
        ws = text.split()
        weights = np.array([len(re.sub(r"\W", "", w)) + 2 for w in ws], float)
        edges = np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * dur
        for i, w in enumerate(ws):
            words.append({"word": w, "line": li, "start": float(t + edges[i]), "end": float(t + edges[i + 1] * 0.98)})
        parts += [a, np.zeros(int(sr * pause), np.float32)]
        t += dur + pause
    return np.concatenate(parts), sr, words


def _trim(a, sr, thresh=0.01):
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    s = max(0, idx[0] - int(0.02 * sr))
    e = min(len(a), idx[-1] + int(0.05 * sr))
    return a[s:e]
