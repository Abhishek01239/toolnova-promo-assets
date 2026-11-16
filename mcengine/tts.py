"""Free, local narration with Kokoro (Apache-2.0): no API key or credits.

Model files (~350 MB) are downloaded once by scripts/get_models.sh into models/.

The default "pip" voice is tuned to sound like the series' energetic young
narrator using only Kokoro's own stock voices. It blends two built-in voices
(Eric + Fenrir), raises the pitch slightly with formant-preserving shifting,
varies pitch, pace and loudness per line from emotion tags, and finishes with
a clean "YouTuber mic" chain (EQ, presence and compression). No voice model is
trained on, or copied from, any other speaker.

Each line is synthesized separately, so line timings are exact. Word timings
inside a line are estimated from word length (good enough for captions).
"""
import os
import re
import subprocess
import numpy as np

from .voice import strip_tags, speakable

MODEL_DIR = os.environ.get("KOKORO_DIR", os.path.join(os.path.dirname(__file__), "..", "models"))

# Named voice presets. "blend" mixes Kokoro style vectors; "pitch" is semitones.
PRESETS = {
    "pip": {"blend": {"am_eric": 0.65, "am_fenrir": 0.35}, "pitch": 0.6, "speed": 1.0},
    "puck": {"blend": {"am_puck": 1.0}, "pitch": 0.0, "speed": 1.12},
}

# delivery per emotion tag -> (speed mult, pause after line s, pitch offset st, gain dB)
STYLE = {
    "shouting": (1.10, 0.30, 4.0, 2.5), "panicked": (1.14, 0.25, 3.2, 1.5),
    "excited": (1.06, 0.40, 2.4, 1.0), "intense": (1.06, 0.30, 2.0, 1.0),
    "determined": (1.00, 0.40, 0.8, 0.5), "curious": (0.98, 0.45, 1.2, 0.0),
    "relieved": (0.96, 0.45, 0.6, 0.0), "laughing": (1.00, 0.45, 1.6, 0.5),
    "suspenseful": (0.90, 0.55, -1.4, -2.5), "quieter": (0.92, 0.55, -1.6, -3.0),
    "whispering": (0.88, 0.60, -2.4, -5.5),
}
DEFAULT_STYLE = (0.98, 0.40, 0.0, 0.0)
CHUNK_PAUSE = 0.32          # dramatic pause at "..." inside a line
CHUNK_WALK = (0.0, 1.6, -1.1, 1.2, -0.6)   # pitch walk across chunks (semitones) for natural emphasis


def _style(line):
    tags = [t.strip().lower() for grp in re.findall(r"\[([^\]]*)\]", line) for t in grp.split(",")]
    for t in tags:
        for k, v in STYLE.items():
            if k in t:
                return v
    text = strip_tags(line)
    if text.endswith("!") or re.search(r"\b[A-Z]{3,}\b", text):   # exclamations / CAPS = more energy
        return (1.04, 0.35, 1.8, 1.0)
    return DEFAULT_STYLE


def _voice_style(k, voice):
    preset = PRESETS.get(voice)
    if preset is None:
        return voice, {"pitch": 0.0, "speed": 1.12}
    vec = sum(w * k.get_voice_style(name) for name, w in preset["blend"].items())
    return vec.astype(np.float32), preset


def _ffmpeg_process(a, sr, semitones=0.0, gain_db=0.0, master=False):
    """Pitch-shift (formant-preserving) and/or apply the mic chain via ffmpeg."""
    filters = []
    if abs(semitones) > 0.01:
        filters.append(f"rubberband=pitch={2 ** (semitones / 12):.5f}:formant=preserved:pitchq=quality")
    if gain_db:
        filters.append(f"volume={gain_db}dB")
    if master:
        filters += ["highpass=f=85", "equalizer=f=220:t=q:w=1.0:g=-2",
                    "equalizer=f=3000:t=q:w=1.2:g=2", "equalizer=f=9500:t=q:w=1.0:g=0.5",
                    "acompressor=threshold=-20dB:ratio=3:attack=4:release=90:makeup=3",
                    "alimiter=limit=0.95"]
    if not filters:
        return a
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                        "-af", ",".join(filters), "-f", "f32le", "-ar", str(sr), "-ac", "1", "-"],
                       input=np.ascontiguousarray(a, np.float32).tobytes(), capture_output=True, check=True)
    out = np.frombuffer(p.stdout, np.float32)
    return out[: len(a)] if len(out) > len(a) else out


def synthesize(lines, voice="pip", speed=None, lang="en-us", lead_in=0.12):
    """-> (audio float32 mono, sample_rate, words[{word,line,start,end}])"""
    from kokoro_onnx import Kokoro
    k = Kokoro(os.path.join(MODEL_DIR, "kokoro-v1.0.onnx"), os.path.join(MODEL_DIR, "voices-v1.0.bin"))
    vstyle, preset = _voice_style(k, voice)
    base_speed = speed or preset["speed"]
    sr = 24000
    parts, words, t = [np.zeros(int(sr * lead_in), np.float32)], [], lead_in
    for li, line in enumerate(lines):
        text = strip_tags(line)
        sp_mult, pause, st, gain = _style(line)
        chunks = [c.strip() for c in re.split(r"(?<=\.\.\.)\s+|(?<=[.!?])\s+(?=[A-Z])", text) if c.strip()]
        pieces = []
        for ci, chunk in enumerate(chunks):
            c, sr = k.create(speakable(chunk), voice=vstyle, speed=base_speed * sp_mult, lang=lang)
            c = _trim(c, sr)
            walk = CHUNK_WALK[ci % len(CHUNK_WALK)] if len(chunks) > 1 else 0.0
            c = _ffmpeg_process(c, sr, semitones=preset["pitch"] + st + walk, gain_db=gain)
            gap = CHUNK_PAUSE if chunk.endswith("...") else 0.12
            pieces += [c, np.zeros(int(sr * gap), np.float32)]
        a = np.concatenate(pieces[:-1])
        dur = len(a) / sr
        ws = text.split()
        weights = np.array([len(re.sub(r"\W", "", w)) + 2 for w in ws], float)
        edges = np.concatenate([[0], np.cumsum(weights)]) / weights.sum() * dur
        for i, w in enumerate(ws):
            words.append({"word": w, "line": li, "start": float(t + edges[i]),
                          "end": float(t + edges[i + 1] * 0.98)})
        parts += [a, np.zeros(int(sr * pause), np.float32)]
        t += dur + pause
    out = _ffmpeg_process(np.concatenate(parts), sr, master=True)
    return out, sr, words


def _trim(a, sr, thresh=0.01):
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0:
        return a
    s = max(0, idx[0] - int(0.02 * sr))
    e = min(len(a), idx[-1] + int(0.05 * sr))
    return a[s:e]
