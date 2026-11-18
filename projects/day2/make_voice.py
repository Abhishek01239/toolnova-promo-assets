"""Narrate Day 2 with the channel voice (puck_classic = Hardcore Parts 2-5 voice).

Each line is synthesized on its own with no time-stretching. Chapter title-card gaps and
action-beat holds from script.HOLDS are inserted as silence. Writes assets/narration.wav and
assets/words.json (absolute word timings) used by story.py.
"""
import json
import os
import sys

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mcengine import tts_classic        # noqa: E402
import script as S                       # noqa: E402

INTRO_GAP = 0.25
CHAPTER_GAP = 3.0
HOLD_SCALE = float(os.environ.get("HOLD_SCALE", "1.0"))

sr = 24000
parts, words, chapters = [np.zeros(int(sr * INTRO_GAP), np.float32)], [], []
t = INTRO_GAP
li = 0
for ci, (num, title, lines) in enumerate(S.CHAPTERS):
    if ci > 0:
        parts.append(np.zeros(int(sr * CHAPTER_GAP), np.float32))
        chapters.append({"num": num, "title": title, "card_start": t, "card_end": t + CHAPTER_GAP})
        t += CHAPTER_GAP
    else:
        chapters.append({"num": num, "title": title, "card_start": None, "card_end": None})
    for _key, line in lines:
        a, sr, ws = tts_classic.synthesize([line], voice="am_puck", lead_in=0.0)
        # synthesize() appends its own per-line pause; keep it, then add any action-beat hold
        hold = S.HOLDS.get(li, 0.0) * HOLD_SCALE
        for w in ws:
            words.append({"text": w["word"], "word": w["word"], "line": li,
                          "start": w["start"] + t, "end": w["end"] + t, "type": "word"})
        parts += [a.astype(np.float32), np.zeros(int(sr * hold), np.float32)]
        t += len(a) / sr + hold
        li += 1
        print(f"line {li}/{len(S.LINES)}  t={t:6.1f}s", flush=True)
audio = np.concatenate(parts)
sf.write(os.path.join(HERE, "assets", "narration.wav"), audio, sr)
json.dump({"words": words, "chapters": chapters, "duration": len(audio) / sr},
          open(os.path.join(HERE, "assets", "words.json"), "w"), indent=0)
print("duration %.1fs (%d:%02d), %d words" % (len(audio) / sr, len(audio) / sr // 60, len(audio) / sr % 60, len(words)))
