"""Narration helpers: script parsing, word-timing alignment, silence-based fallback.

Script lines may contain ElevenLabs v3/v4 audio tags like "[excited]"; those are
spoken as direction only and are stripped from subtitles.
"""
import re
import difflib
import numpy as np


def strip_tags(line):
    return re.sub(r"\[[^\]]*\]\s*", "", line).strip()


# Real abbreviations that SHOULD be spelled out letter by letter.
ACRONYMS = {"TNT", "MLG", "OK", "TV", "XP", "HP", "AFK", "OP", "NPC", "PVP", "DIY", "USA", "UK", "AI"}


def speakable(text):
    """Text for the TTS engine: ALL-CAPS emphasis words ('LIT', 'SKY', 'BOOM') become normal
    words so they are read, not spelled out. Captions keep the original caps."""
    def fix(m):
        w = m.group(0)
        return w if w in ACRONYMS else w.lower()
    return re.sub(r"\b[A-Z]{2,}\b", fix, text)


def script_words(lines):
    """-> list of dicts {word, line} with display text taken from the script."""
    out = []
    for li, line in enumerate(lines):
        for w in strip_tags(line).split():
            out.append({"word": w, "line": li})
    return out


def norm(w):
    w = w.lower().replace("’", "'")
    w = re.sub(r"[^a-z0-9']", "", w)
    nums = {"200": "twohundred", "two": "two", "3": "three"}
    return nums.get(w, w)


def align(lines, asr_words):
    """Give every script word a start/end using ASR word timings.

    asr_words: [{"text"/"word", "start", "end"}]. Unmatched script words are
    interpolated between matched neighbours.
    """
    sw = script_words(lines)
    aw = [w for w in asr_words if (w.get("type", "word") == "word")
          and not w.get("text", w.get("word", "")).startswith("[")
          and not w.get("text", w.get("word", "")).endswith("]")]
    a_tok = [norm(w.get("text", w.get("word", ""))) for w in aw]
    s_tok = [norm(w["word"]) for w in sw]
    sm = difflib.SequenceMatcher(a=s_tok, b=a_tok, autojunk=False)
    times = [None] * len(sw)
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            a = aw[blk.b + k]
            times[blk.a + k] = (float(a["start"]), float(a["end"]))
    # handle replaced spans (e.g. "two hundred" vs "200") proportionally
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and j2 > j1:
            t0, t1 = float(aw[j1]["start"]), float(aw[j2 - 1]["end"])
            n = i2 - i1
            for k in range(n):
                times[i1 + k] = (t0 + (t1 - t0) * k / n, t0 + (t1 - t0) * (k + 1) / n)
    # interpolate anything still missing
    known = [i for i, t in enumerate(times) if t is not None]
    if not known:
        raise ValueError("alignment failed: no matching words")
    for i in range(len(times)):
        if times[i] is None:
            prev = max([k for k in known if k < i], default=None)
            nxt = min([k for k in known if k > i], default=None)
            if prev is None:
                times[i] = (max(0, times[nxt][0] - 0.3), times[nxt][0])
            elif nxt is None:
                times[i] = (times[prev][1], times[prev][1] + 0.3)
            else:
                gap0, gap1 = times[prev][1], times[nxt][0]
                span = nxt - prev
                f0, f1 = (i - prev - 1) / (span - 1 + 1e-9), (i - prev) / (span - 1 + 1e-9)
                times[i] = (gap0 + (gap1 - gap0) * min(f0, 1), gap0 + (gap1 - gap0) * min(f1, 1))
    for w, t in zip(sw, times):
        w["start"], w["end"] = t
    return sw


def line_times(words, n_lines):
    out = []
    for li in range(n_lines):
        ws = [w for w in words if w["line"] == li]
        out.append((ws[0]["start"], ws[-1]["end"]))
    return out


def silence_fallback(audio, sr, lines, thresh_db=-38, min_gap=0.28):
    """Estimate word timings without ASR: split speech at pauses, map pauses to
    line breaks, distribute words by character count inside each line."""
    mono = np.abs(audio).mean(axis=1)
    hop = int(sr * 0.01)
    fr = mono[: len(mono) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-9)
    speech = db > thresh_db
    segs, start = [], None
    for i, s in enumerate(speech):
        if s and start is None:
            start = i
        if not s and start is not None:
            segs.append([start, i]); start = None
    if start is not None:
        segs.append([start, len(speech)])
    merged = []
    for s in segs:
        if merged and (s[0] - merged[-1][1]) * 0.01 < min_gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    # choose the n_lines-1 largest gaps as line boundaries
    gaps = sorted(range(len(merged) - 1), key=lambda i: merged[i + 1][0] - merged[i][1], reverse=True)
    cuts = sorted(gaps[: len(lines) - 1])
    bounds, s0 = [], merged[0][0]
    for c in cuts:
        bounds.append((s0 * 0.01, merged[c][1] * 0.01)); s0 = merged[c + 1][0]
    bounds.append((s0 * 0.01, merged[-1][1] * 0.01))
    words = []
    for li, (line, (t0, t1)) in enumerate(zip(lines, bounds)):
        ws = strip_tags(line).split()
        L = sum(len(w) + 1 for w in ws)
        t = t0
        for w in ws:
            d = (t1 - t0) * (len(w) + 1) / L
            words.append({"word": w, "line": li, "start": t, "end": t + d * 0.92})
            t += d
    return words


def cue(words, token, occurrence=0, at="start"):
    """Time of the n-th occurrence of a word (case/punctuation-insensitive)."""
    tok = norm(token)
    hits = [w for w in words if norm(w["word"]) == tok]
    if len(hits) <= occurrence:
        raise KeyError(f"cue word not found: {token}")
    return hits[occurrence][at]
