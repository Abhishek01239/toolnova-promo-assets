"""Audio engine: synthesized sound effects, procedural original music, mixing.

Everything is generated from code, so all audio is original and free to use.
"""
import subprocess
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 44100


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _rng(seed):
    return np.random.default_rng(seed)


def lowpass(x, fc, order=2):
    sos = signal.butter(order, min(fc, SR / 2 - 100), "low", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def highpass(x, fc, order=2):
    sos = signal.butter(order, fc, "high", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def bandpass(x, lo, hi, order=2):
    sos = signal.butter(order, [lo, min(hi, SR / 2 - 100)], "band", fs=SR, output="sos")
    return signal.sosfilt(sos, x, axis=0)


def env_exp(n, decay):
    return np.exp(-np.arange(n) / SR / decay)


def adsr(n, a=0.01, d=0.1, s=0.7, r=0.2):
    e = np.ones(n) * s
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na = min(na, n); e[:na] = np.linspace(0, 1, na)
    nd = min(nd, n - na); e[na:na + nd] = np.linspace(1, s, nd)
    if nr > 0 and n > nr:
        e[-nr:] *= np.linspace(1, 0, nr)
    return e


def stereo(x, pan=0.0, width=0.0, seed=0):
    """Mono -> stereo with constant-power pan and optional decorrelation width."""
    l = np.cos((pan + 1) * np.pi / 4)
    r = np.sin((pan + 1) * np.pi / 4)
    if width > 0:
        d = int(0.011 * SR * width)
        xr = np.concatenate([np.zeros(d), x[:-d]]) if d else x
        return np.stack([x * l * 1.41, xr * r * 1.41], 1)
    return np.stack([x * l * 1.41, x * r * 1.41], 1)


def reverb(x, decay=1.6, wet=0.25, seed=1):
    n = int(decay * SR)
    r = _rng(seed)
    ir_l = r.normal(size=n) * np.exp(-np.arange(n) / SR / (decay / 5))
    ir_r = r.normal(size=n) * np.exp(-np.arange(n) / SR / (decay / 5))
    ir_l, ir_r = lowpass(ir_l, 6000), lowpass(ir_r, 6000)
    ir_l /= np.sqrt((ir_l ** 2).sum()); ir_r /= np.sqrt((ir_r ** 2).sum())
    if x.ndim == 1:
        x = stereo(x)
    wl = signal.fftconvolve(x[:, 0], ir_l)[:len(x)]
    wr = signal.fftconvolve(x[:, 1], ir_r)[:len(x)]
    return x * (1 - wet) + np.stack([wl, wr], 1) * wet


# ================================================================== SFX
def sfx_pop(seed=0, pitch=1.0):
    t = _t(0.16)
    f = (700 + 500 * np.exp(-t * 40)) * pitch
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * env_exp(len(t), 0.04)
    x += _rng(seed).normal(size=len(t)) * env_exp(len(t), 0.004) * 0.3
    return stereo(x * 0.8)


def sfx_place(seed=0):
    t = _t(0.22)
    n = lowpass(_rng(seed).normal(size=len(t)), 900) * env_exp(len(t), 0.05)
    x = n * 1.8 + np.sin(2 * np.pi * 150 * t) * env_exp(len(t), 0.04) * 0.6
    return stereo(x * 0.7)


def sfx_dig(seed=0):
    out = np.zeros(int(0.35 * SR))
    r = _rng(seed)
    for k in range(3):
        n = int(0.07 * SR)
        b = bandpass(r.normal(size=n), 300 + k * 150, 2500) * env_exp(n, 0.018)
        s = int(k * 0.1 * SR)
        out[s:s + n] += b * (1.6 - 0.3 * k)
    return stereo(out * 0.9)


def sfx_step(seed=0):
    n = int(0.12 * SR)
    x = bandpass(_rng(seed).normal(size=n), 200, 1800) * env_exp(n, 0.025)
    return stereo(x * 0.5)


def sfx_hiss(dur=1.6, seed=0):
    t = _t(dur)
    x = bandpass(_rng(seed).normal(size=len(t)), 2500, 9000)
    e = np.clip(t / dur, 0, 1) ** 0.6 * (1 - np.exp(-(dur - t) * 30))
    trem = 0.8 + 0.2 * np.sin(2 * np.pi * 9 * t)
    return stereo(x * e * trem * 0.9, width=0.3)


def sfx_explosion(seed=0, dur=3.2):
    t = _t(dur)
    r = _rng(seed)
    noise = r.normal(size=len(t))
    # time-varying lowpass: filter in chunks with falling cutoff
    out = np.zeros(len(t))
    chunk = 2048
    zi = None
    for i in range(0, len(t), chunk):
        fc = 200 + 7000 * np.exp(-t[i] * 3.5)
        sos = signal.butter(2, fc, "low", fs=SR, output="sos")
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0
        seg, zi = signal.sosfilt(sos, noise[i:i + chunk], zi=zi)
        out[i:i + chunk] = seg
    body = out * np.exp(-t * 1.4) * 2.2
    sub = np.sin(2 * np.pi * np.cumsum(55 * np.exp(-t * 0.8) + 22) / SR) * np.exp(-t * 1.6) * 1.2
    crack = np.zeros(len(t))
    for _ in range(140):
        i = int(r.uniform(0, 1.6) ** 1.7 * SR)
        if i < len(t) - 300:
            crack[i:i + 300] += r.normal(size=300) * np.exp(-np.arange(300) / 40) * r.uniform(0.2, 1)
    crack = highpass(crack, 1500) * np.exp(-t * 2.0)
    attack = r.normal(size=len(t)) * np.exp(-t * 60) * 1.5
    x = np.tanh((body + sub + crack * 0.6 + attack) * 1.4)
    return reverb(stereo(x, width=0.8), decay=2.0, wet=0.25)


def sfx_whoosh(dur=0.5, seed=0, up=True):
    t = _t(dur)
    x = _rng(seed).normal(size=len(t))
    out = np.zeros(len(t))
    chunk = 1024
    zi = None
    for i in range(0, len(t), chunk):
        u = t[i] / dur
        fc = 400 + 3500 * (u if up else 1 - u)
        sos = signal.butter(2, [fc * 0.6, fc * 1.4], "band", fs=SR, output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        seg, zi = signal.sosfilt(sos, x[i:i + chunk], zi=zi)
        out[i:i + chunk] = seg
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 2
    return stereo(out * e * 2.0, width=0.5)


def sfx_wind(dur, seed=0, intensity=None):
    t = _t(dur)
    r = _rng(seed)
    l = lowpass(r.normal(size=len(t)), 700) * 2.2
    rr = lowpass(r.normal(size=len(t)), 700) * 2.2
    mod = 0.7 + 0.3 * np.sin(2 * np.pi * 0.7 * t + 1) * np.sin(2 * np.pi * 0.31 * t)
    if intensity is not None:
        mod = mod * intensity(t)
    hi = bandpass(r.normal(size=len(t)), 1500, 4000) * 0.4
    return np.stack([(l + hi) * mod, (rr + hi) * mod], 1)


def sfx_splash(seed=0):
    t = _t(1.2)
    r = _rng(seed)
    x = highpass(r.normal(size=len(t)), 800) * np.exp(-t * 5) * 1.3
    x += lowpass(r.normal(size=len(t)), 400) * np.exp(-t * 8) * 1.5
    for _ in range(25):
        s = int(r.uniform(0.02, 0.9) * SR)
        n = int(0.05 * SR)
        f0 = r.uniform(500, 1400)
        tt = np.arange(n) / SR
        b = np.sin(2 * np.pi * (f0 + 3000 * tt) * tt) * np.exp(-tt * 60) * 0.25
        x[s:s + n] += b[:len(x) - s]
    return reverb(stereo(x, width=0.6), 1.2, 0.2)


def sfx_ding(seed=0, base=880):
    t = _t(1.6)
    x = sum(np.sin(2 * np.pi * base * m * t) * np.exp(-t * (2 + m)) * a
            for m, a in ((1, 1), (2.01, 0.4), (3.0, 0.2), (4.2, 0.1)))
    return reverb(stereo(x * 0.5), 1.5, 0.3)


def sfx_heartbeat(seed=0):
    t = _t(0.7)
    def thump(t0, a):
        tt = np.clip(t - t0, 0, None)
        return np.sin(2 * np.pi * 55 * tt) * np.exp(-tt * 14) * (t >= t0) * a
    x = thump(0, 1.0) + thump(0.22, 0.7)
    return stereo(np.tanh(x * 2) * 0.8)


def sfx_riser(dur=2.0, seed=0):
    t = _t(dur)
    f = 120 * 2 ** (3 * t / dur)
    saw = signal.sawtooth(2 * np.pi * np.cumsum(f) / SR) * 0.25
    n = highpass(_rng(seed).normal(size=len(t)), 2000) * 0.5
    e = (t / dur) ** 2
    return reverb(stereo((lowpass(saw, 3000) + n) * e, width=0.5), 1.2, 0.3)


def sfx_impact(seed=0):
    """Trailer-style low hit for dramatic cuts."""
    t = _t(1.8)
    f = 90 * np.exp(-t * 2.5) + 35
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.2)
    x += lowpass(_rng(seed).normal(size=len(t)), 300) * np.exp(-t * 8) * 1.2
    return reverb(stereo(np.tanh(x * 2.2)), 2.2, 0.3)


def sfx_shimmer(seed=0, dur=1.2):
    t = _t(dur)
    x = np.zeros(len(t))
    notes = [1046.5, 1318.5, 1568, 2093, 2637]
    for i, f in enumerate(notes):
        s = int(i * 0.07 * SR)
        tt = t[: len(t) - s]
        x[s:] += np.sin(2 * np.pi * f * tt) * np.exp(-tt * 4) * 0.3
    return reverb(stereo(x, width=0.5), 1.8, 0.45)


def sfx_hurt(seed=0):
    t = _t(0.35)
    f = 420 * np.exp(-t * 3)
    x = signal.square(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 0.35
    return stereo(lowpass(x, 2500))


SFX = {
    "pop": sfx_pop, "place": sfx_place, "dig": sfx_dig, "step": sfx_step, "hiss": sfx_hiss,
    "explosion": sfx_explosion, "whoosh": sfx_whoosh, "splash": sfx_splash, "ding": sfx_ding,
    "heartbeat": sfx_heartbeat, "riser": sfx_riser, "impact": sfx_impact, "shimmer": sfx_shimmer,
    "hurt": sfx_hurt,
}


# ================================================================== MUSIC
A4 = 440.0
NOTE = {n: i for i, n in enumerate(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])}


def midi(name):
    """'C4' -> midi number"""
    n, o = name[:-1], int(name[-1])
    return 12 * (o + 1) + NOTE[n]


def hz(m):
    return A4 * 2 ** ((m - 69) / 12)


CHORDS = {   # root-position triads as midi offsets from C4 (60)
    "C": [60, 64, 67], "G": [55, 59, 62], "Am": [57, 60, 64], "F": [53, 57, 60],
    "Em": [52, 55, 59], "Dm": [50, 53, 57], "E": [52, 56, 59], "Bb": [58, 62, 65],
}


def inst_pluck(f, dur, bright=1.0):
    t = _t(dur)
    x = (signal.square(2 * np.pi * f * t, 0.3) * 0.5 + np.sin(2 * np.pi * f * t))
    x = lowpass(x * np.exp(-t * 7), 1800 * bright + f)
    return x * 0.35


def inst_pad(f, dur):
    t = _t(dur)
    x = sum(signal.sawtooth(2 * np.pi * f * d * t) for d in (0.997, 1.0, 1.004)) / 3
    x = lowpass(x, 1400)
    return x * adsr(len(t), a=min(0.4, dur / 3), d=0.2, s=0.8, r=min(0.5, dur / 3)) * 0.22


def inst_bass(f, dur):
    t = _t(dur)
    x = np.sin(2 * np.pi * f * t) + 0.35 * signal.sawtooth(2 * np.pi * f * t)
    return lowpass(x, 700) * adsr(len(t), 0.005, 0.1, 0.7, 0.05) * 0.5


def inst_stab(f, dur):
    t = _t(dur)
    x = sum(signal.sawtooth(2 * np.pi * f * d * t) for d in (0.995, 1.005)) / 2
    return lowpass(x, 2600) * np.exp(-t * 6) * 0.3


def inst_bell(f, dur):
    t = _t(dur)
    x = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 6)
    return x * np.exp(-t * 2.5) * 0.25


def drum_kick(seed=0, hard=1.0):
    t = _t(0.4)
    f = 45 + 110 * np.exp(-t * 30)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7)
    return np.tanh(x * 1.5 * hard) * 0.9


def drum_snare(seed=0):
    t = _t(0.25)
    n = bandpass(_rng(seed).normal(size=len(t)), 1200, 8000) * np.exp(-t * 18)
    tone = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 25)
    return (n * 0.8 + tone * 0.5) * 0.8


def drum_hat(seed=0, open_=False):
    t = _t(0.25 if open_ else 0.06)
    return highpass(_rng(seed).normal(size=len(t)), 7000) * np.exp(-t * (12 if open_ else 70)) * 0.25


def drum_tom(f=110):
    t = _t(0.5)
    x = np.sin(2 * np.pi * np.cumsum(f * (1 + 0.6 * np.exp(-t * 20))) / SR) * np.exp(-t * 6)
    return x * 0.8


class Score:
    """Accumulates notes into a stereo buffer."""

    def __init__(self, dur):
        self.buf = np.zeros((int(dur * SR) + SR * 3, 2))

    def add(self, t0, x, gain=1.0, pan=0.0):
        if x.ndim == 1:
            x = stereo(x, pan)
        s = int(t0 * SR)
        if s >= len(self.buf) or s < 0:
            return
        n = min(len(x), len(self.buf) - s)
        self.buf[s:s + n] += x[:n] * gain


def compose(sections, total_dur, bpm=118):
    """sections: list of (name, t0, t1). Section styles:
    'adventure', 'build', 'suspense', 'panic', 'freefall', 'silence', 'resolve'."""
    beat = 60.0 / bpm
    dry = Score(total_dur)
    for name, t0, t1 in sections:
        sc = Score(total_dur)
        nb = int(np.ceil((t1 - t0) / beat))
        if name in ("adventure", "build"):
            prog = ["C", "G", "Am", "F"]
            for b in range(nb):
                tb = t0 + b * beat
                ch = CHORDS[prog[(b // 4) % 4]]
                # arpeggio 8ths
                for k in range(2):
                    note = ch[(b * 2 + k) % 3] + 12
                    sc.add(tb + k * beat / 2, inst_pluck(hz(note), beat), 0.8, pan=0.25 * (-1) ** k)
                if b % 4 == 0:
                    for n in ch:
                        sc.add(tb, inst_pad(hz(n), beat * 4), 0.9)
                    sc.add(tb, inst_bass(hz(ch[0] - 24), beat * 2), 1.0)
                if b % 4 == 2:
                    sc.add(tb, inst_bass(hz(ch[0] - 24), beat * 2), 0.9)
                if name == "build":
                    if b % 2 == 0:
                        sc.add(tb, drum_kick(b), 0.9)
                    else:
                        sc.add(tb, drum_snare(b), 0.55)
                    sc.add(tb + beat / 2, drum_hat(b), 0.8)
                    sc.add(tb, drum_hat(b + 99), 0.5)
                    if b % 8 == 7:
                        sc.add(tb + beat / 2, inst_bell(hz(ch[2] + 24), beat * 2), 0.7)
        elif name == "suspense":
            for b in range(0, nb, 4):
                tb = t0 + b * beat
                for n in (57, 60, 64):
                    sc.add(tb, inst_pad(hz(n - 12), beat * 4.2), 0.9)
                sc.add(tb, inst_bass(hz(45 - 12), beat * 4), 0.8)
            for b in range(nb):
                tb = t0 + b * beat
                if b % 2 == 0:
                    sc.add(tb, drum_kick(b, 0.7), 0.6)
                if b % 4 == 3:
                    sc.add(tb, inst_bell(hz(76 + (b // 4 % 2) * 1), beat * 3), 0.5, pan=0.4)
        elif name in ("panic", "freefall"):
            prog = ["Am", "F", "C", "G"] if name == "panic" else ["Am", "F", "Dm", "E"]
            for b in range(nb):
                tb = t0 + b * beat
                ch = CHORDS[prog[(b // 2) % 4]]
                for k in range(4):     # 16th bass
                    sc.add(tb + k * beat / 4, inst_bass(hz(ch[0] - 24), beat / 4), 0.9)
                sc.add(tb, drum_kick(b, 1.3), 1.1)
                if b % 2 == 1:
                    sc.add(tb, drum_snare(b), 0.9)
                for k in range(2):
                    sc.add(tb + k * beat / 2, drum_hat(b * 3 + k), 0.9)
                if b % 2 == 0:
                    for n in ch:
                        sc.add(tb, inst_stab(hz(n), beat * 1.5), 0.8)
                        sc.add(tb, inst_pad(hz(n + 12), beat * 2), 0.6)
                if name == "freefall" and b % 4 == 3:
                    for k in range(4):
                        sc.add(tb + k * beat / 4, drum_tom(150 - 20 * k), 0.6)
        elif name == "resolve":
            prog = ["F", "C", "G", "C"]
            for b in range(nb):
                tb = t0 + b * beat
                ch = CHORDS[prog[(b // 4) % 4]]
                if b % 4 == 0:
                    for n in ch:
                        sc.add(tb, inst_pad(hz(n), beat * 4.3), 1.0)
                    sc.add(tb, inst_bass(hz(ch[0] - 24), beat * 4), 0.7)
                sc.add(tb, inst_pluck(hz(ch[b % 3] + 12), beat * 1.5, 0.6), 0.6)
                if b % 4 == 2:
                    sc.add(tb, inst_bell(hz(ch[2] + 24), beat * 2), 0.5)
        elif name == "silence":
            continue
        # crossfade edges of each section
        a, b_ = int(t0 * SR), int(t1 * SR)
        seg = sc.buf.copy()
        fade = int(0.08 * SR)
        mask = np.zeros(len(seg))
        mask[a:b_] = 1
        if fade:
            mask[a:a + fade] = np.linspace(0, 1, min(fade, len(mask) - a))[: len(mask[a:a + fade])]
            tail_end = min(len(mask), b_ + int(0.9 * SR))
            mask[b_:tail_end] = np.linspace(1, 0, tail_end - b_)
        dry.buf += seg * mask[:, None]
    out = reverb(dry.buf, decay=1.8, wet=0.22)
    return out[: int(total_dur * SR)]


# ================================================================== MIX
def load_audio(path):
    """Decode any audio file via ffmpeg to float stereo at SR."""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, 2).astype(np.float64)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)


def envelope(x, attack=0.01, release=0.35):
    mono = np.abs(x).mean(axis=1) if x.ndim == 2 else np.abs(x)
    # downsample for speed
    hop = 256
    m = mono[: len(mono) // hop * hop].reshape(-1, hop).max(axis=1)
    e = np.zeros_like(m)
    a = np.exp(-hop / SR / attack)
    r = np.exp(-hop / SR / release)
    v = 0.0
    for i, s in enumerate(m):
        v = a * v + (1 - a) * s if s > v else r * v + (1 - r) * s
        e[i] = v
    e = np.repeat(e, hop)
    return np.pad(e, (0, len(mono) - len(e)), mode="edge")


class Mixer:
    def __init__(self, dur):
        self.n = int(dur * SR)
        self.voice = np.zeros((self.n, 2))
        self.music = np.zeros((self.n, 2))
        self.sfx = np.zeros((self.n, 2))

    def _add(self, bus, t0, x, gain):
        s = int(t0 * SR)
        if s >= self.n:
            return
        if s < 0:
            x = x[-s:]
            s = 0
        k = min(len(x), self.n - s)
        bus[s:s + k] += x[:k] * gain

    def add_voice(self, t0, x, target_db=-17.0):
        g = 10 ** ((target_db - rms_db(x[np.abs(x).max(axis=1) > 0.02])) / 20)
        self._add(self.voice, t0, x, g)

    def add_music(self, x, gain_db=-14.0):
        self._add(self.music, 0, x / (np.abs(x).max() + 1e-9), 10 ** (gain_db / 20))

    def add_sfx(self, t0, name_or_arr, gain_db=-6.0, **kw):
        x = SFX[name_or_arr](**kw) if isinstance(name_or_arr, str) else name_or_arr
        x = x / (np.abs(x).max() + 1e-9)
        self._add(self.sfx, t0, x, 10 ** (gain_db / 20))

    def render(self, duck_db=-9.0):
        env = envelope(self.voice, 0.02, 0.45)
        env = np.clip(env / (env.max() + 1e-9) * 3, 0, 1)
        duck = 10 ** (duck_db * env / 20)
        mix = self.voice + self.music * duck[:, None] + self.sfx * (0.6 + 0.4 * duck[:, None])
        # soft-knee limiter
        peak = np.abs(mix).max()
        mix = mix / max(peak, 1e-9) * 1.25
        mix = np.tanh(mix * 1.1) / np.tanh(1.1)
        mix *= 0.93 / np.abs(mix).max()
        fade = int(0.4 * SR)
        mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
        return mix


def write_wav(path, x):
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def sfx_groan(seed=0, dur=1.1):
    """Low zombie moan: buzzy source through two vowel formants with wobble."""
    t = _t(dur)
    r = _rng(seed)
    f0 = r.uniform(70, 95) * (1 + 0.06 * np.sin(2 * np.pi * r.uniform(3, 5) * t)) * (1 - 0.15 * t / dur)
    src = signal.sawtooth(2 * np.pi * np.cumsum(f0) / SR)
    x = bandpass(src, 300, 900) * 1.4 + bandpass(src, 1000, 1500) * 0.5
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 0.7
    return stereo(x * e * 0.8, width=0.2)


def sfx_creak(seed=0, dur=0.9):
    t = _t(dur)
    f = 220 + 160 * np.sin(np.pi * t / dur) + 30 * np.sin(2 * np.pi * 23 * t)
    x = signal.sawtooth(2 * np.pi * np.cumsum(f) / SR) * (0.5 + 0.5 * (np.sin(2 * np.pi * 31 * t) > 0))
    return stereo(bandpass(x, 300, 2500) * np.sin(np.pi * t / dur) * 0.6)


def sfx_crackle(seed=0, dur=1.6):
    t = _t(dur)
    r = _rng(seed)
    x = lowpass(r.normal(size=len(t)), 1200) * 0.5
    for _ in range(int(60 * dur)):
        i = r.integers(0, len(t) - 200)
        x[i:i + 200] += r.normal(size=200) * np.exp(-np.arange(200) / 25) * r.uniform(0.3, 1.0)
    e = np.minimum(1, t * 6) * np.minimum(1, (dur - t) * 4)
    return stereo(highpass(x, 300) * e, width=0.4)


def sfx_water(seed=0, dur=2.5):
    t = _t(dur)
    r = _rng(seed)
    x = bandpass(r.normal(size=len(t)), 400, 3000) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t))
    for _ in range(int(30 * dur)):
        s = int(r.uniform(0, dur - 0.06) * SR)
        n = int(0.04 * SR)
        tt = np.arange(n) / SR
        x[s:s + n] += np.sin(2 * np.pi * (r.uniform(400, 1100) + 2500 * tt) * tt) * np.exp(-tt * 70) * 0.3
    e = np.minimum(1, t * 3) * np.minimum(1, (dur - t) * 2)
    return stereo(x * e * 0.7, width=0.5)


SFX.update({"groan": sfx_groan, "creak": sfx_creak, "crackle": sfx_crackle, "water": sfx_water})
