"""Animation helpers: easing, keyframe tracks, camera rigs, shake, stateless particles."""
import numpy as np


# ------------------------------------------------------------------ easing
def clamp01(x):
    return float(np.clip(x, 0.0, 1.0))


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_in_out(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out(x, p=3):
    x = clamp01(x)
    return 1 - (1 - x) ** p


def ease_in(x, p=2):
    return clamp01(x) ** p


def ease_out_back(x, s=1.70158):
    x = clamp01(x)
    return 1 + (s + 1) * (x - 1) ** 3 + s * (x - 1) ** 2


def lerp(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def seg(t, t0, t1):
    """Normalised progress of t inside [t0, t1]."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return clamp01((t - t0) / (t1 - t0))


class Track:
    """Keyframe track of vectors. keys: [(time, value, easing_name)]"""
    EASE = {"linear": clamp01, "smooth": smooth, "inout": ease_in_out, "out": ease_out, "in": ease_in,
            "back": ease_out_back, "hold": lambda x: 0.0}

    def __init__(self, keys):
        self.keys = sorted(keys, key=lambda k: k[0])

    def __call__(self, t):
        k = self.keys
        if t <= k[0][0]:
            return np.asarray(k[0][1], np.float32)
        for (t0, v0, _), (t1, v1, e) in zip(k, k[1:]):
            if t <= t1:
                return lerp(v0, v1, self.EASE[e]((t - t0) / max(t1 - t0, 1e-6)))
        return np.asarray(k[-1][1], np.float32)


# ------------------------------------------------------------------ noise
def noise1(t, seed=0):
    """Smooth 1D value noise in [-1, 1]."""
    i = np.floor(t)
    f = t - i
    def h(n):
        return np.sin((n + seed * 17.13) * 127.1) * 43758.5453 % 1.0 * 2 - 1
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def shake(t, amp, freq=18.0):
    if amp <= 0:
        return np.zeros(3, np.float32)
    return np.array([noise1(t * freq, 1), noise1(t * freq, 2), noise1(t * freq, 3)], np.float32) * amp


def orbit(center, radius, angle, height):
    c = np.asarray(center, np.float32)
    return c + np.array([np.sin(angle) * radius, height, np.cos(angle) * radius], np.float32)


# ------------------------------------------------------------------ particles
class Burst:
    """A stateless particle burst: positions are a closed-form function of time,
    so any frame can be rendered independently (no simulation state)."""

    def __init__(self, t0, origin, n, speed=(2, 6), life=(0.6, 1.4), size=(0.08, 0.2),
                 colors=((1, 1, 1, 1),), gravity=-9.0, drag=1.5, spread=(1, 1, 1),
                 up_bias=0.0, seed=0, grow=0.0, additive=False, fade=True, box=0.0, vel=None):
        r = np.random.default_rng(seed)
        self.t0 = t0
        self.n = n
        d = r.normal(size=(n, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        d *= np.asarray(spread, np.float32)
        d[:, 1] += up_bias
        sp = r.uniform(*speed, size=(n, 1))
        self.v0 = (d * sp).astype(np.float32)
        if vel is not None:
            self.v0 += np.asarray(vel, np.float32)
        self.p0 = (np.asarray(origin, np.float32) + r.uniform(-box, box, size=(n, 3))).astype(np.float32)
        self.life = r.uniform(*life, size=n).astype(np.float32)
        self.size = r.uniform(*size, size=n).astype(np.float32)
        cols = np.asarray(colors, np.float32)
        self.col = cols[r.integers(0, len(cols), n)]
        self.g, self.k, self.grow, self.additive, self.fade = gravity, drag, grow, additive, fade

    def at(self, t):
        a = t - self.t0
        if a < 0:
            return None
        alive = a < self.life
        if not alive.any():
            return None
        k = self.k
        # velocity with linear drag: v = v0 e^{-kt}; x = v0 (1-e^{-kt})/k ; gravity term approx
        e = (1 - np.exp(-k * a)) / k if k > 0 else a
        pos = self.p0 + self.v0 * e
        pos[:, 1] += 0.5 * self.g * a * a * (1.0 / (1 + k * a * 0.5))
        u = np.clip(a / self.life, 0, 1)
        size = self.size * (1 + self.grow * u)
        col = self.col.copy()
        if self.fade:
            col[:, 3] *= (1 - u) ** 1.2
        out = np.concatenate([pos, size[:, None], col], 1)[alive]
        return out.astype(np.float32)


def gather(bursts, t, additive):
    parts = [b.at(t) for b in bursts if b.additive == additive]
    parts = [p for p in parts if p is not None]
    return np.concatenate(parts) if parts else None
