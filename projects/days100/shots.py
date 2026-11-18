"""All shots of '100 Days', chapter by chapter (chapters live in ch*.py)."""
import numpy as np

import engine as E
import world as WD

PI = np.pi
GY = WD.GY
Y0 = GY + 1                     # walk surface on flat ground
U = np.array([400.0, 40.0, 0.0])
FLOOR = -32
G, B, R, O, Y, Wh, PU = (120, 240, 120), (110, 235, 255), (255, 80, 60), (255, 150, 30), (255, 225, 40), (255, 255, 255), (200, 120, 255)
TOP = (1050, 150)


class X:
    """Helper namespace handed to the chapter builders."""

    def __init__(self, f):
        self.f = f
        self.T, self.C, self.Wd, self.E = f.T, f.C, f.Wd, f.E

    # actors ------------------------------------------------------
    def pip(self, keys, **kw):
        return E.Actor("pip", keys, self.f, **kw)

    def mob(self, kind, keys, **kw):
        return E.Actor(kind, keys, self.f, **kw)

    def biscuit(self, keys, **kw):
        kw.setdefault("tame", True)
        return E.Actor("wolf", keys, self.f, **kw)

    def follow(self, leader, t0, t1, kind="wolf", lag=0.7, side=1.0, back=1.4, loc="over", **kw):
        keys = E.follower(self.f, leader, t0, t1, lag=lag, side=side, back=back, loc=loc)
        if kind == "wolf":
            kw.setdefault("tame", True)
        return E.Actor(kind, keys, self.f, loc=loc, **kw)

    # helpers -----------------------------------------------------
    def g(self, x, z, dy=0.0):
        return np.array([x, self.f.ground(x, z) + dy, z], float)

    def ug(self, x, y, z):
        return np.array([x, y, z], float) + U

    def ng(self, x, y, z):
        return np.array([x, y, z], float) + WD.NO

    def every(self, t0, t1, n):
        return [t0 + (t1 - t0) * (i + 0.5) / n for i in range(n)]

    def span(self, key, a=0.0, b=1.0):
        return self.T(key, a), self.T(key, b)

    def title(self, t0, t1, txt, **kw):
        return (t0, t1, txt, kw)

    def items(self, key, n, a=0.1, b=0.9):
        return self.every(self.T(key, a), self.T(key, b), n)


def look_pan(eye, yaw0, yaw1, t0, t1, pitch=0.05, fov=55):
    e = np.asarray(eye, np.float32)

    def fn(t):
        u = E.ease_in_out(E.seg(t, t0, t1))
        a = yaw0 + (yaw1 - yaw0) * u
        d = np.array([np.sin(a), pitch, np.cos(a)], np.float32)
        return dict(eye=e, target=e + d * 10, fov=fov, roll=0.0)
    return fn


def build(f):
    x = X(f)
    shots = []
    import ch_a, ch_b, ch_c, ch_d
    for mod in (ch_a, ch_b, ch_c, ch_d):
        for name in sorted(n for n in dir(mod) if n.startswith("ch")):
            shots += getattr(mod, name)(x)
    return shots
