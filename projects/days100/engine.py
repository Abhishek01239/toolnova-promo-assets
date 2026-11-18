"""Shot engine for '100 Days'.

A film is a list of Shots. Each shot starts at a narration line (or a fraction of one) and lasts until the
next shot. A shot names a location + stage, a time of day, a camera function, actors (Pip, mobs, animals,
Biscuit) with keyframed paths, props (blocks that pop in / break, flying items, torches, reveals) and
overlay titles. Everything is a pure function of time t, so any frame renders on its own.
"""
import numpy as np

from mcengine.anim import seg, smooth, ease_out, ease_out_back, ease_in_out, Burst, clamp01, shake
from mcengine.geometry import box_vertices, translate, VoxelWorld
from mcengine.models import pip_vertices, creeper_vertices, zombie_vertices, block_entity_vertices, item_vertices, torch_vertices
from mcengine import models100 as M
from mcengine import textures as tx

PI = np.pi


class PathTrack:
    """Piecewise-linear path with cumulative distance (for walk cycles) and velocity."""

    def __init__(self, keys):
        ks = sorted([(float(a), np.array(b, float)) for a, b in keys], key=lambda x: x[0])
        dd = []
        for a, b in ks:
            if dd and a <= dd[-1][0]:
                a = dd[-1][0] + 1e-3
            if dd and np.linalg.norm(b - dd[-1][1]) / max(a - dd[-1][0], 1e-6) > 25:
                dd.append((a - 1e-3, dd[-1][1].copy()))
            dd.append((a, b))
        self.t = np.array([k[0] for k in dd])
        self.p = np.array([k[1] for k in dd], float)
        if len(self.t) == 1:
            self.t = np.r_[self.t, self.t + 1]
            self.p = np.r_[self.p, self.p]
        seglen = np.linalg.norm(np.diff(self.p[:, [0, 2]], axis=0), axis=1)
        seglen[seglen > 8] = 0
        self.d = np.r_[0, np.cumsum(seglen)]

    def pos(self, t):
        if t <= self.t[0]:
            return self.p[0].copy()
        if t >= self.t[-1]:
            return self.p[-1].copy()
        i = np.searchsorted(self.t, t) - 1
        u = (t - self.t[i]) / max(self.t[i + 1] - self.t[i], 1e-6)
        return self.p[i] * (1 - u) + self.p[i + 1] * u

    def dist(self, t):
        return float(np.interp(t, self.t, self.d))

    def vel(self, t):
        v = (self.pos(t + 0.05) - self.pos(t - 0.05)) / 0.1
        return v if np.linalg.norm(v) < 30 else np.zeros(3)


def _sched(spec, t, default=None):
    """spec: value | callable(t) | [(t0, value), ...] (last t0 <= t wins)."""
    if callable(spec):
        return spec(t)
    if isinstance(spec, list):
        v = default
        for t0, val in spec:
            if t >= t0:
                v = val
        return v
    return spec if spec is not None else default


def poof(t, p, seed=0, n=18, col=(0.9, 0.9, 0.9, 0.9)):
    return Burst(t, p, n, speed=(0.6, 2.2), life=(0.5, 1.0), size=(0.18, 0.4), colors=(col, (0.75, 0.75, 0.75, 0.8)),
                 gravity=0.6, drag=2.5, up_bias=0.4, seed=seed, grow=1.5)


def debris(t, p, block, seed=0, n=14):
    cols = {"dirt": (0.52, 0.37, 0.26, 1), "grass": (0.37, 0.6, 0.2, 1), "log": (0.42, 0.33, 0.2, 1),
            "stone": (0.5, 0.5, 0.5, 1), "cobble": (0.45, 0.45, 0.45, 1), "leaves": (0.25, 0.5, 0.15, 1),
            "obsidian": (0.15, 0.08, 0.25, 1), "netherrack": (0.45, 0.17, 0.17, 1), "glowstone": (0.95, 0.8, 0.4, 1),
            "iron_ore": (0.8, 0.6, 0.45, 1), "coal_ore": (0.2, 0.2, 0.2, 1), "deep_diamond": (0.35, 0.9, 0.88, 1),
            "diamond_ore": (0.35, 0.9, 0.88, 1), "planks": (0.62, 0.45, 0.26, 1), "sand": (0.86, 0.8, 0.55, 1),
            "deepslate": (0.3, 0.3, 0.32, 1)}
    c = cols.get(block, (0.5, 0.5, 0.5, 1))
    return Burst(t, p, n, speed=(1.5, 4), life=(0.4, 0.8), size=(0.06, 0.13), colors=(c,), gravity=-14, drag=1.0,
                 up_bias=0.8, seed=seed)


def fire_burst(t0, p, dur, seed=0):
    out = []
    k = 0
    tt = t0
    while tt < t0 + dur:
        out.append(Burst(tt, p, 6, speed=(0.2, 0.8), life=(0.3, 0.6), size=(0.12, 0.26),
                         colors=((1, 0.6, 0.1, 0.9), (1, 0.85, 0.3, 0.9), (1, 0.35, 0.05, 0.8)),
                         gravity=3.0, drag=1.0, up_bias=1.0, seed=seed + k, box=0.3, additive=True))
        tt += 0.15
        k += 1
    return out


def hearts_burst(t, p, seed=0):
    return Burst(t, p, 8, speed=(0.3, 1.0), life=(0.9, 1.6), size=(0.16, 0.24), colors=((1, 0.25, 0.35, 1),),
                 gravity=1.2, drag=1.5, up_bias=1.0, seed=seed, box=0.3)


class Actor:
    KINDS = ("pip", "zombie", "creeper", "spider", "cow", "sheep", "wolf", "villager", "ghast", "baby_cow", "kid")

    def __init__(self, kind, keys, film=None, loc="over", item=None, acts=(), skin=None, yaw=None, pose=None,
                 spawn=None, death=None, hits=(), burn=None, scale=1.0, tame=False, swell=(), climb=None, y_off=None,
                 sit=None, boat=None, walk_scale=None, hidden=None, flash=None, name=None, face_target=None, open_=None):
        self.kind = kind
        pts = []
        for k in keys:
            t, p = k[0], k[1]
            p = tuple(p)
            if len(p) == 2:
                y = film.ground(p[0], p[1], loc) if film is not None else 11.0
                p = (p[0], y, p[1])
            pts.append((t, p))
        self.path = PathTrack(pts)
        self.item, self.acts, self.skin, self.yaw_spec, self.pose_fn = item, list(acts), skin, yaw, pose
        self.spawn, self.death, self.hits, self.burn = spawn, death, list(hits), burn
        self.scale, self.tame, self.swell, self.climb, self.y_off, self.sit = scale, tame, list(swell), climb, y_off, sit
        self.boat, self.walk_scale, self.hidden, self.flash_spec, self.name = boat, walk_scale, hidden, flash, name
        self.face_target, self.open_ = face_target, open_
        self._last_yaw = {}

    def alive(self, t):
        if self.spawn is not None and t < self.spawn:
            return False
        if self.death is not None and t >= self.death:
            return False
        if self.hidden is not None and _sched(self.hidden, t, False):
            return False
        return True

    def pos(self, t):
        p = self.path.pos(t)
        if self.y_off is not None:
            p = p.copy()
            p[1] += _sched(self.y_off, t, 0.0)
        return p

    def yaw(self, t):
        if self.yaw_spec is not None:
            y = _sched(self.yaw_spec, t)
            if y is not None:
                return float(y)
        v = self.path.vel(t)
        sp = float(np.hypot(v[0], v[2]))
        if sp > 0.3:
            return float(np.arctan2(v[0], v[2]))
        if self.face_target is not None:
            tp = self.face_target(t) if callable(self.face_target) else np.asarray(self.face_target)
            p = self.pos(t)
            return float(np.arctan2(tp[0] - p[0], tp[2] - p[2]))
        # hold last moving direction: search back
        for dt in (0.2, 0.5, 1.0, 2.0, 4.0):
            v = self.path.vel(t - dt)
            if np.hypot(v[0], v[2]) > 0.3:
                return float(np.arctan2(v[0], v[2]))
        return 0.0

    def verts(self, t):
        if not self.alive(t):
            return None
        p = self.pos(t)
        yaw = self.yaw(t)
        v = self.path.vel(t)
        sp = float(np.hypot(v[0], v[2]))
        ws = self.walk_scale or {"pip": 3.2, "zombie": 1.5, "creeper": 1.5, "spider": 2.0, "cow": 1.4, "sheep": 1.4,
                                 "wolf": 2.5, "villager": 1.5, "baby_cow": 1.2, "kid": 2.0}.get(self.kind, 2.0)
        pose = {"walk": clamp01(sp / ws), "walk_phase": self.path.dist(t) * (2.4 if self.kind != "wolf" else 4.0), "t": t}
        if self.boat is not None and _sched(self.boat, t, False):
            pose["walk"] = 0.0
        fl = 0.0
        for h in self.hits:
            if h + 0.05 <= t < h + 0.35:
                fl = 0.5
                kb = -0.5 * seg(t, h + 0.05, h + 0.35)
                p = p + kb * np.array([np.sin(yaw), 0, np.cos(yaw)])
        if self.flash_spec is not None:
            fl = max(fl, _sched(self.flash_spec, t, 0.0))
        for a in self.acts:
            if a - 0.05 <= t < a + 0.22:
                pose["punch"] = seg(t, a - 0.05, a + 0.22)
        if self.pose_fn is not None:
            pose.update(self.pose_fn(t) or {})
        k = self.kind
        if k == "pip":
            pose["item"] = _sched(self.item, t)
            skin = _sched(self.skin, t, "pip") or "pip"
            out = pip_vertices(p, yaw, pose, flash=fl, skin=skin)
        elif k == "villager" or k == "kid":
            pose.pop("item", None)
            pose.setdefault("arms_up", 0.0)
            if k == "kid":
                from mcengine.geometry import scale as gs
                out = pip_vertices((0, 0, 0), 0.0, pose, flash=fl, skin="villager")
                from mcengine.geometry import transform_vertices, rot_y
                out = transform_vertices(out, translate(*p) @ rot_y(yaw) @ gs(0.55))
            else:
                out = pip_vertices(p, yaw, pose, flash=fl, skin="villager")
        elif k == "zombie":
            out = zombie_vertices(p, yaw, pose, flash=fl)
        elif k == "creeper":
            sw = 0.0
            for (t0, t1) in self.swell:
                if t0 <= t < t1 + 0.05:
                    sw = max(sw, seg(t, t0, t1) ** 1.5)
                    if (t * 6) % 1 < 0.5:
                        fl = max(fl, 0.45)
            pose["swell"] = sw
            out = creeper_vertices(p, yaw, pose, flash=fl)
        elif k == "spider":
            pose["climb"] = _sched(self.climb, t, 0.0) if self.climb is not None else 0.0
            out = M.spider_vertices(p, yaw, pose, flash=fl, s=self.scale)
        elif k == "cow":
            out = M.cow_vertices(p, yaw, pose, flash=fl, s=self.scale)
        elif k == "baby_cow":
            out = M.cow_vertices(p, yaw, pose, flash=fl, s=0.55 * self.scale)
        elif k == "sheep":
            out = M.sheep_vertices(p, yaw, pose, flash=fl, s=self.scale)
        elif k == "wolf":
            pose["tail"] = t * (14 if self.tame else 3)
            if self.sit is not None:
                pose["sit"] = _sched(self.sit, t, 0.0)
            out = M.wolf_vertices(p, yaw, pose, flash=fl, tame=_sched(self.tame, t, False), s=self.scale)
        elif k == "ghast":
            pose["open"] = bool(_sched(self.open_, t, False)) if self.open_ is not None else False
            out = M.ghast_vertices(p, yaw, pose, flash=fl, s=self.scale)
        else:
            raise KeyError(k)
        if self.boat is not None and _sched(self.boat, t, False):
            bp = p.copy()
            bp[1] -= 0.45
            rock = 0.04 * np.sin(t * 2.3)
            out = np.concatenate([out, M.boat_vertices(bp, yaw, rock)])
        return out

    def auto_fx(self, seed):
        """Particles implied by hits / death / burning."""
        fx = []
        for i, h in enumerate(self.hits):
            p = self.path.pos(h) + np.array([0, 1.2, 0])
            fx.append(Burst(h + 0.05, p, 8, speed=(1, 3), life=(0.3, 0.6), size=(0.08, 0.15), colors=((0.8, 0.1, 0.1, 1),),
                            gravity=-9, drag=1.5, up_bias=0.5, seed=seed + i))
        if self.death is not None and self.kind not in ("pip",):
            fx.append(poof(self.death, self.path.pos(self.death) + np.array([0, 0.8, 0]), seed + 50))
        if self.burn is not None:
            b0, b1 = self.burn
            b1 = min(b1, self.death) if self.death is not None else b1
            tt = b0
            k = 0
            while tt < b1:
                p = self.path.pos(tt) + np.array([0, 1.0, 0])
                fx.append(Burst(tt, p, 5, speed=(0.2, 0.7), life=(0.3, 0.55), size=(0.12, 0.25),
                                colors=((1, 0.6, 0.1, 0.9), (1, 0.85, 0.3, 0.9)), gravity=3.0, drag=1.0, up_bias=1.0,
                                seed=seed + 100 + k, box=0.35, additive=True))
                tt += 0.12
                k += 1
        return fx


def follower(film, leader, t0, t1, lag=0.7, side=1.0, back=1.4, loc="over", step=0.25, stop_dist=1.6):
    """Keys for an animal that trails `leader` (Biscuit)."""
    keys = []
    tt = t0
    last = None
    while tt <= t1 + 1e-6:
        lp = leader.path.pos(tt - lag)
        yaw = leader.yaw(tt - lag)
        fwd = np.array([np.sin(yaw), 0, np.cos(yaw)])
        right = np.array([np.cos(yaw), 0, -np.sin(yaw)])
        p = lp - fwd * back + right * side
        if last is not None and np.linalg.norm((p - last)[[0, 2]]) < 0.05:
            p = last
        y = film.ground(p[0], p[2], loc) if loc == "over" else lp[1]
        keys.append((tt, (p[0], y, p[2])))
        last = p
        tt += step
    return keys


# ------------------------------------------------------------------ props
class Prop:
    def draw(self, t, f):
        return [], [], []

    def fx(self, seed):
        return []

    def sfx(self):
        return []


class Pops(Prop):
    """Blocks that pop into place at given times and then stay (until `until`)."""

    def __init__(self, items, until=None, sound=True, offset=(0, 0, 0), scale=1.0):
        self.items = list(items)          # (t, (x,y,z), block)
        self.until, self.sound, self.off, self.scale = until, sound, np.asarray(offset, float), scale

    def draw(self, t, f):
        dyn, dyn_t = [], []
        if self.until is not None and t >= self.until:
            return dyn, dyn_t, []
        for (tp, p, b) in self.items:
            if t < tp:
                continue
            s = ease_out_back(seg(t, tp, tp + 0.15))
            s = 1.0 if s >= 0.999 else s
            v = block_entity_vertices(b, np.array(p, float) + 0.5 + self.off, max(s * self.scale, 0.05))
            (dyn_t if tx.is_translucent(b) else dyn).append(v)
        return dyn, dyn_t, []

    def sfx(self):
        if not self.sound:
            return []
        out = []
        last = -9
        for i, (tp, p, b) in enumerate(self.items):
            if tp - last > 0.12:
                out.append((tp, "place", -12, dict(seed=i % 7)))
                last = tp
        return out


class Breaks(Prop):
    """Blocks drawn until broken: cracks during the last `dur` seconds, then debris + a drop that flies to `collect`."""

    def __init__(self, items, dur=0.6, collect=None, drop=None, offset=(0, 0, 0), overlay=True, from_t=None):
        self.items = list(items)          # (t_break, (x,y,z), block)
        self.dur, self.collect, self.drop, self.off = dur, collect, drop, np.asarray(offset, float)
        self.overlay, self.from_t = overlay, from_t

    def draw(self, t, f):
        dyn = []
        for (tb, p, b) in self.items:
            c = np.array(p, float) + 0.5 + self.off
            if self.from_t is not None and t < self.from_t:
                continue
            if t < tb:
                s = 1.012 if self.overlay else 1.0
                crack = None
                if t >= tb - self.dur:
                    crack = min(3, int(4 * seg(t, tb - self.dur, tb)))
                dyn.append(block_entity_vertices(b, c, s, crack=crack))
            elif self.drop is not None and t < tb + 0.9:
                u = seg(t, tb, tb + 0.9)
                tgt = self.collect(t) if callable(self.collect) else (c if self.collect is None else np.asarray(self.collect))
                pos = c * (1 - ease_in_out(u)) + (tgt + np.array([0, 1.0, 0])) * ease_in_out(u)
                pos = pos + np.array([0, 0.6 * np.sin(PI * u), 0])
                dyn.append(item_vertices(self.drop, pos, spin=t * 4, s=1.2))
        return dyn, [], []

    def fx(self, seed):
        return [debris(tb, np.array(p, float) + 0.5 + self.off, b, seed + i) for i, (tb, p, b) in enumerate(self.items)]

    def sfx(self):
        out = []
        for i, (tb, p, b) in enumerate(self.items):
            out += [(tb - self.dur * 0.7, "dig", -12, dict(seed=i)), (tb - self.dur * 0.3, "dig", -12, dict(seed=i + 20)),
                    (tb, "dig", -9, dict(seed=i + 40))]
            if self.drop:
                out.append((tb + 0.85, "pop", -15, dict(pitch=1.2)))
        return out


class Fly(Prop):
    """Items flying from a point to a point (crafting results, ingots popping out, trades)."""

    def __init__(self, items):
        self.items = list(items)          # (t0, t1, p0, p1, kind)

    def draw(self, t, f):
        dyn = []
        for (t0, t1, p0, p1, kind) in self.items:
            if t0 <= t < t1:
                u = seg(t, t0, t1)
                p = np.asarray(p0, float) * (1 - ease_in_out(u)) + np.asarray(p1, float) * ease_in_out(u)
                p = p + np.array([0, 0.8 * np.sin(PI * u), 0])
                if kind in ("diamond", "iron", "coal", "log", "tnt", "planks", "cobble", "rawiron", "gravel"):
                    dyn.append(item_vertices(kind, p, spin=t * 5, s=1.3))
                else:
                    tile = {"emerald": "emerald", "wheat": "wheat_item", "bone": "bone", "glowstone": "glowstone"}.get(kind, kind)
                    if tile == "glowstone":
                        dyn.append(block_entity_vertices("glowstone", p, 0.3, (0.5, t * 4, 0)))
                    else:
                        from mcengine.geometry import rot_y, rot_x
                        m = translate(*p) @ rot_y(t * 4) @ rot_x(1.2)
                        dyn.append(box_vertices((0.42, 0.04, 0.42), tile, matrix=m, emissive=0.2))
        return dyn, [], []


class Torches(Prop):
    def __init__(self, items, glow_min_phase=0.0, power=1.0):
        self.items = list(items)          # (t_appear, base(x,y,z))
        self.pmin, self.power = glow_min_phase, power

    def draw(self, t, f):
        dyn, glows = [], []
        for (ta, b) in self.items:
            if t < ta:
                continue
            dyn.append(torch_vertices(b, 0.1 * np.sin(t * 13 + b[0])))
            glows.append([b[0], b[1] + 0.72, b[2], (0.55 + 0.08 * np.sin(t * 9 + b[0])) * self.power, 1.0, 0.75, 0.35, 0.22])
        return dyn, [], glows

    def sfx(self):
        return [(ta, "pop", -15, dict(pitch=0.8)) for (ta, b) in self.items if ta > 0]


class Custom(Prop):
    def __init__(self, fn, fxfn=None, sfxlist=()):
        self.fn, self.fxfn, self.sfxlist = fn, fxfn, list(sfxlist)

    def draw(self, t, f):
        r = self.fn(t)
        if r is None:
            return [], [], []
        if isinstance(r, np.ndarray):
            return [r], [], []
        if isinstance(r, list):
            return r, [], []
        return r

    def fx(self, seed):
        return self.fxfn(seed) if self.fxfn else []

    def sfx(self):
        return self.sfxlist


class Reveal(Prop):
    """Progressive build: `blocks` [(pos, block)] appear in order between t0 and t1.
    Finished part is a cached static mesh (K chunks); the newest blocks pop in dynamically."""

    def __init__(self, key, blocks, t0, t1, K=36, ease=None, offset=(0, 0, 0)):
        self.key, self.blocks, self.t0, self.t1, self.K = key, list(blocks), t0, t1, K
        self.ease = ease or (lambda u: u)
        self.off = np.asarray(offset, float)
        self.n = len(self.blocks)

    def count(self, t):
        return int(self.n * self.ease(seg(t, self.t0, self.t1)))

    def time_of(self, i):
        # inverse of count (numeric)
        lo, hi = self.t0, self.t1
        for _ in range(30):
            mid = (lo + hi) / 2
            if self.count(mid) > i:
                hi = mid
            else:
                lo = mid
        return hi

    def draw(self, t, f):
        n = self.count(t)
        if n <= 0:
            return [], [], []
        k = int(n / max(self.n, 1) * self.K)
        nk = int(self.n * k / self.K)
        dyn, dyn_t = [], []
        if nk > 0:
            name = f"rev_{self.key}_{k}"
            if name not in f._r.static:
                w = VoxelWorld()
                for p, b in self.blocks[:nk]:
                    w.set(p, b)
                o, tr = w.mesh()
                if self.off.any():
                    o = o.copy(); tr = tr.copy()
                    o[:, 0:3] += self.off.astype(np.float32); tr[:, 0:3] += self.off.astype(np.float32)
                f._r.set_static(name, o)
                f._r.set_static(name + "_t", tr)
            f.extra_static.append(name)
            f.extra_static_t.append(name + "_t")
        # dynamic: blocks nk..n, newest popping
        rate = self.n / max(self.t1 - self.t0, 1e-3)
        for i in range(nk, n):
            p, b = self.blocks[i]
            age = (n - 1 - i) / max(rate, 1e-3)
            s = ease_out_back(clamp01(age / 0.15)) if age < 0.15 else 1.0
            v = block_entity_vertices(b, np.array(p, float) + 0.5 + self.off, max(s, 0.05))
            (dyn_t if tx.is_translucent(b) else dyn).append(v)
        return dyn, dyn_t, []

    def sfx(self):
        out = []
        tt = self.t0
        step = max(0.09, (self.t1 - self.t0) / max(self.n, 1))
        step = max(step, 0.13)
        i = 0
        while tt < self.t1:
            out.append((tt, "place", -14, dict(seed=i % 9)))
            tt += step
            i += 1
        return out


# ------------------------------------------------------------------ cameras
def cam_static(eye, target, fov=50, roll=0.0):
    e, g = np.asarray(eye, np.float32), np.asarray(target, np.float32)
    return lambda t: dict(eye=e, target=g, fov=fov, roll=roll)


def cam_dolly(t0, t1, eye0, eye1, tgt0, tgt1, fov=50, fov1=None, ease=ease_in_out, roll=0.0):
    e0, e1 = np.asarray(eye0, np.float32), np.asarray(eye1, np.float32)
    g0, g1 = np.asarray(tgt0, np.float32), np.asarray(tgt1, np.float32)
    f1 = fov if fov1 is None else fov1

    def fn(t):
        u = ease(seg(t, t0, t1))
        return dict(eye=e0 * (1 - u) + e1 * u, target=g0 * (1 - u) + g1 * u, fov=fov * (1 - u) + f1 * u, roll=roll)
    return fn


def cam_orbit(center, radius, height, a0, a1, t0, t1, fov=50, look_y=1.0, ease=ease_in_out, r1=None, h1=None):
    c = np.asarray(center, np.float32)
    rr1 = radius if r1 is None else r1
    hh1 = height if h1 is None else h1

    def fn(t):
        u = ease(seg(t, t0, t1))
        a = a0 + (a1 - a0) * u
        r = radius + (rr1 - radius) * u
        h = height + (hh1 - height) * u
        eye = c + np.array([np.sin(a) * r, h, np.cos(a) * r], np.float32)
        return dict(eye=eye, target=c + np.array([0, look_y, 0], np.float32), fov=fov, roll=0.0)
    return fn


def cam_track(actor, offset, look=(0, 1.2, 0), fov=50, smooth_t=0.0, offset1=None, t0=None, t1=None):
    off0 = np.asarray(offset, np.float32)
    off1 = off0 if offset1 is None else np.asarray(offset1, np.float32)
    lk = np.asarray(look, np.float32)

    def fn(t):
        if smooth_t > 0:
            p = np.mean([actor.pos(t - smooth_t * k / 4) for k in range(5)], axis=0)
        else:
            p = actor.pos(t)
        u = ease_in_out(seg(t, t0, t1)) if t0 is not None else 0.0
        off = off0 * (1 - u) + off1 * u
        return dict(eye=(p + off).astype(np.float32), target=(p + lk).astype(np.float32), fov=fov, roll=0.0)
    return fn


def cam_follow(actor, back=5.0, up=2.5, side=1.0, look=(0, 1.3, 0), fov=50, smooth_t=0.5, look_ahead=1.0):
    """Behind the actor relative to its (smoothed) heading."""
    lk = np.asarray(look, np.float32)

    def fn(t):
        ps = [actor.pos(t - smooth_t * k / 4) for k in range(5)]
        p = np.mean(ps, axis=0)
        yaws = [actor.yaw(t - smooth_t * k / 4) for k in range(5)]
        yaw = float(np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws))))
        fwd = np.array([np.sin(yaw), 0, np.cos(yaw)], np.float32)
        right = np.array([np.cos(yaw), 0, -np.sin(yaw)], np.float32)
        eye = p - fwd * back + right * side + np.array([0, up, 0], np.float32)
        tgt = p + lk + fwd * look_ahead
        return dict(eye=eye.astype(np.float32), target=tgt.astype(np.float32), fov=fov, roll=0.0)
    return fn


def cam_front(actor, dist=3.5, up=1.6, side=0.6, look=(0, 1.4, 0), fov=45, yaw=None, smooth_t=0.3):
    """In front of the actor, looking back at its face."""
    lk = np.asarray(look, np.float32)

    def fn(t):
        p = np.mean([actor.pos(t - smooth_t * k / 4) for k in range(5)], axis=0)
        yy = actor.yaw(t) if yaw is None else yaw
        fwd = np.array([np.sin(yy), 0, np.cos(yy)], np.float32)
        right = np.array([np.cos(yy), 0, -np.sin(yy)], np.float32)
        eye = p + fwd * dist + right * side + np.array([0, up, 0], np.float32)
        return dict(eye=eye.astype(np.float32), target=(p + lk).astype(np.float32), fov=fov, roll=0.0)
    return fn


def with_shake(cam, events):
    """events: [(t0, dur, amp)]"""
    def fn(t):
        c = dict(cam(t))
        a = 0.0
        for t0, d, amp in events:
            if t0 <= t < t0 + d:
                a = max(a, amp * (1 - (t - t0) / d) ** 2)
        if a > 0:
            s = shake(t, a)
            c["eye"] = c["eye"] + s
            c["target"] = c["target"] + s * 0.5
        return c
    return fn


# ------------------------------------------------------------------ shot
class Shot:
    def __init__(self, start, cam, loc="over", stage="P0", phase=0.0, actors=(), props=(), statics=(), fx=(), sfx=(),
                 titles=(), crafts=(), overlay=None, hud=True, crops="gold", post=None, light=None, music=None,
                 hp=None, hotbar=None, shake=(), flash=(), cave=None, torches_on=None, clouds=True, decor=True,
                 nether_tint=False, hud_items=None, label=None, fog=None, torch_reveal=None):
        self.start = start               # key | (key, frac) | absolute float
        self.cam, self.loc, self.stage, self.phase = cam, loc, stage, phase
        self.actors, self.props, self.statics = list(actors), list(props), list(statics)
        self.fx, self.sfx_list = list(fx), list(sfx)
        self.titles, self.crafts, self.overlay = list(titles), list(crafts), overlay
        self.hud, self.crops, self.post, self.light, self.music = hud, crops, post or {}, light, music
        self.hp, self.hotbar, self.shake, self.flash = hp, hotbar, list(shake), list(flash)
        self.cave, self.torches_on, self.clouds, self.decor = cave, torches_on, clouds, decor
        self.label = label
        self.fog = fog
        self.torch_reveal = torch_reveal
        self.t0 = None
        self.t1 = None

    def time_of(self, film):
        s = self.start
        if isinstance(s, (int, float)):
            return float(s)
        if isinstance(s, tuple):
            return film.T(s[0], s[1])
        return film.C(s)
