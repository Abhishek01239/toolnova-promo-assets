"""'Day 2: I Had Until Sunset to Find Diamonds': a complete ~10 min, 16:9 film.

All timing comes from the narration (assets/words.json). T(key, f) is the time f (0..1) of the way
through the line with that key (see script.py). Two sets: SURFACE (the Day 1 base) and UNDER.
"""
import json
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mcengine import textures as tx                                                     # noqa: E402
from mcengine.geometry import VoxelWorld, box_vertices, translate                       # noqa: E402
from mcengine.models import (pip_vertices, creeper_vertices, zombie_vertices, block_entity_vertices,  # noqa: E402
                             torch_vertices, item_vertices)
from mcengine.anim import (seg, smooth, ease_out, ease_in, ease_out_back, ease_in_out, shake, orbit,  # noqa: E402
                           Burst, gather, lerp, clamp01, Track)
from mcengine.lighting import environment                                                # noqa: E402
from mcengine import overlay as ov                                                       # noqa: E402
import script as S                                                                        # noqa: E402
import world as WD                                                                        # noqa: E402
from world import ug, U, FLOOR, GY                                                        # noqa: E402

PI = np.pi
W, H = 1920, 1080
TOWER = np.array([-3.5, 18.0, -3.5])
ROOF_TABLE = (1, 14, -1)
DEMO = np.array([8.5, 11.5, 10.5])
FURNACE = np.array([17.5, 11.5, -24.5])
MOUTH = np.array([18.6, GY, -17.0])
PORTAL_O = np.array([-15.0, GY, -1.0])
PORTAL = [(1, 0), (2, 0), (0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (3, 3), (1, 4), (2, 4)]
COAL = [(10, -8, 3), (11, -9, 3)]
CAVE_TORCHES = [(40.5, 0.6), (47.5, -3.4), (52.5, 7.0), (63.5, 7.0), (57.5, 4.6), (70.5, -2.4), (77.5, 1.6)]
TORCH_STEPS = [4, 8, 12, 16, 20, 24, 28, 31]
GRAVEL_STEP = 18
RUBBLE = (11, 17)          # steps refilled by the collapse


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


class PathTrack:
    """Piecewise-linear path with cumulative distance (for walk cycles) and velocity."""

    def __init__(self, keys):
        ks = sorted([(float(a), np.array(b, float)) for a, b in keys], key=lambda x: x[0])
        dd = []
        for a, b in ks:
            if dd and a <= dd[-1][0]:
                a = dd[-1][0] + 1e-3
            if dd and np.linalg.norm(b - dd[-1][1]) / max(a - dd[-1][0], 1e-6) > 25:
                dd.append((a - 1e-3, dd[-1][1].copy()))     # teleport (cut): hold, then jump
            dd.append((a, b))
        self.t = np.array([k[0] for k in dd])
        self.p = np.array([k[1] for k in dd], float)
        seglen = np.linalg.norm(np.diff(self.p[:, [0, 2]], axis=0), axis=1)
        seglen[seglen > 6] = 0          # teleports don't count as walking
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


class Film:
    def __init__(self, cache_dir):
        d = json.load(open(os.path.join(HERE, "assets", "words.json")))
        self.words = d["words"]
        self.chapters = d["chapters"]
        self.duration = d["duration"] + 0.6
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        n = len(S.LINES)
        self.ls = [min(w["start"] for w in self.words if w["line"] == i) for i in range(n)]
        self.le = [max(w["end"] for w in self.words if w["line"] == i) for i in range(n)]
        self.subs = ov.Subtitles(self.words, {"DIAMOND": (110, 235, 255), "DIAMONDS": (110, 235, 255),
                                              "IRON": (236, 190, 150), "LAVA": (255, 120, 40),
                                              "OBSIDIAN": (190, 140, 255), "CREEPER": (110, 235, 90),
                                              "ZOMBIE": (130, 220, 110), "ZOMBIES": (130, 220, 110),
                                              "RUN": (255, 90, 70), "BOOM": (255, 150, 30),
                                              "FIVE": (110, 235, 255), "FOUR": (110, 235, 255),
                                              "PICKAXE": (110, 235, 255), "NETHER": (200, 90, 255)},
                                 max_words=6, max_chars=34, y=930, size=66, cx=W / 2, max_w=1500)
        self.surface = WD.Surface(cache_dir)
        self._props_cache = {}
        self._build_timeline()
        self._build_events()
        self._build_actors()
        self._build_effects()
        self._red = self._red_vignette()
        self._orange = self._red_vignette((255, 120, 20))

    # ------------------------------------------------------------ timing helpers
    def T(self, key, f=0.0):
        i = S.IDX[key]
        return self.ls[i] + f * (self.le[i] - self.ls[i])

    def C(self, key):
        """Cut time for the shot that starts with this line."""
        return self.T(key, 0.0) - 0.15

    def E(self, key):
        """End of the line's shot (next cut)."""
        i = S.IDX[key]
        return self.C(S.KEYS[i + 1]) if i + 1 < len(S.KEYS) else self.duration

    def Wd(self, key, word, k=0):
        i = S.IDX[key]
        hits = [w for w in self.words if w["line"] == i and norm(w["word"]) == norm(word)]
        return hits[min(k, len(hits) - 1)]["start"] if hits else self.T(key, 0.5)

    def card(self, num):
        c = self.chapters[num - 1]
        return c["card_start"], c["card_end"]

    def key_at(self, t):
        cuts = [self.C(k) for k in S.KEYS]
        i = max(0, int(np.searchsorted(cuts, t, side="right")) - 1)
        return S.KEYS[i]

    # ------------------------------------------------------------ where are we
    def _build_timeline(self):
        under = ["deep"] + S.KEYS[S.IDX["eight"]:S.IDX["climb"] + 1] + ["obsidian", "downthere"]
        self.under_keys = set(under)

    def loc(self, t):
        return "U" if self.key_at(t) in self.under_keys else "S"

    def phase(self, t):
        T, C = self.T, self.C
        if t < C("breakfast"):
            return 3.0
        if t < C("outside"):
            return 0.0
        return float(Track([(C("outside"), 0.95, "linear"), (T("dash", 0.0), 1.05, "smooth"),
                            (T("madeit", 0.6), 1.85, "smooth"), (C("table"), 2.0, "smooth"),
                            (self.duration, 2.0, "linear")])(t)[()])

    # ------------------------------------------------------------ events
    def _build_events(self):
        T, Wd, C = self.T, self.Wd, self.C
        evS, evU = [], []
        # ---------------- surface: iron ore in the hill, mined on the counted words
        self.iron_pos = self.surface.iron
        for p in self.iron_pos:
            evS.append((-1.0, p, "iron_ore"))
        words = ["One", "Two", "Three"]
        t_ir = [Wd("iron_mine", w) for w in words]
        t_ir += [T("iron_mine", f) for f in (0.62, 0.78, 0.93)]
        self.iron_times = t_ir
        for tt, p in zip(t_ir, self.iron_pos):
            evS.append((tt + 0.25, p, None))
        # door sealed when Pip makes it home, opened to go out at night
        self.t_seal = T("sealdoor", 0.15)
        self.t_open = T("meet", 0.25)
        for y in (11, 12):
            evS.append((self.t_seal + 0.15 * (y - 11), (0, y, 4), "cobble"))
            evS.append((self.t_open + 0.1 * (y - 11), (0, y, 4), None))
        # ---------------- underground
        for k in range(1, WD.N_STEPS + 1):
            for y in range(-k, -k + 4):
                for z in range(3):
                    evU.append((-1.0, (k, y, z), "deepslate" if y < -17 else "stone"))
        for p in COAL:
            evU.append((-1.0, p, "coal_ore"))
        for x in range(*WD.LAVA_X):
            for z in range(*WD.BRIDGE_Z):
                evU.append((-1.0, (x, FLOOR - 1, z), "lava"))
        for p in WD.ORES:
            evU.append((-1.0, p, "deep_diamond"))
        # dig schedule: (start key/frac, end key/frac) for step ranges
        sched = [((1, 8), ("stairs", 0.15), ("coal", 0.0)), ((9, 11), ("coal", 0.05), ("coal", 0.9)),
                 ((12, 15), ("light", 0.1), ("torches", 0.0)), ((16, 18), ("torches", 0.1), ("gravel", 0.2)),
                 ((19, 22), ("trick", 0.25), ("deepslate", 0.0)), ((23, 29), ("deepslate", 0.05), ("breakthrough", 0.2)),
                 ((30, 31), ("breakthrough", 0.25), ("breakthrough", 0.55))]
        self.dig_t = {}
        for (k0, k1), a, b in sched:
            ta, tb = T(*a), T(*b)
            n = k1 - k0 + 1
            for i, k in enumerate(range(k0, k1 + 1)):
                self.dig_t[k] = ta + (tb - ta) * (i + 0.5) / n
        self.dig_t[32] = Wd("breakthrough", "nothing")
        for k, tt in self.dig_t.items():
            for y in range(-k, -k + 4):
                for z in range(3):
                    evU.append((tt, (k, y, z), None))
        self.coal_t = [T("coal", 0.45), T("coal", 0.65)]
        for tt, p in zip(self.coal_t, COAL):
            evU.append((tt, p, None))
        # gravel falls into step 18 right after it is dug, then gets dug out
        self.t_gravel = T("gravel", 0.35)
        self.gravel_blocks = [(GRAVEL_STEP, y, z) for y in (-18, -17) for z in range(3)]
        for p in self.gravel_blocks:
            evU.append((self.t_gravel + 0.45, p, "gravel"))
        self.t_gravel_dig = [T("trick", 0.05), T("trick", 0.15)]
        for i, p in enumerate(self.gravel_blocks):
            evU.append((self.t_gravel_dig[0 if p[1] == -17 else 1], p, None))
        # obsidian pours
        self.pour_t = [T("pour1", 0.3), T("pour2", 0.35), T("pour2", 0.75), T("careful", 0.85), T("bridge", 0.25)]
        for tt, (x0, x1) in zip(self.pour_t, WD.POURS):
            for x in range(x0, x1):
                for z in range(*WD.BRIDGE_Z):
                    evU.append((tt + 0.35 + 0.05 * (x - x0), (x, FLOOR - 1, z), "obsidian"))
        # diamonds
        self.ore_t = [T("first", 0.55), Wd("rest", "Second"), Wd("rest", "third"), Wd("rest", "fourth")]
        for tt, p in zip(self.ore_t, WD.ORES):
            evU.append((tt, p, None))
        # creeper blast collapses the stairs below Pip
        self.t_boom = Wd("boom", "BOOM") + 0.05
        self.rubble = []
        rng = np.random.default_rng(12)
        for k in range(RUBBLE[0], RUBBLE[1] + 1):
            for y in range(-k, -k + 4):
                for z in range(3):
                    b = "gravel" if rng.random() < 0.4 else ("cobble" if y > -17 else "deepslate")
                    tt = self.t_boom + 0.5 + 0.9 * rng.random() + 0.08 * (k - RUBBLE[0])
                    self.rubble.append((tt, (k, y, z), b))
        evU += self.rubble
        evS.sort(key=lambda e: e[0])
        evU.sort(key=lambda e: e[0])
        self.ev = {"S": evS, "U": evU}
        self.ev_t = {k: np.array([e[0] for e in v]) for k, v in self.ev.items()}

    def props_world(self, t, where):
        ev, et = self.ev[where], self.ev_t[where]
        n_set = int(np.searchsorted(et, t - 0.14, side="right"))
        n_all = int(np.searchsorted(et, t, side="right"))
        key = (where, n_set)
        if key not in self._props_cache:
            w = VoxelWorld()
            for (_, p, b) in ev[:n_set]:
                w.set(p, b)
            if where == "U":
                w2 = VoxelWorld()
                for p, b in w.blocks.items():
                    w2.set(p, b)
                self._props_cache[key] = w2
            else:
                self._props_cache[key] = w
        pops = [e for e in ev[n_set:n_all] if e[2] is not None and e[0] > 0]
        return self._props_cache[key], key, pops

    # ------------------------------------------------------------ actors
    def _build_actors(self):
        T, Wd, C, E = self.T, self.Wd, self.C, self.E
        y = GY
        k = []
        tw = TOWER
        # Ch1: tower at sunrise, then the farm, then the demo spot
        k += [(0.0, tw + (0.6, 0, 0.6)), (C("breakfast") - 0.001, tw + (0.6, 0, 0.6)),
              (C("breakfast"), (3.5, y, 11.3)), (C("neediron") - 0.001, (3.5, y, 11.3)),
              (C("neediron"), DEMO + (0, -0.5, 1.75)), (C("hill") - 0.001, DEMO + (0, -0.5, 1.75))]
        # Ch2: walk to the hill
        k += [(C("hill"), (6.0, y, 10.5)), (T("hill", 0.45), (11.0, y, 3.0)), (T("hill", 1.0), (15.5, y, -9.0)),
              (T("stillthere", 0.6), MOUTH + (0, 0, 0.4)), (T("stonepick", 1.0), MOUTH + (0, 0, 0.4)),
              (Wd("iron_mine", "One") - 0.9, MOUTH + (0, 0, 0.4)), (Wd("iron_mine", "One") - 0.1, (20.0, y, -21.0)), (T("why6", 1.0), (20.0, y, -21.0)),
              (C("raw"), (18.5, y, -21.2)), (T("raw", 0.8), (17.5, y, -22.5)), (T("bucket", 1.0), (17.5, y, -22.5)),
              (T("stairsready", 0.15), (17.8, y, -19.0)), (T("stairsready", 0.6), MOUTH + (0.9, 0, -0.5)),
              (T("stairsready", 1.0), MOUTH + (4.5, 0, -0.5))]
        # Ch3: the staircase (follow the dig front)
        k += [(C("eight"), ug(-1.4, 0, 1.2)), (T("stairs", 0.1), ug(-0.4, 0, 1.2))]
        for st in range(1, WD.N_STEPS + 1):
            td = self.dig_t[st]
            if st == GRAVEL_STEP:
                k += [(td + 0.35, ug(st + 0.3, -st, 1.2)), (self.t_gravel + 0.15, ug(st + 0.3, -st, 1.2)),
                      (self.t_gravel + 0.45, ug(st - 0.6, -(st - 1), 1.2)),
                      (self.t_gravel_dig[1] + 0.3, ug(st - 0.6, -(st - 1), 1.2))]
                continue
            k.append((td + 0.35, ug(st + 0.3, -st, 1.2)))
        # Ch4: cavern
        tb = self.dig_t[32]
        k += [(tb + 1.2, ug(33.0, FLOOR, 1.5)), (T("cavern", 1.0), ug(38.0, FLOOR, 1.5)),
              (T("ceiling", 0.8), ug(48.0, FLOOR, 3.0)), (T("size", 1.0), ug(55.5, FLOOR, 5.6)),
              (T("fill", 0.1), ug(56.0, FLOOR, 6.4)), (T("night", 1.0), ug(56.0, FLOOR, 6.4)),
              (T("orange", 0.0), ug(60.0, FLOOR, 4.5)), (T("orange", 1.0) + 0.8, ug(84.0, FLOOR, 3.5))]
        # Ch5: lava chamber, the bridge built pour by pour
        k += [(C("lava"), ug(90.6, FLOOR, 3.5)), (self.pour_t[0], ug(91.2, FLOOR, 3.5))]
        for i in range(1, 5):
            edge = WD.POURS[i - 1][1] - 0.8
            k += [(self.pour_t[i - 1] + 0.9, ug(WD.POURS[i - 1][0] + 0.2, FLOOR, 3.5)),
                  (self.pour_t[i] - 0.6, ug(edge, FLOOR, 3.5))]
            if i == 3:     # the slip: steps onto lava and jumps back
                ts = T("slip", 0.35)
                k += [(ts - 0.4, ug(edge, FLOOR, 3.5)), (ts, ug(edge + 0.9, FLOOR - 0.25, 3.5)),
                      (T("fire", 0.45), ug(edge + 0.9, FLOOR - 0.25, 3.5)), (T("fire", 0.6), ug(edge - 0.6, FLOOR, 3.5)),
                      (T("careful", 0.3), ug(edge - 0.6, FLOOR, 3.5)), (T("careful", 0.7), ug(edge - 0.1, FLOOR, 3.5)),
                      (self.pour_t[3] - 0.05, ug(edge - 0.1, FLOOR, 3.5))]
        k += [(self.pour_t[4] + 1.2, ug(104.6, FLOOR, 3.5)), (T("bridge", 1.0), ug(104.6, FLOOR, 3.5))]
        # Ch6: the ledge
        k += [(C("four") + 0.3, ug(108.0, FLOOR, 4.0)), (T("four", 0.6), ug(114.4, FLOOR, 4.0)),
              (T("time", 1.0), ug(114.4, FLOOR, 4.0))]
        # Ch7: back across, lost, torches, creeper
        k += [(C("ranback"), ug(112.0, FLOOR, 3.5)), (T("ranback", 0.35), ug(91.0, FLOOR, 3.5)),
              (T("ranback", 0.7), ug(76.0, FLOOR, 3.0)), (T("ranback", 0.9), ug(64.0, FLOOR, 0.5)),
              (T("same", 1.0), ug(64.0, FLOOR, 0.5)),
              (T("dead", 0.3), ug(49.5, FLOOR, -15.5)), (T("dead", 0.45), ug(49.5, FLOOR, -14.5)),
              (T("dead", 0.7), ug(52.0, FLOOR, -4.0)), (T("dead", 1.0), ug(60.5, FLOOR, -9.0)),
              (T("wrongway", 1.0), ug(60.0, FLOOR, -3.0)), (T("remember", 1.0), ug(60.0, FLOOR, -3.0)),
              (T("follow", 1.0) + 1.0, ug(36.0, FLOOR, 1.4)), (T("explode", 1.0), ug(35.0, FLOOR, 1.4))]
        # Ch8: up the stairs, the blast, the climb, the dash home
        tr = C("run") + 0.3
        steps_up = list(range(32, -1, -1))
        t_top = T("climb", 0.95)
        kb = 14            # step Pip is on at the blast
        for st in steps_up:
            if st >= kb:
                tt = tr + (self.t_boom - tr) * (32 - st) / (32 - kb)
            else:
                tt = self.t_boom + 0.9 + (t_top - self.t_boom - 0.9) * (kb - st) / kb
                if T("collapse", 0.0) < tt < T("climb", 0.0):
                    tt = None
            if tt is not None:
                k.append((tt, ug(st + 0.5, -st, 1.4)))
        k += [(T("sealed", 0.0), ug(kb - 0.5, -(kb - 1), 1.4)), (T("climb", 0.0), ug(kb - 0.5, -(kb - 1), 1.4)),
              (T("climb", 0.97), ug(-3.0, 0, 1.4))]
        k += [(C("outside"), (21.5, y, -17.5)), (T("outside", 0.5), (16.5, y, -17.0)), (T("outside", 1.0), (15.0, y, -15.0)),
              (T("dash", 0.25), (12.0, y, -6.0)), (T("dash", 0.5), (10.0, y, 4.0)), (T("dash", 0.62), (7.0, y, 8.5)),
              (T("dash", 0.75), (0.5, y, 8.5)), (T("dash", 0.85), (0.5, y, 6.5)), (T("dash", 0.95), (0.5, y, 3.0)),
              (T("madeit", 1.0), (0.5, y, 2.0))]
        # Ch9: the roof at night
        k += [(C("table") - 0.001, (0.5, y, 2.0)), (C("table"), (1.5, 14.0, 1.0)), (T("cue", 0.3), (1.5, 14.0, 1.0)), (T("cue", 1.0), (2.6, 14.0, 3.2))]
        # Ch10: night two by the bridge
        k += [(C("meet") - 0.001, (2.6, 14.0, 3.2)), (C("meet"), (0.5, y, 2.0)), (T("meet", 0.5), (0.5, y, 6.5)), (T("meet", 1.0), (1.4, y, 8.3)),
              (T("z1", 1.0), (1.4, y, 8.3)), (T("z2", 0.25), (7.4, y, 7.0)), (T("z2", 1.0), (7.4, y, 7.0)),
              (T("z3", 0.15), (2.0, y, 8.6)), (T("knock", 1.0), (2.0, y, 8.6)),
              (C("tower") - 0.001, (2.0, y, 8.6)), (C("tower"), tw + (0.6, 0, 0.6)), (self.duration, tw + (0.6, 0, 0.6))]
        self.pip_path = PathTrack(k)
        self.hidden = [(C("deep"), self.E("deep")), (C("obsidian"), self.E("downthere")), (C("portal"), self.E("portal"))]

        # ---------------- mobs
        self.mobs = []

        def mob(kind, keys, spawn, death=None, **kw):
            m = dict(kind=kind, path=PathTrack(keys), spawn=spawn, death=death, **kw)
            self.mobs.append(m)
            return m
        # cavern zombie behind the lake
        tz = C("groan")
        self.z_cave = mob("z", [(tz, ug(44.0, FLOOR, -3.0)), (T("zombie", 0.6), ug(53.6, FLOOR, 4.2)),
                                (T("swing", 1.0), ug(54.6, FLOOR, 5.0))],
                          tz, death=Wd("down", "down") + 0.05,
                          hits=[Wd("swing", "Once"), Wd("swing", "Twice"), Wd("swing", "Three")])
        # distant cave zombies in the 'always night' shot
        for i, (x, z) in enumerate([(72.0, -6.0), (75.0, 9.0), (46.0, 11.0)]):
            mob("z", [(C("night"), ug(x, FLOOR, z)), (self.E("night"), ug(x + 1.5, FLOOR, z - 1.0))], C("night"),
                death=self.E("night"))
        # the creeper that follows Pip up the stairs
        tc = C("hiss")
        ck = [(tc, ug(48.0, FLOOR, 2.0)), (T("explode", 1.0), ug(38.0, FLOOR, 1.4))]
        for st in range(32, 0, -1):
            tt = tr + 0.9 + (self.t_boom - tr - 0.9) * (32 - st) / (32 - (kb + 4))
            if tt > self.t_boom:
                break
            ck.append((tt, ug(st + 0.5, -st, 1.4)))
        ck.append((self.t_boom, ug(kb + 4.5, -(kb + 4), 1.4)))
        self.creeper = mob("c", ck, tc, death=self.t_boom)
        # surface zombies rising during the dash (they stay in the field)
        self.dash_z = []
        for i, (x, z) in enumerate([(15.0, -2.0), (17.0, 6.0), (13.0, 11.0)]):
            ts = T("zombies", 0.1 + 0.25 * i)
            self.dash_z.append(mob("z", [(ts, (x, y, z)), (T("madeit", 1.0), (x - 3.0, y, z - 1.0))], ts,
                                   death=C("table"), rise=True))
        # night two: three zombies
        self.n2 = []
        z1k = [(C("cue"), (14.0, y, 9.5)), (T("night2", 1.0), (6.5, y, 8.6)), (T("meet", 1.0), (3.1, y, 8.4)),
               (T("z1", 1.0), (3.1, y, 8.4))]
        self.n2.append(mob("z", z1k, C("cue"), death=Wd("z1", "Gone") + 0.05,
                           hits=[Wd("z1", "One"), Wd("z1", "Two"), Wd("z1", "Three")]))
        z2k = [(C("cue"), (17.0, y, 1.0)), (T("night2", 1.0), (8.0, y, -2.0)), (T("z1", 1.0), (7.5, y, 2.5)),
               (T("z2", 0.55), (7.4, y, 5.2)), (Wd("z2", "Big"), (7.4, y, 5.2)),
               (Wd("z2", "Big") + 0.4, (6.5, 9.9, 4.6)), (T("z2", 1.0), (6.5, 9.4, 4.6))]
        self.n2.append(mob("z", z2k, C("cue"), death=self.E("z2") - 0.2, hits=[Wd("z2", "Big") - 0.1], sink=True))
        z3k = [(C("cue"), (18.0, y, 12.0)), (T("night2", 1.0), (9.0, y, 10.0)), (T("z2", 1.0), (2.0, y, 9.0)),
               (T("z3", 0.1), (0.5, y, 7.4)), (T("z3", 0.4), (0.5, y, 6.4)),
               (T("z3", 0.55), (1.5, 9.9, 6.5)), (T("z3", 1.0), (1.5, 9.4, 6.5))]
        self.n2.append(mob("z", z3k, C("cue"), death=self.E("z3"), hits=[T("z3", 0.42)], sink=True))

    def hidden_pip(self, t):
        return any(a <= t < b for a, b in self.hidden)

    def pip_state(self, t):
        T, Wd, C = self.T, self.Wd, self.C
        key = self.key_at(t)
        pos = self.pip_path.pos(t)
        v = self.pip_path.vel(t)
        sp = float(np.hypot(v[0], v[2]))
        pose = {"walk": clamp01(sp / 3.2), "walk_phase": self.pip_path.dist(t) * 2.4}
        yaw = float(np.arctan2(v[0], v[2])) if sp > 0.4 else None
        if yaw is None:
            yaw = {"hook": PI * 0.25, "recap": PI * 0.25, "halfheart": PI * 0.25, "hook2": PI * 0.25,
                   "need3": PI * 0.25, "sunset": PI * 0.75, "morning": PI * 0.75,
                   "neediron": PI, "stonepick": -PI / 2, "raw": PI, "furnace": PI, "ingots": PI, "smelt": PI,
                   "ironpick": PI, "bucket": PI * 0.8, "cavern": PI / 2, "ceiling": PI / 2,
                   "glow": 0.0, "size": 0.0, "fill": 0.0, "groan": 0.0,
                   "sat": -PI / 2, "time": -PI / 2, "same": PI * 0.3, "wrongway": PI, "remember": PI * 0.7,
                   "hiss": -PI / 2, "creeper": PI / 2, "explode": PI / 2,
                   "table": PI, "recipe": PI, "dpick": PI, "dsword": PI, "glow2": PI, "toolfact": PI, "eat": PI,
                   "night2": PI / 2, "meet": PI / 2, "z1": PI / 2, "z2": PI, "z3": PI,
                   "knock": PI, "zero": 0.4, "tower": PI * 0.25, "plan3": PI * 0.25, "outro": PI * 0.25}.get(key, None)
            if yaw is None:
                if self.loc(t) == "U":
                    yaw = PI / 2
                elif t >= C("iron_found") and t < C("raw"):
                    yaw = PI / 2
                else:
                    yaw = 0.0
        item = None
        # tools in hand over the day
        if t >= C("neediron") and t < C("hill"):
            item = "pickaxe:stone"
        if C("hill") <= t < T("stonepick", 0.9):
            item = "pickaxe"
        if T("stonepick", 0.9) <= t < T("ironpick", 0.6):
            item = "pickaxe:stone"
        if t >= T("ironpick", 0.6):
            item = "pickaxe:iron"
        if C("raw") <= t < C("ironpick"):
            item = None
        if C("bucket") <= t < T("bucket", 0.95):
            item = None
        if C("fill") <= t < C("groan") or C("goodthing") <= t < C("bridge") + 2.0:
            item = "water_bucket" if t < T("fill", 0.4) and t < C("groan") else "bucket"
            if C("goodthing") <= t:
                # full between pours, empty right after pouring
                item = "water_bucket"
                for tp in self.pour_t:
                    if tp <= t < tp + 0.9:
                        item = "bucket"
        if C("fill") <= t < C("groan"):
            item = "water_bucket" if t >= T("fill", 0.4) else "bucket"
        if C("torches") <= t < T("torches", 1.0) or C("follow") <= t < C("hiss"):
            item = "torch"
        if C("table") <= t:
            item = None
        if T("dpick", 0.0) <= t < C("dsword"):
            item = "pickaxe:diamond"
        if T("dsword", 0.6) <= t:
            item = "sword"
        if C("tower") <= t:
            item = "pickaxe:diamond" if t >= C("plan3") else "sword"
        if C("breakfast") <= t < C("neediron"):
            item = None
        if t < C("breakfast"):
            item = None
        pose["item"] = item
        # actions
        acts = list(self.iron_times) + list(self.dig_t.values()) + self.coal_t + self.t_gravel_dig + self.ore_t
        acts += [T("breakfast", f) for f in (0.4, 0.55, 0.7, 0.85)]
        acts += [T("neediron", f) for f in (0.25, 0.35, 0.45)]
        acts += self.z_cave["hits"] + [h for m in self.n2 for h in m["hits"]]
        for tp in acts:
            if tp - 0.05 <= t < tp + 0.2:
                pose["punch"] = seg(t, tp - 0.05, tp + 0.2)
        for tp in self.pour_t + [T("fill", 0.3)]:
            if tp - 0.2 <= t < tp + 0.6:
                pose["reach"] = np.sin(PI * seg(t, tp - 0.2, tp + 0.6))
        # eating
        for te in (T("bread", 0.05), T("eat", 0.05)):
            if te <= t < te + 1.4:
                pose["punch"] = 0.35 + 0.15 * np.sin((t - te) * 18)
                pose["head_pitch"] = -0.2
        # raising the diamond pickaxe
        if T("dpick", 0.05) <= t < T("dpick", 1.0) + 1.0:
            pose["arms_up"] = 0.9
            pose["flail_phase"] = 0.0
        # looks
        if key in ("ceiling",):
            pose["head_pitch"] = 0.55
        if key in ("same", "wrongway"):
            pose["head_yaw"] = 0.9 * np.sin((t - C(key)) * 2.2)
        if key in ("groan",):
            pose["head_yaw"] = -0.9 * smooth(seg(t, T("groan", 0.5), T("groan", 0.9)))
        if key in ("why6", "dontknow"):
            pose["head_pitch"] = -0.15
            pose["head_yaw"] = 0.3
        if key in ("careful",):
            pose["lean"] = 0.35
        if key in ("hiss",):
            pose["head_yaw"] = 1.2 * smooth(seg(t, T("hiss", 0.6), T("hiss", 0.9)))
        if key in ("zero", "madeit", "five"):
            pose["wave"] = 1.0
            pose["wave_phase"] = t * 12
        if key == "outro":
            pose["wave"] = smooth(seg(t, C("outro"), C("outro") + 0.4))
            pose["wave_phase"] = t * 10
        # the slip: on fire
        ts = T("slip", 0.35)
        if ts <= t < T("fire", 0.6):
            pose["arms_up"] = 0.8
            pose["flail_phase"] = t * 9
        if T("fire", 0.45) <= t < T("fire", 0.6):
            u = seg(t, T("fire", 0.45), T("fire", 0.6))
            pos = pos.copy()
            pos[1] += 1.0 * np.sin(PI * u) + 0.25 * u
        if key == "sat":
            pos = pos.copy()
            pos[1] -= 0.0
        return pos, yaw, pose

    # ------------------------------------------------------------ effects
    def _build_effects(self):
        T, Wd, C = self.T, self.Wd, self.C
        B = self.bursts = []
        sm = [(0.3, 0.3, 0.32, 0.7), (0.45, 0.45, 0.45, 0.6)]

        def chips(tt, p, col, n=24, seed=0):
            B.append(Burst(tt, p, n, speed=(1.5, 4), life=(0.4, 0.8), size=(0.06, 0.12), colors=col,
                           gravity=-14, seed=seed, box=0.4))

        def poof(tt, p, seed):
            B.append(Burst(tt, p, 60, speed=(0.5, 2.5), life=(0.6, 1.3), size=(0.25, 0.6),
                           colors=[(0.85, 0.85, 0.85, 0.85), (0.6, 0.6, 0.6, 0.8)], gravity=1.0, drag=2.0,
                           grow=1.6, seed=seed, box=0.5))
        stone = [(0.5, 0.5, 0.5, 1), (0.35, 0.35, 0.35, 1)]
        deep = [(0.3, 0.3, 0.33, 1), (0.22, 0.22, 0.25, 1)]
        for i, (tt, p) in enumerate(zip(self.iron_times, self.iron_pos)):
            chips(tt + 0.25, (p[0] + 0.3, p[1] + 0.5, p[2] + 0.5), stone + [(0.85, 0.62, 0.44, 1)], seed=10 + i)
        for k, tt in self.dig_t.items():
            chips(tt, ug(k + 0.3, -k + 1.3, 1.5), deep if k > 17 else stone, n=30, seed=100 + k)
        for i, tt in enumerate(self.coal_t):
            p = COAL[i]
            chips(tt, ug(p[0] + 0.5, p[1] + 0.5, p[2] - 0.1), stone + [(0.08, 0.08, 0.08, 1)], seed=150 + i)
        # gravel dust
        B.append(Burst(self.t_gravel + 0.45, ug(GRAVEL_STEP + 0.5, -17.5, 1.5), 80, speed=(1, 4), life=(0.6, 1.4),
                       size=(0.1, 0.25), colors=[(0.55, 0.52, 0.5, 0.9)], gravity=-6, seed=160, box=1.2))
        for i, tt in enumerate(self.t_gravel_dig):
            chips(tt, ug(GRAVEL_STEP + 0.5, -17.0 - i, 1.5), [(0.55, 0.52, 0.5, 1)], n=40, seed=165 + i)
        # demo: the diamond ore breaks into nothing
        self.t_demo_break = T("neediron", 0.47)
        B.append(Burst(self.t_demo_break, tuple(DEMO), 70, speed=(1.5, 5), life=(0.4, 0.9), size=(0.06, 0.12),
                       colors=stone + [(0.4, 0.9, 0.9, 1)], gravity=-14, seed=170, box=0.4))
        # harvest
        for i in range(4):
            chips(T("breakfast", 0.4 + 0.15 * i) + 0.1, (2.5 + i * 0.6, 9.6, 13.2), [(0.85, 0.75, 0.3, 1), (0.4, 0.7, 0.2, 1)],
                  n=14, seed=180 + i)
        # furnace smoke & fire
        tf0, tf1 = T("furnace", 0.55), T("smelt", 1.0)
        tt, kk = tf0, 0
        while tt < tf1:
            B.append(Burst(tt, tuple(FURNACE + (0, 0.7, 0)), 3, speed=(0.2, 0.6), life=(1.5, 2.5), size=(0.25, 0.5),
                           colors=[(0.5, 0.5, 0.52, 0.45)], gravity=1.4, drag=0.8, grow=1.8, seed=200 + kk, box=0.3))
            B.append(Burst(tt, tuple(FURNACE + (0, -0.2, 0.55)), 4, speed=(0.3, 1.0), life=(0.3, 0.6), size=(0.05, 0.1),
                           colors=[(1, 0.7, 0.2, 1), (1, 0.45, 0.1, 1)], gravity=2.0, seed=600 + kk, additive=True, box=0.2))
            tt += 0.35
            kk += 1
        # cavern drips
        tt, kk = C("ceiling"), 0
        while tt < C("groan"):
            x = WD.LAKE_C[0] + 5 * np.sin(kk * 1.7)
            z = WD.LAKE_C[1] + 2.5 * np.cos(kk * 2.3)
            B.append(Burst(tt, ug(x, FLOOR - 0.9, z), 10, speed=(0.3, 1.2), life=(0.3, 0.6), size=(0.04, 0.07),
                           colors=[(0.6, 0.75, 1.0, 0.9)], gravity=-8, up_bias=1.5, seed=700 + kk, box=0.1))
            tt += 0.9
            kk += 1
        # zombie hits and poofs
        for m in self.mobs:
            for j, h in enumerate(m.get("hits", [])):
                p = m["path"].pos(h)
                B.append(Burst(h + 0.1, (p[0], p[1] + 1.2, p[2]), 14, speed=(1, 3), life=(0.2, 0.4), size=(0.06, 0.12),
                               colors=[(1, 0.3, 0.3, 1), (1, 1, 1, 1)], gravity=-6, seed=800 + j + int(h * 7) % 97,
                               additive=True, box=0.3))
            if m["death"] is not None and m.get("hits"):
                p = m["path"].pos(m["death"])
                poof(m["death"], (p[0], p[1] + 0.9, p[2]), seed=900 + int(m["death"] * 3) % 97)
            if m.get("sink"):
                p = m["path"].pos(m["death"] - 1.0)
                B.append(Burst(m["death"] - 1.2, (p[0], GY - 0.9, p[2]), 50, speed=(1, 3.5), life=(0.4, 0.9),
                               size=(0.08, 0.16), colors=[(0.6, 0.75, 1.0, 0.9), (1, 1, 1, 0.9)], gravity=-9,
                               up_bias=1.5, seed=950 + int(m["death"]) % 37, box=0.4))
            if m.get("rise"):
                p = m["path"].pos(m["spawn"])
                B.append(Burst(m["spawn"], (p[0], GY + 0.1, p[2]), 30, speed=(1, 3), life=(0.5, 1.0),
                               size=(0.06, 0.12), colors=[(0.4, 0.28, 0.18, 1), (0.3, 0.45, 0.2, 1)], gravity=-12,
                               up_bias=1.2, seed=980 + int(m["spawn"]) % 13, box=0.4))
        # lava: embers + pour steam
        tt, kk = C("orange"), 0
        while tt < C("ranback") + 6:
            x = WD.LAVA_X[0] + (kk * 3.7) % (WD.LAVA_X[1] - WD.LAVA_X[0])
            z = WD.LAVA_Z[0] + (kk * 5.3) % (WD.LAVA_Z[1] - WD.LAVA_Z[0])
            B.append(Burst(tt, ug(x, FLOOR - 0.05, z), 6, speed=(0.5, 2.0), life=(0.6, 1.4), size=(0.05, 0.1),
                           colors=[(1, 0.6, 0.15, 1), (1, 0.85, 0.3, 1)], gravity=1.2, up_bias=2.0, drag=0.6,
                           seed=1000 + kk, additive=True, box=0.4))
            tt += 0.22
            kk += 1
        for i, tp in enumerate(self.pour_t):
            x0, x1 = WD.POURS[i]
            c = ug((x0 + x1) / 2, FLOOR - 0.3, 3.5)
            B.append(Burst(tp + 0.25, c, 120, speed=(1, 4), life=(1.0, 2.2), size=(0.4, 1.0),
                           colors=[(0.9, 0.9, 0.92, 0.6), (0.75, 0.75, 0.78, 0.5)], gravity=2.0, drag=1.4, grow=2.0,
                           seed=1500 + i, box=1.2))
        # Pip on fire
        ts = T("slip", 0.35)
        tt, kk = ts, 0
        while tt < T("fire", 0.65):
            p = self.pip_path.pos(tt)
            B.append(Burst(tt, (p[0], p[1] + 1.0, p[2]), 8, speed=(0.3, 1.2), life=(0.3, 0.6), size=(0.12, 0.3),
                           colors=[(1, 0.7, 0.2, 1), (1, 0.4, 0.05, 1), (1, 0.9, 0.4, 1)], gravity=3, drag=1,
                           seed=1600 + kk, additive=True, box=0.35))
            tt += 0.08
            kk += 1
        # diamonds
        for i, tt in enumerate(self.ore_t):
            p = WD.ORES[i]
            chips(tt, ug(p[0] - 0.1, p[1] + 0.5, p[2] + 0.5), deep + [(0.4, 0.95, 0.95, 1)], n=36, seed=1700 + i)
            B.append(Burst(tt, ug(p[0] - 0.1, p[1] + 0.5, p[2] + 0.5), 40, speed=(1, 3), life=(0.5, 1.0),
                           size=(0.05, 0.1), colors=[(0.6, 1, 1, 1), (1, 1, 1, 1)], gravity=0, drag=2,
                           seed=1750 + i, additive=True, box=0.3))
        tt, kk = C("sparkle"), 0
        while tt < self.ore_t[-1]:
            for j, p in enumerate(WD.ORES):
                if tt < self.ore_t[j]:
                    B.append(Burst(tt, ug(p[0] - 0.05, p[1] + 0.5, p[2] + 0.5), 3, speed=(0.2, 0.8), life=(0.4, 0.8),
                                   size=(0.04, 0.08), colors=[(0.6, 1, 1, 1), (1, 1, 1, 1)], gravity=0.4, drag=2,
                                   seed=1800 + kk * 4 + j, additive=True, box=0.5))
            tt += 0.3
            kk += 1
        # creeper blast on the stairs
        P = tuple(self.creeper["path"].pos(self.t_boom) + np.array([0, 1.0, 0]))
        B.append(Burst(self.t_boom, P, 220, speed=(3, 13), life=(0.35, 1.0), size=(0.4, 1.2),
                       colors=[(1, 0.95, 0.6, 1), (1, 0.6, 0.15, 1), (1, 0.35, 0.05, 1)], gravity=3, drag=3, grow=1.2,
                       seed=1900, additive=True))
        B.append(Burst(self.t_boom, P, 160, speed=(2, 8), life=(1.4, 3.5), size=(0.7, 1.6),
                       colors=[(0.25, 0.24, 0.24, 0.8), (0.4, 0.38, 0.36, 0.75)], gravity=1.0, drag=1.8, grow=2.0,
                       seed=1901))
        for i, (tt, p, b) in enumerate(self.rubble[::3]):
            B.append(Burst(tt, ug(p[0] + 0.5, p[1] + 0.6, p[2] + 0.5), 8, speed=(1, 3), life=(0.5, 1.0),
                           size=(0.1, 0.25), colors=[(0.5, 0.48, 0.46, 0.8)], gravity=-4, seed=1950 + i, box=0.5))
        # diamond tool sparkles
        for i, tt in enumerate((T("dpick", 0.0), T("dsword", 0.6))):
            B.append(Burst(tt, (1.5, 15.6, 0.6), 80, speed=(1, 3.5), life=(0.6, 1.3), size=(0.05, 0.11),
                           colors=[(0.6, 1, 1, 1), (1, 1, 1, 1)], gravity=0, drag=2, seed=2000 + i, additive=True, box=0.3))
        # portal sparkle
        tt, kk = T("portal", 0.6), 0
        while tt < self.E("portal"):
            B.append(Burst(tt, tuple(PORTAL_O + (2.0, 2.5, 0.5)), 6, speed=(0.2, 0.9), life=(0.8, 1.6),
                           size=(0.08, 0.18), colors=[(0.7, 0.3, 1.0, 1), (0.9, 0.6, 1.0, 1)], gravity=0.3, drag=1.5,
                           seed=2100 + kk, additive=True, box=1.0))
            tt += 0.12
            kk += 1

    def _red_vignette(self, col=(200, 0, 0)):
        yy = np.linspace(-1, 1, H)[:, None]
        xx = np.linspace(-1, 1, W)[None, :]
        r = np.clip((np.sqrt(xx ** 2 * 0.7 + yy ** 2 * 1.1) - 0.5) / 0.6, 0, 1) ** 1.5
        a = np.zeros((H, W, 4), np.uint8)
        a[..., 0], a[..., 1], a[..., 2] = col
        a[..., 3] = (r * 255).astype(np.uint8)
        return a

    # ------------------------------------------------------------ camera
    def camera(self, t):
        T, Wd, C = self.T, self.Wd, self.C
        key = self.key_at(t)
        u = seg(t, C(key), self.E(key))
        pip, yaw, _ = self.pip_state(t)
        P = pip + np.array([0, 1.25, 0])
        fwd = np.array([np.sin(yaw), 0, np.cos(yaw)])
        side = np.array([np.cos(yaw), 0, -np.sin(yaw)])
        L = ug
        fov, roll = 55.0, 0.0
        eye, tgt = pip + (4, 2.5, 4), P

        def fight(m):
            zp = m["path"].pos(t)
            mid = (pip + zp) / 2
            d = zp - pip
            d[1] = 0
            n = np.linalg.norm(d)
            pr = np.array([-d[2], 0, d[0]]) / (n if n > 1e-3 else 1)
            return mid + pr * 4.2 + np.array([0, 1.8, 0]), mid + np.array([0, 1.0, 0])

        def stairs_behind(d=2.4):
            return pip + np.array([-d, d + 1.5, 0.45]), pip + np.array([1.6, 0.4, 0.0])

        def stairs_above(d=2.4):
            return pip + np.array([-d, d + 1.35, 0.4]), pip + np.array([0.0, 1.1, 0.0])

        # ---------------- chapter 1
        if key == "hook":
            uu = ease_out(u, 3)
            eye = lerp(pip + fwd * 1.5 + (0, 1.62, 0), pip + fwd * 7 + side * 3 + (0, 3.5, 0), uu)
            tgt = lerp(pip + (0, 1.6, 0), P, uu)
            fov = 48 + 10 * uu
        elif key == "recap":
            eye = orbit((0.5, 11, 6), 28.0, 0.6 + 0.9 * u, 15.0)
            tgt = (0.5, 10.5, 8.0)
            fov = 58
        elif key == "halfheart":
            eye = pip + fwd * 3.0 + side * 0.8 + (0, 1.4, 0)
            tgt = P + (0, 0.2, 0)
            fov = 48
        elif key == "hook2":
            eye = pip + fwd * 2.3 - side * 1.3 + (0, 1.5, 0)
            tgt = P + (0, 0.3, 0)
            fov = 46
        elif key == "need3":
            eye = pip + fwd * 3.9 + (0, 1.5, 0)
            v = P - eye
            rc = np.cross(v, (0, 1, 0))
            rc /= np.linalg.norm(rc)
            tgt = P + rc * 0.8
            fov = 50
        elif key == "deep":
            uu = smooth(u)
            eye = lerp(L(95, -28.5, -2.5), L(103, -29.6, 0.5), uu)
            tgt = L(113, -31.2, 3.8)
            fov = 60
        elif key == "sunset":
            eye = pip - fwd * 3.0 + side * 1.0 + (0, 1.7, 0)
            tgt = pip + fwd * 8 + (0, 2.0, 0)
            fov = 60
        elif key == "morning":
            eye = orbit((0.5, 12, 0.5), 21.0, 2.4 + 0.8 * u, 11.0)
            tgt = (0.5, 12.5, 0.5)
            fov = 58
        elif key == "breakfast":
            eye = lerp(np.array([8.0, 14.0, 16.5]), np.array([7.0, 13.2, 15.5]), u)
            tgt = (3.0, 10.4, 13.2)
        elif key == "wheat":
            eye = lerp(np.array([5.5, 11.9, 20.0]), np.array([4.5, 11.5, 19.0]), u)
            tgt = (0.8, 9.0, 15.5)
            fov = 52
        elif key in ("bread", "regen"):
            eye = pip + (0.7, 1.5, 2.7) + (0, 0, -0.4 * u if key == "regen" else 0)
            tgt = P + (0, 0.2, 0)
            fov = 48
        elif key == "plan":
            eye = pip + (4.5, 2.6, 3.5)
            tgt = P + (-1.0, 0, 0)
        elif key == "dontknow":
            eye = pip + (-1.3, 1.55, 2.3)
            tgt = P + (0, 0.25, 0)
            fov = 45
        elif key == "neediron":
            eye = np.array([11.8, 12.6, 11.9])
            tgt = (8.5, 11.6, 11.3)
            fov = 50
        # ---------------- chapter 2
        elif key == "hill":
            eye = lerp(np.array([-3.0, 17.0, 16.0]), np.array([3.0, 16.5, 2.0]), u)
            tgt = P
            fov = 58
        elif key == "stillthere":
            eye = np.array([13.0, 13.2, -12.5])
            tgt = (20.0, 12.0, -17.5)
        elif key == "stonepick":
            eye = pip + (-2.6, 1.5, 1.8)
            tgt = P + (0.2, 0.1, -0.6)
            fov = 50
        elif key == "iron_found":
            uu = smooth(u)
            eye = lerp(np.array([16.6, 12.9, -20.6]), np.array([18.2, 12.5, -21.0]), uu)
            tgt = (21.8, 12.0, -21.0)
            fov = 50 - 8 * uu
        elif key == "iron_mine":
            eye = np.array([17.6, 13.2, -17.6])
            tgt = (21.2, 12.0, -21.0)
        elif key == "why6":
            eye = pip + (-1.2, 1.6, 2.5)
            tgt = P + (0.6, 0.2, 0)
            fov = 46
        elif key == "ironcommon":
            uu = smooth(u)
            eye = lerp(np.array([-4.0, 36.0, 34.0]), np.array([6.0, 32.0, 24.0]), uu)
            tgt = (24, 13, -16)
            fov = 55
        elif key == "raw":
            eye = np.array([21.8, 13.0, -18.2])
            tgt = P
        elif key == "furnace":
            eye = np.array([20.8, 12.9, -20.9])
            tgt = (17.4, 11.6, -23.7)
        elif key == "ingots":
            eye = np.array([18.9, 12.1, -21.7])
            tgt = (17.5, 11.5, -24.0)
            fov = 50
        elif key == "smelt":
            eye = np.array([13.8, 13.6, -19.6])
            tgt = (17.5, 11.6, -23.5)
        elif key == "ironpick":
            eye = pip + (2.2, 1.5, -1.1)
            tgt = P + (0, 0.1, -0.2)
            fov = 48
        elif key == "bucket":
            eye = pip + (-2.3, 1.45, -0.9)
            tgt = P + (0, 0.1, -0.2)
            fov = 48
        elif key == "stairsready":
            eye = np.array([12.5, 13.2, -11.8])
            tgt = P
        # ---------------- chapter 3 (stairs)
        elif key == "eight":
            eye = L(-6.4, 2.7, 1.5)
            tgt = L(0.5, 0.9, 1.4)
            fov = 62
        elif key == "rule1":
            eye = L(-2.9, 3.3, 2.4)
            tgt = L(0.6, 0.1, 1.0)
            fov = 60
        elif key == "coal" and t < self.coal_t[-1] + 0.8 and t > self.coal_t[0] - 1.5:
            eye = L(8.7, -6.3, 0.45)
            tgt = L(10.9, -8.0, 3.0)
            fov = 60
        elif key == "light":
            eye = pip + (0.42, 1.55, 1.0)
            tgt = L(-6.5, 9.0, 1.5)
            fov = 72
        elif key == "deepslate" and u < 0.55:
            eye = pip + (0.4, 1.9, 0.4)
            tgt = pip + (0.7, 1.0, 1.8)
            fov = 62
        elif key in ("stairs", "coal", "torches", "gravel", "dodge", "trick", "deepslate", "breakthrough"):
            eye, tgt = stairs_behind()
            fov = 66
            if key == "breakthrough" and t > self.dig_t[32] - 0.2:
                uu = smooth(seg(t, self.dig_t[32] - 0.2, self.E("breakthrough")))
                eye = lerp(eye, pip + (0.0, 1.9, 0.3), uu)
                tgt = lerp(tgt, L(60, -27, 4), uu)
        # ---------------- chapter 4 (cavern)
        elif key == "cavern":
            uu = smooth(u)
            eye = lerp(L(35.0, -29.2, 1.6), L(40.0, -27.5, 0.5), uu)
            tgt = L(60, -26, 4)
            fov = 70
        elif key == "ceiling":
            eye = L(46, -30.4, 2.0)
            tgt = lerp(L(56, -27, 4), L(58, -18, 4), smooth(u))
            fov = 68
        elif key == "glow":
            eye = L(58.0, -30.2, 4.0)
            tgt = L(58.0, -32.6, 12.0)
            fov = 62
        elif key == "size":
            eye = orbit(L(60, -28, 4), 15.0, 3.6 + 0.7 * u, 5.0)
            tgt = L(60, -27, 4)
            fov = 66
        elif key == "fill":
            eye = L(58.6, -30.3, 4.0)
            tgt = pip + (0, 0.6, 1.0)
        elif key == "groan":
            eye = L(57.6, -30.4, 8.4)
            tgt = L(50, -30.8, -1)
            fov = 60
        elif key == "zombie":
            zp = self.z_cave["path"].pos(t)
            eye = zp + (1.8, 1.45, 2.0)
            tgt = zp + (0, 1.35, 0)
            fov = 50
        elif key in ("swing", "down"):
            eye, tgt = fight(self.z_cave)
            fov = 55
        elif key == "night":
            eye = L(66, -29.0, 12.0)
            tgt = L(58, -30.0, -4)
            fov = 66
        elif key == "rotten":
            eye = pip + (-1.6, 1.5, -1.8)
            tgt = P
            fov = 50
        elif key == "orange":
            eye = pip + (-4.4, 2.1, 0.7)
            tgt = P + (3, 0, 0)
            fov = 62
        # ---------------- chapter 5 (lava)
        elif key == "lava":
            eye = L(89.2, -28.6, 0.0)
            tgt = L(104, -32, 5)
            fov = 68
        elif key == "lose":
            eye = lerp(L(96.0, -30.6, -2.6), L(100.0, -30.6, -2.6), u)
            tgt = L(99, -32, 6)
            fov = 60
        elif key == "sparkle":
            eye = L(91.0, -30.2, 3.0)
            tgt = L(116, -31.2, 3.9)
            fov = 22 - 6 * smooth(u)
        elif key == "noway":
            eye = pip + (-2.6, 1.9, 0.9)
            tgt = pip + (4.0, 0.5, -0.4)
            fov = 60
        elif key == "goodthing":
            eye = pip + (1.7, 1.35, -1.7)
            tgt = P + (0, 0.05, 0)
            fov = 48
        elif key == "hot":
            eye = L(91.0, -31.0, 9.4)
            tgt = L(100, -32, 2)
            fov = 62
        elif key == "obsfact":
            eye = L(90.0, -28.0, 9.0)
            tgt = L(101, -32, 3)
            fov = 60
        elif key in ("pour1", "pour2", "bridge") and not (key == "bridge" and u > 0.5):
            eye = np.array([pip[0] + 1.6, U[1] + FLOOR + 3.3, U[2] - 3.3])
            tgt = np.array([pip[0] + 1.6, U[1] + FLOOR + 0.3, U[2] + 3.5])
            fov = 60
        elif key in ("slip", "fire"):
            eye = pip + (2.4, 1.5, -2.6)
            tgt = P
            fov = 52
        elif key == "wrong":
            eye = pip + (1.9, 1.5, 0.7)
            tgt = P + (0, 0.2, 0)
            fov = 45
        elif key == "careful":
            eye = pip + (2.2, 1.0, -2.8)
            tgt = pip + (1.0, 0.7, 0)
            fov = 58
        elif key == "bridge":
            eye = L(91.0, -27.0, -3.5)
            tgt = L(100, -32, 3.5)
            fov = 62
        # ---------------- chapter 6 (diamonds)
        elif key == "four":
            eye = L(112.2, -30.2, 1.0)
            tgt = L(116, -31.3, 3.9)
            fov = 50
        elif key in ("first", "rest"):
            eye = pip + (-1.7, 1.8, -1.4) + (0.3 * u if key == "rest" else 0, 0, 0)
            tgt = L(116, -31.4, 3.9)
            fov = 55
        elif key == "ironmatter":
            eye = L(113.0, -30.6, 2.0)
            tgt = L(116, -31.4, 3.9)
            fov = 52
        elif key in ("five", "sword"):
            eye = pip + (1.6, 1.6, -2.9)
            tgt = P + (0, 0.5, 0)
            fov = 52
        elif key == "sat":
            eye = pip + (1.9, 1.7, 1.6)
            tgt = pip + (-6.0, 0.6, -0.5)
            fov = 58
        elif key == "time":
            eye = pip + (-2.0, 1.5, 0.9)
            tgt = P + (0, 0.2, 0)
            fov = 46
        # ---------------- chapter 7 (lost)
        elif key == "ranback":
            eye = pip + (3.6, 2.0, 0.6)
            tgt = P + (-2, 0, 0)
            fov = 62
        elif key == "same":
            eye = orbit(pip, 4.2, 1.0 + 3.6 * u, 1.9)
            tgt = P
            fov = 60
        elif key == "dead":
            eye = pip + (0.0, 2.3, 3.2)
            tgt = P + (0, 0, -2)
            fov = 62
        elif key == "wrongway":
            eye = pip + (2.8, 2.0, 2.8)
            tgt = P
        elif key == "remember":
            eye = pip + (1.3, 1.5, 1.7)
            tgt = P + (0, 0.2, 0)
            fov = 46
        elif key == "follow":
            eye = pip + (-3.8, 2.1, 0.2)
            tgt = P
            fov = 60
        elif key == "hiss":
            cp = self.creeper["path"].pos(t)
            eye = L(33.6, -29.7, 3.6)
            tgt = (pip + cp) / 2 + (0, 1.0, 0)
            fov = 58
        elif key == "creeper":
            cp = self.creeper["path"].pos(t)
            eye = cp + (-1.6, 1.3, 1.5)
            tgt = cp + (0, 1.1, 0)
            fov = 50
        elif key == "explode":
            eye = pip + (1.5, 1.5, 1.6)
            tgt = P
            fov = 50
        # ---------------- chapter 8 (escape)
        elif key in ("run", "sprint", "boom", "climb"):
            eye, tgt = stairs_above()
            fov = 68
        elif key == "collapse":
            eye = pip + (-1.3, 2.9, 1.5)
            tgt = pip + (4.0, -2.0, 0.0)
            fov = 66
        elif key == "sealed":
            kb = 14
            eye = L(kb - 3.0, -(kb - 3) + 1.5, 2.0)
            tgt = L(kb + 2.0, -(kb + 2) + 1.5, 1.5)
            fov = 62
        elif key == "outside":
            eye = np.array([12.5, 12.7, -14.0])
            tgt = P
            fov = 60
        elif key == "dash":
            eye = lerp(np.array([22.0, 18.0, -6.0]), np.array([9.0, 16.0, 17.0]), smooth(u))
            tgt = P
            fov = 60
        elif key == "zombies":
            eye = np.array([20.0, 13.6, 15.0])
            tgt = (14.5, 11.8, 5.0)
            fov = 58
        elif key in ("madeit", "sealdoor"):
            eye = np.array([1.6, 12.7, 11.8])
            tgt = (0.5, 11.8, 4.5)
            fov = 55
        # ---------------- chapter 9 (roof)
        elif key == "table":
            eye = np.array([5.0, 16.2, 4.6])
            tgt = (1.5, 14.8, -0.2)
        elif key in ("recipe", "dsword"):
            eye = np.array([3.9, 15.8, 2.6])
            tgt = (1.0, 15.0, 0.0)
            fov = 52
        elif key == "dpick":
            eye = np.array([3.3, 14.9, -0.9])
            tgt = (1.5, 15.7, 1.0)
            fov = 55
        elif key == "glow2":
            eye = pip + (1.5, 1.2, -1.5)
            tgt = P + (0.1, -0.35, -0.2)
            fov = 50
        elif key == "toolfact":
            eye = orbit(pip, 4.6, 0.8 + 0.9 * u, 2.0)
            tgt = P
        elif key == "eat":
            eye = pip + (0.6, 1.5, -2.3)
            tgt = P + (0, 0.2, 0)
            fov = 46
        elif key == "cue":
            eye = np.array([1.0, 16.5, -0.6])
            tgt = lerp(np.array([3.0, 14.5, 4.0]), np.array([10.0, 11.5, 12.0]), smooth(u))
            fov = 58
        # ---------------- chapter 10 (night two)
        elif key == "night2":
            eye = np.array([5.5, 13.2, 16.5])
            tgt = (8.0, 11.5, 6.0)
            fov = 60
        elif key == "meet":
            eye = np.array([4.6, 13.0, 11.6])
            tgt = P
        elif key == "z1":
            eye, tgt = fight(self.n2[0])
        elif key == "z2":
            eye, tgt = fight(self.n2[1])
        elif key == "z3":
            eye, tgt = fight(self.n2[2])
        elif key == "knock":
            eye = np.array([5.0, 14.0, 13.5])
            tgt = (1.0, 10.6, 6.5)
        elif key == "zero":
            eye = pip + (1.5, 1.5, 2.6)
            tgt = P + (0, 0.2, 0)
            fov = 48
        elif key == "tower":
            eye = pip + (3.6 + 0.4 * u, 2.4, -0.9)
            tgt = P + (0, 0.3, 0)
            fov = 55
        elif key in ("obsidian", "downthere"):
            uu = seg(t, C("obsidian"), self.E("downthere"))
            eye = lerp(L(90.0, -30.0, 3.5), L(100.0, -28.5, 1.0), smooth(uu))
            tgt = lerp(L(104, -32, 3.5), L(110, -32, 4.0), smooth(uu))
            fov = 60
        elif key == "plan3":
            eye = pip + (2.0, 1.5, 0.5)
            tgt = P + (0, 0.2, 0)
            fov = 48
        elif key == "portal":
            eye = PORTAL_O + (7.5, 3.6, 9.5)
            tgt = PORTAL_O + (2.0, 2.4, 0.5)
            fov = 55
        elif key == "outro":
            uu = ease_in_out(u)
            eye = lerp(pip + (2.6, 1.6, 2.6), np.array([16.0, 26.0, 22.0]), uu)
            tgt = lerp(P, np.array([0.5, 12.0, 0.5]), uu)
        eye = np.asarray(eye, np.float32)
        amp = 0.0
        for t0, a, d in ((self.t_boom, 0.9, 1.8),):
            if t0 <= t < t0 + d:
                amp += a * (1 - (t - t0) / d) ** 2
        if T("collapse", 0.0) <= t < T("collapse", 1.0):
            amp += 0.12
        if C("run") <= t < C("boom") or C("dash") <= t < self.E("dash"):
            amp += 0.04
        eye = eye + shake(t, amp)
        return dict(eye=eye, target=np.asarray(tgt, np.float32), fov=float(fov), roll=roll)

    def cuts(self):
        out = [self.C(k) for k in S.KEYS[1:]]
        return [c for c in out if c > 0.5]

    # ------------------------------------------------------------ torches
    def stair_torches(self):
        out = []
        for st in TORCH_STEPS:
            out.append((self.dig_t[st] + 0.6, ug(st + 0.5, -st, 2.6)))
        for i, (x, z) in enumerate(CAVE_TORCHES):
            out.append((self.T("ceiling", 0.1) + 0.5 * i, ug(x, FLOOR, z)))
        return out

    def cave_light(self, t, cam_eye):
        lx = cam_eye[0] - U[0]
        lit = sum(1 for tt, _ in self.stair_torches() if t >= tt)
        k = 0.74 + 0.035 * min(lit, 12)
        if lx < 6:
            k += 0.25 * clamp01((6 - lx) / 8)
        light = dict(dir=(0.3, 0.9, 0.2), sun=(0.22 * k, 0.19 * k, 0.13 * k),
                     ambient=(0.6 * k, 0.5 * k, 0.4 * k), exposure=1.2)
        fog = dict(color=(0.03, 0.028, 0.025), start=10, end=58)
        lava = clamp01((lx - 80) / 8)
        if lava > 0:
            light["ambient"] = tuple(np.array(light["ambient"]) * (1 - lava) + np.array([0.72, 0.44, 0.27]) * lava)
            light["sun"] = tuple(np.array(light["sun"]) * (1 - lava) + np.array([0.45, 0.22, 0.08]) * lava)
            fog = dict(color=tuple(np.array([0.03, 0.028, 0.025]) * (1 - lava) + np.array([0.16, 0.06, 0.02]) * lava),
                       start=12, end=60)
        sky = dict(zenith=(0.02, 0.02, 0.02), horizon=(0.03, 0.025, 0.02), below=(0.02, 0.02, 0.02),
                   stars=0.0, sun_vis=0.0, moon_vis=0.0, sun_dir=(0.3, 0.9, 0.2))
        if lx < 2:          # the corridor at the top sees daylight through the shaft
            sky.update(zenith=(0.45, 0.62, 0.95), horizon=(0.75, 0.85, 1.0), below=(0.5, 0.6, 0.8))
        return sky, light, fog

    # ------------------------------------------------------------ frame spec
    def frame_spec(self, t, cam=None):
        T, Wd, C = self.T, self.Wd, self.C
        cam = cam or self.camera(t)
        where = self.loc(t)
        key = self.key_at(t)
        post = dict(bloom=0.34, vignette=0.32, saturation=1.12)
        if where == "S":
            ph = self.phase(t)
            sky, light, fog = environment(ph, 40, 300)
            if ph >= 1.8:
                light = dict(light)
                light["ambient"] = tuple(np.array(light["ambient"]) + np.array([0.07, 0.06, 0.04]))
                fog = dict(fog, start=30, end=200)
            static, static_t = ["far", "surface", "clouds"], ["far_t", "surface_t"]
            if key == "portal":
                post["tint"] = (1.05, 0.9, 1.2)
        else:
            sky, light, fog = self.cave_light(t, cam["eye"])
            static, static_t = ["under"], ["under_t"]
            post["vignette"] = 0.5
            if key == "deep":
                post["tint"] = (1.15, 0.85, 0.75)
                post["saturation"] = 0.85
        for t0, d, col in ((self.t_boom, 1.2, (1.4, 0.8, 0.3)),):
            if t0 <= t < t0 + d:
                kk = (1 - (t - t0) / d) ** 2
                light = dict(light)
                light["ambient"] = tuple(np.array(light["ambient"]) + np.array(col) * kk)
        r = self._r
        dyn, dyn_t = [], []
        pw, pkey, pops = self.props_world(t, where)
        kk = f"props_{pkey[0]}_{pkey[1]}"
        if kk not in r.static:
            o, tr = pw.mesh()
            if where == "U":
                o = o.copy()
                tr = tr.copy()
                o[:, 0:3] += U.astype(np.float32)
                tr[:, 0:3] += U.astype(np.float32)
            r.set_static(kk, o)
            r.set_static(kk + "_t", tr)
        static.append(kk)
        static_t.append(kk + "_t")
        off = U if where == "U" else np.zeros(3)
        for (tp, p, b) in pops:
            s = ease_out_back(seg(t, tp, tp + 0.14))
            v = block_entity_vertices(b, np.array(p) + 0.5 + off, max(s, 0.05))
            (dyn_t if tx.is_translucent(b) else dyn).append(v)
        # ---- Pip
        hurt = (T("slip", 0.35) <= t < T("fire", 0.6) and (t * 6) % 1 < 0.5)
        if not self.hidden_pip(t):
            pos, yaw, pose = self.pip_state(t)
            dyn.append(pip_vertices(pos, yaw, pose, flash=0.45 if hurt else 0.0))
        flash_col = (1, 0.15, 0.15) if hurt else (1, 1, 1)
        # ---- mobs
        for m in self.mobs:
            if t < m["spawn"] - 0.01 or (m["death"] is not None and t >= m["death"]):
                continue
            p = m["path"].pos(t).copy()
            if (p[0] > 300) != (where == "U"):
                continue
            v = m["path"].vel(t)
            sp = float(np.hypot(v[0], v[2]))
            pip_p = self.pip_path.pos(t)
            yawm = float(np.arctan2(v[0], v[2])) if sp > 0.2 else float(np.arctan2(pip_p[0] - p[0], pip_p[2] - p[2]))
            fl = 0.0
            for h in m.get("hits", []):
                if h + 0.08 <= t < h + 0.35:
                    fl = 0.5
                    p = p + 0.35 * np.array([np.sin(yawm), 0, np.cos(yawm)]) * -seg(t, h + 0.08, h + 0.35)
            if m.get("rise"):
                p[1] = GY - 1.9 * (1 - smooth(seg(t, m["spawn"], m["spawn"] + 1.0)))
            if m["kind"] == "z":
                dyn.append(zombie_vertices(p, yawm, {"walk": clamp01(sp / 1.5) * 0.8,
                                                     "walk_phase": m["path"].dist(t) * 2.5}, flash=fl))
            else:
                fl = 0.0
                sw = 0.0
                if t >= C("creeper"):
                    fl = 0.45 if (t * 5) % 1 < 0.5 else 0.0
                    sw = 0.25
                if t >= self.t_boom - 0.6:
                    sw = seg(t, self.t_boom - 0.6, self.t_boom) ** 1.5
                dyn.append(creeper_vertices(p, yawm, {"walk": clamp01(sp / 1.5) * 0.8,
                                                      "walk_phase": m["path"].dist(t) * 3, "swell": sw}, flash=fl))
        # ---- props by location
        glows = []
        if where == "S":
            ph = self.phase(t)
            for b in self.surface.torches:
                dyn.append(torch_vertices(b, 0.1 * np.sin(t * 13 + b[0])))
                if ph > 0.8:
                    glows.append([b[0], b[1] + 0.72, b[2], 0.55 + 0.08 * np.sin(t * 9 + b[0]), 1.0, 0.75, 0.35, 0.22])
            for b in [(19.6, GY, -19.4), (19.6, GY, -15.6)]:
                dyn.append(torch_vertices(b, 0.1 * np.sin(t * 11 + b[2])))
                if ph > 0.8:
                    glows.append([b[0], b[1] + 0.72, b[2], 0.5, 1.0, 0.75, 0.35, 0.2])
            harvested = self._harvested(t)
            for i, cp in enumerate(self.surface.crops):
                if i in harvested:
                    continue
                dyn.append(box_vertices((0.8, 0.7, 0.8), "crop", matrix=translate(*cp), offset=(0, 0.35, 0)))
            if C("neediron") <= t < self.t_demo_break:
                crack = min(3, int(4 * seg(t, T("neediron", 0.2), self.t_demo_break)))
                dyn.append(block_entity_vertices("diamond_ore", tuple(DEMO), 1.0, crack=crack))
            if t >= T("furnace", 0.45):
                s = ease_out_back(seg(t, T("furnace", 0.45), T("furnace", 0.45) + 0.2))
                m = translate(*FURNACE)
                from mcengine.geometry import scale as gscale
                dyn.append(box_vertices((1, 1, 1), ["furnace_side", "furnace_side", "furnace_side", "furnace_side",
                                                    "furnace_front", "furnace_side"], matrix=m @ gscale(max(s, 0.05)),
                                        emissive=0.2 if T("furnace", 0.55) <= t < T("smelt", 1.0) else 0.0))
                if T("furnace", 0.55) <= t < T("smelt", 1.0):
                    glows.append([FURNACE[0], FURNACE[1] - 0.2, FURNACE[2] + 0.6, 0.9, 1.0, 0.6, 0.2, 0.3])
            if t >= C("table") - 5:
                tb = np.array(ROOF_TABLE) + 0.5
                dyn.append(block_entity_vertices("crafting", tuple(tb), 1.0))
            if key == "portal":
                for i, (a, b) in enumerate(PORTAL):
                    t0 = T("portal", 0.05 + 0.05 * i)
                    if t >= t0:
                        s = ease_out_back(seg(t, t0, t0 + 0.2))
                        dyn.append(block_entity_vertices("obsidian", tuple(PORTAL_O + (a + 0.5, b + 0.5, 0.5)),
                                                         max(s, 0.05)))
                if t >= T("portal", 0.6):
                    k = smooth(seg(t, T("portal", 0.6), T("portal", 0.8)))
                    for yy in range(3):
                        for xx in range(2):
                            glows.append([PORTAL_O[0] + 1.5 + xx, PORTAL_O[1] + 1.5 + yy, PORTAL_O[2] + 0.5,
                                          1.0 * k, 0.7, 0.3, 1.0, 0.35])
        else:
            for (tt, b) in self.stair_torches():
                if t >= tt:
                    dyn.append(torch_vertices(b, 0.1 * np.sin(t * 11 + b[0])))
                    glows.append([b[0], b[1] + 0.72, b[2], 0.55 + 0.08 * np.sin(t * 9 + b[0]), 1.0, 0.75, 0.35, 0.22])
                    lx, lz = b[0] - U[0], b[2] - U[2]
                    if ((lx - WD.LAKE_C[0]) / (WD.LAKE_R[0] + 3)) ** 2 + ((lz - WD.LAKE_C[1]) / (WD.LAKE_R[1] + 3)) ** 2 < 1:
                        my = 2 * (U[1] + FLOOR) - (b[1] + 0.72)
                        glows.append([b[0], my, b[2] + 1.2, 0.45, 1.0, 0.75, 0.35, 0.14])
            # lava glow
            for x in range(WD.LAVA_X[0], WD.LAVA_X[1], 3):
                for z in range(WD.LAVA_Z[0], WD.LAVA_Z[1], 3):
                    p = ug(x + 1.5, FLOOR + 0.1, z + 1.5)
                    glows.append([p[0], p[1], p[2], 1.8 + 0.2 * np.sin(t * 2 + x + z), 1.0, 0.45, 0.1, 0.10])
            # falling gravel
            tg = self.t_gravel
            if tg <= t < tg + 0.5:
                u = seg(t, tg, tg + 0.45) ** 2
                for z in range(3):
                    for j, yb in enumerate((-17, -18)):
                        y0 = -14.5 + j
                        yy = y0 + (yb + 0.5 - y0) * u
                        dyn.append(block_entity_vertices("gravel", tuple(ug(GRAVEL_STEP + 0.5, yy, z + 0.5)), 1.0))
            # diamond ore cracks while mining
            for i, (tt, p) in enumerate(zip(self.ore_t, WD.ORES)):
                if tt - 0.7 <= t < tt:
                    c = min(3, int(4 * seg(t, tt - 0.7, tt)))
                    dyn.append(box_vertices((1, 1, 1), f"crack{c}", matrix=translate(*ug(p[0] + 0.5, p[1] + 0.5, p[2] + 0.5)),
                                            inflate=0.006))
            # water poured on lava (brief flowing water)
            for i, tp in enumerate(self.pour_t):
                if tp <= t < tp + 0.4:
                    x0, x1 = WD.POURS[i]
                    s = seg(t, tp, tp + 0.4)
                    for x in range(x0, x1):
                        dyn_t.append(box_vertices((1, 0.25 * (1 - s) + 0.05, 3), "water_flow",
                                                  matrix=translate(*ug(x + 0.5, FLOOR + 0.1, 3.5)), alpha=0.45))
        dyn += self._flying_items(t, where)
        # ---- particles
        parts = gather(self.bursts, t, False)
        adds = gather(self.bursts, t, True)
        if glows:
            g = np.array(glows, np.float32)
            adds = g if adds is None else np.concatenate([adds, g])
        wf = 0.0
        for t0, a, d in ((self.t_boom, 0.8, 0.35),):
            if t0 <= t < t0 + d:
                wf = max(wf, a * (1 - (t - t0) / d))
        post["white_flash"] = wf
        return dict(camera=cam, sky=sky, light=light, fog=fog, static=static, static_trans=static_t,
                    dynamic=np.concatenate(dyn) if dyn else None,
                    dynamic_trans=np.concatenate(dyn_t) if dyn_t else None,
                    particles=parts, particles_add=adds, flash_col=flash_col, post=post)

    def _harvested(self, t):
        T = self.T
        if t < T("breakfast", 0.4):
            return set()
        crops = self.surface.crops
        near = sorted(range(len(crops)), key=lambda i: (crops[i][0] - 3.2) ** 2 + (crops[i][2] - 13.6) ** 2)[:4]
        n = sum(1 for f in (0.4, 0.55, 0.7, 0.85) if t >= T("breakfast", f) + 0.1)
        return set(near[:n])

    def _flying_items(self, t, where):
        T, Wd, C = self.T, self.Wd, self.C
        out = []
        pip, yaw, _ = self.pip_state(t)
        hand = pip + np.array([0, 1.0, 0])

        def fly(kind, t0, p0, dur=0.7, s=1.0):
            if t0 <= t < t0 + dur:
                u = seg(t, t0, t0 + dur)
                p = lerp(np.asarray(p0, float), hand, ease_in(u, 2)) + np.array([0, 0.8 * np.sin(PI * u), 0])
                out.append(item_vertices(kind, p, spin=t * 6, s=s))
        if where == "S":
            for tt, p in zip(self.iron_times, self.iron_pos):
                fly("rawiron", tt + 0.25, np.array(p) + 0.5)
            crops = self.surface.crops
            for j, f in enumerate((0.4, 0.55, 0.7, 0.85)):
                near = sorted(range(len(crops)), key=lambda i: (crops[i][0] - 3.2) ** 2 + (crops[i][2] - 13.6) ** 2)[:4]
                cp = crops[near[j]]
                if T("breakfast", f) + 0.1 <= t < T("breakfast", f) + 0.8:
                    u = seg(t, T("breakfast", f) + 0.1, T("breakfast", f) + 0.8)
                    p = lerp(np.array(cp) + (0, 0.4, 0), hand, ease_in(u, 2)) + np.array([0, 0.8 * np.sin(PI * u), 0])
                    out.append(box_vertices((0.3, 0.3, 0.05), "crop", matrix=translate(*p)))
            for i in range(6):
                t0 = T("ingots", 0.1 + 0.13 * i)
                fly("iron", t0, FURNACE + (0, -0.1, 0.6), 0.6, 1.2)
            if key_is(self, t, ("hook", "recap", "halfheart", "hook2", "need3", "sunset")):
                fw = np.array([np.sin(yaw), 0, np.cos(yaw)])
                sd = np.array([np.cos(yaw), 0, -np.sin(yaw)])
                p = pip + np.array([0.0, 1.05 + 0.05 * np.sin(t * 3), 0.0]) + 0.55 * fw - 0.3 * sd
                out.append(item_vertices("diamond", p, spin=t * 2.5, s=0.9))
        else:
            for tt, p in zip(self.coal_t, COAL):
                fly("coal", tt, ug(p[0] + 0.5, p[1] + 0.5, p[2] + 0.2))
            for i, (tt, p) in enumerate(zip(self.ore_t, WD.ORES)):
                fly("diamond", tt, ug(p[0] - 0.1, p[1] + 0.5, p[2] + 0.5), 0.6, 1.4)
            if key_is(self, t, ("five", "sword", "sat")):
                n = 5
                for i in range(n):
                    a = t * 1.6 + 2 * PI * i / n
                    p = pip + np.array([0.75 * np.cos(a), 2.25 + 0.1 * np.sin(t * 3 + i), 0.75 * np.sin(a)])
                    out.append(item_vertices("diamond", p, spin=t * 3 + i, s=0.9))
        return out

    # ------------------------------------------------------------ overlay
    def hearts(self, t):
        T, Wd = self.T, self.Wd
        hp = 0.5
        if t >= T("regen", 0.0):
            hp = 0.5 + 9.5 * smooth(seg(t, T("regen", 0.0), T("regen", 0.9)))
        if t >= T("slip", 0.5):
            hp = 9.0
        if t >= T("fire", 0.0):
            hp = 8.0
        if t >= Wd("fire", "three"):
            hp = 7.0
        if t >= T("eat", 0.1):
            hp = 7.0 + 3.0 * smooth(seg(t, T("eat", 0.1), T("eat", 0.8)))
        return float(np.round(hp * 2) / 2)

    def diamonds(self, t):
        n = 1 + sum(1 for tt in self.ore_t if t >= tt + 0.6)
        if t >= self.T("dpick", 0.0):
            n -= 3
        if t >= self.T("dsword", 0.6):
            n -= 2
        return n

    def draw_overlay(self, t, layer):
        T, Wd, C = self.T, self.Wd, self.C
        tp = ov.title_pop
        TOP = (W / 2, 150)
        key = self.key_at(t)
        where = self.loc(t)
        # damage vignette (lava)
        if T("slip", 0.35) <= t < T("fire", 0.9):
            k = 0.6 + 0.2 * np.sin(t * 14)
            if t > T("fire", 0.6):
                k *= 1 - seg(t, T("fire", 0.6), T("fire", 0.9))
            a = self._orange.copy()
            a[..., 3] = (a[..., 3] * k * 0.55).astype(np.uint8)
            layer.alpha_composite(Image.fromarray(a))
        # HUD
        hidden_hud = key in ("deep", "obsidian", "downthere", "portal")
        if t >= 0.8 and not hidden_hud:
            hp = self.hearts(t)
            shk = 1.0 if T("slip", 0.35) <= t < T("fire", 0.8) else 0.0
            ov.draw_hearts(layer, hp, (300, 40), px=5, shake_amt=shk, t=t)
            items, sel = self._hotbar(t)
            ov.draw_hotbar(layer, items, sel, (W / 2, H - 70), slot=72)
            nd = self.diamonds(t)
            if nd > 0:
                d = ImageDraw.Draw(layer)
                d.rounded_rectangle([W - 300, 120, W - 40, 200], radius=14, fill=(0, 0, 0, 140))
                ov.paste_emoji(layer, "💎", (W - 245, 160), 54)
                ov.draw_text(layer, f"x {nd}", (W - 140, 160), 58, fill=(110, 235, 255), shadow=False)
        # daylight clock
        if T("sunset", 0.3) <= t < T("madeit", 0.6) and not hidden_hud:
            secs = float(np.interp(t, [T("sunset", 0.3), C("eight"), C("time"), T("time", 0.6), C("outside"),
                                       T("dash", 0.0), T("madeit", 0.5)],
                                   [600, 480, 200, 180, 45, 30, 0]))
            ov.countdown(layer, secs, (W - 170, 70), size=60)
        # depth meter underground
        if where == "U" and not hidden_hud:
            pip = self.pip_path.pos(t)
            ly = pip[1] - U[1]
            yv = 64 + ly * (118 / 32) if ly < 0 else 64
            ov.draw_text(layer, f"Y: {int(round(yv))}", (W - 170, 250), 52,
                         fill=(255, 225, 40) if yv < -40 else (255, 255, 255))
        # plan checklist
        if T("plan", 0.0) <= t < C("ranback") and key not in ("deep",):
            self._checklist(layer, t)
        # chapter cards
        for n in range(2, 11):
            c0, c1 = self.card(n)
            ov.chapter_card(layer, t, c0 - 0.25, (c1 - c0) + 0.5, n, self.chapters[n - 1]["title"], W, H)
        self._crafting(layer, t)
        # ---- titles
        G, B, R, O, Y, Wh = (120, 240, 120), (110, 235, 255), (255, 80, 60), (255, 150, 30), (255, 225, 40), (255, 255, 255)
        tp(layer, "DAY 2", t, 0.3, T("recap", 0.0), TOP, 140, fill=Y, emoji="☀️", emoji_side="right")
        tp(layer, "NIGHT 1: SURVIVED", t, T("recap", 0.05), T("halfheart", 0.0), TOP, 100, fill=G, emoji="✅", emoji_side="right")
        tp(layer, "1 / 3", t, Wd("need3", "one"), T("deep", 0.0), (W * 0.72, 250), 110, fill=B, emoji="💎", emoji_side="left")
        tp(layer, "THE BOTTOM OF THE WORLD", t, T("deep", 0.3), T("sunset", 0.0) - 0.1, TOP, 86, fill=O, emoji="🔥", emoji_side="right")
        tp(layer, "10 MINUTES", t, Wd("sunset", "Ten"), T("morning", 0.0), TOP, 120, fill=Y, emoji="☀️", emoji_side="right")
        if T("wheat", 0.55) <= t < T("bread", 0.0):
            tp(layer, "3", t, T("wheat", 0.55), T("bread", 0.0), (W / 2 - 260, 180), 120, fill=Y, emoji="🌾", emoji_side="right")
            tp(layer, "=", t, T("wheat", 0.65), T("bread", 0.0), (W / 2, 180), 120, fill=Wh)
            tp(layer, "1", t, T("wheat", 0.75), T("bread", 0.0), (W / 2 + 220, 180), 120, fill=Y, emoji="🍞", emoji_side="right")
        tp(layer, "FUN FACT", t, T("regen", 0.0), T("regen", 0.4), TOP, 100, fill=Y, emoji="💡", emoji_side="right")
        tp(layer, "FULL HUNGER = HEALING", t, T("regen", 0.4), T("plan", 0.0), TOP, 90, fill=G, emoji="❤️", emoji_side="right")
        if T("neediron", 0.2) <= t < self.E("neediron"):
            tp(layer, "STONE PICKAXE", t, T("neediron", 0.2), self.t_demo_break + 0.2, TOP, 90, fill=Wh, emoji="⛏️", emoji_side="right")
            tp(layer, "NO DIAMOND!", t, self.t_demo_break + 0.2, T("neediron", 0.78), TOP, 110, fill=R, emoji="❌", emoji_side="right")
            tp(layer, "NEED IRON", t, T("neediron", 0.78), self.E("neediron"), TOP, 110, fill=(236, 190, 150), emoji="✅", emoji_side="right")
        tp(layer, "IRON!", t, Wd("iron_found", "IRON"), T("iron_mine", 0.0), TOP, 130, fill=(236, 190, 150), emoji="⛏️", emoji_side="right")
        for i, tt in enumerate(self.iron_times):
            end = self.iron_times[i + 1] if i < 5 else T("why6", 0.0)
            tp(layer, f"{i + 1} / 6", t, tt + 0.25, end + 0.25, (W / 2, 300), 120, fill=(236, 190, 150))
        if key == "why6":
            ov.paste_emoji(layer, "🤔", TOP, 130)
        tp(layer, "IRON: NEAR THE SURFACE", t, T("ironcommon", 0.0), T("ironcommon", 0.5), TOP, 86, fill=(236, 190, 150), emoji="⛰️", emoji_side="right")
        tp(layer, "DIAMONDS: THE BOTTOM", t, T("ironcommon", 0.5), self.E("ironcommon"), TOP, 86, fill=B, emoji="💎", emoji_side="right")
        tp(layer, "6 IRON INGOTS", t, T("ingots", 0.85), T("smelt", 0.0), TOP, 100, fill=(220, 220, 228), emoji="✨", emoji_side="right")
        tp(layer, "1 COAL = 8 ITEMS", t, T("smelt", 0.1), self.E("smelt"), TOP, 100, fill=Wh, emoji="🔥", emoji_side="right")
        tp(layer, "8:00", t, T("eight", 0.0), self.E("eight"), (W / 2, 300), 140, fill=Y, emoji="⏰", emoji_side="left")
        tp(layer, "NEVER DIG STRAIGHT DOWN", t, T("rule1", 0.2), self.E("rule1"), TOP, 86, fill=R, emoji="❌", emoji_side="right")
        tp(layer, "COAL!", t, self.coal_t[0], T("coal", 1.0), TOP, 120, fill=(220, 220, 220), emoji="🪨", emoji_side="right")
        tp(layer, "TORCHES ON THE RIGHT", t, Wd("torches", "right"), self.E("torches"), TOP, 90, fill=Y, emoji="➡️", emoji_side="right")
        tp(layer, "GRAVEL!", t, self.t_gravel + 0.3, self.E("gravel"), TOP, 130, fill=R, emoji="😱", emoji_side="right")
        if key == "trick":
            tp(layer, "GOING DOWN: RIGHT", t, T("trick", 0.25), self.E("trick"), (W / 2, 130), 80, fill=Y, emoji="⬇️", emoji_side="left")
            tp(layer, "GOING UP: LEFT", t, T("trick", 0.5), self.E("trick"), (W / 2, 240), 80, fill=G, emoji="⬆️", emoji_side="left")
        tp(layer, "DEEPSLATE", t, Wd("deepslate", "deepslate"), self.E("deepslate"), TOP, 120, fill=(170, 170, 190))
        tp(layer, "IT'S ALWAYS NIGHT", t, T("night", 0.25), self.E("night"), TOP, 100, fill=(170, 190, 255), emoji="🌙", emoji_side="right")
        tp(layer, "ROTTEN FLESH", t, T("rotten", 0.15), self.E("rotten"), TOP, 100, fill=(200, 120, 90), emoji="🍖", emoji_side="right")
        tp(layer, "LAVA", t, T("lava", 0.0), self.E("lava"), TOP, 150, fill=O, emoji="🔥", emoji_side="right")
        tp(layer, "DIAMONDS!", t, Wd("sparkle", "DIAMONDS"), self.E("sparkle"), TOP, 130, fill=B, emoji="💎", emoji_side="right")
        if key == "obsfact" and t >= T("obsfact", 0.3):
            tp(layer, "WATER + LAVA = OBSIDIAN", t, T("obsfact", 0.3), self.E("obsfact"), TOP, 90, fill=(190, 140, 255), emoji="💧", emoji_side="left")
        for i, tt in enumerate(self.pour_t):
            tp(layer, f"POUR {i + 1}", t, tt, tt + 1.4, (W / 2, 300), 110, fill=(120, 180, 255))
        tp(layer, "ON FIRE!", t, T("slip", 0.5), T("fire", 0.6), TOP, 130, fill=O, emoji="🔥", emoji_side="right")
        tp(layer, "CROUCH = NO FALLING", t, T("careful", 0.15), T("careful", 0.85), TOP, 90, fill=Y, emoji="💡", emoji_side="right")
        tp(layer, "OBSIDIAN BRIDGE", t, T("bridge", 0.5), self.E("bridge"), TOP, 100, fill=(190, 140, 255), emoji="🌉", emoji_side="right")
        tp(layer, "4 DIAMOND ORES!", t, Wd("four", "FOUR"), self.E("four"), TOP, 110, fill=B, emoji="💎", emoji_side="right")
        tp(layer, "5 DIAMONDS", t, Wd("five", "FIVE", 0), self.E("five"), TOP, 130, fill=B, emoji="💎", emoji_side="right")
        if key == "sword":
            tp(layer, "3 = PICKAXE", t, T("sword", 0.0), self.E("sword"), (W / 2, 130), 80, fill=B, emoji="⛏️", emoji_side="left")
            tp(layer, "2 = SWORD", t, T("sword", 0.45), self.E("sword"), (W / 2, 240), 80, fill=B, emoji="🗡️", emoji_side="left")
        tp(layer, "3:00 LEFT", t, Wd("time", "Three"), self.E("time"), TOP, 130, fill=R, emoji="⏰", emoji_side="right")
        if key == "same":
            ov.paste_emoji(layer, "❓", (W / 2 + 120 * np.sin(t * 3), 170), 130)
        tp(layer, "DEAD END", t, T("dead", 0.25), T("dead", 0.6), TOP, 110, fill=R, emoji="🚫", emoji_side="right")
        tp(layer, "TORCHES ON THE LEFT", t, Wd("remember", "left"), self.E("remember"), TOP, 90, fill=G, emoji="⬅️", emoji_side="left")
        tp(layer, "Sssssss...", t, T("hiss", 0.6), self.E("hiss"), TOP, 130, fill=(110, 235, 90))
        tp(layer, "1.5 SECONDS", t, T("explode", 0.4), self.E("explode"), TOP, 110, fill=Wh, emoji="⏱️", emoji_side="right")
        tp(layer, "RUN!", t, T("run", 0.0), self.E("run") + 0.6, TOP, 170, fill=R, emoji="🏃", emoji_side="right")
        tp(layer, "BOOM!", t, self.t_boom, T("collapse", 0.0), TOP, 160, fill=O, emoji="💥", emoji_side="right")
        tp(layer, "30 SECONDS", t, Wd("dash", "Thirty"), T("dash", 0.5), TOP, 110, fill=R, emoji="⏰", emoji_side="right")
        tp(layer, "MADE IT!", t, T("madeit", 0.0), self.E("madeit"), TOP, 130, fill=G, emoji="✅", emoji_side="right")
        tp(layer, "DIAMOND PICKAXE!", t, T("dpick", 0.0), self.E("dpick"), TOP, 120, fill=B, emoji="⛏️", emoji_side="right")
        tp(layer, "DIAMOND SWORD", t, T("dsword", 0.6), self.E("dsword"), TOP, 110, fill=B, emoji="🗡️", emoji_side="right")
        if key == "toolfact":
            for i, (txt, em) in enumerate((("FASTER", "⚡"), ("LASTS LONGER", "🛡️"), ("FEW HITS PER ZOMBIE", "🧟"))):
                tp(layer, txt, t, T("toolfact", 0.05 + 0.3 * i), self.E("toolfact"), (W / 2, 110 + 105 * i), 76,
                   fill=B, emoji=em, emoji_side="left")
        tp(layer, "NIGHT 2", t, Wd("night2", "Night"), self.E("night2"), TOP, 130, fill=(170, 190, 255), emoji="🌙", emoji_side="right")
        for m, key2 in zip(self.n2, ("z1", "z2", "z3")):
            if m["death"] is not None:
                tp(layer, "GONE!" if key2 == "z1" else "SPLASH!", t, m["death"] - (0 if key2 == "z1" else 1.0),
                   m["death"] + 0.9, TOP, 130, fill=G if key2 == "z1" else (120, 180, 255),
                   emoji="💨" if key2 == "z1" else "💦", emoji_side="right")
        tp(layer, "KNOCKBACK", t, T("knock", 0.1), self.E("knock"), TOP, 110, fill=Wh, emoji="💨", emoji_side="right")
        tp(layer, "0 DAMAGE", t, Wd("zero", "Zero"), self.E("zero"), TOP, 120, fill=G, emoji="🛡️", emoji_side="right")
        tp(layer, "ONLY DIAMOND MINES OBSIDIAN", t, T("obsidian", 0.35), self.E("obsidian"), TOP, 80, fill=(190, 140, 255), emoji="⛏️", emoji_side="right")
        tp(layer, "10 OBSIDIAN = NETHER PORTAL", t, T("portal", 0.2), self.E("portal"), TOP, 80, fill=(200, 90, 255), emoji="🟪", emoji_side="right")
        if t >= C("outro") + 0.3:
            t0 = C("outro") + 0.3
            a = ease_out_back(seg(t, t0, t0 + 0.35))
            ov.draw_text(layer, "BUILD A NETHER PORTAL?", (W / 2, 330), 100, fill=(200, 90, 255), scale=max(a, 0.05))
            ov.paste_emoji(layer, "🟪", (W / 2, 190), 130 * max(a, 0.05))
            b = smooth(seg(t, t0 + 0.4, t0 + 0.8))
            if b > 0:
                ov.draw_text(layer, "COMMENT BELOW  •  DAY 3 SOON", (W / 2, 450), 60, alpha=b)
                ov.paste_emoji(layer, "👇", (W / 2, 560 + 10 * np.sin(t * 7)), 90, alpha=b)
        self.subs.draw(layer, t)

    def _hotbar(self, t):
        T, Wd, C = self.T, self.Wd, self.C
        bread = "e:🍞" if T("bread", 0.0) <= t < T("bread", 0.0) + 0.01 else None
        items = ["e:⛏️", "cobble", "e:🕯️" if t >= T("stairsready", 0.0) else None,
                 "e:🍞" if (T("stairsready", 0.0) <= t < T("eat", 0.3)) else None,
                 ("iron_ore" if self.iron_times[0] + 0.4 <= t < T("ingots", 0.1) else
                  ("t:ingot" if T("ingots", 0.1) <= t < T("bucket", 0.6) else None)),
                 "e:🪣" if t >= T("bucket", 0.6) else None,
                 "e:💎" if self.diamonds(t) > 0 else None,
                 "e:🗡️" if t >= T("dsword", 0.6) else None,
                 "furnace" if T("furnace", 0.1) <= t < T("furnace", 0.45) else None]
        del bread
        sel = 0
        if T("torches", 0.0) <= t < T("torches", 1.0) or C("follow") <= t < C("hiss"):
            sel = 2
        if T("bread", 0.0) <= t < T("bread", 0.6) or T("eat", 0.0) <= t < T("eat", 0.6):
            sel = 3
        if C("fill") <= t < C("groan") or C("goodthing") <= t < C("four"):
            sel = 5
        if C("table") <= t < T("dsword", 0.6):
            sel = 6
        if t >= T("dsword", 0.6) and t < C("plan3"):
            sel = 7
        if T("furnace", 0.1) <= t < T("furnace", 0.45):
            sel = 8
        return items, sel

    def _checklist(self, layer, t):
        T, Wd, C = self.T, self.Wd, self.C
        rows = [("IRON", T("ironpick", 0.6)), ("GO DEEP", self.dig_t[32]), ("2 DIAMONDS", self.ore_t[1] + 0.6),
                ("HOME", T("madeit", 0.0))]
        a = smooth(seg(t, T("plan", 0.0), T("plan", 0.15)))
        n_show = 1 + sum(1 for w in ("two", "three", "four") if t >= Wd("plan", w))
        if t >= C("dontknow"):
            n_show = 4
        d = ImageDraw.Draw(layer)
        x0, y0 = 40, 110
        d.rounded_rectangle([x0, y0, x0 + 350, y0 + 60 + 58 * n_show], radius=16, fill=(0, 0, 0, int(140 * a)))
        ov.draw_text(layer, "THE PLAN", (x0 + 175, y0 + 32), 40, fill=(255, 225, 40), alpha=a, shadow=False)
        for i, (name, td) in enumerate(rows[:n_show]):
            y = y0 + 90 + 58 * i
            done = t >= td
            ov.paste_emoji(layer, "✅" if done else "⬜", (x0 + 45, y), 40, alpha=a)
            ov.draw_text(layer, name, (x0 + 195, y), 40, fill=(150, 255, 150) if done else (255, 255, 255),
                         alpha=a, shadow=False)

    def _crafting(self, layer, t):
        T, Wd, C = self.T, self.Wd, self.C
        st = "e:🥢"
        D = "e:💎"
        I = "t:ingot"
        recipes = [
            # (t0, t1, grid, result, label)
            (Wd("need3", "three"), T("need3", 1.0) + 0.2, [D, None, None, None, st, None, None, st, None], None, "1 / 3"),
            (Wd("stonepick", "Three"), T("stonepick", 1.0) + 0.3,
             ["cobble", "cobble", "cobble", None, st, None, None, st, None], "e:⛏️", "STONE PICKAXE"),
            (T("furnace", 0.0), T("furnace", 0.45),
             ["cobble", "cobble", "cobble", "cobble", None, "cobble", "cobble", "cobble", "cobble"], "furnace", "FURNACE"),
            (T("ironpick", 0.0), T("ironpick", 1.0) + 0.5, [I, I, I, None, st, None, None, st, None], "e:⛏️", "IRON PICKAXE"),
            (T("bucket", 0.25), T("bucket", 1.0) + 0.3, [I, None, I, None, I, None, None, None, None], "e:🪣", "BUCKET"),
            (T("recipe", 0.0), T("dpick", 0.4), [D, D, D, None, st, None, None, st, None], "e:⛏️", "DIAMOND PICKAXE"),
            (T("dsword", 0.0), T("dsword", 1.0) + 0.3, [None, D, None, None, D, None, None, st, None], "e:🗡️", "DIAMOND SWORD"),
        ]
        for t0, t1, grid, res, label in recipes:
            if not (t0 <= t < t1):
                continue
            appear = seg(t, t0, t0 + 0.35) * (1 - seg(t, t1 - 0.25, t1))
            g = list(grid)
            showres = res
            if label == "DIAMOND PICKAXE":
                n = int(1 + 5 * seg(t, T("recipe", 0.05), T("recipe", 0.9)))
                order = [0, 1, 2, 4, 7]
                g = [x if (i in order[:n]) else None for i, x in enumerate(grid)]
                showres = res if t >= T("dpick", 0.0) else None
            if label == "1 / 3":
                g = [D, None, None, None, None, None, None, None, None]
                g2 = [D, "e:❔", "e:❔", None, st, None, None, st, None]
                g = g2 if t > t0 + 0.6 else g
            ov.crafting_grid(layer, (W * 0.70, H / 2 - 40), g, showres, appear=appear, cell=96)
            ov.draw_text(layer, label, (W * 0.70, H / 2 + 140), 60, fill=(255, 225, 40), alpha=appear)

    # ------------------------------------------------------------ audio
    def sfx_events(self):
        T, Wd, C = self.T, self.Wd, self.C
        ev = [(0.0, "whoosh", -10, dict(dur=0.8)), (0.3, "pop", -12, {})]
        for n in range(2, 11):
            c0, c1 = self.card(n)
            ev += [(c0 - 0.3, "whoosh", -8, dict(dur=0.7)), (c0 + 0.2, "impact", -10, {}),
                   (c1 - 0.4, "whoosh", -11, dict(dur=0.5, up=False))]
        ev += [(T("deep", 0.2), "riser", -12, dict(dur=2.5)), (Wd("sunset", "Ten"), "ding", -12, {})]
        for f in (0.4, 0.55, 0.7, 0.85):
            ev.append((T("breakfast", f) + 0.1, "dig", -12, dict(seed=int(f * 10))))
            ev.append((T("breakfast", f) + 0.7, "pop", -14, dict(pitch=1.2)))
        for te in (T("bread", 0.05), T("eat", 0.05)):
            for j in range(3):
                ev.append((te + 0.35 * j, "step", -12, dict(seed=40 + j)))
        ev.append((T("regen", 0.9), "ding", -12, dict(base=990)))
        for w in ("one", "two", "three", "four"):
            ev.append((Wd("plan", w), "pop", -12, dict(pitch=1.1)))
        for f in (0.25, 0.35, 0.45):
            ev.append((T("neediron", f) + 0.1, "dig", -9, dict(seed=int(f * 10))))
        ev += [(self.t_demo_break, "pop", -9, dict(pitch=0.7)), (T("neediron", 0.78), "ding", -10, {})]
        for i, tt in enumerate(self.iron_times):
            ev += [(tt + 0.1, "dig", -8, dict(seed=60 + i)), (tt + 0.25, "pop", -11, dict(pitch=0.9 + 0.05 * i))]
        ev += [(Wd("iron_found", "IRON"), "shimmer", -9, {}), (T("stonepick", 0.9), "ding", -10, {}),
               (T("furnace", 0.45), "place", -8, {}), (T("furnace", 0.55), "crackle", -10, dict(dur=4.0))]
        for i in range(6):
            ev.append((T("ingots", 0.1 + 0.13 * i), "pop", -10, dict(pitch=1.1 + 0.04 * i)))
        ev += [(T("ironpick", 0.6), "ding", -8, {}), (T("bucket", 0.95), "ding", -10, dict(base=660))]
        for k, tt in self.dig_t.items():
            ev.append((tt, "dig", -8 if k % 2 else -10, dict(seed=100 + k)))
            ev.append((tt + 0.35, "step", -15, dict(seed=200 + k)))
        for tt in self.coal_t:
            ev.append((tt, "dig", -8, dict(seed=300)))
        ev.append((self.coal_t[0], "ding", -12, dict(base=660)))
        for tt, _ in self.stair_torches():
            ev.append((tt, "pop", -14, dict(pitch=0.8)))
        ev += [(self.t_gravel, "creak", -9, {}), (self.t_gravel + 0.45, "explosion", -14, dict(seed=5, dur=1.2)),
               (self.t_gravel + 0.2, "whoosh", -10, dict(dur=0.4, up=False))]
        for tt in self.t_gravel_dig:
            ev.append((tt, "dig", -8, dict(seed=310)))
        ev += [(self.dig_t[32], "whoosh", -6, dict(dur=1.2)), (self.dig_t[32] + 0.1, "impact", -10, {}),
               (T("cavern", 0.1), "shimmer", -14, {})]
        tt = C("ceiling")
        while tt < C("groan"):
            ev.append((tt, "ding", -26, dict(base=1500 + 300 * np.sin(tt))))
            tt += 0.9
        ev += [(T("fill", 0.3), "water", -10, dict(dur=1.2)), (T("groan", 0.5), "groan", -6, dict(seed=1)),
               (T("zombie", 0.1), "groan", -6, dict(seed=2))]
        for m in self.mobs:
            for h in m.get("hits", []):
                ev.append((h + 0.08, "hurt", -9, dict(seed=int(h))))
            if m["death"] is not None and m.get("hits"):
                ev.append((m["death"], "pop", -8, dict(pitch=0.7)))
            if m.get("sink"):
                ev.append((m["death"] - 1.2, "water", -8, dict(dur=1.0)))
            if m.get("rise"):
                ev += [(m["spawn"], "dig", -12, dict(seed=5)), (m["spawn"] + 0.4, "groan", -11, dict(seed=7))]
        ev += [(C("night"), "groan", -16, dict(seed=8)), (T("night", 0.6), "groan", -17, dict(seed=9)),
               (T("orange", 0.4), "riser", -16, dict(dur=3.0)), (C("lava"), "crackle", -12, dict(dur=6.0)),
               (C("hot"), "crackle", -10, dict(dur=5.0)), (Wd("sparkle", "DIAMONDS"), "shimmer", -7, {})]
        for i, tp in enumerate(self.pour_t):
            ev += [(tp, "water", -9, dict(dur=0.8)), (tp + 0.25, "hiss", -9, dict(dur=1.2, seed=i)),
                   (tp + 0.45, "place", -11, dict(seed=70 + i))]
        ev += [(T("slip", 0.35), "crackle", -6, dict(dur=2.0)), (T("slip", 0.5), "hurt", -6, {}),
               (T("fire", 0.0), "hurt", -7, dict(seed=1)), (Wd("fire", "three"), "hurt", -7, dict(seed=2)),
               (T("wrong", 0.3), "heartbeat", -6, {}), (T("wrong", 0.7), "heartbeat", -7, dict(seed=1)),
               (T("bridge", 0.5), "ding", -8, {}), (Wd("four", "FOUR"), "shimmer", -6, {})]
        for i, tt in enumerate(self.ore_t):
            ev += [(tt - 0.6, "dig", -10, dict(seed=80 + i)), (tt - 0.3, "dig", -10, dict(seed=90 + i)),
                   (tt, "pop", -8, dict(pitch=1.3)), (tt + 0.05, "shimmer", -12, {})]
        ev += [(Wd("five", "FIVE", 0), "ding", -6, {}), (Wd("time", "Three"), "heartbeat", -6, {})]
        tt = C("ranback")
        while tt < T("ranback", 0.9):
            ev.append((tt, "step", -13, dict(seed=int(tt * 7))))
            tt += 0.25
        ev += [(T("dead", 0.3), "impact", -14, {}), (T("hiss", 0.6), "hiss", -5, dict(dur=self.t_boom - T("hiss", 0.6)))]
        tt = C("run")
        while tt < self.t_boom:
            ev.append((tt, "step", -12, dict(seed=int(tt * 7))))
            tt += 0.22
        ev += [(self.t_boom, "explosion", -2, {}), (T("collapse", 0.0), "explosion", -9, dict(seed=3, dur=2.5))]
        for i, (tt, p, b) in enumerate(self.rubble[::4]):
            ev.append((tt, "place", -14, dict(seed=i)))
        ev += [(C("outside"), "whoosh", -8, dict(dur=0.8)), (C("outside"), "shimmer", -12, {})]
        tt = C("dash")
        while tt < T("dash", 0.95):
            ev.append((tt, "step", -13, dict(seed=int(tt * 7))))
            tt += 0.25
        ev += [(T("madeit", 0.0), "ding", -7, {})]
        for y in (11, 12):
            ev.append((self.t_seal + 0.15 * (y - 11), "place", -9, dict(seed=y)))
            ev.append((self.t_open + 0.1 * (y - 11), "dig", -10, dict(seed=y)))
        ev += [(T("recipe", 0.05), "pop", -10, {}), (T("dpick", 0.0), "shimmer", -5, {}), (T("dpick", 0.0), "impact", -9, {}),
               (T("dsword", 0.6), "shimmer", -6, {}), (T("cue", 0.4), "groan", -8, dict(seed=11)),
               (T("cue", 0.75), "groan", -10, dict(seed=12)), (Wd("night2", "Night"), "impact", -10, {}),
               (T("night2", 0.5), "groan", -10, dict(seed=13)), (Wd("zero", "Zero"), "ding", -8, {}),
               (T("portal", 0.6), "shimmer", -8, {}), (C("outro") + 0.3, "pop", -10, {})]
        for i in range(len(PORTAL)):
            ev.append((T("portal", 0.05 + 0.05 * i), "place", -12, dict(seed=90 + i)))
        return ev

    def music_sections(self):
        T, C = self.T, self.C
        c = [None, None] + [self.card(n) for n in range(2, 11)]
        return [("adventure", 0.0, C("deep")), ("suspense", C("deep"), C("sunset")), ("adventure", C("sunset"), c[2][0]),
                ("build", c[2][0], c[3][0]), ("suspense", c[3][0], C("gravel")), ("silence", C("gravel"), T("gravel", 0.3)),
                ("panic", T("gravel", 0.3), C("dodge") + 1.0), ("suspense", C("dodge") + 1.0, c[4][0]),
                ("suspense", c[4][0], C("zombie")), ("panic", C("zombie"), C("down") + 1.5),
                ("suspense", C("down") + 1.5, C("slip")),
                ("panic", C("slip"), C("wrong")), ("silence", C("wrong"), C("careful")),
                ("suspense", C("careful"), c[6][0]),
                ("adventure", c[6][0], C("time")), ("suspense", C("time"), C("hiss")), ("silence", C("hiss"), C("run")),
                ("freefall", C("run"), self.t_boom), ("silence", self.t_boom, self.t_boom + 1.2), ("suspense", self.t_boom + 1.2, C("outside")),
                ("panic", C("outside"), C("madeit")), ("resolve", C("madeit"), c[10][0]),
                ("panic", c[10][0], C("zero")), ("resolve", C("zero"), self.duration)]


def key_is(f, t, keys):
    return f.key_at(t) in keys
