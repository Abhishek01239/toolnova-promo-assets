"""'I Had 10 Minutes to Survive My First Night': a complete ~5 min, 16:9 film.

Everything is driven by the narration's line timings (assets/words.json):
T(i, f) is the time f (0..1) of the way through line i.
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
from mcengine.geometry import VoxelWorld, box_vertices, translate, rot_y               # noqa: E402
from mcengine.models import (pip_vertices, creeper_vertices, zombie_vertices, block_entity_vertices,  # noqa: E402
                             torch_vertices, chest_vertices, item_vertices)
from mcengine.anim import (seg, smooth, ease_out, ease_in, ease_out_back, ease_in_out, shake, orbit,  # noqa: E402
                           Burst, gather, lerp, clamp01, Track)
from mcengine.lighting import environment                                                # noqa: E402
from mcengine import overlay as ov                                                       # noqa: E402
import script as S                                                                        # noqa: E402

PI = np.pi
W, H = 1920, 1080
GY = 11                     # ground surface height (grass blocks at y=10)
CAVE = np.array([200.0, 40.0, 0.0])   # origin of the cave interior set
TREE = (7, -7)
BOULDER = [(9, 11, -3), (10, 11, -3), (9, 12, -3)]
HILL = (27, -18)
TNT_X = [-5, -3, -1, 1, 3, 5]
TNT_Z = 17
CRATER_C = (0.0, 17.0)


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


class PathTrack:
    """Piecewise-linear path with cumulative distance (for walk cycles) and velocity."""

    def __init__(self, keys):
        self.k = sorted(keys, key=lambda x: x[0])
        self.t = np.array([k[0] for k in self.k])
        self.p = np.array([k[1] for k in self.k], float)
        d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.p[:, [0, 2]], axis=0), axis=1))]
        self.d = d

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
        return (self.pos(t + 0.05) - self.pos(t - 0.05)) / 0.1


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
        self.subs = ov.Subtitles(self.words, {"TNT": (255, 80, 60), "DIAMOND": (110, 235, 255),
                                              "COAL": (200, 200, 200), "BOOM": (255, 150, 30),
                                              "ZOMBIES": (130, 220, 110), "CREEPER": (110, 235, 90),
                                              "CREEPERS": (110, 235, 90), "SURVIVED": (120, 240, 120),
                                              "TEN": (255, 221, 0)},
                                 max_words=6, max_chars=34, y=930, size=66, cx=W / 2, max_w=1500)
        self._build_events()
        self._build_actors()
        self._build_effects()
        self._ground_cache = {}
        self._props_cache = {}
        self._red = self._red_vignette()

    # ------------------------------------------------------------ timing helpers
    def T(self, i, f=0.0):
        return self.ls[i] + f * (self.le[i] - self.ls[i])

    def Wd(self, i, word, k=0):
        hits = [w for w in self.words if w["line"] == i and norm(w["word"]) == norm(word)]
        return hits[min(k, len(hits) - 1)]["start"] if hits else self.T(i, 0.5)

    def card(self, num):
        c = self.chapters[num - 1]
        return c["card_start"], c["card_end"]

    # ------------------------------------------------------------ world: static
    def build_static(self, r):
        self._r = r
        p = os.path.join(self.cache_dir, "far.npz")
        if os.path.exists(p):
            dd = np.load(p)
            o, tr = dd["o"], dd["t"]
        else:
            o, tr = self._far_world().mesh()
            o = np.concatenate([o, self._far_ring()])
            np.savez_compressed(p, o=o, t=tr)
        r.set_static("far", o)
        r.set_static("far_t", tr)
        p = os.path.join(self.cache_dir, "cave.npz")
        if os.path.exists(p):
            dd = np.load(p)
            o = dd["o"]
        else:
            o, _ = self._cave_world().mesh()
            np.savez_compressed(p, o=o)
        r.set_static("cave", o)
        # clouds
        from scipy.ndimage import gaussian_filter
        rng = np.random.default_rng(3)
        f = gaussian_filter(rng.normal(size=(30, 30)), 1.4)
        f /= f.std()
        cw = VoxelWorld()
        for gx in range(30):
            for gz in range(30):
                if f[gx, gz] > 0.7:
                    cw.set((gx - 15, 0, gz - 15), "cloud")
        _, cv = cw.mesh(alpha_override=0.75)
        cv[:, 0:3] = cv[:, 0:3] * np.array([12, 3, 12], np.float32) + np.array([0, 70, 0], np.float32)
        r.set_static("clouds", cv)

    def _height(self, x, z):
        h = (10 + 2.5 * np.sin(x * 0.11 + 0.4) * np.cos(z * 0.09) + 1.5 * np.sin((x + z) * 0.06)
             + 6 * max(0.0, np.sin(x * 0.04 + 2.0) * np.cos(z * 0.05 + 0.3)) ** 2)
        r = max(abs(x), abs(z))
        k = smooth((r - 34) / 14)
        return int(round(10 * (1 - k) + h * k))

    def _far_world(self):
        """Hills and forest outside the local 68x68 patch."""
        w = VoxelWorld()
        R = 64
        Hm = {}
        for x in range(-R, R):
            for z in range(-R, R):
                Hm[(x, z)] = self._height(x, z)
        for (x, z), h in Hm.items():
            if -34 <= x < 34 and -34 <= z < 34:
                continue
            nmin = min(Hm.get((x + a, z + b), h) for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            for y in range(min(nmin, h) - 1, h + 1):
                w.set((x, y, z), "grass" if y == h else ("dirt" if y >= h - 2 else "stone"))
        rng = np.random.default_rng(11)
        trees = []
        for _ in range(500):
            x, z = rng.integers(-R + 3, R - 3, 2)
            if max(abs(x), abs(z)) < 36 and not (max(abs(x), abs(z)) > 18 and x < -14):
                continue
            if max(abs(x), abs(z)) < 18:
                continue
            trees.append((int(x), int(z)))
        for (x, z) in trees[:170]:
            h = Hm.get((x, z), 10)
            self._tree(w, x, h, z, rng)
        return w

    def _tree(self, w, x, h, z, rng, th=None):
        th = th or int(rng.integers(4, 6))
        for y in range(h + 1, h + 1 + th):
            w.set((x, y, z), "log")
        top = h + th
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for dy in range(-1, 2):
                    if (abs(dx) + abs(dz) + max(0, dy) <= 3) and (dx, dz) != (0, 0) or dy == 1 and abs(dx) + abs(dz) <= 1:
                        if w.get((x + dx, top + dy, z + dz)) is None:
                            w.set((x + dx, top + dy, z + dz), "leaves")

    def _far_ring(self, R=64, extent=640, q=8):
        from mcengine.geometry import _face_uv, FACE_CORNERS, FACE_NORMALS, TRI_ORDER
        rng = np.random.default_rng(5)
        uvg = _face_uv("grass_top")
        quads = []
        for x in range(-extent, extent, q):
            for z in range(-extent, extent, q):
                if -R <= x < R and -R <= z < R:
                    continue
                y = 12.0 + (2.0 if np.sin(x * 0.013) * np.cos(z * 0.017) > 0.3 else 0.0)
                c = FACE_CORNERS[2] * np.array([q, 0, q], np.float32) + np.array([x, y, z], np.float32)
                v = np.zeros((6, 12), np.float32)
                sh = rng.uniform(0.75, 0.95)
                for k, ci in enumerate(TRI_ORDER):
                    v[k, 0:3] = c[ci]; v[k, 3:5] = uvg[ci]; v[k, 5:8] = FACE_NORMALS[2]; v[k, 8] = sh; v[k, 10] = 1
                quads.append(v)
        return np.concatenate(quads)

    def _ground(self, version):
        """Local 68x68 patch: version 0 base, 1 trench, 2 blast gap, 3 crater, 4 farm."""
        if version in self._ground_cache:
            return self._ground_cache[version]
        w = VoxelWorld()
        for x in range(-34, 34):
            for z in range(-34, 34):
                h = self._height(x, z)
                for y in range(h - 2, h + 1):
                    w.set((x, y, z), "grass" if y == h else "dirt")
        # hill with tunnel
        hx, hz = HILL
        for x in range(hx - 8, hx + 8):
            for z in range(hz - 8, hz + 8):
                d = np.hypot(x - hx, z - hz)
                top = int(10 + max(0, 8 - d) * 1.0)
                for y in range(10, top + 1):
                    w.set((x, y, z), "grass" if y == top else ("stone" if d < 6 else "dirt"))
        for x in range(hx - 7, hx + 3):
            for z in (hz, hz + 1):
                for y in (11, 12):
                    w.set((x, y, z), None)
        if version >= 1:     # trench for the moat
            for x in range(-6, 7):
                for z in range(-6, 7):
                    if max(abs(x), abs(z)) == 6:
                        w.set((x, 10, z), None)
        if version >= 2:     # creeper blast took a bite out of the east side
            for x in range(5, 8):
                for z in range(-1, 2):
                    w.set((x, 10, z), "dirt")
        if version >= 3:     # TNT crater
            for x in range(-10, 11):
                for z in range(11, 24):
                    e = ((x - CRATER_C[0]) / 9.0) ** 2 + ((z - CRATER_C[1]) / 5.0) ** 2
                    if e < 1:
                        depth = int(round(3 * (1 - e) + 0.4))
                        for y in range(10 - depth + 1, 11):
                            w.set((x, y, z), None)
                        w.set((x, 10 - depth, z), "dirt" if version == 3 else "farmland")
                        if version == 4 and x == 0:
                            w.set((x, 10 - depth, z), "water")
        if version >= 4:     # moat gap repaired
            for z in range(-1, 2):
                w.set((6, 10, z), None)
        self._ground_cache[version] = w
        return w

    def ground_version(self, t):
        v = 0
        if t >= self.T(22, 0.05):
            v = 1
        if t >= self.t_wallboom:
            v = 2
        if t >= self.t_boom:
            v = 3
        if t >= self.T(50, 0.55):
            v = 4
        return v

    def _cave_world(self):
        w = VoxelWorld()
        ox, oy, oz = CAVE.astype(int)
        for x in range(ox - 1, ox + 18):
            for z in range(oz - 6, oz + 7):
                for y in range(oy - 1, oy + 7):
                    inside = (ox <= x <= ox + 16) and (oz - 4 <= z <= oz + 4) and (oy <= y <= oy + 4)
                    if inside:
                        continue
                    if x == ox - 1 and oz <= z <= oz + 1 and oy <= y <= oy + 1:
                        continue                       # entrance opening (daylight beyond)
                    if (x, y, z) == (ox + 4, oy + 1, oz - 5):
                        continue                       # slot for the coal ore entity
                    b = "stone"
                    rr = (x * 7 + z * 13 + y * 5) % 23
                    if rr == 0:
                        b = "cobble"
                    if rr == 5 and y == oy - 1:
                        b = "dirt"
                    w.set((x, y, z), b)
        # rubble and pillars for depth
        for (x, z) in [(ox + 5, oz + 3), (ox + 9, oz - 3), (ox + 12, oz + 4)]:
            for y in range(oy, oy + 5):
                w.set((x, y, z), "stone")
        w.set((ox + 7, oy, oz + 3), "cobble")
        w.set((ox + 11, oy, oz - 4), "cobble")
        return w

    # ------------------------------------------------------------ events (props world)
    def _build_events(self):
        T, Wd = self.T, self.Wd
        ev = []      # (time, pos, block or None)
        # initial props: tree + boulder
        rng = np.random.default_rng(4)
        tw = VoxelWorld()
        self._tree(tw, TREE[0], 10, TREE[1], rng, th=4)
        for p, b in tw.blocks.items():
            ev.append((-1.0, p, b))
        for p in BOULDER:
            ev.append((-1.0, p, "stone"))
        # punching logs off the tree, bottom up
        self.punch_times = [T(4, 0.35), T(4, 0.55), T(4, 0.75), T(4, 0.92), T(5, 0.1), T(5, 0.3), T(5, 0.5), T(5, 0.7)]
        self.log_breaks = [T(4, 0.6), T(4, 0.95), T(5, 0.35), T(5, 0.75)]
        for k, tb in enumerate(self.log_breaks):
            ev.append((tb, (TREE[0], 11 + k, TREE[1]), None))
        # mining the boulder (line 7)
        self.mine_times = [T(7, 0.25), T(7, 0.5), T(7, 0.75)]
        for k, tm in enumerate(self.mine_times):
            ev.append((tm + 0.2, BOULDER[[2, 0, 1][k]], None))
        # -------- base build
        build = []
        walls = [(x, y, z) for y in (11, 12) for x in range(-4, 5) for z in range(-4, 5)
                 if max(abs(x), abs(z)) == 4 and not (x == 0 and z == 4)]
        walls.sort(key=lambda p: (p[1], np.arctan2(p[2], p[0])))
        roof = [(x, 13, z) for x in range(-4, 5) for z in range(-4, 5)]
        roof.sort(key=lambda p: (abs(p[0]) + abs(p[2])) * -1)
        self.final_roof = (0, 13, 0)
        roof.remove(self.final_roof)
        t0, t1 = T(17, 0.6), T(18, 0.75)
        for i, p in enumerate(walls):
            build.append((t0 + (t1 - t0) * i / len(walls), p, "cobble"))
        t0, t1 = T(18, 0.75), T(19, 0.4)
        for i, p in enumerate(roof):
            build.append((t0 + (t1 - t0) * i / len(roof), p, "planks"))
        tower = [(x, y, z) for y in range(14, 17) for x in (-4, -3) for z in (-4, -3)]
        tower += [(x, 17, z) for x in range(-5, -1) for z in range(-5, -1)]
        tower += [(x, 18, z) for x, z in ((-5, -5), (-2, -5), (-5, -2), (-2, -2))]
        t0, t1 = T(20, 0.05), T(20, 0.95)
        for i, p in enumerate(tower):
            build.append((t0 + (t1 - t0) * i / len(tower), p, "cobble" if p[1] < 17 or p[1] == 18 else "planks"))
        ring = [(x, 10, z) for x in range(-6, 7) for z in range(-6, 7) if max(abs(x), abs(z)) == 6 and (x, z) != (0, 6)]
        ring.sort(key=lambda p: np.arctan2(p[2], p[0]))
        t0, t1 = T(22, 0.45), T(22, 0.98)
        for i, p in enumerate(ring):
            build.append((t0 + (t1 - t0) * i / len(ring), p, "water"))
        build.append((T(22, 0.3), (0, 10, 6), "planks"))         # bridge
        build.append((self.Wd(23, "roof"), self.final_roof, "planks"))
        ev += build
        self.build_events = build
        # creeper blast on the east wall
        self.t_wallboom = Wd(33, "BOOM") + 0.05
        self.wall_center = np.array([4.5, 12.0, 0.5])
        # blast removes everything within 2.6 of the wall centre (computed at runtime from state)
        self.blast_removed = None
        # repairs + second wall (line 50)
        t0, t1 = T(50, 0.05), T(50, 0.5)
        rep = [(4, y, z) for y in (11, 12) for z in (-1, 0, 1)] + [(x, 13, z) for x in (3, 4) for z in (-1, 0, 1)]
        rep += [(6, 10, z) for z in (-1, 0, 1)]
        outer = [(x, y, z) for y in (11, 12) for x in range(-5, 6) for z in range(-5, 6)
                 if max(abs(x), abs(z)) == 5 and not (x == 0 and z == 5)]
        outer.sort(key=lambda p: (p[1], np.arctan2(p[2], p[0])))
        rep_all = [(p, "water" if p[1] == 10 else ("planks" if p[1] == 13 else "cobble")) for p in rep]
        rep_all += [(p, "cobble") for p in outer]
        t1 = T(50, 0.95)
        for i, (p, b) in enumerate(rep_all):
            ev.append((t0 + (t1 - t0) * i / len(rep_all), p, b))
        self.repair_events = rep_all
        ev.sort(key=lambda e: e[0])
        self.events = ev
        self.ev_t = np.array([e[0] for e in ev])
        # TNT placement (line 39: counting words)
        counts = ["one", "two", "three", "four", "five", "six"]
        self.tnt_times = [Wd(39, c) for c in counts]
        self.t_fuse = Wd(40, "lit")
        self.t_boom = self.T(41, 0.0) + 0.05
        # torches: (time, base position)
        self.torches = []
        for i, (x, z) in enumerate([(-4, -4), (4, -4), (-4, 4), (4, 4)]):
            self.torches.append((T(23, 0.05 + 0.08 * i), (x + 0.5, 14.0, z + 0.5)))
        for i, (x, z) in enumerate([(-5, -5), (-2, -5), (-5, -2), (-2, -2)]):
            self.torches.append((T(23, 0.4 + 0.05 * i), (x + 0.5, 19.0, z + 0.5)))
        ox, oy, oz = CAVE
        self.cave_torches = [(T(11, 0.15), (ox + 3.5, oy, oz + 3.6)), (T(11, 0.45), (ox + 6.5, oy, oz - 3.6)),
                             (T(11, 0.7), (ox + 10.5, oy, oz + 3.6)), (T(12, 0.4), (ox + 14.5, oy, oz - 3.6))]

    def props_world(self, t):
        """World of placed blocks at time t (settled ones) + list of popping blocks."""
        n_set = int(np.searchsorted(self.ev_t, t - 0.14, side="right"))
        n_all = int(np.searchsorted(self.ev_t, t, side="right"))
        boom = t >= self.t_wallboom
        key = (n_set, boom)
        if key not in self._props_cache:
            w = VoxelWorld()
            for (te, p, b) in self.events[:n_set]:
                if boom and te < self.t_wallboom and self.blast_removed is not None and p in self.blast_removed:
                    continue
                w.set(p, b)
            if boom and self.blast_removed is None:
                self.blast_removed = set(p for p in w.blocks if np.linalg.norm(np.array(p) + 0.5 - self.wall_center) < 2.8
                                         and p[1] >= 10)
                for p in self.blast_removed:
                    w.set(p, None)
            self._props_cache[key] = w
        pops = [e for e in self.events[n_set:n_all] if e[2] is not None]
        return self._props_cache[key], key, pops

    # ------------------------------------------------------------ actors
    def _build_actors(self):
        T, Wd = self.T, self.Wd
        y = GY
        ox, oy, oz = CAVE
        # Pip path keys (time, pos)
        k = [
            (0.0, (0.5, y, 1.5)), (T(3, 0.15), (0.5, y, 1.5)), (T(4, 0.2), (TREE[0] + 0.5, y, TREE[1] + 1.7)),
            (T(6, 0.0), (TREE[0] + 0.5, y, TREE[1] + 1.7)), (T(7, 0.1), (BOULDER[0][0] - 0.2, y, BOULDER[0][2] + 1.9)),
            (T(7, 1.0), (BOULDER[0][0] - 0.2, y, BOULDER[0][2] + 1.9)),
            (T(8, 0.1), (11.0, y, -6.0)), (T(8, 0.95), (HILL[0] - 8.0, y, HILL[1] + 1.0)),
            # cave interior (teleport at cut)
            (T(9, 0.0) - 0.001, (HILL[0] - 8.0, y, HILL[1] + 1.0)), (T(9, 0.0), (ox + 0.8, oy, oz + 0.3)),
            (T(10, 0.0), (ox + 2.5, oy, oz - 1.6)), (T(10, 1.0), (ox + 4.2, oy, oz - 2.4)),
            (T(11, 1.0), (ox + 8.0, oy, oz + 0.5)), (T(12, 0.9), (ox + 12.0, oy, oz + 0.5)),
            (T(14, 1.0), (ox + 12.0, oy, oz + 0.5)), (T(15, 0.6), (ox + 13.2, oy, oz - 0.8)),
            (T(16, 0.25), (ox + 13.2, oy, oz - 0.8)), (T(16, 0.7), (ox + 3.0, oy, oz + 0.5)),
            # running out of the tunnel
            (T(16, 0.7) + 0.001, (HILL[0] - 6.0, y, HILL[1] + 1.0)), (T(16, 1.0) + 1.0, (HILL[0] - 13.0, y, HILL[1] + 4.0)),
            # base site
            (T(17, 0.0), (0.5, y, 6.0)), (T(17, 0.6), (0.5, y, 5.6)), (T(18, 0.5), (5.8, y, 0.5)),
            (T(19, 0.0), (5.8, y, 3.5)), (T(19, 0.9), (5.7, y, 2.0)),
            (T(20, 0.0), (-5.6, y, -1.5)), (T(20, 0.95), (-5.6, y, -1.5)),
            (T(21, 0.0), (-3.5, 18.0, -3.5)), (T(21, 1.0), (-3.5, 18.0, -3.5)),
            (T(22, 0.0), (-2.5, y, 7.6)), (T(22, 1.0), (2.5, y, 7.6)),
            (T(23, 0.0), (0.5, 14.0, 1.5)), (T(25, 1.0), (0.5, 14.0, 1.5)),
            (T(26, 0.0), (-5.6, y, -3.5)), (T(26, 0.5), (-5.6, y, -3.5)), (T(26, 0.9), (-3.5, 18.0, -3.5)),
            (T(37, 0.12), (-3.5, 18.0, -3.5)),
        ]
        # jump off the tower and the chase into the field
        tj = T(37, 0.12)
        self.t_jump = tj
        k += [(tj + 0.5, (-5.4, y, -5.4)), (tj + 1.6, (-5.4, y, 4.8)), (tj + 2.3, (0.5, y, 5.4)),
              (tj + 2.9, (0.5, y, 8.0)), (T(38, 0.4), (2.0, y, 14.0)), (T(38, 1.0), (3.0, y, 19.5)),
              (self.tnt_times[0] - 0.3, (-5.0, y, 19.0))]
        for i, tt in enumerate(self.tnt_times):
            k.append((tt, (TNT_X[i] + 0.5, y, 18.8)))
        k += [(self.tnt_times[-1] + 0.25, (5.5, y, 18.8)), (T(40, 0.12), (0.5, y, 8.0)), (T(40, 0.2), (0.5, y, 6.0)),
              (T(40, 0.28), (2.0, y, 5.6)), (T(44, 1.0), (2.0, y, 5.6)), (T(45, 0.0), (-3.5, 18.0, -3.5)), (T(49, 1.0), (-3.5, 18.0, -3.5)),
              (T(50, 0.0), (7.8, y, 0.5)), (T(50, 0.5), (7.8, y, 8.5)), (T(50, 0.95), (2.0, y, 8.6)),
              (T(51, 0.0), (1.5, y, 9.0)),
              (self.duration, (1.5, y, 9.0))]
        self.pip_path = PathTrack(k)
        # ------------- zombies
        rng = np.random.default_rng(21)
        spawn_ts = [Wd(27, "first"), Wd(27, "another")] + [T(27, 0.72 + 0.06 * i) for i in range(5)]
        angs = [2.2, 1.5, 2.9, 0.9, 3.5, 0.3, 4.2]
        self.zombies = []
        tm0, tm1 = T(28, 0.0), T(28, 0.9)
        for i, (ts, a) in enumerate(zip(spawn_ts, angs)):
            R0 = 17 + rng.uniform(-2, 3)
            p0 = np.array([np.sin(a) * R0, y, np.cos(a) * R0])
            dv = np.array([np.sin(a), 0, np.cos(a)])
            dv = dv / max(abs(dv[0]), abs(dv[2]))
            edge = np.array([dv[0] * 7.4, y, dv[2] * 7.4])
            back = np.array([dv[0] * 9.2, y, dv[2] * 9.2])
            keys = [(ts, p0), (tm0 + 0.3 * i, p0), (tm1 + 0.3 * i, edge)]
            tb = tm1 + 0.3 * i
            j = 0
            while tb < self.t_wallboom:
                keys.append((tb + 1.3, back if j % 2 == 0 else edge))
                tb += 1.3
                j += 1
            self.zombies.append({"spawn": ts, "keys": keys, "a": a, "id": i})
        # after the wall blast, 4 zombies head for the hole
        hole_path = [np.array([9.0, y, 0.5]), np.array([6.0, y, 0.5]), np.array([4.5, y, 0.5]), np.array([1.5, y, 0.5])]
        tb = T(34, 0.0)
        for j, z in enumerate(self.zombies[:4]):
            z["keys"] = [kk for kk in z["keys"] if kk[0] < tb - 0.5]
            last = np.array(z["keys"][-1][1])
            z["keys"].append((tb + 0.1 * j, last))
            z["keys"].append((tb + 1.2 + 0.4 * j, hole_path[0] + np.array([0, 0, (j - 1.5) * 0.7])))
            for q, hp in enumerate(hole_path[1:]):
                z["keys"].append((tb + 2.0 + 0.4 * j + 0.8 * q, hp + np.array([0, 0, (j - 1.5) * 0.5])))
        # chase: everybody follows Pip with a lag after he jumps, then crowds the TNT line
        self.creepers = []
        for c in range(3):
            ts = T(38, 0.05 + 0.08 * c)
            p0 = np.array([13.0 + 2 * c, y, 12.0 + 3 * c])
            self.creepers.append({"spawn": ts, "keys": [(ts, p0)], "id": c})
        mobs = self.zombies + self.creepers
        targets = []
        for i, m in enumerate(mobs):
            col = i % 6
            targets.append(np.array([TNT_X[col] + 0.5 + rng.uniform(-0.6, 0.6), y, TNT_Z + 0.5 + rng.uniform(-1.6, 1.6)]))
        for i, m in enumerate(mobs):
            lag = 1.6 + 0.25 * i
            t_start = tj + 0.8 if m in self.zombies else m["spawn"] + 0.2
            m["keys"] = [kk for kk in m["keys"] if kk[0] < t_start]
            if not m["keys"]:
                m["keys"] = [(t_start, m["keys_p0"] if "keys_p0" in m else np.array([12.0, y, 12.0]))]
            tt = t_start
            off = np.array([(i % 4 - 1.5) * 1.3, 0, (i // 4) * 1.2])
            while tt < self.tnt_times[-1]:
                p = self.pip_path.pos(max(tt - lag, tj)) + off
                p[1] = y
                m["keys"].append((tt + 0.5, p))
                tt += 0.5
            m["keys"].append((self.t_boom - 0.6, targets[i]))
            m["death"] = self.t_boom + 0.02 * i
            m["fly_v"] = np.array([rng.uniform(-6, 6), rng.uniform(9, 15), rng.uniform(-6, 6)])
            m["fly_spin"] = rng.uniform(-6, 6, 3)
        for m in mobs:
            ks = sorted([(float(a), np.array(b, float)) for a, b in m["keys"]], key=lambda x: x[0])
            dedup = []
            for a, b in ks:
                if dedup and a <= dedup[-1][0]:
                    a = dedup[-1][0] + 1e-3
                dedup.append((a, b))
            m["path"] = PathTrack(dedup)
        # survivors who burn at sunrise
        self.survivors = []
        for i, (x, z) in enumerate([(-15.0, 9.0), (-12.0, 15.0), (-17.0, 1.0)]):
            ts = T(45, 0.0)
            keys = [(ts, (x, y, z)), (ts + 12, (x + 3.5, y, z - 1.5))]
            self.survivors.append({"path": PathTrack(keys), "spawn": ts - 2.0, "ignite": Wd(46, "caught") + 0.9 * i,
                                   "id": i})

    def pip_state(self, t):
        T, Wd = self.T, self.Wd
        pos = self.pip_path.pos(t)
        v = self.pip_path.vel(t)
        sp = float(np.hypot(v[0], v[2]))
        pose = {"walk": clamp01(sp / 3.2), "walk_phase": self.pip_path.dist(t) * 2.4}
        yaw = float(np.arctan2(v[0], v[2])) if sp > 0.4 else None
        item = None
        # facing overrides when standing
        if yaw is None:
            if T(4, 0.2) <= t < T(6, 0.0):
                yaw = PI
            elif T(7, 0.1) <= t < T(8, 0.1):
                yaw = PI
            elif T(10, 0.0) <= t < T(11, 0.0):
                yaw = PI
            elif T(13, 0.0) <= t < T(15, 0.0):
                yaw = PI / 2
            elif T(15, 0.6) <= t < T(16, 0.3):
                yaw = PI / 2 + 0.5
            elif T(21, 0.0) <= t < T(22, 0.0):
                yaw = PI * 0.75
            elif self.t_fuse - 1.0 <= t < T(44, 1.0):
                yaw = 0.0
            elif T(45, 0.0) <= t < T(50, 0.0):
                yaw = -PI / 2 - 0.4
            else:
                yaw = 0.0 if t < T(26, 0) else PI * 0.25
        # punches / mining / placing
        for tp in self.punch_times + self.mine_times + [self.T(10, f) for f in (0.2, 0.4, 0.6, 0.8)] + \
                [T(16, 0.1), T(16, 0.2)] + self.tnt_times:
            if tp - 0.05 <= t < tp + 0.2:
                pose["punch"] = seg(t, tp - 0.05, tp + 0.2)
        if T(17, 0.6) <= t < T(20, 1.0) or T(22, 0.0) <= t < T(23, 1.0) or T(50, 0.0) <= t < T(50, 1.0):
            pose["punch"] = (t * 5.0) % 1.0
        # items
        if T(6, 0.8) <= t < T(17, 0.0):
            item = "pickaxe"
        if T(11, 0.0) <= t < T(12, 0.5):
            item = "torch"
        if T(17, 0.0) <= t < T(24, 0.0):
            item = None
        if T(35, 0.3) <= t < self.t_fuse + 0.3:
            item = "tnt" if t < self.tnt_times[-1] + 0.3 else "torch"
        if T(52, 0.0) <= t:
            item = None
        # waves / celebrations / looks
        if T(6, 1.0) <= t < T(7, 0.0):
            pose["wave"] = 1.0
            pose["wave_phase"] = t * 12
        if T(48, 0.2) <= t < T(48, 1.0):
            pose["wave"] = np.sin(PI * seg(t, T(48, 0.2), T(48, 1.0)))
            pose["wave_phase"] = t * 14
        if T(53, 0.0) <= t:
            pose["wave"] = smooth(seg(t, T(53, 0.0), T(53, 0.2)))
            pose["wave_phase"] = t * 10
            yaw = 0.35
        if T(1, 0.0) <= t < T(1, 1.0):
            pose["head_yaw"] = 0.8 * np.sin((t - T(1, 0)) * 2.5)
        if T(5, 0.6) <= t < T(6, 0.0):
            pose["head_pitch"] = 0.7
        if T(12, 0.6) <= t < T(15, 1.0) or T(10, 0.0) <= t < T(10, 1.0):
            pose["head_pitch"] = -0.25
        if T(33, 0.0) <= t < T(34, 1.0):
            pose["head_pitch"] = -0.45
            pose["head_yaw"] = -0.8
        # jump from the tower
        tj = self.t_jump
        if tj <= t < tj + 0.5:
            u = seg(t, tj, tj + 0.5)
            pos = pos.copy()
            pos[1] = 18.0 * (1 - u) + GY * u + 1.2 * np.sin(PI * u)
            pose["arms_up"] = 0.6
            pose["flail_phase"] = t * 8
        # diving behind the wall
        if self.t_fuse + 0.4 <= t < self.t_boom + 1.4:
            pose["lean"] = 0.55 * smooth(seg(t, self.t_fuse + 0.4, self.t_fuse + 0.8))
            pose["walk"] = 0
        pose["item"] = item
        return pos, yaw, pose

    # ------------------------------------------------------------ effects
    def _build_effects(self):
        T, Wd = self.T, self.Wd
        B = self.bursts = []
        ox, oy, oz = CAVE
        # tree punching chips + log breaks
        for i, tp in enumerate(self.punch_times):
            hk = min(3, sum(1 for tb in self.log_breaks if tb < tp))
            B.append(Burst(tp + 0.12, (TREE[0] + 0.5, 11.5 + hk, TREE[1] + 1.0), 10, speed=(1, 3), life=(0.3, 0.6),
                           size=(0.05, 0.1), colors=[(0.42, 0.33, 0.2, 1)], gravity=-12, seed=10 + i, box=0.2))
        for i, tm in enumerate(self.mine_times):
            p = BOULDER[[2, 0, 1][i]]
            B.append(Burst(tm + 0.2, (p[0] + 0.5, p[1] + 0.6, p[2] + 0.5), 30, speed=(1.5, 4), life=(0.4, 0.8),
                           size=(0.06, 0.12), colors=[(0.5, 0.5, 0.5, 1), (0.35, 0.35, 0.35, 1)], gravity=-14,
                           seed=30 + i, box=0.4))
        # cave: coal reveal sparkle, torches glow, chest dust, diamond sparkle
        B.append(Burst(Wd(10, "COAL"), (ox + 4.5, oy + 1.5, oz - 3.6), 30, speed=(0.5, 2), life=(0.5, 1.0),
                       size=(0.05, 0.09), colors=[(1, 1, 1, 1)], gravity=0, drag=2, seed=40, additive=True, box=0.4))
        B.append(Burst(T(13, 0.2), (ox + 14.5, oy + 0.8, oz + 0.5), 40, speed=(0.3, 1.2), life=(1.0, 2.0),
                       size=(0.05, 0.12), colors=[(0.8, 0.7, 0.55, 0.7)], gravity=0.4, drag=1.5, seed=41, box=0.6))
        t = T(15, 0.0)
        k = 0
        while t < T(16, 0.3):
            B.append(Burst(t, (ox + 14.5, oy + 0.5, oz - 1.5), 6, speed=(0.3, 1.0), life=(0.5, 1.0),
                           size=(0.04, 0.08), colors=[(0.6, 1, 1, 1), (1, 1, 1, 1)], gravity=0.5, drag=2,
                           seed=50 + k, additive=True, box=0.5))
            t += 0.25
            k += 1
        B.append(Burst(Wd(15, "DIAMOND"), (ox + 14.5, oy + 0.5, oz - 1.5), 60, speed=(1, 3), life=(0.6, 1.2),
                       size=(0.05, 0.1), colors=[(0.6, 1, 1, 1), (1, 1, 1, 1)], gravity=0, drag=2, seed=59,
                       additive=True, box=0.3))
        # zombie dirt when rising
        for z in self.zombies:
            a = z["keys"][0][1]
            B.append(Burst(z["spawn"], (a[0], GY + 0.1, a[2]), 30, speed=(1, 3), life=(0.5, 1.0), size=(0.06, 0.12),
                           colors=[(0.4, 0.28, 0.18, 1), (0.3, 0.45, 0.2, 1)], gravity=-12, up_bias=1.2,
                           seed=60 + z["id"], box=0.4))
        # wall blast
        E = tuple(self.wall_center)
        tb = self.t_wallboom
        B.append(Burst(tb, E, 200, speed=(3, 13), life=(0.35, 1.0), size=(0.4, 1.2),
                       colors=[(1, 0.95, 0.6, 1), (1, 0.6, 0.15, 1), (1, 0.35, 0.05, 1)], gravity=3, drag=3, grow=1.2,
                       seed=70, additive=True))
        B.append(Burst(tb, E, 140, speed=(2, 8), life=(1.4, 3.0), size=(0.7, 1.8),
                       colors=[(0.25, 0.24, 0.24, 0.75), (0.4, 0.38, 0.36, 0.7)], gravity=1.6, drag=1.8, grow=2.2,
                       seed=71))
        # TNT fuse sparks along the line from Pip to the TNT
        for i in range(24):
            ti = self.t_fuse + 0.05 * i
            u = i / 23
            p = (1.5 * (1 - u) + 0.5 * u, GY + 0.1, 6.0 * (1 - u) + TNT_Z * u)
            B.append(Burst(ti, p, 6, speed=(0.5, 2), life=(0.2, 0.4), size=(0.04, 0.08),
                           colors=[(1, 0.85, 0.4, 1)], gravity=-4, seed=80 + i, additive=True))
        # the big chain explosion
        for i, x in enumerate(TNT_X):
            te = self.t_boom + 0.11 * i
            P = (x + 0.5, GY + 0.5, TNT_Z + 0.5)
            B.append(Burst(te, P, 220, speed=(4, 18), life=(0.4, 1.3), size=(0.6, 1.8),
                           colors=[(1, 0.95, 0.6, 1), (1, 0.65, 0.15, 1), (1, 0.35, 0.05, 1)], gravity=3, drag=3,
                           grow=1.3, seed=100 + i, additive=True))
            B.append(Burst(te, P, 120, speed=(2, 9), life=(2.0, 4.5), size=(1.0, 2.6),
                           colors=[(0.25, 0.24, 0.24, 0.75), (0.4, 0.38, 0.36, 0.7), (0.12, 0.12, 0.12, 0.8)],
                           gravity=1.5, drag=1.5, grow=2.5, seed=110 + i))
            B.append(Burst(te, P, 120, speed=(10, 28), life=(0.6, 1.6), size=(0.08, 0.16),
                           colors=[(1, 0.9, 0.4, 1)], gravity=-16, drag=0.8, seed=120 + i, additive=True))
        for c in self.creepers:
            P = tuple(c["path"].pos(self.t_boom - 0.1) + np.array([0, 1.0, 0]))
            B.append(Burst(self.t_boom + 0.35 + 0.1 * c["id"], P, 160, speed=(3, 14), life=(0.4, 1.0),
                           size=(0.5, 1.4), colors=[(1, 0.95, 0.6, 1), (0.6, 1, 0.4, 1)], gravity=3, drag=3, grow=1.2,
                           seed=130 + c["id"], additive=True))
        # lingering crater smoke
        t, k = self.t_boom + 1.0, 0
        while t < T(44, 1.0):
            B.append(Burst(t, (CRATER_C[0] + 0.5, GY - 1, CRATER_C[1] + 0.5), 4, speed=(0.3, 1.5), life=(3, 5),
                           size=(2.0, 3.5), colors=[(0.3, 0.3, 0.32, 0.45)], gravity=1.2, drag=0.6, grow=1.5,
                           seed=200 + k, box=5.0))
            t += 0.4
            k += 1
        # mobs' death poofs
        for m in self.zombies + self.creepers:
            B.append(Burst(m["death"] + 1.0, (0, 0, 0), 0, seed=1))  # placeholder (positions added in frame)
        # burning survivors
        for s in self.survivors:
            t = s["ignite"]
            k = 0
            while t < s["ignite"] + 1.8:
                p = s["path"].pos(t)
                B.append(Burst(t, (p[0], GY + 1.0, p[2]), 8, speed=(0.3, 1.2), life=(0.3, 0.7), size=(0.12, 0.3),
                               colors=[(1, 0.7, 0.2, 1), (1, 0.4, 0.05, 1), (1, 0.9, 0.4, 1)], gravity=3, drag=1,
                               seed=300 + 10 * s["id"] + k, additive=True, box=0.35))
                t += 0.1
                k += 1
            p = s["path"].pos(s["ignite"] + 1.8)
            B.append(Burst(s["ignite"] + 1.8, (p[0], GY + 0.8, p[2]), 50, speed=(0.5, 2.5), life=(0.8, 1.6),
                           size=(0.3, 0.7), colors=[(0.7, 0.7, 0.7, 0.8)], gravity=1, drag=2, grow=1.5,
                           seed=400 + s["id"]))
        # block placement puffs
        for i, (tp, p, b) in enumerate(self.build_events[::4]):
            B.append(Burst(tp, (p[0] + 0.5, p[1] + 1.0, p[2] + 0.5), 5, speed=(0.5, 1.5), life=(0.25, 0.45),
                           size=(0.05, 0.09), colors=[(1, 1, 1, 0.9)], gravity=0, drag=3, seed=500 + i,
                           additive=True, box=0.4))
        # crater crops popping (farm)
        self.crops = []
        rng = np.random.default_rng(9)
        for x in range(-7, 8):
            for z in range(13, 22):
                if x == 0:
                    continue
                e = ((x - CRATER_C[0]) / 9.0) ** 2 + ((z - CRATER_C[1]) / 5.0) ** 2
                if e < 0.8 and (x + z) % 2 == 0:
                    depth = int(round(3 * (1 - e) + 0.4))
                    self.crops.append((T(50, 0.6) + rng.uniform(0, 0.9), (x + 0.5, 11 - depth, z + 0.5)))

    def _red_vignette(self):
        yy = np.linspace(-1, 1, H)[:, None]
        xx = np.linspace(-1, 1, W)[None, :]
        r = np.clip((np.sqrt(xx ** 2 * 0.7 + yy ** 2 * 1.1) - 0.5) / 0.6, 0, 1) ** 1.5
        a = np.zeros((H, W, 4), np.uint8)
        a[..., 0] = 200
        a[..., 3] = (r * 255).astype(np.uint8)
        return a

    # ------------------------------------------------------------ camera
    def in_cave(self, t):
        return self.T(9, 0.0) <= t < self.T(16, 0.7)

    def camera(self, t):
        T, Wd = self.T, self.Wd
        pip, _, _ = self.pip_state(t)
        P = pip + np.array([0, 1.2, 0])
        ox, oy, oz = CAVE
        fov, roll = 55.0, 0.0
        cards = [self.card(n) for n in range(2, 7)]

        def L(i):
            return T(i, 0.0) - 0.15

        if t < L(1):                                  # hook: face close-up -> reveal
            u = ease_out(seg(t, 0, L(1)), 3)
            eye = lerp(pip + (0.05, 1.65, 1.5), pip + (8, 6, 12), u)
            tgt = lerp(pip + (0, 1.6, 0), pip + (0, 1.0, 0), u)
            fov = 50 + 10 * u
        elif t < L(2):
            a = 0.6 + 1.2 * seg(t, L(1), L(2))
            eye = orbit(pip, 6.0, a, 2.6)
            tgt = P
        elif t < L(3):                                # night preview
            u = seg(t, L(2), L(3))
            eye = lerp((-2, 14, 26), (2, 12.5, 22), u)
            tgt = (0, 12, 10)
        elif t < L(4):
            eye = pip + np.array([5.5, 2.0, 3.0])
            tgt = P
        elif t < L(5):
            eye = (TREE[0] + 6.0, 14.2, TREE[1] + 6.0)
            tgt = (TREE[0] + 0.5, 12.6, TREE[1] + 1.0)
        elif t < L(6):
            u = seg(t, L(5), L(6))
            eye = (TREE[0] + 2.4, 12.4, TREE[1] + 2.8)
            tgt = lerp((TREE[0] + 0.5, 12.2, TREE[1] + 0.5), (TREE[0] + 0.5, 15.8, TREE[1] + 0.5), smooth(seg(u, 0.55, 1)))
            fov = 50
        elif t < L(7):
            u = seg(t, L(6), L(7))
            eye = pip + np.array([0.3 - 0.6 * u, 1.4, 5.2])
            tgt = pip + np.array([2.3, 1.1, 0.0])
            fov = 52
        elif t < cards[0][0]:
            eye = (5.0, 12.8, -0.6)
            tgt = (9.3, 11.8, -2.4)
        elif t < L(9):                                # card + approach to the cave
            u = seg(t, cards[0][0], L(9))
            eye = lerp((4, 15, -8), (9, 13.2, -12.5), smooth(u))
            tgt = lerp((HILL[0] - 5, 14, HILL[1]), (HILL[0] - 7, 12.2, HILL[1] + 1), smooth(u))
        elif t < L(10):                               # inside looking out
            eye = (ox + 7.0, oy + 2.2, oz + 2.5)
            tgt = (ox + 0.5, oy + 1.2, oz + 0.3)
            fov = 60
        elif t < L(11):
            u = seg(t, L(10), L(11))
            eye = (ox + 6.5 - u, oy + 2.0, oz + 1.8)
            tgt = (ox + 4.2, oy + 1.2, oz - 3.6)
            fov = 58
        elif t < L(12):
            a = -0.5 + 0.8 * seg(t, L(11), L(12))
            eye = pip + np.array([np.sin(a) * 4.0, 1.8, np.cos(a) * 3.0])
            tgt = P
        elif t < L(13):
            u = smooth(seg(t, L(12), L(13)))
            eye = (ox + 5 + 5 * u, oy + 2.3, oz + 2.6)
            tgt = (ox + 14.5, oy + 0.6, oz + 0.5)
            fov = 55 - 10 * u
        elif t < L(14):
            u = ease_out(seg(t, L(13), L(14)))
            eye = (ox + 12.0 - 0.6 * u, oy + 1.9, oz + 2.8 - 0.6 * u)
            tgt = (ox + 14.5, oy + 0.6, oz + 0.5)
            fov = 50
        elif t < L(15):
            eye = pip + np.array([-2.4, 1.5, 2.6])
            tgt = (ox + 14.5, oy + 0.6, oz + 0.5)
        elif t < L(16):
            u = smooth(seg(t, L(15), L(16)))
            eye = (ox + 11.8 + 0.8 * u, oy + 1.5 - 0.3 * u, oz - 3.3)
            tgt = (ox + 14.5, oy + 0.6, oz - 1.5)
            fov = 50 - 14 * u
        elif t < T(16, 0.7):
            eye = (ox + 1.0, oy + 1.8, oz + 2.5)
            tgt = P
            fov = 62
        elif t < cards[1][0]:                         # running out into daylight
            eye = (HILL[0] - 16, 12.6, HILL[1] + 8)
            tgt = P
        elif t < L(18):                               # card + base site
            u = seg(t, cards[1][0], L(18))
            eye = lerp((10, 15, 16), (8, 13, 13), u)
            tgt = (0.5, 11.5, 0.5)
        elif t < L(19):
            a = 0.4 + 1.4 * seg(t, L(18), L(19))
            eye = orbit((0.5, 11, 0.5), 13.0, a, 5.0)
            tgt = (0.5, 12.0, 0.5)
        elif t < L(20):
            u = seg(t, L(19), L(20))
            eye = (9.0 - 0.6 * u, 12.4, 1.5 + 2 * u)
            tgt = (4.5, 12.0, 0.5)
            fov = 50
        elif t < L(21):
            u = seg(t, L(20), L(21))
            eye = (-9.5, 11.6, 3.5)
            tgt = lerp((-3.5, 13.0, -3.5), (-3.5, 17.5, -3.5), smooth(u))
            fov = 55
        elif t < L(22):
            eye = (-7.4, 21.2, -7.4)
            tgt = (8, 15.0, 8)
            fov = 62
        elif t < L(23):
            u = seg(t, L(22), L(23))
            eye = orbit((0.5, 11, 0.5), 15.0, 0.2 + 0.6 * u, 10.0)
            tgt = (0.5, 10.5, 0.5)
        elif t < cards[2][0]:
            u = ease_in_out(seg(t, L(23), cards[2][0]))
            eye = orbit((0.5, 11, 0.5), 11 + 8 * u, 0.9 + 1.2 * u, 4 + 6 * u)
            tgt = (0.5, 12.5, 0.5)
        elif t < L(25):                               # card + sunset
            u = seg(t, cards[2][0], L(25))
            eye = lerp((-4, 13.5, 24), (-2, 13.0, 20), u)
            tgt = (6, 14, -20)
            fov = 60
        elif t < L(26):
            u = seg(t, L(25), L(26))
            eye = orbit((0.5, 11, 0.5), 20.0, 2.6 + 0.4 * u, 6.0)
            tgt = (0.5, 12, 0.5)
        elif t < L(27):
            eye = (-8.5, 15.0, -0.5)
            tgt = P
        elif t < L(28):
            zp = self.zombies[0]["path"].pos(t)
            u = seg(t, L(27), L(28))
            if u < 0.45:
                eye = zp + np.array([2.2, 1.3, 2.4])
                tgt = zp + np.array([0, 0.8, 0])
            else:
                eye = (-9.0, 24.5, -9.0)
                tgt = (6, 11, 8)
                fov = 60
        elif t < L(29):
            zp = self.zombies[1]["path"].pos(t)
            eye = zp + np.array([-2.5, 1.0, 3.5])
            tgt = zp + np.array([0, 1.2, -3])
        elif t < L(30):
            eye = pip + np.array([1.6, 0.6, 2.8])
            tgt = P
        elif t < L(31):
            eye = (7.0, 15.5, 13.0)
            tgt = (3.0, 11, 6.5)
            fov = 55
        elif t < L(32):
            u = smooth(seg(t, L(31), L(32)))
            eye = (14 - 4 * u, 12.5, 6 - 2 * u)
            tgt = (5.0, 11.8, 0.5)
            fov = 55 - 8 * u
        elif t < L(33):
            cp = self._creeper_wall_pos(t)
            u = seg(t, L(32), L(33))
            if u < 0.5:
                eye = cp + np.array([2.8, 1.2, 2.2])
                tgt = cp + np.array([0, 1.0, 0])
            else:
                eye = pip + np.array([1.4, 0.4, 2.4])
                tgt = P
        elif t < L(34):
            eye = (16, 15, 9)
            tgt = (4.5, 11.5, 0.5)
            fov = 58
        elif t < cards[3][0]:
            eye = (-8.0, 25.0, -7.0)
            tgt = (4.0, 11, 1.0)
            fov = 56
        elif t < L(36):                               # card + Pip remembers the TNT
            eye = pip + np.array([0.0, 1.0, 3.3])
            tgt = P
            fov = 50
        elif t < L(37):
            u = seg(t, L(36), L(37))
            eye = (-1.0 + 1.5 * u, 21.5, 4.5)
            tgt = (-2.8, 17.0, -2.8)
            fov = 56
        elif t < L(38):                               # chase: wide tracking, then behind Pip in the field
            if pip[2] < 6.5 and pip[0] < -2:
                eye = pip + np.array([-7.5, 6.5, 2.0])
                tgt = P
                fov = 60
            else:
                eye = pip + np.array([6.5, 4.0, -1.0])
                tgt = P + np.array([0, 0, 1.5])
                fov = 62
        elif t < L(39):                               # chase: Pip running toward camera with horde
            eye = pip + np.array([0.8, 1.6, 6.0])
            tgt = P + np.array([0, -0.2, -3])
            fov = 60
        elif t < L(40):                               # placing the TNT line
            eye = (0.5, 17.5, 29.0)
            tgt = (0.5, 11.0, 17.5)
            fov = 55
        elif t < self.t_boom:
            u = seg(t, L(40), self.t_boom)
            if u < 0.6:
                eye = (3.4, 15.0, 4.4)
                tgt = (0.6, 11.4, 15.0)
                fov = 64
            else:
                eye = (-12, 15, 30)
                tgt = (0.5, 11.5, 17)
                fov = 55
        elif t < L(42):
            eye = (-16, 18, 34)
            tgt = (0.5, 12.5, 17)
            fov = 58
        elif t < L(43):
            a = 3.6 + 0.6 * seg(t, L(42), L(43))
            eye = orbit((0.5, 11, 17), 22.0, a, 9.0)
            tgt = (0.5, 12.5, 17)
        elif t < L(44):
            u = smooth(seg(t, L(43), L(44)))
            eye = lerp((0.5, 26, 32), (0.5, 22, 30), u)
            tgt = (0.5, 9, 16)
        elif t < cards[4][0]:
            eye = (4.6, 12.8, 12.0)
            tgt = (1.8, 11.9, 6.0)
        elif t < L(46):                               # card + sunrise with survivors
            u = seg(t, cards[4][0], L(46))
            eye = (-4, 14, 20)
            tgt = lerp((-14, 12, 7), (-16, 13, 4), u)
            fov = 58
        elif t < L(47):
            s = self.survivors[min(2, int(seg(t, L(46), L(47)) * 3))]
            sp = s["path"].pos(t)
            eye = sp + np.array([3.5, 1.4, 3.0])
            tgt = sp + np.array([0, 1.0, 0])
        elif t < L(48):
            eye = pip + np.array([-3.4, 1.8, 0.8])
            tgt = P
        elif t < L(49):
            eye = pip + np.array([-3.4, 0.9, 0.0])
            tgt = P
        elif t < L(50):                               # sunrise silhouette
            eye = pip + np.array([3.4, 0.9, 0.0])
            tgt = pip + np.array([-6, 1.8, 1.5])
            fov = 55
        elif t < L(51):
            u = seg(t, L(50), L(51))
            eye = orbit((0.5, 11, 6), 22.0, 0.3 + 0.5 * u, 11.0)
            tgt = (0.5, 11, 6)
        elif t < L(52):
            u = ease_in_out(seg(t, L(51), L(52)))
            eye = orbit((0.5, 11, 6), 26.0, 0.8 + 1.0 * u, 14.0)
            tgt = (0.5, 12, 4)
        elif t < L(53):
            eye = pip + np.array([1.4, 0.9, 2.2])
            tgt = pip + np.array([0.6, 1.3, 0.5])
            fov = 45
        else:
            u = ease_in_out(seg(t, L(53), self.duration))
            eye = lerp(pip + np.array([3.0, 1.6, 5.0]), np.array([14.0, 24.0, 30.0]), u)
            tgt = lerp(P, np.array([0.5, 12, 3]), u)
        eye = np.asarray(eye, np.float32)
        amp = 0.0
        for t0, a, d in ((self.t_wallboom, 0.45, 1.2), (self.t_boom, 1.2, 2.2), (self.t_boom + 0.55, 0.6, 1.4)):
            if t0 <= t < t0 + d:
                amp += a * (1 - (t - t0) / d) ** 2
        if T(37, 0.2) <= t < T(39, 0.0):
            amp += 0.05
        eye = eye + shake(t, amp)
        return dict(eye=eye, target=np.asarray(tgt, np.float32), fov=float(fov), roll=roll)

    def cuts(self):
        out = [self.T(i, 0.0) - 0.15 for i in range(1, len(S.LINES))]
        return [c for c in out if c > 0.5]

    def _creeper_wall_pos(self, t):
        T = self.T
        p0 = np.array([8.5, GY, -3.5])
        p1 = np.array([5.3, GY, 0.5])
        return lerp(p0, p1, smooth(seg(t, T(31, 0.5), T(32, 0.5))))

    # ------------------------------------------------------------ lighting phase
    def phase(self, t):
        T = self.T
        return float(Track([(0, 0, "linear"), (T(21, 0.0), 0, "linear"), (T(24, 0.6), 1.0, "smooth"),
                            (T(25, 0.2), 1.3, "smooth"), (T(25, 0.9), 2.0, "smooth"),
                            (T(45, 0.1), 2.0, "linear"), (T(45, 0.95), 3.0, "smooth")])(t)[()])

    # ------------------------------------------------------------ frame spec
    def frame_spec(self, t, cam=None):
        T, Wd = self.T, self.Wd
        cam = cam or self.camera(t)
        cave = self.in_cave(t)
        night_preview = T(2, 0.0) - 0.15 <= t < T(3, 0.0) - 0.15
        ph = 2.0 if night_preview else self.phase(t)
        sky, light, fog = environment(ph, 40, 300)
        post = dict(bloom=0.32, vignette=0.32, saturation=1.12)
        if night_preview:
            post["tint"] = (0.85, 0.8, 1.15)
            post["saturation"] = 0.7
        if cave:
            lit = sum(1 for tt, _ in self.cave_torches if t >= tt)
            k = 0.55 + 0.14 * lit
            sky = dict(sky)
            sky.update(zenith=(0.02, 0.02, 0.02), horizon=(0.03, 0.025, 0.02), below=(0.02, 0.02, 0.02),
                       stars=0.0, sun_vis=0.0, moon_vis=0.0)
            light = dict(dir=(0.3, 0.9, 0.2), sun=(0.25 * k, 0.2 * k, 0.12 * k),
                         ambient=(0.62 * k, 0.5 * k, 0.36 * k), exposure=1.2)
            fog = dict(color=(0.03, 0.025, 0.02), start=6, end=34)
            post["vignette"] = 0.55
        if ph >= 1.8 and not cave:
            light = dict(light)
            light["ambient"] = tuple(np.array(light["ambient"]) + np.array([0.06, 0.04, 0.0]))
            fog = dict(fog, start=30, end=200)
        for t0, d, col in ((self.t_wallboom, 1.0, (1.2, 0.7, 0.3)), (self.t_boom, 1.8, (1.6, 0.9, 0.35))):
            if t0 <= t < t0 + d:
                kk = (1 - (t - t0) / d) ** 2
                light = dict(light)
                light["ambient"] = tuple(np.array(light["ambient"]) + np.array(col) * kk)

        dyn, dyn_t = [], []
        static = ["far", "cave"]
        static_t = ["far_t", "clouds"]
        gv = self.ground_version(t)
        gk = f"ground_{gv}"
        if gk not in self._r.static:
            o, tr = self._ground(gv).mesh()
            self._r.set_static(gk, o)
            self._r.set_static(gk + "_t", tr)
        static.append(gk)
        static_t.append(gk + "_t")
        pw, pkey, pops = self.props_world(t)
        kk = f"props_{pkey[0]}_{int(pkey[1])}"
        if kk not in self._r.static:
            o, tr = pw.mesh()
            self._r.set_static(kk, o)
            self._r.set_static(kk + "_t", tr)
        static.append(kk)
        static_t.append(kk + "_t")
        for (tp, p, b) in pops:
            s = ease_out_back(seg(t, tp, tp + 0.14))
            v = block_entity_vertices(b, np.array(p) + 0.5, max(s, 0.05))
            (dyn_t if tx.is_translucent(b) else dyn).append(v)
        # floor planks (thin) before the roof covers them
        if T(18, 0.3) <= t:
            n = int(49 * seg(t, T(18, 0.3), T(18, 0.75)))
            for i in range(n):
                x, z = i % 7 - 3, i // 7 - 3
                dyn.append(box_vertices((1, 0.06, 1), "planks", matrix=translate(x + 0.5, GY + 0.03, z + 0.5)))
        # ---- Pip
        pos, yaw, pose = self.pip_state(t)
        hurt = any(t0 <= t < t0 + 0.35 for t0 in (self.t_wallboom + 0.1, Wd(38, "clipped")))
        dyn.append(pip_vertices(pos, yaw, pose, flash=0.5 if hurt else 0.0))
        flash_col = (1, 0.15, 0.15) if hurt else (1, 1, 1)
        # ---- zombies & creepers
        for z in self.zombies:
            if t < z["spawn"] - 0.01:
                continue
            p = z["path"].pos(t)
            rise = smooth(seg(t, z["spawn"], z["spawn"] + 1.0))
            v = z["path"].vel(t)
            sp = float(np.hypot(v[0], v[2]))
            yawz = float(np.arctan2(v[0], v[2])) if sp > 0.2 else float(np.arctan2(-p[0], -p[2]))
            if t >= z["death"]:
                dt = t - z["death"]
                if dt > 1.0:
                    continue
                p = p + z["fly_v"] * dt + np.array([0, -9 * dt * dt, 0])
                m = translate(*p)
                vz = zombie_vertices((0, 0, 0), yawz, {"walk": 0})
                from mcengine.geometry import transform_vertices, rot_x, rot_z
                transform_vertices(vz, m @ rot_x(z["fly_spin"][0] * dt) @ rot_z(z["fly_spin"][2] * dt))
                dyn.append(vz)
                continue
            p = p.copy()
            p[1] = GY - 1.9 * (1 - rise)
            dyn.append(zombie_vertices(p, yawz, {"walk": clamp01(sp / 1.5) * 0.8, "walk_phase": z["path"].dist(t) * 2.5}))
        for c in self.creepers:
            if t < c["spawn"]:
                continue
            p = c["path"].pos(t)
            v = c["path"].vel(t)
            sp = float(np.hypot(v[0], v[2]))
            yawc = float(np.arctan2(v[0], v[2])) if sp > 0.2 else 0.0
            fl = 0.0
            if self.t_fuse <= t < self.t_boom:
                fl = 0.45 if ((t - self.t_fuse) * 5) % 1 < 0.5 else 0.0
            if t >= c["death"]:
                continue
            dyn.append(creeper_vertices(p, yawc, {"walk": clamp01(sp / 1.5) * 0.8,
                                                  "walk_phase": c["path"].dist(t) * 3, "swell": 0.3 if fl else 0},
                                        flash=fl))
        if night_preview:
            for i, (x, z, kind) in enumerate([(-3, 12, "z"), (2, 14, "c"), (5, 11, "z"), (-6, 15, "z")]):
                yy = np.arctan2(-x, -z)
                if kind == "z":
                    dyn.append(zombie_vertices((x + 0.5, GY, z + 0.5), yy, {"walk": 0.6, "walk_phase": t * 4 + i}))
                else:
                    dyn.append(creeper_vertices((x + 0.5, GY, z + 0.5), yy, {}))
        # the creeper at the wall
        if T(31, 0.4) <= t < self.t_wallboom:
            cp = self._creeper_wall_pos(t)
            u = seg(t, T(32, 0.3), self.t_wallboom)
            fl = 0.45 if (t > T(32, 0.3) and ((t - T(32, 0.3)) * (2 + 6 * u)) % 1 < 0.5) else 0.0
            dyn.append(creeper_vertices(cp, -PI / 2, {"swell": u ** 1.5, "walk": 0.6 if u <= 0 else 0,
                                                      "walk_phase": t * 8}, flash=fl))
        # survivors (burn at sunrise)
        for s in self.survivors:
            if not (s["spawn"] <= t < s["ignite"] + 1.8):
                continue
            p = s["path"].pos(t)
            fl = 0.35 if t >= s["ignite"] and (t * 10) % 1 < 0.5 else 0.0
            dyn.append(zombie_vertices(p, PI / 2 + 0.3, {"walk": 0.5, "walk_phase": t * 5}, flash=fl))
        # ---- items, TNT, torches, chest, crops
        for i, tt in enumerate(self.tnt_times):
            if tt <= t < self.t_boom + 0.11 * i:
                fl = 0.6 if (t > self.t_fuse + 0.4 and (t * 6) % 1 < 0.5) else 0.0
                s = ease_out_back(seg(t, tt, tt + 0.15))
                dyn.append(block_entity_vertices("tnt", (TNT_X[i] + 0.5, GY + 0.5 * max(s, 0.05), TNT_Z + 0.5),
                                                 max(s, 0.05), flash=fl))
        for (tt, base) in self.torches:
            if t >= tt and not (self.t_wallboom <= t and base[0] > 3 and abs(base[2] - 0.5) < 5 and t < T(50, 0.5)
                                and base[1] < 15 and False):
                s = ease_out_back(seg(t, tt, tt + 0.15))
                dyn.append(torch_vertices(base, 0.1 * np.sin(t * 13 + base[0])) if s > 0.99 else
                           torch_vertices(base, 0.0))
        ox, oy, oz = CAVE
        if cave or T(8, 0) <= t < T(17, 0):
            for (tt, base) in self.cave_torches:
                if t >= tt:
                    dyn.append(torch_vertices(base, 0.1 * np.sin(t * 11 + base[0])))
            lid = smooth(seg(t, T(13, 0.1), T(13, 0.55)))
            dyn.append(chest_vertices((ox + 14.5, oy + 0.5, oz + 0.5), -PI / 2, lid))
            if t < T(16, 0.2):
                dyn.append(block_entity_vertices("diamond_ore", (ox + 14.5, oy + 0.5, oz - 1.5), 1.0))
            # a loose stone in front of the coal; mining it reveals the ore, then the coal is mined too
            if t < Wd(10, "COAL") - 0.3:
                dyn.append(block_entity_vertices("stone", (ox + 4.5, oy + 1.5, oz - 3.5), 1.0))
            if t < Wd(10, "COAL") + 0.6:
                dyn.append(block_entity_vertices("coal_ore", (ox + 4.5, oy + 1.5, oz - 4.5), 1.0))
        for (tt, cp) in self.crops:
            if t >= tt:
                s = ease_out_back(seg(t, tt, tt + 0.2))
                dyn.append(box_vertices((0.8 * s, 0.7 * s, 0.8 * s), "crop", matrix=translate(*cp),
                                        offset=(0, 0.35 * s, 0)))
        dyn += self._flying_items(t)
        # ---- particles
        parts = gather(self.bursts, t, False)
        adds = gather(self.bursts, t, True)
        extra = self._torch_glow(t, cave)
        if extra is not None:
            adds = extra if adds is None else np.concatenate([adds, extra])
        wf = 0.0
        for t0, a, d in ((self.t_wallboom, 0.6, 0.3), (self.t_boom, 1.0, 0.4)):
            if t0 <= t < t0 + d:
                wf = max(wf, a * (1 - (t - t0) / d))
        post["white_flash"] = wf
        return dict(camera=cam, sky=sky, light=light, fog=fog, static=static, static_trans=static_t,
                    dynamic=np.concatenate(dyn) if dyn else None,
                    dynamic_trans=np.concatenate(dyn_t) if dyn_t else None,
                    particles=parts, particles_add=adds, flash_col=flash_col, post=post)

    def _torch_glow(self, t, cave):
        pts = []
        src = self.cave_torches if cave else (self.torches if self.phase(t) > 0.8 else [])
        for (tt, b) in src:
            if t >= tt:
                pts.append([b[0], b[1] + 0.72, b[2], 0.55 + 0.08 * np.sin(t * 9 + b[0]), 1.0, 0.75, 0.35, 0.22])
        return np.array(pts, np.float32) if pts else None

    def _flying_items(self, t):
        T, Wd = self.T, self.Wd
        out = []
        pip, _, _ = self.pip_state(t)
        hand = pip + np.array([0, 1.0, 0])

        def fly(kind, t0, p0, dur=0.7, s=1.0):
            if t0 <= t < t0 + dur:
                u = seg(t, t0, t0 + dur)
                p = lerp(p0, hand, ease_in(u, 2)) + np.array([0, 0.8 * np.sin(PI * u), 0])
                out.append(item_vertices(kind, p, spin=t * 6, s=s))

        for k, tb in enumerate(self.log_breaks):
            fly("log", tb, np.array([TREE[0] + 0.5, 11.5 + k, TREE[1] + 0.5]))
        for k, tm in enumerate(self.mine_times):
            p = BOULDER[[2, 0, 1][k]]
            fly("cobble", tm + 0.2, np.array([p[0] + 0.5, p[1] + 0.5, p[2] + 0.5]))
        ox, oy, oz = CAVE
        fly("coal", Wd(10, "COAL"), np.array([ox + 4.5, oy + 1.5, oz - 3.6]))
        # TNT bursting out of the chest
        tt = Wd(13, "TNT")
        for i in range(6):
            t0 = tt + 0.08 * i
            if t0 <= t < T(14, 0.5) + 0.1 * i:
                u = seg(t, t0, t0 + 0.6)
                a = -0.8 + 0.32 * i
                p0 = np.array([ox + 14.5, oy + 1.0, oz + 0.5])
                p1 = p0 + np.array([-1.3, 1.2, np.sin(a) * 1.8])
                p = lerp(p0, p1, ease_out(u)) + np.array([0, 0.1 * np.sin(t * 3 + i), 0])
                if t > T(14, 0.3):
                    p = lerp(p, hand, ease_in(seg(t, T(14, 0.3) + 0.1 * i, T(14, 0.5) + 0.1 * i), 2))
                out.append(item_vertices("tnt", p, spin=t * 2 + i, s=1.2))
        fly("diamond", T(16, 0.2), np.array([ox + 14.5, oy + 0.6, oz - 1.5]), 0.5, 1.4)
        if T(52, 0.0) <= t < T(53, 0.0):
            p = pip + np.array([0.0, 1.9 + 0.08 * np.sin(t * 3), 0.6])
            out.append(item_vertices("diamond", p, spin=t * 2.5, s=1.6))
        return out

    # ------------------------------------------------------------ overlay
    def draw_overlay(self, t, layer):
        T, Wd = self.T, self.Wd
        tp = ov.title_pop
        TOP = (W / 2, 150)
        # hurt vignette
        for t0 in (self.t_wallboom + 0.1, Wd(38, "clipped")):
            if t0 <= t < t0 + 0.9:
                k = 1 - seg(t, t0, t0 + 0.9)
                a = self._red.copy()
                a[..., 3] = (a[..., 3] * k * 0.45).astype(np.uint8)
                layer.alpha_composite(Image.fromarray(a))
        # HUD: hearts + hotbar
        hp = 10.0
        if t >= self.t_wallboom + 0.15:
            hp = 5.0
        if t >= Wd(38, "clipped") + 0.1:
            hp = 0.5
        shk = 1.0 if any(t0 <= t < t0 + 0.7 for t0 in (self.t_wallboom + 0.15, Wd(38, "clipped") + 0.1)) else 0.0
        if t >= 1.0:
            ov.draw_hearts(layer, hp, (300, 40), px=5, shake_amt=shk, t=t)
        items, sel = self._hotbar(t)
        if items is not None:
            ov.draw_hotbar(layer, items, sel, (W / 2, H - 70), slot=72)
        # countdown clock
        if 0.5 <= t < T(25, 0.9):
            secs = float(np.interp(t, [0.2, self.T(8, 0.0), self.T(17, 0.0), self.T(24, 0.0), self.T(24, 0.9)],
                                   [600, 540, 420, 75, 0]))
            ov.countdown(layer, secs, (W - 170, 70), size=60)
        # plan checklist
        if T(3, 0.3) <= t < T(24, 0.0):
            self._checklist(layer, t)
        # chapter cards
        for n in range(2, 7):
            c0, c1 = self.card(n)
            ov.chapter_card(layer, t, c0 - 0.25, (c1 - c0) + 0.5, n, self.chapters[n - 1]["title"], W, H)
        # titles
        tp(layer, "10 MINUTES", t, Wd(0, "TEN"), T(1, 0.0), TOP, 120, fill=(255, 225, 40), emoji="☀️", emoji_side="right")
        tp(layer, "ZOMBIES", t, Wd(2, "zombies"), Wd(2, "skeletons"), TOP, 110, fill=(130, 220, 110), emoji="🧟", emoji_side="right")
        tp(layer, "SKELETONS", t, Wd(2, "skeletons"), Wd(2, "creepers"), TOP, 110, fill=(230, 230, 230), emoji="💀", emoji_side="right")
        tp(layer, "CREEPERS", t, Wd(2, "creepers"), T(3, 0.0), TOP, 110, fill=(110, 235, 90), emoji="💥", emoji_side="right")
        tp(layer, "FUN FACT", t, T(5, 0.0), T(5, 0.7), TOP, 100, fill=(255, 225, 40), emoji="💡", emoji_side="right")
        if T(5, 0.75) <= t < T(6, 0.0):
            ov.paste_emoji(layer, "🤔", TOP, 130)
        self._crafting(layer, t)
        tp(layer, "FIRST PICKAXE!", t, T(6, 1.0), T(7, 0.0) - 0.1, TOP, 110, fill=(255, 225, 40), emoji="⛏️", emoji_side="right")
        tp(layer, "COAL!", t, Wd(10, "COAL"), T(11, 0.0), TOP, 120, fill=(220, 220, 220), emoji="🪨", emoji_side="right")
        tp(layer, "6× TNT", t, Wd(13, "TNT"), T(14, 0.6), TOP, 120, fill=(255, 80, 60), emoji="🧨", emoji_side="right")
        tp(layer, "DIAMOND!", t, Wd(15, "DIAMOND"), T(16, 0.6), TOP, 120, fill=(110, 235, 255), emoji="💎", emoji_side="right")
        if T(19, 0.1) <= t < T(20, 0.0):
            self._height_marker(layer, t)
        tp(layer, "BASE COMPLETE", t, Wd(23, "DONE"), T(24, 0.0) - 2.6, TOP, 110, fill=(120, 240, 120), emoji="✅", emoji_side="right")
        tp(layer, "NIGHT 1", t, Wd(25, "dark"), T(26, 0.5), TOP, 120, fill=(170, 190, 255), emoji="🌙", emoji_side="right")
        tp(layer, "Sssssss...", t, Wd(31, "Ssssss") if any(norm(w["word"]) == "ssssss" for w in self.words) else T(31, 0.8),
           T(32, 0.2), TOP, 120, fill=(110, 235, 90))
        tp(layer, "BOOM!", t, self.t_wallboom, T(34, 0.0), TOP, 150, fill=(255, 150, 30), emoji="💥", emoji_side="right")
        tp(layer, "½ HEART", t, Wd(38, "clipped") + 0.1, T(39, 0.0), TOP, 110, fill=(255, 80, 70), emoji="❤️", emoji_side="right")
        for i, tt in enumerate(self.tnt_times):
            end = self.tnt_times[i + 1] if i < 5 else T(40, 0.0)
            tp(layer, str(i + 1), t, tt, end, (W / 2, 300), 170, fill=(255, 80, 60))
        tp(layer, "KABOOM!", t, self.t_boom, T(42, 0.0), TOP, 170, fill=(255, 150, 30), emoji="💥", emoji_side="right")
        tp(layer, "6 TNT = 0 MONSTERS", t, T(44, 0.0), T(44, 1.0), TOP, 100, fill=(255, 225, 40), emoji="🏆", emoji_side="right")
        tp(layer, "ZOMBIES BURN IN DAYLIGHT", t, Wd(46, "burn"), T(47, 0.0), TOP, 90, fill=(255, 160, 60), emoji="🔥", emoji_side="right")
        tp(layer, "CREEPERS DON'T BURN", t, T(47, 0.0), T(48, 0.0), TOP, 90, fill=(110, 235, 90), emoji="💡", emoji_side="right")
        tp(layer, "SURVIVED NIGHT 1", t, Wd(48, "SURVIVED"), T(49, 0.0), TOP, 110, fill=(120, 240, 120), emoji="✅", emoji_side="right")
        if t >= T(53, 0.0):
            a = ease_out_back(seg(t, T(53, 0.0), T(53, 0.0) + 0.35))
            ov.draw_text(layer, "WHAT SHOULD I BUILD WITH IT?", (W / 2, 330), 96, fill=(110, 235, 255),
                         scale=max(a, 0.05))
            ov.paste_emoji(layer, "💎", (W / 2, 190), 140 * max(a, 0.05))
            b = smooth(seg(t, T(53, 0.4), T(53, 0.8)))
            if b > 0:
                ov.draw_text(layer, "COMMENT BELOW  •  DAY 2 SOON", (W / 2, 450), 60, alpha=b)
                ov.paste_emoji(layer, "👇", (W / 2, 560 + 10 * np.sin(t * 7)), 90, alpha=b)
        # subtitles
        self.subs.draw(layer, t)

    def _hotbar(self, t):
        T, Wd = self.T, self.Wd
        if t < T(1, 0.3):
            return None, 0
        items = ["log" if t >= self.log_breaks[0] else None,
                 "planks" if t >= Wd(6, "sixteen") else None,
                 "e:⛏️" if t >= T(6, 0.8) else None,
                 "cobble" if t >= self.mine_times[0] + 0.3 else None,
                 "coal_ore" if t >= Wd(10, "COAL") + 0.3 else None,
                 "e:🕯️" if t >= T(11, 0.1) else None,
                 "tnt" if t >= T(14, 0.5) else None,
                 "e:💎" if t >= T(16, 0.3) else None]
        sel = 0
        if t >= T(6, 0.8):
            sel = 2
        if T(11, 0.0) <= t < T(12, 0.5):
            sel = 5
        if T(17, 0.0) <= t < T(24, 0.0) or T(50, 0.0) <= t < T(51, 0.0):
            sel = 3
        if T(35, 0.3) <= t < self.t_boom:
            sel = 6
        if t >= T(52, 0.0):
            sel = 7
        return items, sel

    def _checklist(self, layer, t):
        T, Wd = self.T, self.Wd
        rows = [("WOOD", T(6, 1.0)), ("STONE", self.mine_times[-1] + 0.3), ("BASE", Wd(23, "DONE"))]
        a = smooth(seg(t, T(3, 0.3), T(3, 0.5)))
        d = ImageDraw.Draw(layer)
        x0, y0 = 40, 110
        d.rounded_rectangle([x0, y0, x0 + 300, y0 + 60 + 58 * 3], radius=16, fill=(0, 0, 0, int(140 * a)))
        ov.draw_text(layer, "THE PLAN", (x0 + 150, y0 + 32), 40, fill=(255, 225, 40), alpha=a, shadow=False)
        for i, (name, td) in enumerate(rows):
            y = y0 + 90 + 58 * i
            done = t >= td
            ov.paste_emoji(layer, "✅" if done else "⬜", (x0 + 45, y), 40, alpha=a)
            ov.draw_text(layer, name, (x0 + 170, y), 42, fill=(150, 255, 150) if done else (255, 255, 255),
                         alpha=a, shadow=False)

    def _crafting(self, layer, t):
        T, Wd = self.T, self.Wd
        i = 6
        steps = [(Wd(i, "Four"), ["log", None, None, None, None, None, None, None, None], "planks"),
                 (Wd(i, "crafting"), ["planks", "planks", None, "planks", "planks", None, None, None, None], "planks"),
                 (Wd(i, "sticks"), [None, "planks", None, None, "planks", None, None, None, None], "e:🥢"),
                 (Wd(i, "pickaxe"), ["planks", "planks", "planks", None, "e:🥢", None, None, "e:🥢", None], "e:⛏️")]
        t_end = T(6, 1.0) + 0.1
        if not (steps[0][0] <= t < t_end):
            return
        cur = max([s for s in steps if s[0] <= t], key=lambda s: s[0])
        appear = seg(t, steps[0][0], steps[0][0] + 0.35) * (1 - seg(t, t_end - 0.25, t_end))
        ov.crafting_grid(layer, (W * 0.66, H / 2 - 40), cur[1], cur[2], appear=appear, cell=100)
        labels = {0: "16 PLANKS", 1: "CRAFTING TABLE", 2: "STICKS", 3: "PICKAXE!"}
        idx = steps.index(cur)
        ov.draw_text(layer, labels[idx], (W * 0.66, H / 2 + 150), 64, fill=(255, 225, 40), alpha=appear)

    def _height_marker(self, layer, t):
        """'2 BLOCKS TALL' measuring bracket next to the wall (screen space, approximate)."""
        a = smooth(seg(t, self.T(19, 0.1), self.T(19, 0.3)))
        d = ImageDraw.Draw(layer)
        x, y0, y1 = W * 0.62, H * 0.30, H * 0.66
        col = (255, 225, 40, int(255 * a))
        d.line([(x, y0), (x, y1)], fill=col, width=8)
        d.line([(x - 30, y0), (x + 30, y0)], fill=col, width=8)
        d.line([(x - 30, y1), (x + 30, y1)], fill=col, width=8)
        ov.draw_text(layer, "2 BLOCKS", (x + 190, (y0 + y1) / 2), 80, fill=(255, 225, 40), alpha=a)

    # ------------------------------------------------------------ audio
    def sfx_events(self):
        T, Wd = self.T, self.Wd
        ev = [(0.0, "whoosh", -10, dict(dur=0.8)), (Wd(0, "TEN"), "pop", -12, {})]
        for n in range(2, 7):
            c0, c1 = self.card(n)
            ev += [(c0 - 0.3, "whoosh", -8, dict(dur=0.7)), (c0 + 0.2, "impact", -10, {}),
                   (c1 - 0.4, "whoosh", -11, dict(dur=0.5, up=False))]
        for tp in self.punch_times:
            ev.append((tp + 0.1, "dig", -9, dict(seed=int(tp * 10))))
        for tb in self.log_breaks:
            ev.append((tb, "pop", -12, dict(pitch=0.9)))
        for s in [Wd(6, "Four"), Wd(6, "crafting"), Wd(6, "sticks"), Wd(6, "pickaxe")]:
            ev.append((s, "pop", -10, dict(pitch=1.2)))
        ev.append((T(6, 1.0), "ding", -10, {}))
        for tm in self.mine_times:
            ev.append((tm + 0.1, "dig", -8, dict(seed=int(tm * 10))))
        for f in (0.2, 0.4, 0.6, 0.8):
            ev.append((T(10, f) + 0.1, "dig", -8, dict(seed=int(f * 10))))
        ev += [(Wd(10, "COAL"), "ding", -12, dict(base=660)), (Wd(13, "TNT"), "impact", -8, {}),
               (T(13, 0.1), "creak", -8, {}), (Wd(15, "DIAMOND"), "shimmer", -6, {}),
               (T(16, 0.2), "pop", -9, dict(pitch=1.4)), (Wd(23, "DONE"), "ding", -8, {}),
               (T(22, 0.45), "water", -10, dict(dur=2.6))]
        for i, (tt, _) in enumerate(self.torches + self.cave_torches):
            ev.append((tt, "pop", -14, dict(pitch=0.8)))
        for i, (tp, p, b) in enumerate(self.build_events + [(T(50, 0) + 0.04 * j, p, b)
                                                            for j, (p, b) in enumerate(self.repair_events)]):
            if i % 3 == 0:
                ev.append((tp, "place", -15, dict(seed=i)))
        ev += [(Wd(25, "dark"), "shimmer", -14, {})]
        for z in self.zombies:
            ev.append((z["spawn"], "dig", -12, dict(seed=z["id"])))
            ev.append((z["spawn"] + 0.4, "groan", -10, dict(seed=z["id"])))
        for k in range(6):
            ev.append((T(28, 0.0) + 1.3 * k, "groan", -13, dict(seed=10 + k)))
        ev += [(T(31, 0.55), "hiss", -6, dict(dur=self.t_wallboom - T(31, 0.55))),
               (self.t_wallboom, "explosion", -2, {}), (self.t_wallboom + 0.15, "hurt", -9, {})]
        for k in range(4):
            ev.append((T(34, 0.1) + 0.9 * k, "groan", -11, dict(seed=20 + k)))
        ev += [(self.t_jump, "whoosh", -9, dict(dur=0.5, up=False)), (self.t_jump + 0.5, "place", -10, {})]
        tt = self.t_jump + 0.6
        while tt < self.tnt_times[0]:
            ev.append((tt, "step", -14, dict(seed=int(tt * 7))))
            tt += 0.28
        ev += [(Wd(38, "clipped") + 0.1, "hurt", -7, {}), (Wd(38, "clipped") + 0.3, "heartbeat", -6, {})]
        for i, t0 in enumerate(self.tnt_times):
            ev.append((t0, "place", -9, dict(seed=50 + i)))
        ev += [(self.t_fuse, "hiss", -9, dict(dur=self.t_boom - self.t_fuse))]
        for i in range(6):
            ev.append((self.t_boom + 0.11 * i, "explosion", -1 if i == 0 else -6, dict(seed=i)))
        for c in self.creepers:
            ev.append((self.t_boom + 0.35 + 0.1 * c["id"], "explosion", -7, dict(seed=10 + c["id"])))
        ev.append((T(44, 0.0), "ding", -10, {}))
        for s in self.survivors:
            ev += [(s["spawn"] + 1.0, "groan", -13, dict(seed=30 + s["id"])), (s["ignite"], "crackle", -8, {}),
                   (s["ignite"] + 1.8, "pop", -10, dict(pitch=0.7))]
        ev += [(Wd(48, "SURVIVED"), "ding", -7, {}), (T(52, 0.0), "shimmer", -9, {}), (T(53, 0.0), "pop", -10, {})]
        return ev

    def music_sections(self):
        T = self.T
        c = [self.card(n) for n in range(2, 7)]
        return [("adventure", 0.0, c[0][0]), ("suspense", c[0][0], c[1][0]), ("build", c[1][0], c[2][0]),
                ("suspense", c[2][0], T(31, 0.4)), ("silence", T(31, 0.4), T(32, 0.2)),
                ("panic", T(32, 0.2), c[3][0]), ("freefall", c[3][0], self.t_fuse),
                ("silence", self.t_fuse, self.t_boom + 0.4), ("suspense", self.t_boom + 2.5, c[4][0]),
                ("resolve", c[4][0], self.duration)]
