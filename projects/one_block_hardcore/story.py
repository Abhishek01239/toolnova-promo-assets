"""'ONE BLOCK in the sky — HARDCORE' : scene direction for the first Short.

Every event (cuts, explosions, title pops, SFX) is anchored to a word in the
narration via `cue()`, so picture and voice stay in sync whatever the take.
"""
import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mcengine import textures as tx                                   # noqa: E402
from mcengine.geometry import VoxelWorld, box_vertices, translate, VERT  # noqa: E402
from mcengine.models import pip_vertices, creeper_vertices, block_entity_vertices  # noqa: E402
from mcengine.anim import (Track, seg, smooth, ease_out, ease_in, ease_out_back, ease_in_out,  # noqa: E402
                           shake, orbit, Burst, gather, lerp, clamp01, noise1)
from mcengine.lighting import environment                              # noqa: E402
from mcengine import overlay as ov                                     # noqa: E402
from mcengine.voice import cue as _cue                                 # noqa: E402

TITLE = "ONE BLOCK. HARDCORE."

LINES = [
    "[excited] I spawned on ONE block... two hundred blocks up in the sky. In HARDCORE.",
    "One life. No ground. Just me, a grass block, and a very... long... fall.",
    "[curious] But this block was special. Every time I broke it, it gave me something new.",
    "Wood. Stone. Water. Little by little, I built a home in the clouds.",
    "[suspenseful, quieter] Then, on night three, the block gave me something I did NOT expect...",
    "[whispering] A Creeper.",
    "[panicked] No sword. No space. Nowhere to run.",
    "[determined] So I grabbed my water bucket... and I jumped.",
    "[shouting] My island EXPLODED above me! Two hundred blocks. Falling. Falling!",
    "[intense] One second left. Water. DOWN!",
    "[relieved, laughing] I survived. Hardcore. Zero hearts to spare.",
    "[excited] But my home is gone... so tomorrow, I climb back up. Think I can make it?",
]

KEYWORDS = {
    "HARDCORE": (255, 70, 60), "CREEPER": (110, 235, 90), "WATER": (90, 180, 255),
    "EXPLODED": (255, 150, 30), "DOWN": (90, 180, 255), "SURVIVED": (120, 240, 120),
    "ONE": (255, 221, 0), "FALL": (255, 150, 30),
}

P0 = np.array([0.5, 200.0, 0.5], np.float32)          # Pip's spawn (top of the one block)
CENTER = (0, 199, 0)
LAND = (5, 7, 0)                                      # terrain block Pip lands next to (top y=8)
CLOUD_Y = 150
PI = np.pi


def tile_color(name):
    a = np.asarray(tx.tile_image(name, 16)).reshape(-1, 4)
    a = a[a[:, 3] > 0]
    return tuple((a[:, :3].mean(0) / 255).tolist())


class OneBlockHardcore:
    def __init__(self, words, cache_dir):
        self.words = words
        c = lambda w, k=0, at="start": _cue(words, w, k, at)
        self.c = c
        T = self.T = {}
        T["one"] = c("ONE"); T["two0"] = c("two", 0); T["hard"] = c("HARDCORE.")
        T["life"] = c("One", 1); T["ground"] = c("ground.") - 0.12; T["just"] = c("Just")
        T["very"] = c("very..."); T["fall"] = c("fall.")
        T["but"] = c("But", 0); T["special"] = c("special."); T["every"] = c("Every")
        T["new"] = c("new.")
        T["wood"] = c("Wood."); T["stone"] = c("Stone."); T["water"] = c("Water.", 0)
        T["little"] = c("Little"); T["clouds_end"] = c("clouds.", 0, "end")
        T["then"] = c("Then,"); T["night"] = c("night"); T["theblock"] = c("block", 3)
        T["not"] = c("NOT"); T["creeper"] = c("Creeper.")
        T["sword"] = c("No", 1); T["space"] = c("No", 2); T["nowhere"] = c("Nowhere")
        T["so"] = c("So"); T["grabbed"] = c("grabbed"); T["bucket"] = c("water", 1)
        T["andi"] = c("and", 1); T["jumped"] = c("jumped.")
        T["island"] = c("island") - 0.1; T["exploded"] = c("EXPLODED")
        T["two1"] = c("two", 1); T["falling0"] = c("Falling.", 0); T["falling1"] = c("Falling!", 0)
        T["second"] = c("One", 2); T["water2"] = c("water", 2); T["down"] = c("DOWN!")
        T["survived"] = c("survived."); T["hard2"] = c("Hardcore.", 0); T["zero"] = c("Zero")
        T["but2"] = c("But", 1); T["gone"] = c("gone..."); T["tomorrow"] = c("tomorrow,")
        T["think"] = c("Think")
        T["end"] = words[-1]["end"]
        self.duration = T["end"] + 1.5
        T["jump"] = T["jumped"] + 0.02
        T["fuse"] = T["grabbed"] + 0.25
        T["land"] = T["down"] + 0.14

        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._plan()
        self._center_timeline()
        self._effects()
        self.subs = ov.Subtitles(words, KEYWORDS, max_words=3, max_chars=15, y=1180, size=104)
        self._red_vig = self._make_red_vignette()

    # ================================================================ world
    def build_static(self, r):
        """Create cached meshes on renderer r (terrain, clouds)."""
        tp = os.path.join(self.cache_dir, "terrain.npz")
        if os.path.exists(tp):
            d = np.load(tp)
            o, t = d["o"], d["t"]
        else:
            o, t = self._terrain().mesh()
            np.savez_compressed(tp, o=o, t=t)
        o = np.concatenate([o, self._far_ground()])
        r.set_static("terrain", o)
        r.set_static("terrain_t", t)
        cw = VoxelWorld()
        rng = np.random.default_rng(42)
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(rng.normal(size=(29, 29)), 1.3)
        field /= field.std()
        for gx in range(-14, 15):
            for gz in range(-14, 15):
                v = field[gx + 14, gz + 14]
                if v > 0.45 and not (abs(gx) <= 3 and abs(gz) <= 3):
                    cw.set((gx, 0, gz), "cloud")
        _, cv = cw.mesh(alpha_override=0.82)
        cv[:, 0:3] = cv[:, 0:3] * np.array([10, 2, 10], np.float32) + np.array([0, CLOUD_Y, 0], np.float32)
        r.set_static("clouds", cv)

    def _far_ground(self, R=56, extent=560, q=8):
        """Low-detail ground ring (8x8-block quads) out to the horizon, so the
        terrain never shows an edge when seen from 200 blocks up."""
        from mcengine.geometry import _face_uv, FACE_CORNERS, FACE_NORMALS, TRI_ORDER
        rng = np.random.default_rng(5)
        quads = []
        uvg = _face_uv("grass_top")
        for x in range(-extent, extent, q):
            for z in range(-extent, extent, q):
                if -R <= x < R and -R <= z < R:
                    continue
                y = 6.0 + (1.0 if (np.sin(x * 0.013) * np.cos(z * 0.017) > 0.3) else 0.0)
                shade = rng.uniform(0.82, 1.0)
                c = FACE_CORNERS[2] * np.array([q, 0, q], np.float32) + np.array([x, y, z], np.float32)
                v = np.zeros((6, 12), np.float32)
                for k, ci in enumerate(TRI_ORDER):
                    v[k, 0:3] = c[ci]; v[k, 3:5] = uvg[ci]; v[k, 5:8] = FACE_NORMALS[2]
                    v[k, 8] = shade; v[k, 10] = 1.0
                quads.append(v)
        return np.concatenate(quads)

    def _terrain(self):
        w = VoxelWorld()
        R = 56
        xs = np.arange(-R, R)
        H = {}
        for x in xs:
            for z in xs:
                h = (6 + 2.6 * np.sin(x * 0.13 + 0.4) * np.cos(z * 0.11) + 1.8 * np.sin((x + z) * 0.07)
                     + 7 * max(0.0, np.sin(x * 0.045 + 2.0) * np.cos(z * 0.05 + 0.3)) ** 2)
                dl = np.hypot(x - LAND[0], z - LAND[2])
                if dl < 20:
                    k = smooth((dl - 10) / 10)
                    h = h * k + 7 * (1 - k)
                H[(x, z)] = int(round(h))
        for (x, z), h in H.items():
            nmin = min(H.get((x + dx, z + dz), h) for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            for y in range(min(nmin, h) - 1, h + 1):
                name = "grass" if y == h else ("dirt" if y >= h - 2 else "stone")
                if h <= 3 and y == h:
                    name = "sand"
                w.set((x, y, z), name)
            for y in range(h + 1, 4):                 # lakes
                w.set((x, y, z), "water")
        rng = np.random.default_rng(7)
        for _ in range(60):
            x, z = rng.integers(-R + 3, R - 3, 2)
            if np.hypot(x - LAND[0], z - LAND[2]) < 14 or H[(x, z)] <= 3:
                continue
            h = H[(x, z)]
            th = rng.integers(4, 6)
            for y in range(h + 1, h + 1 + th):
                w.set((x, y, z), "log")
            top = h + th
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    for dy in range(-1, 2):
                        if abs(dx) + abs(dz) + abs(dy) <= 3 and (dx, dz) != (0, 0) or dy == 1:
                            if w.get((x + dx, top + dy, z + dz)) is None:
                                w.set((x + dx, top + dy, z + dz), "leaves")
        return w

    def _plan(self):
        """The 'home in the clouds' build order (time-lapse)."""
        plan = []
        ring = lambda r: [(x, z) for x in range(-r, r + 1) for z in range(-r, r + 1)
                          if max(abs(x), abs(z)) == r]
        for (x, z) in ring(1):
            plan.append(((x, 199, z), "grass"))
        for x in range(-1, 2):
            for z in range(-1, 2):
                plan.append(((x, 198, z), "dirt"))
        for (x, z) in ring(2):
            plan.append(((x, 199, z), "grass"))
        for (x, z) in ring(2):
            plan.append(((x, 198, z), "dirt"))
        for x in range(-1, 2):
            for z in range(-1, 2):
                plan.append(((x, 197, z), "stone"))
        for (x, z) in ring(3):
            plan.append(((x, 199, z), "grass"))
        plan.append(((0, 196, 0), "cobble"))
        for y in (200, 201, 202):
            plan.append(((-2, y, -2), "log"))
        for y in (202, 203):
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if y == 202 and dx == 0 and dz == 0:
                        continue
                    plan.append(((-2 + dx, y, -2 + dz), "leaves"))
        plan.append(((-2, 204, -2), "leaves"))
        for y in (200, 201):
            for x in (-3, -2, -1):
                for z in (1, 2, 3):
                    if (x, z) == (-2, 2) or (x, z) == (-1, 2):
                        continue
                    name = "glass" if (y == 201 and (x, z) in ((-2, 3), (-3, 2))) else "planks"
                    plan.append(((x, y, z), name))
        for x in (-3, -2, -1):
            for z in (1, 2, 3):
                plan.append(((x, 202, z), "planks"))
        plan += [((1, 199, -3), "water"), ((2, 199, -3), "water")]
        T = self.T
        t0, t1 = T["little"] + 0.05, T["clouds_end"] - 0.15
        self.plan = [(t0 + (t1 - t0) * i / len(plan), p, b) for i, (p, b) in enumerate(plan)]
        self._prefix = {}
        # blocks blown away by the creeper
        self.E = np.array([2.6, 200.9, 0.5], np.float32)
        final = self._world_at_k(len(self.plan))
        self.removed = {p: b for p, b in final.blocks.items()
                        if np.linalg.norm(np.array(p) + 0.5 - self.E) < 3.4}
        rng = np.random.default_rng(11)
        self.debris = []
        for p, b in list(self.removed.items()) + [(CENTER, "grass")]:
            c = np.array(p, np.float32) + 0.5
            d = c - self.E + rng.normal(size=3) * 0.4
            d /= np.linalg.norm(d) + 1e-6
            v = d * rng.uniform(6, 14) + np.array([0, rng.uniform(3, 8), 0])
            self.debris.append((b, c, v, rng.uniform(-6, 6, 3), rng.uniform(0.55, 1.0)))
        for _ in range(26):          # small chunks
            b = rng.choice(["dirt", "grass", "stone", "planks"])
            d = rng.normal(size=3); d /= np.linalg.norm(d)
            self.debris.append((b, self.E.copy(), d * rng.uniform(8, 20) + np.array([0, 5, 0]),
                                rng.uniform(-9, 9, 3), rng.uniform(0.2, 0.35)))

    def _world_at_k(self, k):
        if k in self._prefix:
            return self._prefix[k]
        base_k = max([kk for kk in self._prefix if kk <= k], default=0)
        w = self._prefix[base_k].copy() if base_k in self._prefix else VoxelWorld()
        for _, p, b in self.plan[base_k:k]:
            w.set(p, b)
        self._prefix[k] = w
        return w

    def _center_timeline(self):
        T = self.T
        ev = [(0.0, "grass")]
        tb = T["new"] - 0.85
        cyc = ["sand", "planks", "cobble", "tnt", "obsidian", "lucky", "grass"]
        for i, b in enumerate(cyc):
            ev.append((tb + i * (T["wood"] - 0.1 - tb) / len(cyc), b))
        ev += [(T["wood"], "log"), (T["stone"], "stone"), (T["water"], "grass")]
        self.night_break = T["theblock"] + 0.55
        ev.append((self.night_break, "grass"))
        self.center_events = ev
        # punch times (each a 0.22s swing)
        self.punches = [T["every"] + 0.15, T["every"] + 0.42, tb - 0.12,
                        T["wood"] - 0.18, T["stone"] - 0.18, T["water"] - 0.18,
                        T["theblock"] - 0.05, T["theblock"] + 0.3]
        self.breaks = [(tb, "grass"), (T["wood"], "lucky"), (T["stone"], "log"), (T["water"], "stone"),
                       (self.night_break, "grass")]

    def center_type(self, t):
        cur, since = "grass", 0.0
        for te, b in self.center_events:
            if t >= te:
                cur, since = b, te
        return cur, since

    # ================================================================ effects
    def _effects(self):
        T = self.T
        B = self.bursts = []
        top = (0.5, 200.05, 0.5)
        for i, (tb, prev) in enumerate(self.breaks):
            col = tile_color(tx.BLOCKS[prev][0]) + (1.0,)
            col2 = tile_color(tx.BLOCKS[prev][1]) + (1.0,)
            B.append(Burst(tb, top, 50, speed=(1.5, 4.5), life=(0.5, 1.0), size=(0.07, 0.14),
                           colors=[col, col2], gravity=-14, drag=1.0, up_bias=1.2, seed=100 + i, box=0.4))
        # lucky shimmer
        B.append(Burst(T["special"], (0.5, 199.8, 0.5), 90, speed=(0.6, 2.2), life=(0.8, 1.6),
                       size=(0.05, 0.11), colors=[(1, 0.9, 0.3, 1), (1, 1, 0.7, 1)], gravity=1.5,
                       drag=1.2, box=0.6, seed=3, additive=True))
        # sparkle on each regenerate during the cycle
        for i, (te, b) in enumerate(self.center_events[1:]):
            B.append(Burst(te, (0.5, 199.9, 0.5), 18, speed=(1, 3), life=(0.3, 0.6), size=(0.05, 0.09),
                           colors=[(1, 1, 0.8, 1)], gravity=0, drag=3, seed=200 + i, additive=True, box=0.5))
        # ominous green fumes at night
        t = T["theblock"] + 0.6
        k = 0
        while t < T["creeper"]:
            B.append(Burst(t, (0.5, 200.0, 0.5), 10, speed=(0.3, 1.0), life=(0.9, 1.6), size=(0.08, 0.16),
                           colors=[(0.2, 0.8, 0.25, 0.9), (0.1, 0.5, 0.15, 0.9)], gravity=1.6, drag=1.5,
                           box=0.45, seed=300 + k, additive=True))
            t += 0.18; k += 1
        # creeper spawn burst
        B.append(Burst(T["creeper"] - 0.05, (0.5, 200.1, 0.5), 70, speed=(2, 6), life=(0.5, 1.1),
                       size=(0.08, 0.16), colors=[(0.3, 0.75, 0.3, 1), (0.5, 0.35, 0.25, 1)], gravity=-12,
                       up_bias=1.0, seed=8, box=0.4))
        B.append(Burst(T["creeper"] - 0.05, (0.5, 200.4, 0.5), 40, speed=(1, 3), life=(0.8, 1.4),
                       size=(0.25, 0.5), colors=[(0.5, 0.5, 0.5, 0.6)], gravity=1.0, grow=1.5, seed=9, box=0.3))
        # explosion
        te = T["exploded"]
        E = tuple(self.E)
        B.append(Burst(te, E, 260, speed=(3, 16), life=(0.35, 1.1), size=(0.5, 1.4),
                       colors=[(1, 0.95, 0.6, 1), (1, 0.65, 0.15, 1), (1, 0.35, 0.05, 1)],
                       gravity=3, drag=3.2, grow=1.2, seed=20, additive=True))
        B.append(Burst(te, E, 200, speed=(2, 9), life=(1.6, 3.6), size=(0.8, 2.0),
                       colors=[(0.25, 0.24, 0.24, 0.75), (0.4, 0.38, 0.36, 0.7), (0.12, 0.12, 0.12, 0.8)],
                       gravity=1.8, drag=1.8, grow=2.4, seed=21))
        B.append(Burst(te, E, 180, speed=(10, 26), life=(0.6, 1.5), size=(0.08, 0.16),
                       colors=[(1, 0.9, 0.4, 1)], gravity=-16, drag=0.8, seed=22, additive=True))
        # lingering wreck smoke (seen from the ground at the end)
        t, k = te + 0.6, 0
        while t < self.duration:
            B.append(Burst(t, (1.0, 199.5, 0.5), 5, speed=(0.5, 2), life=(4, 6), size=(2.5, 4.5),
                           colors=[(0.3, 0.3, 0.32, 0.55)], gravity=2.0, drag=0.4, grow=1.6, seed=400 + k,
                           box=2.5))
            B.append(Burst(t, (1.0, 199.5, 0.5), 2, speed=(0.2, 1), life=(1, 2), size=(0.6, 1.0),
                           colors=[(1, 0.5, 0.1, 0.9)], gravity=1.0, drag=1, seed=900 + k, box=2.0,
                           additive=True))
            t += 0.5; k += 1
        # splash
        td = T["land"]
        L = (LAND[0] + 0.5, 8.6, LAND[2] + 0.5)
        B.append(Burst(td, L, 240, speed=(3, 10), life=(0.5, 1.3), size=(0.08, 0.24),
                       colors=[(0.55, 0.75, 1, 0.95), (0.9, 0.95, 1, 0.95), (0.3, 0.55, 0.95, 0.9)],
                       gravity=-16, drag=0.6, up_bias=1.8, spread=(1, 0.6, 1), seed=30))
        B.append(Burst(td, L, 60, speed=(1, 3), life=(0.6, 1.2), size=(0.3, 0.6),
                       colors=[(1, 1, 1, 0.6)], gravity=0.5, drag=2, grow=1.5, seed=31))
        # block placement puffs during the time-lapse
        for i, (tp, p, b) in enumerate(self.plan[::3]):
            B.append(Burst(tp, (p[0] + 0.5, p[1] + 1.0, p[2] + 0.5), 6, speed=(0.5, 1.5), life=(0.25, 0.45),
                           size=(0.05, 0.09), colors=[(1, 1, 1, 0.9)], gravity=0, drag=3, seed=500 + i,
                           additive=True, box=0.4))

    def _make_red_vignette(self):
        h, w = 1920, 1080
        yy = np.linspace(-1, 1, h)[:, None]
        xx = np.linspace(-1, 1, w)[None, :]
        r = np.clip((np.sqrt(xx ** 2 * 1.1 + yy ** 2 * 0.7) - 0.45) / 0.65, 0, 1) ** 1.5
        a = np.zeros((h, w, 4), np.uint8)
        a[..., 0] = 200
        a[..., 3] = (r * 255).astype(np.uint8)
        return Image.fromarray(a, "RGBA")

    # ================================================================ actors
    def pip_state(self, t):
        T = self.T
        pose = {}
        pos = P0.copy()
        yaw = 0.0
        item = None
        for tp in self.punches:
            if tp <= t < tp + 0.24:
                pose["punch"] = seg(t, tp, tp + 0.24)
        if t < T["little"]:
            if T["life"] <= t < T["ground"]:
                pose["head_yaw"] = 0.7 * np.sin((t - T["life"]) * 5)
            if T["just"] <= t < T["very"]:
                pose["wave"] = smooth(seg(t, T["just"], T["just"] + 0.25)) * (1 - smooth(seg(t, T["very"] - 0.25, T["very"])))
                pose["wave_phase"] = t * 11
            if T["very"] - 0.3 <= t < T["but"]:
                pose["head_pitch"] = 0.65 * smooth(seg(t, T["very"] - 0.3, T["very"] + 0.2))
            if T["every"] - 0.4 <= t < T["little"]:
                pose["head_pitch"] = -0.55
                pose["lean"] = 0.12
            if T["special"] <= t < T["every"] - 0.4:
                pose["head_pitch"] = -0.5 * smooth(seg(t, T["special"], T["special"] + 0.3))
        elif t < T["clouds_end"] + 0.1:
            # time-lapse: busy building, spinning around
            u = seg(t, T["little"], T["little"] + 0.5)
            pos = lerp(P0, (1.5, 200, 0.5), smooth(u))
            yaw = -PI / 2 + (t - T["little"]) * 3.0
            pose["punch"] = (t * 6) % 1.0
            pose["walk"] = 0.6 * (1 - u) if u < 1 else 0.0
            pose["walk_phase"] = t * 12
        else:
            pos = np.array([1.5, 200, 0.5], np.float32)
            yaw = -PI / 2
            if t >= T["theblock"] - 0.3:
                pose.setdefault("head_pitch", -0.45)
            if T["creeper"] - 0.1 <= t < T["sword"]:
                h = seg(t, T["creeper"], T["creeper"] + 0.35)
                pos = pos + np.array([0, 0.45 * np.sin(PI * h), 0], np.float32)
                pose["head_pitch"] = 0.1
                pose["lean"] = -0.15
            if T["sword"] <= t < T["space"]:
                pose["head_pitch"] = -0.35
                pose["head_yaw"] = -0.55
                pose["reach"] = smooth(seg(t, T["sword"], T["sword"] + 0.2))
            if t >= T["space"]:
                u = seg(t, T["space"], T["nowhere"] + 0.7)
                pos = np.array([1.5 + 1.85 * ease_in_out(u), 200, 0.5], np.float32)
                pose["walk"] = 0.7 if u < 1 else 0.0
                pose["walk_phase"] = -t * 10
                pose["lean"] = -0.1
                pose["head_pitch"] = 0.0
            if t >= T["so"]:
                item = "water_bucket" if t >= T["bucket"] - 0.15 else None
                if T["so"] <= t < T["bucket"]:
                    pose["head_yaw"] = -1.3 * np.sin(PI * seg(t, T["so"], T["bucket"]))   # glance at the drop
                turn = smooth(seg(t, T["andi"] - 0.35, T["jump"] - 0.02))
                yaw = -PI / 2 + PI * turn * -1
            if t >= T["jump"]:
                dt = t - T["jump"]
                x = 3.35 + 2.0 * ease_out(seg(t, T["jump"], T["jump"] + 1.2), 2)
                if t < T["second"]:
                    y = 200 + 4.5 * dt - 5.3 * dt * dt
                elif t < T["down"]:
                    u = seg(t, T["second"], T["down"])
                    y = 30 + (9.4 - 30) * ease_in(u, 1.15)
                    x = 5.5
                elif t < T["land"]:
                    y = 9.4 + (8.0 - 9.4) * seg(t, T["down"], T["land"])
                    x = 5.5
                else:
                    y, x = 8.0, 5.5
                pos = np.array([x, y, 0.5], np.float32)
                yaw = PI / 2 * -1 + PI   # facing +x
                if t < T["land"]:
                    pose["arms_up"] = smooth(seg(t, T["jump"], T["jump"] + 0.3))
                    pose["flail_phase"] = t * (6 if t < T["second"] else 2.2)
                    pose["roll"] = 0.18 * np.sin(t * 2.3)
                    pose["head_pitch"] = 0.5 if t < T["two1"] else -0.4
                    if t >= T["water2"] - 0.1:
                        pose["arms_up"] = 1 - smooth(seg(t, T["water2"] - 0.1, T["water2"] + 0.15))
                        pose["reach"] = smooth(seg(t, T["water2"] - 0.1, T["water2"] + 0.15))
                        pose["head_pitch"] = -0.6
                else:
                    item = "bucket" if t < T["but2"] else None
                    sq = seg(t, T["land"], T["land"] + 0.3)
                    pose["lean"] = 0.25 * np.sin(PI * sq)
                    yaw = PI / 2 * -1 + PI - 0.9 * smooth(seg(t, T["land"] + 0.3, T["land"] + 0.8))
                    if T["survived"] <= t < T["hard2"] + 0.3:
                        pose["wave"] = np.sin(PI * seg(t, T["survived"], T["hard2"] + 0.3))
                        pose["wave_phase"] = t * 14
                    if T["but2"] - 0.2 <= t < T["tomorrow"]:
                        pose["head_pitch"] = 0.8 * smooth(seg(t, T["but2"] - 0.2, T["but2"] + 0.3))
                    if t >= T["tomorrow"] - 0.2:
                        pos, yaw, pose = self._pillar(t, pose)
        pose["item"] = item
        return pos, yaw, pose

    def _pillar(self, t, pose):
        T = self.T
        t0 = T["tomorrow"] - 0.2
        u = seg(t, t0, t0 + 0.45)
        x = 5.5 + 1.0 * smooth(u)
        y = 8.0
        pose["walk"] = 0.8 if u < 1 else 0
        pose["walk_phase"] = t * 12
        yaw = PI / 2
        for k, tj in enumerate(self.pillar_times()):
            if t >= tj - 0.2:
                h = seg(t, tj - 0.2, tj + 0.25)
                y = 8.0 + k + (1.0 if h >= 1 else 1.25 * np.sin(PI * 0.5 * h / 0.62) if h < 0.62
                               else 1.0 + 0.25 * np.cos(PI * 0.5 * (h - 0.62) / 0.38))
                pose["punch"] = h if h < 1 else 0
                pose["head_pitch"] = -0.6
        if t >= T["think"] - 0.2:
            yaw = PI / 2 - 1.2 * smooth(seg(t, T["think"] - 0.2, T["think"] + 0.3))
            pose["head_pitch"] = 0.35
            pose["wave"] = smooth(seg(t, T["think"], T["think"] + 0.3))
            pose["wave_phase"] = t * 11
        return np.array([x, y, 0.5], np.float32), yaw, pose

    def pillar_times(self):
        T = self.T
        return [T["tomorrow"] + 0.45, T["tomorrow"] + 1.05]

    def creeper_state(self, t):
        T = self.T
        if not (T["creeper"] - 0.06 <= t < T["exploded"]):
            return None
        rise = ease_out_back(seg(t, T["creeper"] - 0.06, T["creeper"] + 0.32))
        x = 0.5
        pose = {"rise": 0.35 + 0.65 * rise}
        if t >= T["space"]:
            u = seg(t, T["space"], T["nowhere"] + 0.8)
            x = 0.5 + 1.6 * ease_in_out(u)
            pose["walk"] = 0.8 if u < 1 else 0.0
            pose["walk_phase"] = t * 9
        if t >= T["jump"]:
            x = 2.1 + 0.5 * smooth(seg(t, T["jump"], T["jump"] + 0.6))
            pose["head_pitch"] = -0.5
        flash = 0.0
        if t >= T["fuse"]:
            u = seg(t, T["fuse"], T["exploded"])
            pose["swell"] = u ** 1.5 + 0.08 * np.sin(t * 30) * u
            ph = (t - T["fuse"]) * (2.5 + 6 * u)
            flash = 0.8 if (ph % 1.0) < 0.5 else 0.0
        y = 199.25 + 0.75 * min(rise, 1.0)
        jitter = np.array([np.sin(t * 50) * 0.02, 0, np.cos(t * 43) * 0.02]) if t >= T["fuse"] else 0
        return np.array([x, y, 0.5]) + jitter, PI / 2, pose, flash

    # ================================================================ camera
    def camera(self, t):
        T = self.T
        pip, _, _ = self.pip_state(t)
        fov, roll = 60.0, 0.0
        head = P0 + (0, 1.6, 0)
        if t < T["two0"]:
            u = ease_out(seg(t, 0, T["two0"] - 0.05), 3)
            eye = lerp(P0 + (0.05, 1.65, 1.35), (6.5, 203.8, 9.5), u)
            tgt = lerp(head, P0 + (0, 0.4, 0), u)
            fov = 55 + 10 * u
        elif t < T["hard"]:
            u = smooth(seg(t, T["two0"], T["hard"]))
            eye = lerp((3.2, 205.5, 4.4), (2.2, 203.2, 3.0), u)
            tgt = lerp((-1.2, 182, -2.4), (-1.0, 176, -2.2), u)
            fov = 72
        elif t < T["life"]:
            u = ease_out(seg(t, T["hard"], T["life"]))
            eye = lerp((2.6, 199.0, 3.2), (1.9, 199.4, 2.4), u)
            tgt = P0 + (0, 1.2, 0)
            fov = 62
        elif t < T["ground"]:
            a = 0.3 + 0.7 * seg(t, T["life"], T["ground"])
            eye = orbit(P0, 4.2, a, 1.4)
            tgt = P0 + (0, 1.0, 0)
        elif t < T["just"]:
            u = smooth(seg(t, T["ground"], T["just"]))
            eye = lerp((1.75, 201.4, 1.7), (1.65, 201.0, 1.55), u)
            tgt = (3.2, 186, 3.4)
            fov = 78
        elif t < T["very"] - 0.05:
            a = -0.6 + 0.5 * seg(t, T["just"], T["very"])
            eye = orbit(P0, 6.5, a, 1.6)
            tgt = P0 + (0, 0.9, 0)
            fov = 55
        elif t < T["but"] - 0.05:
            u = ease_in_out(seg(t, T["very"] - 0.05, T["but"] - 0.05))
            d = 3.0 + 12.0 * u                        # dolly-zoom (vertigo) looking straight down
            fov = np.degrees(2 * np.arctan(2.75 / d))
            eye = P0 + (0.02, d, 0.05)
            tgt = P0 + (0, -10, 0)
            roll = 0.9 * u
        elif t < T["every"] - 0.05:
            a = 2.2 + 0.9 * seg(t, T["but"], T["every"])
            eye = orbit((0.5, 199.5, 0.5), 3.9, a, 1.6)
            tgt = (0.5, 200.5, 0.5)
            fov = 52
        elif t < T["wood"] - 0.05:
            u = seg(t, T["every"], T["wood"])
            eye = orbit(P0, 3.6, 0.9 + 0.25 * u, 0.6)
            tgt = P0 + (0, 0.6, 0)
            fov = 55
        elif t < T["stone"] - 0.05:
            eye = orbit(P0, 2.6 - 0.3 * seg(t, T["wood"], T["stone"]), -0.8, 0.2)
            tgt = P0 + (0, 0.2, 0)
            fov = 55
        elif t < T["water"] - 0.05:
            eye = orbit(P0, 2.4, 2.4 + 0.2 * seg(t, T["stone"], T["water"]), 1.8)
            tgt = P0 + (0, 0.2, 0)
            fov = 55
        elif t < T["little"] - 0.03:
            eye = orbit(P0, 3.2, 0.2, 1.3)
            tgt = P0 + (0, 0.9, 0)
            fov = 55
        elif t < T["clouds_end"] + 0.05:
            u = ease_in_out(seg(t, T["little"], T["clouds_end"] + 0.05))
            eye = orbit((0.5, 200, 0.5), 5 + 12 * u, 0.5 + 1.3 * u, 2 + 6 * u)
            tgt = (0.3, 200.5 - 1.0 * u, 0.3)
            fov = 58
        elif t < T["theblock"] - 0.1:
            u = smooth(seg(t, T["clouds_end"], T["theblock"]))
            eye = orbit((0.5, 200, 0.5), 18 - 4 * u, 2.0 + 0.4 * u, 5 - 2 * u)
            tgt = (0.3, 200, 0.3)
            fov = 55
        elif t < T["creeper"] - 0.08:
            u = seg(t, T["theblock"] - 0.1, T["creeper"])
            eye = lerp((3.9, 202.7, 3.0), (3.3, 202.3, 2.5), smooth(u))   # over Pip's shoulder
            tgt = (0.5, 200.2, 0.5)
            fov = 50 if t < T["not"] else 50 - 16 * ease_out(seg(t, T["not"], T["not"] + 0.18))
        elif t < T["sword"] - 0.05:
            u = ease_out(seg(t, T["creeper"] - 0.08, T["sword"]))
            eye = lerp((2.0, 200.9, 2.4), (1.45, 201.0, 1.75), u)
            tgt = (0.5, 201.2, 0.5)
            fov = 55
        elif t < T["space"] - 0.05:
            eye = lerp((0.4, 202.4, 3.0), (0.55, 202.3, 2.8), seg(t, T["sword"], T["space"]))
            tgt = (1.5, 201.3, 0.5)
            fov = 46
        elif t < T["nowhere"] - 0.05:
            u = seg(t, T["space"], T["nowhere"])
            eye = (1.2, 209.0, 0.7)
            tgt = (1.2, 200, 0.5)
            fov = 52
            roll = 0.3 * u
        elif t < T["so"] - 0.03:
            eye = lerp((1.4, 202.7, 1.3), (1.7, 202.6, 1.2), seg(t, T["nowhere"], T["so"]))
            tgt = (4.8, 196.0, 0.5)
            fov = 70
        elif t < T["andi"] - 0.05:
            eye = lerp((2.4, 201.3, 3.6), (2.7, 201.3, 3.0), seg(t, T["so"], T["andi"]))
            tgt = (2.8, 200.9, 0.5)
            fov = 52
        elif t < T["island"]:
            eye = (1.3, 202.2, -2.8)
            tgt = pip + (0, 0.8, 0)
            fov = 62
        elif t < T["two1"] - 0.05:
            eye = pip + (1.4, -5.0, 2.6)
            tgt = lerp((1.0, 200.0, 0.4), pip + (0, 1.5, 0), 0.25)
            fov = 72
        elif t < T["falling0"] - 0.05:
            eye = pip + (0.7, 3.3, 1.3)
            tgt = pip + (0, -40, 0)
            fov = 70
        elif t < T["second"] - 0.03:
            a = 1.0 + 2.6 * seg(t, T["falling0"], T["second"])
            eye = pip + (3.6 * np.sin(a), 0.6, 3.6 * np.cos(a))
            tgt = pip + (0, 0.4, 0)
            fov = 66
            roll = 0.25 * np.sin(t * 1.7)
        elif t < T["down"] - 0.02:
            eye = pip + np.array([2.6, 2.0, 3.3], np.float32)
            tgt = pip + np.array([0, 0.1, 0], np.float32)
            fov = 64
        elif t < T["survived"] - 0.05:
            eye = (9.0, 9.3, 4.8)
            tgt = lerp((5.5, 9.4, 0.5), (5.5, 9.0, 0.5), smooth(seg(t, T["down"], T["land"])))
            fov = 60 - 8 * ease_out(seg(t, T["down"], T["down"] + 0.25))
        elif t < T["but2"] - 0.05:
            a = 0.9 + 0.7 * seg(t, T["survived"], T["but2"])
            eye = orbit((5.5, 8, 0.5), 4.2, a, 1.3)
            tgt = (5.5, 9.3, 0.5)
            fov = 55
        elif t < T["tomorrow"] - 0.25:
            u = smooth(seg(t, T["but2"], T["tomorrow"]))
            eye = (3.6, 9.2, 3.0)
            tgt = lerp((5.5, 9.8, 0.5), (1.0, 199.5, 0.5), u)
            fov = 60 - 47 * u
        elif t < T["think"] - 0.05:
            eye = (7.0, 10.0, 5.6)
            tgt = (6.3, 9.3, 0.5)
            fov = 55
        else:
            u = ease_in_out(seg(t, T["think"] - 0.05, self.duration))
            eye = lerp((8.3, 11.2, 4.0), (10.5, 13.5, 6.5), u)
            tgt = (6.4, 10.8, 0.5)
            fov = 55
        eye = np.asarray(eye, np.float32)
        amp = 0.0
        for t0, a, d in ((T["hard"], 0.12, 0.5), (T["creeper"], 0.08, 0.4), (T["exploded"], 0.9, 1.4),
                         (T["land"], 0.25, 0.6), (T["not"], 0.04, 0.3)):
            if t0 <= t < t0 + d:
                amp += a * (1 - (t - t0) / d) ** 2
        if T["jump"] < t < T["second"]:
            amp += 0.04
        eye = eye + shake(t, amp)
        return dict(eye=eye, target=np.asarray(tgt, np.float32), fov=float(fov), roll=float(roll))

    def cuts(self):
        T = self.T
        keys = ["two0", "hard", "life", "ground", "just", "but", "every", "wood", "stone", "water", "little",
                "theblock", "creeper", "sword", "space", "nowhere", "so", "andi", "island", "two1", "falling0",
                "second", "down", "survived", "but2", "think"]
        return [T[k] for k in keys]

    # ================================================================ frame
    def phase(self, t):
        T = self.T
        return float(Track([(0, 0, "linear"), (T["little"] + 0.3, 0, "linear"),
                            (T["clouds_end"], 1.0, "smooth"), (T["night"] + 0.2, 2.0, "smooth"),
                            (T["gone"], 2.0, "linear"), (T["think"], 3.0, "smooth")])(t)[()])

    def frame_spec(self, t, cam=None):
        T = self.T
        cam = cam or self.camera(t)
        # fog per location
        if t < T["island"]:
            fs_, fe_ = 70, 340
        elif t < T["second"]:
            fs_, fe_ = 45, 420
        elif T["but2"] <= t < T["tomorrow"] - 0.25:
            fs_, fe_ = 250, 900
        else:
            fs_, fe_ = 35, 240
        sky, light, fog = environment(self.phase(t), fs_, fe_)
        if T["exploded"] <= t < T["exploded"] + 1.2:        # explosion light
            k = (1 - (t - T["exploded"]) / 1.2) ** 2
            light["ambient"] = tuple(np.array(light["ambient"]) + np.array([1.4, 0.8, 0.3]) * k)
        if T["land"] <= t:                                 # moonlit ground scenes slightly brighter
            light["exposure"] = light["exposure"] * 1.1

        dyn, dyn_t = [], []
        pos, yaw, pose = self.pip_state(t)
        hurt = T["exploded"] + 0.1 <= t < T["exploded"] + 0.45
        dyn.append(pip_vertices(pos, yaw, pose, flash=0.55 if hurt else 0.0))
        cs = self.creeper_state(t)
        flash_col = (1, 1, 1)
        if hurt:
            flash_col = (1, 0.1, 0.1)
        if cs is not None:
            cpos, cyaw, cpose, cflash = cs
            dyn.append(creeper_vertices(cpos, cyaw, cpose, flash=cflash))
        # the one block
        exploded = t >= T["exploded"]
        if not exploded:
            ctype, since = self.center_type(t)
            s = 0.7 + 0.3 * ease_out_back(seg(t, since, since + 0.16)) if since > 0 else 1.0
            off = np.zeros(3)
            rot = (0, 0, 0)
            cf = 0.0
            if T["special"] <= t < T["special"] + 0.9:
                cf = 0.45 * np.sin(PI * seg(t, T["special"], T["special"] + 0.9))
                flash_col = (1.0, 0.85, 0.3)
            if self.night_break + 0.1 <= t < T["creeper"]:
                k = seg(t, self.night_break, T["creeper"])
                off = np.array([np.sin(t * 61), np.sin(t * 47) * 0.5, np.cos(t * 53)]) * 0.04 * (0.3 + k)
                rot = (np.sin(t * 37) * 0.05 * k, 0, np.cos(t * 41) * 0.05 * k)
            crack = None
            for tp in self.punches:
                pass
            nxt = [tb for tb, _ in self.breaks if tb > t]
            if nxt and nxt[0] - t < 0.75 and any(tp <= t for tp in self.punches if nxt[0] - 0.8 < tp < nxt[0]):
                crack = int(min(3, (0.75 - (nxt[0] - t)) / 0.75 * 4))
            dyn.append(block_entity_vertices(ctype, np.array([0.5, 199.5, 0.5]) + off, s, rot, flash=cf,
                                             crack=crack))
        # island
        static = ["terrain"]
        static_t = ["terrain_t", "clouds"]
        k_set = sum(1 for tp, _, _ in self.plan if tp <= t - 0.15)
        k_all = sum(1 for tp, _, _ in self.plan if tp <= t)
        if k_set > 0:
            key = f"isl_{k_set}_{int(exploded)}"
            if key not in self._r.static:
                w = self._world_at_k(k_set).copy()
                if exploded:
                    for p in self.removed:
                        w.set(p, None)
                o, tr = w.mesh()
                self._r.set_static(key, o)
                self._r.set_static(key + "_t", tr)
            static.append(key)
            static_t.append(key + "_t")
        for tp, p, b in self.plan[k_set:k_all]:
            s = ease_out_back(seg(t, tp, tp + 0.15))
            v = block_entity_vertices(b, np.array(p) + 0.5, max(s, 0.05))
            (dyn_t if tx.is_translucent(b) else dyn).append(v)
        # bucket item flying from the block to Pip
        tw = T["water"]
        if tw <= t < tw + 0.55:
            u = seg(t, tw, tw + 0.55)
            p = lerp((0.5, 200.3, 0.5), (0.5, 201.6, 0.5), ease_out(min(u * 2, 1)))
            if u > 0.5:
                p = lerp(p, (0.15, 201.0, 0.9), ease_in((u - 0.5) * 2))
            m = translate(*p)
            from mcengine.geometry import rot_y
            m = m @ rot_y(t * 9)
            dyn.append(box_vertices((0.32, 0.32, 0.32), ["bucket_side", "bucket_side", "bucket_top",
                                                         "bucket_side", "bucket_side", "bucket_side"],
                                    matrix=m, emissive=0.2))
        # debris
        if exploded:
            dt = t - T["exploded"]
            for b, c0, v0, spin, sc in self.debris:
                p = c0 + v0 * dt * 0.85 + np.array([0, -0.5 * 18 * dt * dt * 0.6, 0])
                if p[1] < 9 or dt > 9:
                    continue
                if np.linalg.norm(p - cam["eye"]) < 0.6:
                    continue
                dyn.append(block_entity_vertices(b, p, sc, tuple(spin * dt)))
        # landing water
        if t >= T["down"]:
            u = ease_out_back(seg(t, T["down"], T["down"] + 0.18))
            src = block_entity_vertices("water", np.array([LAND[0] + 0.5, 8.5 - 0.5 * (1 - u), LAND[2] + 0.5]),
                                        max(u, 0.05), alpha=0.85)
            dyn_t.append(src)
            for i, (dx, dz) in enumerate(((-1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1))):
                uu = smooth(seg(t, T["down"] + 0.06 + 0.04 * i, T["down"] + 0.3 + 0.04 * i))
                if uu <= 0:
                    continue
                m = translate(LAND[0] + 0.5 + dx, 8.0, LAND[2] + 0.5 + dz)
                dyn_t.append(box_vertices((uu, 0.32 * uu, uu), "water_flow", matrix=m, offset=(0, 0.16 * uu, 0),
                                          alpha=0.8))
        for k, tj in enumerate(self.pillar_times()):
            if t >= tj:
                s = ease_out_back(seg(t, tj, tj + 0.15))
                dyn.append(block_entity_vertices("dirt", np.array([6.5, 8.5 + k, 0.5]), max(s, 0.05)))
        # particles
        bursts = self.bursts
        parts = [gather(bursts, t, False)]
        adds = [gather(bursts, t, True)]
        if T["jump"] + 0.6 <= t < T["land"]:
            adds.append(self._wind_dust(t, pos))
        parts = [p for p in parts if p is not None]
        adds = [p for p in adds if p is not None]

        post = dict(bloom=0.3, vignette=0.38, saturation=1.12)
        wf = 0.0
        for t0, a, d in ((T["exploded"], 1.0, 0.35), (T["land"], 0.45, 0.25), (T["hard"], 0.25, 0.2),
                         (T["creeper"] - 0.05, 0.3, 0.2)):
            if t0 <= t < t0 + d:
                wf = max(wf, a * (1 - (t - t0) / d))
        post["white_flash"] = wf
        if T["hard"] <= t < T["hard"] + 0.6:
            k = 1 - seg(t, T["hard"], T["hard"] + 0.6)
            post["tint"] = (1 + 0.25 * k, 1 - 0.25 * k, 1 - 0.25 * k)
        if T["second"] <= t < T["down"]:
            post["saturation"] = 0.55
            post["vignette"] = 0.65
            post["tint"] = (0.95, 1.0, 1.1)
        return dict(camera=cam, sky=sky, light=light, fog=fog, static=static, static_trans=static_t,
                    dynamic=np.concatenate(dyn) if dyn else None,
                    dynamic_trans=np.concatenate(dyn_t) if dyn_t else None,
                    particles=np.concatenate(parts) if parts else None,
                    particles_add=np.concatenate(adds) if adds else None,
                    flash_col=flash_col, post=post)

    def _wind_dust(self, t, pip):
        rng = np.random.default_rng(77)
        n = 140
        off = rng.uniform(-9, 9, (n, 3)).astype(np.float32)
        speed = 40 if t < self.T["second"] else 9
        off[:, 1] = ((off[:, 1] + t * speed) % 18) - 9
        pos = pip + off
        a = 0.35 if t < self.T["second"] else 0.2
        return np.concatenate([pos, np.full((n, 1), 0.035), np.tile([1, 1, 1, a], (n, 1))], 1).astype(np.float32)

    # ================================================================ overlay
    def draw_overlay(self, t, layer):
        T = self.T
        if T["jump"] + 0.6 <= t < T["second"] and t >= T["two1"] - 0.05:
            ov.speed_lines(layer, t, 0.75)
        if T["exploded"] + 0.1 <= t < T["exploded"] + 1.0:
            k = 1 - seg(t, T["exploded"] + 0.1, T["exploded"] + 1.0)
            a = np.asarray(self._red_vig).copy()
            a[..., 3] = (a[..., 3] * k * 0.45).astype(np.uint8)
            layer.alpha_composite(Image.fromarray(a))
        # HUD
        if T["hard"] <= t < T["think"]:
            hp = 10.0 if t < T["exploded"] + 0.15 else 0.5
            s_in = ease_out_back(seg(t, T["hard"], T["hard"] + 0.25))
            sh = 1.0 if (T["exploded"] + 0.15 <= t < T["exploded"] + 0.8) else 0.0
            if s_in > 0.05:
                ov.draw_hearts(layer, hp, (540, 200 - (1 - min(s_in, 1)) * 120), px=6, shake_amt=sh, t=t)
        if T["water"] + 0.4 <= t < T["think"]:
            items = ["e:🪣"]
            items += ["grass" if t >= T["little"] else None, "log" if t >= T["wood"] else None,
                      "stone" if t >= T["stone"] else None]
            items += ["planks", "glass"] if t >= T["little"] else []
            sel = 0 if t >= T["grabbed"] else 1
            if t >= T["tomorrow"]:
                sel = 1
            a = smooth(seg(t, T["water"] + 0.4, T["water"] + 0.7))
            ov.draw_hotbar(layer, items, sel, (540, 300), slot=84, alpha=a)
        # titles
        tp = ov.title_pop
        tp(layer, "ONE BLOCK", t, T["one"], T["two0"] - 0.02, (540, 560), 128, fill=(255, 225, 40), emoji="☝️")
        tp(layer, "200 BLOCKS UP", t, T["two0"], T["hard"] - 0.03, (540, 560), 100, emoji="☁️")
        tp(layer, "HARDCORE", t, T["hard"], T["life"] - 0.05, (540, 560), 140, fill=(255, 60, 50), emoji="💀",
           emoji_side="right")
        tp(layer, "1 LIFE", t, T["life"], T["ground"], (540, 560), 120, emoji="❤️", emoji_side="right")
        tp(layer, "WOOD", t, T["wood"], T["stone"] - 0.02, (540, 560), 120, emoji="🪵", emoji_side="right")
        tp(layer, "STONE", t, T["stone"], T["water"] - 0.02, (540, 560), 120, emoji="🪨", emoji_side="right")
        tp(layer, "WATER", t, T["water"], T["little"] - 0.02, (540, 560), 120, fill=(110, 190, 255), emoji="💧",
           emoji_side="right")
        span = (T["clouds_end"] - T["little"]) / 3
        for d in range(3):
            tp(layer, f"DAY {d + 1}", t, T["little"] + d * span, T["little"] + (d + 1) * span - 0.02, (540, 560), 110,
               emoji="☀️" if d < 2 else "🌅", emoji_side="right")
        tp(layer, "NIGHT 3", t, T["night"], T["theblock"] + 0.6, (540, 560), 120, fill=(170, 190, 255), emoji="🌙",
           emoji_side="right")
        if T["creeper"] <= t < T["sword"]:
            a = ease_out_back(seg(t, T["creeper"], T["creeper"] + 0.25))
            ov.paste_emoji(layer, "😱", (540, 600), 210 * max(a, 0.05))
        if T["sword"] <= t < T["space"] - 0.03:
            a = ease_out_back(seg(t, T["sword"], T["sword"] + 0.2))
            from PIL import ImageDraw
            rr = 125 * max(a, 0.05)
            ImageDraw.Draw(layer).ellipse([540 - rr, 610 - rr, 540 + rr, 610 + rr], fill=(255, 255, 255, 215),
                                          outline=(0, 0, 0, 255), width=6)
            ov.paste_emoji(layer, "🗡️", (540, 610), 170 * max(a, 0.05))
            b = ease_out_back(seg(t, T["sword"] + 0.4, T["sword"] + 0.6))
            if b > 0.05:
                ov.paste_emoji(layer, "❌", (540, 610), 230 * b)
        tp(layer, "BOOM!", t, T["exploded"], T["two1"] - 0.05, (540, 600), 170, fill=(255, 160, 30), emoji="💥",
           emoji_side="right")
        if T["two1"] - 0.05 <= t < T["second"]:
            pos, _, _ = self.pip_state(t)
            col = (255, 255, 255) if pos[1] > 80 else (255, 90, 70)
            a = smooth(seg(t, T["two1"] - 0.05, T["two1"] + 0.1))
            ov.draw_text(layer, f"Y: {int(pos[1])}", (540, 560), 120, fill=col, alpha=a)
            ov.paste_emoji(layer, "⬇️", (300 - 10, 560), 90, alpha=a)
        tp(layer, "1 SECOND LEFT", t, T["second"], T["down"], (540, 560), 104, fill=(255, 70, 60), emoji="⏱️",
           emoji_side="left")
        tp(layer, "CLUTCH!", t, T["down"] + 0.1, T["survived"] - 0.02, (540, 560), 150, fill=(110, 200, 255),
           emoji="💦", emoji_side="right")
        tp(layer, "SURVIVED", t, T["survived"], T["hard2"] - 0.02, (540, 560), 130, fill=(120, 240, 120),
           emoji="✅", emoji_side="right")
        if T["zero"] - 0.1 <= t < T["but2"]:
            a = ease_out_back(seg(t, T["zero"] - 0.1, T["zero"] + 0.15))
            blink = 0.5 if (t * 6) % 1 < 0.5 else 0.0
            if a > 0.05:
                ov.draw_hearts(layer, blink, (540, 640), px=max(1, int(13 * a)), n=1)
                ov.draw_text(layer, "½ HEART LEFT", (540, 820), 80 * a, fill=(255, 90, 80))
        tp(layer, "HOME: GONE", t, T["gone"] - 0.1, T["tomorrow"] - 0.2, (540, 560), 110, fill=(255, 255, 255),
           emoji="💔", emoji_side="right")
        if t >= T["think"]:
            a = ease_out_back(seg(t, T["think"], T["think"] + 0.3))
            ov.draw_text(layer, "PART 2?", (540, 640), 170, fill=(255, 225, 40), scale=max(a, 0.05),
                         rot=np.sin(t * 5) * 2)
            b = smooth(seg(t, T["think"] + 0.3, T["think"] + 0.6))
            if b > 0:
                ov.draw_text(layer, "FOLLOW TO FIND OUT", (540, 800), 74, alpha=b)
                ov.paste_emoji(layer, "👇", (540, 930 + 12 * np.sin(t * 8)), 110, alpha=b)
        # subtitles last (on top)
        if t < T["think"]:
            self.subs.draw(layer, t)
        else:
            self.subs.draw(layer, t)

    # ================================================================ audio
    def sfx_events(self):
        T = self.T
        ev = [(0.0, "whoosh", -10, dict(dur=0.7)), (T["one"], "pop", -12, {}),
              (T["two0"] - 0.15, "whoosh", -10, dict(dur=0.6, up=False)), (T["hard"], "impact", -3, {}),
              (T["ground"] - 0.1, "whoosh", -12, dict(dur=0.5)),
              (T["very"] - 0.05, "riser", -12, dict(dur=T["but"] - T["very"])),
              (T["fall"] - 0.1, "whoosh", -8, dict(dur=0.8, up=False)),
              (T["special"], "shimmer", -6, {})]
        for tp in self.punches:
            ev.append((tp + 0.1, "dig", -8, dict(seed=int(tp * 10))))
        for i, (te, b) in enumerate(self.center_events[1:]):
            ev.append((te, "pop", -11, dict(seed=i, pitch=0.8 + 0.08 * i)))
        ev += [(T["water"] + 0.45, "ding", -12, {})]
        for i, (tp, p, b) in enumerate(self.plan):
            if i % 3 == 0:
                ev.append((tp, "place", -14, dict(seed=i)))
        ev += [(T["night"], "shimmer", -16, {}), (T["not"], "whoosh", -10, dict(dur=0.35)),
               (T["not"] + 0.3, "heartbeat", -6, {}), (T["not"] + 1.0, "heartbeat", -5, {}),
               (T["creeper"] - 0.05, "impact", -2, {}), (T["creeper"] - 0.05, "dig", -8, {})]
        for i in range(5):
            ev.append((T["space"] + i * 0.3, "step", -14, dict(seed=i)))
        ev += [(T["fuse"], "hiss", -6, dict(dur=T["exploded"] - T["fuse"])),
               (T["bucket"] - 0.15, "pop", -12, dict(pitch=0.7)),
               (T["jump"] - 0.05, "whoosh", -8, dict(dur=0.6)),
               (T["exploded"], "explosion", 0, {}), (T["exploded"] + 0.15, "hurt", -8, {}),
               (T["two1"] - 0.1, "whoosh", -9, dict(dur=0.5, up=False)),
               (T["falling0"] - 0.1, "whoosh", -9, dict(dur=0.6)),
               (T["second"], "heartbeat", -3, {}), (T["second"] + 0.7, "heartbeat", -3, {}),
               (T["water2"], "pop", -10, dict(pitch=0.6)),
               (T["down"], "splash", -2, {}), (T["land"], "impact", -9, {}),
               (T["survived"], "ding", -9, {}), (T["zero"], "heartbeat", -8, {})]
        for tj in self.pillar_times():
            ev.append((tj, "place", -9, {}))
        ev.append((T["think"], "pop", -10, dict(pitch=1.2)))
        return ev

    def wind(self, A):
        """Falling wind bed, as one array starting at the jump."""
        T = self.T
        dur = T["land"] - T["jump"] + 0.4
        def inten(tt):
            t = tt + T["jump"]
            k = np.clip((t - T["jump"]) / 1.5, 0, 1)
            k = np.where(t > T["second"], 0.35, k)
            return k
        x = A.sfx_wind(dur, intensity=inten)
        f = int(0.3 * A.SR)
        x[-f:] *= np.linspace(1, 0, f)[:, None]
        return x

    def music_sections(self):
        T = self.T
        return [("adventure", 0.0, T["wood"] - 0.05), ("build", T["wood"] - 0.05, T["then"]),
                ("suspense", T["then"], T["creeper"] - 0.1), ("silence", T["creeper"] - 0.1, T["sword"]),
                ("panic", T["sword"], T["jump"]), ("freefall", T["jump"], T["second"]),
                ("silence", T["second"], T["survived"] - 0.15), ("resolve", T["survived"] - 0.15, self.duration)]
