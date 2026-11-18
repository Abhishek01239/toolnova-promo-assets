"""'I Survived 100 Days': the Film. Timing comes from the narration (assets/words.json); the picture is a
list of Shots built in shots.py and rendered by the generic engine in engine.py."""
import bisect
import json
import os
import re
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from mcengine import textures as tx                                                     # noqa: E402
from mcengine.geometry import VoxelWorld                                                # noqa: E402
from mcengine.anim import seg, smooth, clamp01, ease_out_back, gather                   # noqa: E402
from mcengine.lighting import environment                                                # noqa: E402
from mcengine.models import torch_vertices                                              # noqa: E402
from mcengine import overlay as ov                                                       # noqa: E402
import script as S                                                                        # noqa: E402
import world as WD                                                                        # noqa: E402
import engine as E                                                                        # noqa: E402

W, H = 1920, 1080
PI = np.pi
U = np.array([400.0, 40.0, 0.0])
FLOOR = -32


def norm(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def ug(x, y, z):
    return np.array([x, y, z], float) + U


def ng(x, y, z):
    return np.array([x, y, z], float) + WD.NO


def env(ph, fs, fe):
    """environment() extended: phase in (3, 4] blends dawn back into full day."""
    if ph <= 3.0:
        return environment(ph, fs, fe)
    k = smooth(min(ph - 3.0, 1.0))
    a, b = environment(3.0, fs, fe), environment(0.0, fs, fe)
    out = []
    for da, db in zip(a, b):
        d = {}
        for key in da:
            va, vb = da[key], db[key]
            if isinstance(va, tuple):
                d[key] = tuple(float(p * (1 - k) + q * k) for p, q in zip(va, vb))
            elif isinstance(va, float):
                d[key] = float(va * (1 - k) + vb * k)
            else:
                d[key] = vb
        out.append(d)
    sd = np.array(out[0]["sun_dir"]); sd /= np.linalg.norm(sd)
    out[0]["sun_dir"] = tuple(sd)
    ld = np.array(out[1]["dir"]); ld /= np.linalg.norm(ld)
    out[1]["dir"] = tuple(ld)
    return tuple(out)


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
        self.cutlist = [self.C(k) for k in S.KEYS]
        G, B, R, Y = (130, 230, 110), (110, 235, 255), (255, 90, 70), (255, 225, 40)
        self.subs = ov.Subtitles(self.words, {
            "DIAMOND": B, "DIAMONDS": B, "IRON": (236, 190, 150), "LAVA": (255, 120, 40), "OBSIDIAN": (190, 140, 255),
            "CREEPER": G, "CREEPERS": G, "ZOMBIE": G, "ZOMBIES": G, "SPIDER": (230, 80, 80), "SPIDERS": (230, 80, 80),
            "BISCUIT": (255, 200, 120), "NETHER": (255, 90, 60), "GHAST": (240, 240, 240), "CASTLE": Y, "VILLAGE": Y,
            "VILLAGERS": Y, "EMERALDS": (90, 230, 120), "EMERALD": (90, 230, 120), "GLOWSTONE": Y, "BOOM": (255, 150, 30),
            "HUNDRED": Y, "FIFTY": Y, "WHEAT": (230, 200, 90), "TORCHES": (255, 190, 90), "PORTAL": (200, 110, 255),
            "BED": R, "WOLF": (230, 230, 230), "STONE": (200, 200, 200), "COAL": (160, 160, 160),
        }, max_words=6, max_chars=34, y=930, size=66, cx=W / 2, max_w=1500)
        self.terrain = WD.Terrain()
        self._r = None
        self.extra_static, self.extra_static_t = [], []
        self._decor_torches = {}
        self.vdoors = None
        import shots
        self.shots = shots.build(self)
        for s in self.shots:
            s.t0 = s.time_of(self)
        self.shots.sort(key=lambda s: s.t0)
        self.shots[0].t0 = 0.0
        for a, b in zip(self.shots, self.shots[1:]):
            a.t1 = b.t0
        self.shots[-1].t1 = self.duration
        self.starts = [s.t0 for s in self.shots]
        for i, s in enumerate(self.shots):
            s.bursts = list(s.fx)
            for j, a in enumerate(s.actors):
                s.bursts += a.auto_fx(1000 * i + 37 * j)
            for j, p in enumerate(s.props):
                s.bursts += p.fx(1000 * i + 500 + 17 * j)
        self._red = self._vignette((200, 0, 0))
        self._orange = self._vignette((255, 120, 20))

    # ------------------------------------------------------------ timing helpers
    def T(self, key, f=0.0):
        i = S.IDX[key]
        return self.ls[i] + f * (self.le[i] - self.ls[i])

    def C(self, key):
        return self.T(key, 0.0) - 0.15

    def E(self, key):
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
        i = max(0, bisect.bisect_right(self.cutlist, t) - 1)
        return S.KEYS[i]

    def shot_at(self, t):
        i = max(0, bisect.bisect_right(self.starts, t) - 1)
        return self.shots[i]

    def day_at(self, t):
        i = S.IDX[self.key_at(t)]
        d = 1
        for k in S.KEYS[:i + 1]:
            if k in S.DAY_AT:
                d = S.DAY_AT[k]
        return d

    def chapter_at(self, t):
        n = 1
        for c in self.chapters[1:]:
            if t >= c["card_start"]:
                n = c["num"]
        return n

    def ground(self, x, z, loc="over"):
        if loc == "over":
            return self.terrain.ground(x, z)
        if loc == "under":
            return U[1] + FLOOR
        if not hasattr(self, "_nh"):
            self._nh = WD.nether_heights(self.cache_dir)
        i, j = int(np.floor(x - WD.NO[0])) + 70, int(np.floor(z - WD.NO[2])) + 70
        if 0 <= i < 140 and 0 <= j < 140:
            return float(self._nh[i, j])
        return 17.0

    # ------------------------------------------------------------ static meshes
    def ensure(self, name):
        r = self._r
        if name in r.static:
            return
        o, t = self._load_static(name)
        r.set_static(name, o)
        r.set_static(name + "_t", t)

    def _load_static(self, name):
        C = self.cache_dir
        if name == "terrain":
            return WD.cached_mesh(C, "terrain", lambda: WD.build_terrain_world(self.terrain))
        if name == "far":
            return WD.cached_mesh(C, "farring", WD.far_ring)
        if name.startswith("pad_"):
            st = name[4:]
            return WD.cached_mesh(C, name, lambda: WD.stage_world(st))
        if name.startswith("decor_"):
            _, st, crops = name.split("_")
            v, torches = WD.stage_decor(st, crops)
            self._decor_torches[st] = torches
            return v, np.zeros((0, 12), np.float32)
        if name == "clouds":
            from scipy.ndimage import gaussian_filter
            rng = np.random.default_rng(3)
            g = gaussian_filter(rng.normal(size=(40, 40)), 1.4)
            g /= g.std()
            cw = VoxelWorld()
            for gx in range(40):
                for gz in range(40):
                    if g[gx, gz] > 0.7:
                        cw.set((gx - 20, 0, gz - 20), "cloud")
            _, cv = cw.mesh(alpha_override=0.75)
            cv[:, 0:3] = cv[:, 0:3] * np.array([12, 3, 12], np.float32) + np.array([10, 70, 40], np.float32)
            return np.zeros((0, 12), np.float32), cv
        if name == "hut":
            return WD.cached_mesh(C, "hut", WD.hut_world)
        if name == "hut_open":
            return WD.cached_mesh(C, "hut_open", lambda: WD.hut_world(True))
        if name in ("lighthouse", "lighthouse_glow"):
            return WD.cached_mesh(C, name, lambda: WD.lighthouse_world(self.terrain, glow=name.endswith("glow"))[0])
        if name.startswith("pool_"):
            return WD.cached_mesh(C, name, lambda: WD.lava_pool_world(name[5:]))
        if name == "under":
            p = os.path.join(C, "under.npz")
            if not os.path.exists(p):
                shutil.copy(os.path.join(HERE, "assets", "under_cave.npz"), p)
            d = np.load(p)
            return d["o"], d["t"]
        if name == "nether":
            def mk():
                w, gl = WD.build_nether()
                return w
            return WD.cached_mesh(C, "nether", mk, offset=WD.NO)
        if name == "village_decor":
            from mcengine import models100 as M
            v = [M.crop_vertices(c, "crop_gold", 0.8) for c in WD.village_crops()]
            for (x0, z0, wd, dp, side) in WD.VILLAGE_HOUSES:
                dx, dz = {"s": ((x0 + x0 + wd - 1) // 2, z0), "w": (x0, (z0 + z0 + dp - 1) // 2)}[side]
                if side == "w":
                    v.append(M.door_vertices((dx + 0.0, WD.GY + 1, dz), PI / 2))
                else:
                    v.append(M.door_vertices((dx, WD.GY + 1, dz), 0.0))
            return np.concatenate(v), np.zeros((0, 12), np.float32)
        raise KeyError(name)

    def stage_torches(self, stage, crops):
        self.ensure(f"decor_{stage}_{crops}")
        if stage not in self._decor_torches:
            self._decor_torches[stage] = WD.stage_decor(stage, crops)[1]
        return self._decor_torches[stage]

    # ------------------------------------------------------------ lighting
    def cave_light(self, t, shot, cam_eye):
        lx = cam_eye[0] - U[0]
        k = 0.95 if shot.cave is None else shot.cave
        light = dict(dir=(0.3, 0.9, 0.2), sun=(0.22 * k, 0.19 * k, 0.13 * k), ambient=(0.6 * k, 0.5 * k, 0.4 * k), exposure=1.2)
        fog = dict(color=(0.03, 0.028, 0.025), start=10, end=58)
        lava = clamp01((lx - 80) / 8)
        if lava > 0:
            light["ambient"] = tuple(np.array(light["ambient"]) * (1 - lava) + np.array([0.72, 0.44, 0.27]) * lava)
            light["sun"] = tuple(np.array(light["sun"]) * (1 - lava) + np.array([0.45, 0.22, 0.08]) * lava)
            fog = dict(color=tuple(np.array([0.03, 0.028, 0.025]) * (1 - lava) + np.array([0.16, 0.06, 0.02]) * lava),
                       start=12, end=60)
        sky = dict(zenith=(0.02, 0.02, 0.02), horizon=(0.03, 0.025, 0.02), below=(0.02, 0.02, 0.02),
                   stars=0.0, sun_vis=0.0, moon_vis=0.0, sun_dir=(0.3, 0.9, 0.2))
        if lx < 2:
            sky.update(zenith=(0.45, 0.62, 0.95), horizon=(0.75, 0.85, 1.0), below=(0.5, 0.6, 0.8))
        return sky, light, fog

    def nether_light(self):
        sky = dict(zenith=(0.25, 0.04, 0.03), horizon=(0.45, 0.1, 0.06), below=(0.35, 0.08, 0.05),
                   stars=0.0, sun_vis=0.0, moon_vis=0.0, sun_dir=(0.2, 0.9, 0.3))
        light = dict(dir=(0.2, 0.9, 0.3), sun=(0.35, 0.18, 0.12), ambient=(0.82, 0.55, 0.48), exposure=1.1)
        fog = dict(color=(0.42, 0.09, 0.06), start=18, end=120)
        return sky, light, fog

    def _vignette(self, col):
        yy = np.linspace(-1, 1, H)[:, None]
        xx = np.linspace(-1, 1, W)[None, :]
        r = np.clip((np.sqrt(xx ** 2 * 0.7 + yy ** 2 * 1.1) - 0.5) / 0.6, 0, 1) ** 1.5
        a = np.zeros((H, W, 4), np.uint8)
        a[..., 0], a[..., 1], a[..., 2] = col
        a[..., 3] = (r * 255).astype(np.uint8)
        return a

    def phase(self, t, shot=None):
        shot = shot or self.shot_at(t)
        return float(E._sched(shot.phase, t, 0.0))

    # ------------------------------------------------------------ frame
    def frame_spec(self, t, cam=None):
        shot = self.shot_at(t)
        cam = cam or shot.cam(t)
        for t0, d, amp in shot.shake:
            if t0 <= t < t0 + d:
                a = amp * (1 - (t - t0) / d) ** 2
                s = E.shake(t, a)
                cam = dict(cam)
                cam["eye"] = np.asarray(cam["eye"]) + s
                cam["target"] = np.asarray(cam["target"]) + s * 0.5
        post = dict(bloom=0.34, vignette=0.32, saturation=1.12)
        post.update(shot.post)
        glows = []
        dyn, dyn_t = [], []
        self.extra_static, self.extra_static_t = [], []
        if shot.loc == "over":
            ph = self.phase(t, shot)
            sky, light, fog = env(ph, 45, 330)
            if ph >= 1.6:
                k = clamp01((ph - 1.6) / 0.4)
                light = dict(light)
                light["ambient"] = tuple(np.array(light["ambient"]) + np.array([0.08, 0.07, 0.05]) * k)
                fog = dict(fog, start=35, end=220)
            static = ["far", "terrain", f"pad_{shot.stage}"]
            static_t = ["far_t", "terrain_t", f"pad_{shot.stage}_t"]
            if shot.clouds:
                static_t.append("clouds_t")
                self.ensure("clouds")
            if shot.decor:
                crops = E._sched(shot.crops, t, "gold")
                dn = f"decor_{shot.stage}_{crops}"
                static.append(dn)
                self.ensure(dn)
                tor = self.stage_torches(shot.stage, crops)
                lit = shot.torches_on if shot.torches_on is not None else ph > 0.8
                if shot.torch_reveal is not None:
                    c0 = np.array([12.0, 14.0])
                    tor = sorted(tor, key=lambda b: np.arctan2(b[2] - c0[1], b[0] - c0[0]))
                    tor = tor[:int(len(tor) * seg(t, *shot.torch_reveal))]
                for b in tor:
                    dyn.append(torch_vertices(b, 0.1 * np.sin(t * 13 + b[0])))
                    if lit:
                        glows.append([b[0], b[1] + 0.72, b[2], 0.55 + 0.08 * np.sin(t * 9 + b[0]), 1.0, 0.75, 0.35, 0.2])
            static.append("village_decor")
            self.ensure("village_decor")
            for nm in ("far", "terrain", f"pad_{shot.stage}"):
                self.ensure(nm)
        elif shot.loc == "under":
            sky, light, fog = self.cave_light(t, shot, cam["eye"])
            static, static_t = ["under"], ["under_t"]
            self.ensure("under")
            post["vignette"] = 0.48
            for x in range(92, 106, 3):
                for z in range(-4, 11, 3):
                    p = ug(x + 1.5, FLOOR + 0.1, z + 1.5)
                    glows.append([p[0], p[1], p[2], 1.8 + 0.2 * np.sin(t * 2 + x + z), 1.0, 0.45, 0.1, 0.10])
        else:
            sky, light, fog = self.nether_light()
            static, static_t = ["nether"], ["nether_t"]
            self.ensure("nether")
            post["vignette"] = 0.45
            post["saturation"] = 1.18
            for x in range(-66, 66, 12):
                for z in range(-66, 66, 12):
                    p = ng(x, 1.1, z)
                    glows.append([p[0], p[1], p[2], 3.0, 1.0, 0.45, 0.1, 0.07])
        for nm in shot.statics:
            self.ensure(nm)
            static.append(nm)
            static_t.append(nm + "_t")
        if shot.fog:
            fog = dict(fog)
            fog.update(shot.fog(t) if callable(shot.fog) else shot.fog)
        if shot.loc == "over" and shot.stage in ("P5", "P6", "P7", "P8"):
            for (x, y, z) in WD.portal_inner():
                glows.append([x + 0.5, y + 0.5, z + 0.5, 1.3 + 0.15 * np.sin(t * 3 + y), 0.7, 0.3, 1.0, 0.25])
        if shot.light:
            light = dict(light)
            light.update(shot.light(t) if callable(shot.light) else shot.light)
        for a in shot.actors:
            v = a.verts(t)
            if v is not None and len(v):
                dyn.append(v)
        for p in shot.props:
            d, dt, g = p.draw(t, self)
            dyn += d
            dyn_t += dt
            glows += g
        static += self.extra_static
        static_t += self.extra_static_t
        parts = gather(shot.bursts, t, False)
        adds = gather(shot.bursts, t, True)
        if glows:
            g = np.array(glows, np.float32)
            adds = g if adds is None else np.concatenate([adds, g])
        wf = 0.0
        for t0, a, dd in shot.flash:
            if t0 <= t < t0 + dd:
                wf = max(wf, a * (1 - (t - t0) / dd))
        post["white_flash"] = wf
        flash_col = (1, 1, 1)
        return dict(camera=cam, sky=sky, light=light, fog=fog, static=static, static_trans=static_t,
                    dynamic=np.concatenate(dyn) if dyn else None,
                    dynamic_trans=np.concatenate(dyn_t) if dyn_t else None,
                    particles=parts, particles_add=adds, flash_col=flash_col, post=post)

    def cuts(self):
        return [s.t0 for s in self.shots[1:] if s.t0 > 0.5]

    # ------------------------------------------------------------ overlay
    def hearts(self, t, shot):
        if shot.hp is None:
            return 10.0
        return float(np.round(E._sched(shot.hp, t, 10.0) * 2) / 2)

    def draw_card(self, layer, t, n):
        c0, c1 = self.card(n)
        t0, dur = c0 - 0.25, (c1 - c0) + 0.5
        if not (t0 <= t < t0 + dur):
            return False
        u = (t - t0) / dur
        cover = min(1.0, u / 0.18) if u < 0.5 else min(1.0, (1 - u) / 0.18)
        B = 80
        cols, rows = W // B + 1, H // B + 1
        rng = np.random.default_rng(7 + n)
        order = rng.random((rows, cols))
        d = ImageDraw.Draw(layer)
        for r in range(rows):
            for c in range(cols):
                if order[r, c] < cover:
                    shade = 34 + int(10 * ((r + c) % 2))
                    d.rectangle([c * B, r * B, c * B + B - 1, r * B + B - 1], fill=(shade, 52 + shade // 3, 30, 255))
        if cover > 0.85:
            a = min(1.0, (cover - 0.85) / 0.15)
            pop = max(ease_out_back(clamp01((u - 0.12) / 0.25)), 0.05)
            ch = self.chapters[n - 1]
            label, title = ch.get("label", ""), ch["title"]
            if label != title:
                ov.draw_text(layer, label, (W / 2, H / 2 - 110), 92, fill=(255, 225, 40), alpha=a, scale=pop)
                ov.draw_text(layer, title, (W / 2, H / 2 + 30), 132, fill=(255, 255, 255), alpha=a, scale=pop)
            else:
                ov.draw_text(layer, title, (W / 2, H / 2 - 20), 190, fill=(255, 225, 40), alpha=a, scale=pop)
            ov.draw_text(layer, f"CHAPTER {n}", (W / 2, H / 2 + 150), 44, fill=(200, 200, 200), alpha=a, scale=pop)
        return cover > 0.95

    def draw_overlay(self, t, layer):
        shot = self.shot_at(t)
        covered = False
        hud = shot.hud and t >= 0.6
        if shot.overlay:
            shot.overlay(layer, t)
        if hud:
            day = self.day_at(t)
            ph = self.phase(t, shot) if shot.loc == "over" else (2.0 if shot.loc == "under" else 2.0)
            d = ImageDraw.Draw(layer)
            d.rounded_rectangle([34, 28, 380, 128], radius=18, fill=(0, 0, 0, 150))
            emo = "🌙" if 1.5 < ph < 2.55 else "☀️"
            if shot.loc == "nether":
                emo = "🔥"
            if shot.loc == "under":
                emo = "⛏️"
            ov.paste_emoji(layer, emo, (88, 78), 58)
            # pop when the day changes
            i = S.IDX[self.key_at(t)]
            pop = 1.0
            for k in S.KEYS[:i + 1][::-1]:
                if k in S.DAY_AT:
                    tc = self.C(k)
                    if t - tc < 0.4:
                        pop = 1.0 + 0.35 * np.sin(PI * clamp01((t - tc) / 0.4))
                    break
            ov.draw_text(layer, f"DAY {day}", (240, 78), 64, fill=(255, 225, 40), scale=pop, shadow=False)
            hp = self.hearts(t, shot)
            ov.draw_hearts(layer, hp, (232, 146), px=4, t=t, shake_amt=1.0 if hp <= 3 else 0.0)
            items, sel = self.hotbar(t, shot)
            if items:
                ov.draw_hotbar(layer, items, sel, (W / 2, H - 62), slot=66)
        for n in range(2, len(self.chapters) + 1):
            if self.draw_card(layer, t, n):
                covered = True
        if covered:
            return
        G, B, R, O, Y, Wh = (120, 240, 120), (110, 235, 255), (255, 80, 60), (255, 150, 30), (255, 225, 40), (255, 255, 255)
        for spec in shot.titles:
            t0, t1, txt = spec[0], spec[1], spec[2]
            kw = spec[3] if len(spec) > 3 else {}
            col = kw.get("color", Y)
            pos = kw.get("pos", (W / 2 + 90, 150))
            ov.title_pop(layer, txt, t, t0, t1, pos, kw.get("size", 100), fill=col, emoji=kw.get("emoji"),
                         emoji_side=kw.get("side", "right"), rot=kw.get("rot", 0.0))
        for (t0, t1, grid, result) in shot.crafts:
            if t0 <= t < t1:
                ap = min(seg(t, t0, t0 + 0.3), seg(t1 - t, 0, 0.25) * 1.0)
                ov.crafting_grid(layer, (W * 0.72, H * 0.42), grid, result, appear=ap, cell=96)
        self.subs.draw(layer, t)

    def hotbar(self, t, shot):
        if shot.hotbar is not None:
            v = shot.hotbar(t) if callable(shot.hotbar) else shot.hotbar
            if v is None:
                return [], 0
            if isinstance(v, tuple):
                return v
            return v, 0
        ch = self.chapter_at(t)
        bars = {
            1: ["e:⛏️", "e:🗡️", "cobble", "planks", "crafting"],
            2: ["e:🗡️", "e:⛏️", "dirt", "cobble", "planks", "crafting"],
            3: ["e:🗡️", "e:⛏️", "e:🪓", "planks", "log", "e:🛏️", "e:🍎"],
            4: ["e:🗡️", "e:⛏️", "e:🌾", "e:🍞", "planks", "e:🪣", "e:🛏️"],
            5: ["e:🗡️", "e:⛏️", "e:🔥", "coal_ore", "iron_ore", "t:ingot", "e:🍞", "furnace"],
            6: ["e:🗡️", "e:⛏️", "t:bone", "e:🍞", "e:🥩", "t:ingot"],
            7: ["e:🗡️", "e:⛏️", "e:🌾", "e:🍞", "e:🥩", "t:bone"],
            8: ["e:🗡️", "e:⛏️", "cobble", "log", "glass", "e:🔥", "e:🍞"],
            9: ["e:🗡️", "e:⛏️", "e:🪣", "e:🔥", "cobble", "e:🍞", "e:💎"],
            10: ["e:🗡️", "e:⛏️", "e:🛶", "e:🌾", "t:emerald", "e:🍞", "e:💎"],
            11: ["e:🗡️", "e:⛏️", "t:emerald", "e:🍞", "e:💎", "e:🔥"],
            12: ["e:🗡️", "e:⛏️", "stone_bricks", "cobble", "glass", "e:🔥", "e:🍞"],
            13: ["e:🗡️", "e:⛏️", "e:🪣", "obsidian", "t:flint_steel", "e:🍞", "e:💎"],
            14: ["e:🗡️", "e:⛏️", "cobble", "e:🍞", "e:🥩", "obsidian"],
            15: ["e:🗡️", "e:⛏️", "cobble", "glowstone", "e:🍞", "e:🥩"],
            16: ["e:🗡️", "e:⛏️", "e:💎", "enchant", "e:📖", "e:🍞"],
            17: ["e:🗡️", "e:⛏️", "stone_bricks", "glass", "e:🪣", "e:🔥", "e:🍞"],
            18: ["e:🗡️", "e:⛏️", "stone_bricks", "e:🍞", "e:🥩", "e:🪣"],
            19: ["e:🗡️", "e:⛏️", "stone_bricks", "e:🌱", "e:🌾", "e:🍞"],
            20: ["e:🗡️", "e:⛏️", "stone_bricks", "e:🌾", "e:🍞", "e:💎"],
        }
        items = bars.get(ch, [])
        sel = 0
        pip = next((a for a in shot.actors if a.kind == "pip"), None)
        if pip is not None:
            it = E._sched(pip.item, t)
            if it and str(it).startswith("pickaxe"):
                sel = 1
        return items, sel

    # ------------------------------------------------------------ audio
    def sfx_events(self):
        ev = [(0.0, "whoosh", -10, dict(dur=0.8)), (0.3, "pop", -12, {})]
        for n in range(2, len(self.chapters) + 1):
            c0, c1 = self.card(n)
            ev += [(c0 - 0.3, "whoosh", -8, dict(dur=0.7)), (c0 + 0.2, "impact", -10, {}),
                   (c1 - 0.4, "whoosh", -11, dict(dur=0.5, up=False))]
        for k in S.DAY_AT:
            if k != "spawn":
                ev.append((self.C(k) + 0.05, "ding", -17, dict(base=1320)))
        for s in self.shots:
            ev += list(s.sfx_list)
            for p in s.props:
                ev += [e for e in p.sfx() if s.t0 - 0.2 <= e[0] < s.t1 + 0.5]
            for a in s.actors:
                for h in a.hits:
                    if s.t0 <= h < s.t1:
                        ev.append((h + 0.05, "hurt", -10, dict(seed=int(h * 10) % 50)))
                if a.death is not None and s.t0 <= a.death < s.t1 and a.kind != "pip":
                    ev.append((a.death, "pop", -10, dict(pitch=0.7)))
                for aa in a.acts:
                    if s.t0 <= aa < s.t1:
                        ev.append((aa + 0.08, "whoosh", -22, dict(dur=0.15)))
        return ev

    def music_sections(self):
        out = []
        cur = None
        for s in self.shots:
            m = s.music
            if m is None:
                continue
            if cur and cur[0] == m:
                cur[2] = s.t1
            else:
                if cur:
                    out.append(tuple(cur))
                cur = [m, s.t0, s.t1]
        if cur:
            out.append(tuple(cur))
        # fill gaps: a section lasts until the next starts
        fixed = []
        for i, (m, a, b) in enumerate(out):
            nb = out[i + 1][1] if i + 1 < len(out) else self.duration
            fixed.append((m, a, nb))
        return fixed
