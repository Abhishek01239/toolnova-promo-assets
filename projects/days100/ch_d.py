"""Chapters 16-20: diamond armor, the castle, the horde, home, day 100."""
import numpy as np

import engine as E
import world as WD
from shots import PI, GY, Y0, U, FLOOR, G, B, R, O, Y, Wh, PU, TOP
from ch_a import timelapse, circle_keys, ug, CAVE_TORCHES, HC
from ch_b import cave_torches, boat_keys, VILLAGERS
from ch_c import fireworks, stats_panel, explosion_fx
from mcengine.models import block_entity_vertices
from mcengine import models100 as M
from mcengine import overlay as ov

F = FLOOR
BC = np.array([12.0, GY + 1.0, 14.0])        # castle centre
S = WD.SPAWN


def diamond_counter(t_hits, base=0, goal=24, until=None):
    def fn(layer, t):
        if until is not None and t > until:
            return
        n = base + sum(1 for tt in t_hits if t >= tt)
        n = min(n, goal)
        from PIL import ImageDraw
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle([1540, 250, 1880, 335], radius=16, fill=(0, 0, 0, 150))
        ov.paste_emoji(layer, "💎", (1600, 292), 56)
        ov.draw_text(layer, f"{n} / {goal}", (1750, 292), 58, fill=(110, 235, 255) if n < goal else (120, 240, 120), shadow=False)
    return fn


def ch16(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    # 1 goal
    pip1 = x.pip([(0, (10.5, 5.0))], skin="pip_iron", item="pickaxe:diamond", yaw=PI)
    out.append(E.Shot("goal24", E.cam_front(pip1, dist=3.4, up=0.5, side=1.0, fov=46), stage="P5", phase=0.0, actors=[pip1],
                      overlay=diamond_counter([], 0), music="adventure",
                      titles=[x.title(Wd("goal24", "armor"), Ee("goal24"), "DIAMOND ARMOR", color=B, emoji="🛡️")]))
    # 2-3 strip mining in the side tunnel
    tx_ = 70
    ores = []
    for i, zz in enumerate(range(-8, -19, -1)):
        side = 69 if i % 2 == 0 else 73
        ores.append((side, F + (i % 2), zz))
    t0, t1 = T("mining", 0.1), Ee("strip") - 0.5
    tb = x.every(t0, t1, len(ores))
    keys = [(C("mining"), ug(71.5, F, -6.0))] + [(tb[i] - 0.4, ug(71.5, F, ores[i][2] + 0.5)) for i in range(len(ores))]
    pip2 = x.pip(keys, skin="pip_iron", item="pickaxe:diamond", acts=[t - 0.15 for t in tb] + [t - 0.4 for t in tb],
                 yaw=lambda t: None)
    br = E.Breaks([(tb[i], ug(*ores[i]), "deep_diamond") for i in range(len(ores))], dur=0.4, collect=lambda t: pip2.pos(t), drop="diamond")
    hits = []
    for i, tt in enumerate(tb):
        hits += [tt + 0.6, tt + 0.7]
    tor = E.Torches([(0, ug(71.5, F, -7.0)), (0, ug(71.5, F, -12.0)), (0, ug(71.5, F, -17.0))])
    out.append(E.Shot("mining", E.cam_follow(pip2, back=3.6, up=1.6, side=0.0, look=(0, 0.9, 0), fov=60, look_ahead=2.0), loc="under",
                      actors=[pip2], props=[br, tor], overlay=diamond_counter(hits[:22], 0), cave=1.0,
                      titles=[x.title(Wd("mining", "tunnel"), Ee("mining"), "TUNNEL AFTER TUNNEL", color=Wh, emoji="⛏️", size=86)]))
    out.append(E.Shot("strip", E.cam_static(ug(71.5, F + 2.6, -4.0), ug(71.5, F + 0.6, -16.0), fov=58), loc="under", actors=[pip2],
                      props=[br, tor], overlay=diamond_counter(hits[:22], 0), cave=1.0,
                      titles=[x.title(Wd("strip", "strip"), Wd("strip", "Long"), "STRIP MINING", color=Y, emoji="💡"),
                              x.title(Wd("strip", "two"), Ee("strip"), "2 TALL • DEEP", color=Wh, emoji="📏")]))
    # 4 found 24
    pip4 = x.pip([(0, ug(71.5, F, -17.0))], skin="pip_iron", item="pickaxe:diamond", yaw=0.0,
                 pose=lambda t: {"arms_up": 0.8, "flail_phase": t * 3})
    fly = E.Fly([(C("found24") + 0.12 * i, C("found24") + 0.12 * i + 0.7, ug(71.5 + 1.5 * np.cos(i), F + 2.5, -17.0 + 1.5 * np.sin(i)),
                  ug(71.5, F + 1.2, -17.0), "diamond") for i in range(12)])
    out.append(E.Shot("found24", E.cam_front(pip4, dist=3.2, up=0.4, side=0.6, fov=50), loc="under", actors=[pip4], props=[fly, tor],
                      overlay=diamond_counter([], 24), cave=1.0, music="resolve",
                      titles=[x.title(Wd("found24", "Twenty-four"), Ee("found24"), "24 DIAMONDS!", color=B, emoji="💎", size=120)],
                      sfx=[(C("found24") + 0.12 * i + 0.7, "pop", -13, dict(pitch=1.2 + 0.03 * i)) for i in range(12)] +
                          [(Wd("found24", "Twenty-four"), "shimmer", -6, {})]))
    # 5 craft the armor
    words = ["Helmet", "Chestplate", "Leggings", "Boots"]
    tw = [Wd("craftarmor", w) for w in words]
    tfull = Wd("craftarmor", "armor")
    pip5 = x.pip([(0, (10.5, 4.5))], skin=[(0, "pip_iron"), (tw[3], "pip_dia")], item="sword:iron", yaw=PI,
                 pose=lambda t: {"arms_up": 0.7 * E.smooth(E.seg(t, tfull, tfull + 0.3)), "flail_phase": 0.0})
    out.append(E.Shot("craftarmor", E.cam_orbit(np.array([10.5, GY + 1.0, 4.5]), 3.8, 1.6, PI - 0.9, PI + 0.9, C("craftarmor"), Ee("craftarmor"),
                                                fov=48, look_y=1.1), stage="P5", phase=0.0, actors=[pip5], music="build",
                      fx=[E.Burst(tw[3], (10.5, GY + 2.2, 4.5), 60, speed=(1, 3), life=(0.6, 1.2), size=(0.06, 0.14),
                                  colors=((0.5, 1, 1, 1), (1, 1, 1, 1)), gravity=-1, drag=2, up_bias=0.5, seed=1600, additive=True)],
                      titles=[x.title(tw[i], tw[i + 1] if i < 3 else tfull, w.upper(), color=B, emoji="💎") for i, w in enumerate(words)] +
                             [x.title(tfull, Ee("craftarmor"), "FULL DIAMOND ARMOR", color=B, emoji="🛡️", size=96)],
                      sfx=[(tt, "place", -8, dict(seed=i)) for i, tt in enumerate(tw)] + [(tw[3], "shimmer", -5, {})]))
    # 6 enchanting table
    ep = np.array([7.5, GY + 1.5, 4.5])
    te = T("enchant", 0.65)
    glyphs = [E.Burst(te + 0.2 * k, ep + (np.cos(k) * 2.0, 1.2, np.sin(k) * 2.0), 3, speed=(0.1, 0.3), life=(0.8, 1.2), size=(0.06, 0.1),
                      colors=((0.7, 0.4, 1, 1), (0.9, 0.8, 1, 1)), gravity=0.0, drag=0.5, seed=1610 + k, additive=True,
                      vel=(-np.cos(k) * 1.6, -0.6, -np.sin(k) * 1.6)) for k in range(int((Ee("glow") - te) / 0.2))]
    table = E.Custom(lambda t: [block_entity_vertices("enchant", ep, max(0.05, E.ease_out_back(E.seg(t, te, te + 0.2))))] if t > te else None,
                     sfxlist=[(te, "place", -8, {}), (te + 0.2, "shimmer", -10, {})])
    pip6 = x.pip([(0, (9.6, 4.5))], skin="pip_dia", item="sword", yaw=-PI / 2)
    out.append(E.Shot("enchant", E.cam_static((9.5, GY + 2.6, 1.0), (8.2, GY + 1.0, 4.6), fov=52), stage="P5", phase=0.0, actors=[pip6],
                      props=[table], fx=glyphs,
                      crafts=[(T("enchant", 0.0), te, [None, "e:📖", None, "e:💎", "obsidian", "e:💎", "obsidian", "obsidian", "obsidian"], "enchant")],
                      titles=[x.title(te, Ee("enchant"), "ENCHANTING TABLE", color=PU, emoji="✨", size=90)]))
    # 7 glowing sword
    tg = T("glow", 0.1)
    pip7 = x.pip([(0, (9.6, 4.5))], skin="pip_dia", item=[(0, "sword"), (tg, "sword:ench")], yaw=PI * 0.8,
                 pose=lambda t: {"arms_up": 0.6 * E.smooth(E.seg(t, tg, tg + 0.3)), "flail_phase": 0.0})
    out.append(E.Shot("glow", E.cam_front(pip7, dist=2.8, up=0.5, side=-0.8, fov=46, look=(0, 1.6, 0)), stage="P5", phase=0.0, actors=[pip7],
                      props=[table], fx=glyphs,
                      titles=[x.title(Wd("glow", "Sharpness"), Ee("glow"), "SHARPNESS III", color=PU, emoji="✨", size=110)],
                      sfx=[(tg, "shimmer", -6, {})]))
    # 8 test: one hit, flies back
    th = Wd("test", "hit") + 0.1
    zp0 = np.array([4.0, f.ground(4.0, -8.0), -8.0])
    def ztrack(t):
        return None
    zk = [(C("test"), (-2.0, -12.0)), (th, (3.0, -6.4)), (th + 0.6, (0.0, GY + 2.5, -11.0)), (th + 1.0, (-1.5, -13.0))]
    zom = x.mob("zombie", zk, death=th + 1.1, hits=[th])
    pip8 = x.pip([(0, (4.0, -4.8))], skin="pip_dia", item="sword:ench", yaw=PI + 0.35, acts=[th - 0.1])
    out.append(E.Shot("test", E.cam_static((8.0, GY + 2.4, -4.0), (1.5, GY + 1.4, -8.5), fov=52), stage="P5", phase=1.2, actors=[zom, pip8],
                      titles=[x.title(th, Ee("test"), "ONE HIT!", color=PU, emoji="💥", size=120)],
                      sfx=[(th, "impact", -5, {}), (th, "whoosh", -8, dict(dur=0.5))]))
    # 9 day 70 at the old hut
    pip9 = x.pip([(0, (S[0] + 3.4, S[2] - 0.5))], skin="pip_dia", item="sword:ench", yaw=PI / 2 + 0.3)
    out.append(E.Shot("day70", E.cam_static((S[0] + 7.5, GY + 1.8, S[2] - 3.0), (S[0] + 1.5, GY + 1.6, S[2] + 0.2), fov=50), stage="P5",
                      phase=0.95, statics=["hut_open"], actors=[pip9],
                      titles=[x.title(Wd("day70", "anymore"), Ee("day70"), "NOT ANYMORE", color=Y, emoji="😎", size=110)]))
    return out


def castle_parts():
    w = WD.VoxelWorld()
    WD.castle(w)
    x0, x1, z0, z1 = WD.CASTLE
    kx0, kx1, kz0, kz1 = WD.KEEP
    walls, towers, keep, water, bridge = [], [], [], [], []
    for p, b in w.blocks.items():
        xx, yy, zz = p
        if b == "water":
            water.append((p, b))
            continue
        if b == "planks" and yy == GY:
            bridge.append((p, b))
            continue
        if yy <= GY:
            continue
        if kx0 <= xx < kx1 and kz0 <= zz < kz1:
            keep.append((p, b))
        elif any(abs(xx - cx) <= 2 and abs(zz - cz) <= 2 for cx, cz in ((x0, z0), (x1, z0), (x0, z1), (x1, z1))):
            towers.append((p, b))
        else:
            walls.append((p, b))
    walls.sort(key=lambda kv: (np.arctan2(kv[0][2] - 14.0, kv[0][0] - 12.0), kv[0][1]))
    towers.sort(key=lambda kv: (kv[0][1], np.arctan2(kv[0][2] - 14.0, kv[0][0] - 12.0)))
    keep.sort(key=lambda kv: (kv[0][1], np.arctan2(kv[0][2] - 14.0, kv[0][0] - 12.0)))
    water.sort(key=lambda kv: np.arctan2(kv[0][2] - 14.0, kv[0][0] - 12.0))
    return walls, towers, keep, water, bridge


def ch17(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    walls, towers, keep, water, bridge = castle_parts()
    # 1 project
    pip1 = x.pip([(0, (9.0, GY + 7.0, 9.5))], skin="pip_dia", item="sword:ench", yaw=PI * 0.8)
    bis1 = x.biscuit([(0, (10.2, GY + 7.0, 9.2))], yaw=PI * 0.8, sit=1.0)
    out.append(E.Shot("project", E.cam_static((15.0, GY + 11.5, 17.0), (6.0, GY + 3.0, -2.0), fov=58), stage="P5", phase=0.0, actors=[pip1, bis1],
                      music="build", titles=[x.title(Wd("project", "castle"), Ee("project"), "A CASTLE", color=Y, emoji="🏰", size=130)]))
    # 2 walls
    tw0, tw1 = T("walls", 0.1), Ee("walls") - 0.4
    rw = E.Reveal("cwalls", walls, tw0, tw1, K=40)
    rw_done = E.Reveal("cwalls", walls, -100, -99, K=40)
    tt0, tt1 = T("towers", 0.05), Ee("bricks") - 0.4
    rt = E.Reveal("ctowers", towers, tt0, tt1, K=40)
    rt_done = E.Reveal("ctowers", towers, -100, -99, K=40)
    tk0, tk1 = T("keep", 0.05), Ee("keep") - 0.3
    rk = E.Reveal("ckeep", keep, tk0, tk1, K=40)
    rk_done = E.Reveal("ckeep", keep, -100, -99, K=40)
    tm0, tm1 = T("moat", 0.2), Ee("moat") - 0.3
    rm = E.Reveal("cmoat", water, tm0, tm1, K=24)
    rm_done = E.Reveal("cmoat", water, -100, -99, K=24)
    pipw = x.pip(circle_keys((12.0, 14.0), 17.5, C("walls"), Ee("walls"), turns=1.0, a0=-PI / 2), skin="pip_dia", item="stone_bricks",
                 acts=x.every(tw0, tw1, int((tw1 - tw0) / 0.3)))
    out.append(E.Shot("walls", E.cam_orbit(BC, 40.0, 22.0, -0.5, 0.4, C("walls"), Ee("walls"), fov=50), stage="P6pre",
                      phase=timelapse(tw0, tw1, 1.0, 0.0), actors=[pipw], props=[rw],
                      titles=[x.title(Wd("walls", "Thirty"), Wd("walls", "battlements"), "30 x 30", color=Wh, emoji="📐"),
                              x.title(Wd("walls", "battlements"), Ee("walls"), "BATTLEMENTS", color=Y, emoji="🏰")]))
    tower_c = [(-3, -1), (27, -1), (-3, 29), (27, 29)]
    kt = []
    for i, (cx, cz) in enumerate(tower_c):
        a = tt0 + (tt1 - tt0) * i / 4
        b = tt0 + (tt1 - tt0) * (i + 1) / 4
        kt += [(a, (cx + 0.5, GY + 1 + 2 * 0, cz - 2.8 if cz < 10 else cz + 2.8)), (b - 0.2, (cx + 0.5, GY + 12.0, cz + 0.5))]
    pipt = x.pip(kt, skin="pip_dia", item="stone_bricks", acts=x.every(tt0, tt1, int((tt1 - tt0) / 0.3)))
    out.append(E.Shot("towers", E.cam_orbit(BC, 42.0, 18.0, 0.4, 1.4, C("towers"), Ee("towers"), fov=52), stage="P6pre",
                      phase=timelapse(tt0, tt1, 1.0, 0.0), actors=[pipt], props=[rw_done, rt],
                      titles=[x.title(Wd("towers", "four"), Ee("towers"), "4 TOWERS", color=Wh, emoji="🗼")]))
    out.append(E.Shot("bricks", E.cam_static((32.0, GY + 9.0, 35.0), (27.0, GY + 6.0, 29.0), fov=52), stage="P6pre",
                      phase=0.0, actors=[pipt], props=[rw_done, rt],
                      titles=[x.title(Wd("bricks", "stone"), Ee("bricks"), "STONE BRICKS", color=(210, 210, 210), emoji="🧱")]))
    pipk = x.pip(circle_keys((12.0, 14.0), 7.5, C("keep"), Ee("keep"), turns=1.2, a0=-PI / 2), skin="pip_dia", item="stone_bricks",
                 acts=x.every(tk0, tk1, int((tk1 - tk0) / 0.3)))
    bisk = x.biscuit([(0, (12.0, 4.0))], yaw=0.0, sit=1.0)
    out.append(E.Shot("keep", E.cam_orbit(BC + (0, 4, 0), 24.0, 10.0, 1.2, 2.0, C("keep"), Ee("keep"), fov=52), stage="P6pre",
                      phase=timelapse(tk0, tk1, 0.8, 0.0), actors=[pipk, bisk], props=[rw_done, rt_done, rk],
                      titles=[x.title(Wd("keep", "keep"), Wd("keep", "Biscuit"), "THE KEEP", color=Y, emoji="🏯"),
                              x.title(Wd("keep", "Biscuit"), Ee("keep"), "+ BISCUIT'S SPOT", color=(255, 200, 120), emoji="🐶", size=86)]))
    pipm = x.pip([(C("moat"), (12.5, -4.0)), (Ee("moat"), (-6.0, 10.0))], skin="pip_dia", item="water_bucket")
    out.append(E.Shot("moat", E.cam_static((-14.0, GY + 9.0, -14.0), (2.0, GY + 1.0, 4.0), fov=55), stage="P6pre", phase=0.0, actors=[pipm],
                      props=[rw_done, rt_done, rk_done, rm],
                      titles=[x.title(Wd("moat", "moat"), Ee("moat"), "MOAT", color=B, emoji="🌊", size=120)],
                      sfx=[(T("moat", 0.3) + 0.6 * k, "water", -14, dict(dur=0.8)) for k in range(6)]))
    tbr = x.every(T("bridge", 0.1), Ee("bridge") - 0.4, len(bridge))
    bridge.sort(key=lambda kv: (-kv[0][2], kv[0][0]))
    pops = E.Pops([(tbr[i], p, b) for i, (p, b) in enumerate(bridge)], scale=1.0)
    pipbr = x.pip([(C("bridge"), (12.5, 1.5)), (Ee("bridge"), (12.5, -1.6))], skin="pip_dia", item="planks", acts=tbr)
    out.append(E.Shot("bridge", E.cam_static((17.0, GY + 4.0, -9.0), (12.5, GY + 1.5, -1.0), fov=52), stage="P6pre", phase=0.0, actors=[pipbr],
                      props=[rw_done, rt_done, rk_done, rm_done, pops],
                      titles=[x.title(Wd("bridge", "only"), Ee("bridge"), "THE ONLY WAY IN", color=Y, emoji="🌉", size=96)]))
    # 8 inside: tour of the courtyard
    cows = [x.mob("cow", [(0, (1.0 + 1.6 * i, 22.0 + 1.5 * (i % 2)))], yaw=0.7 * i) for i in range(3)]
    pipi = x.pip([(C("inside"), (12.5, 2.0)), (Ee("inside"), (16.5, 8.0))], skin="pip_dia", item="sword:ench")
    bisi = x.follow(pipi, C("inside"), Ee("inside"))
    out.append(E.Shot("inside", E.cam_dolly(C("inside"), Ee("inside"), (12.0, GY + 26.0, -8.0), (12.0, GY + 26.0, 30.0), (12.0, GY + 1.0, 8.0),
                                            (10.0, GY + 1.0, 22.0), fov=58), stage="P6", phase=0.0, actors=cows + [pipi, bisi],
                      titles=[x.title(Wd("inside", "farm"), Ee("inside"), "SAFE BEHIND STONE", color=G, emoji="🛡️", size=90)]))
    # 9 ten days: time-lapse
    out.append(E.Shot("tendays", E.cam_orbit(BC, 48.0, 26.0, 2.0, 2.8, C("tendays"), Ee("tendays"), fov=50), stage="P6",
                      phase=timelapse(C("tendays"), Ee("tendays"), 2.0, 0.0), actors=cows,
                      titles=[x.title(Wd("tendays", "Thousands"), Ee("tendays"), "THOUSANDS OF BLOCKS", color=Wh, emoji="🧱", size=86)]))
    # 10 reveal
    out.append(E.Shot("reveal", E.cam_orbit(BC, 36.0, 10.0, -2.6, -1.6, C("reveal"), Ee("reveal"), fov=55, r1=48.0, h1=24.0), stage="P6",
                      phase=1.0, actors=cows, fx=fireworks(T("reveal", 0.6), 6, BC, seed=1700, spread=16), music="resolve",
                      titles=[x.title(Wd("reveal", "castle"), Ee("reveal"), "MY CASTLE", color=Y, emoji="🏰", size=150)],
                      sfx=[(T("reveal", 0.6) + 0.45 * k, "explosion", -18, dict(seed=k, dur=0.8)) for k in range(6)]))
    return out


def ring_pts(n, r0, r1, seed, a0=0.0, a1=2 * PI):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        a = rng.uniform(a0, a1)
        out.append(((BC[0] + r0 * np.cos(a) + rng.uniform(-3, 3), BC[2] + r0 * np.sin(a) + rng.uniform(-3, 3)),
                    (BC[0] + r1 * np.cos(a), BC[2] + r1 * np.sin(a))))
    return out


def ch18(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    night = 2.0
    redt = dict(tint=(1.08, 0.92, 0.92))
    x0, x1, z0, z1 = WD.CASTLE
    # 1 night 85
    out.append(E.Shot("night85", E.cam_static((-12.0, GY + 2.0, 34.0), (-12.0 + 24.0, GY + 2.0 + 19.0, 34.0 - 30.0), fov=64), stage="P6", phase=night, post=redt,
                      music="suspense", titles=[x.title(Wd("night85", "wrong"), Ee("night85"), "SOMETHING'S WRONG", color=R, emoji="🌕", size=96)],
                      sfx=[(T("night85", 0.6), "heartbeat", -8, {}), (T("night85", 0.85), "heartbeat", -9, dict(seed=1))]))
    # 2 the horde approaches
    zs = [x.mob("zombie", [(C("horde"), a), (Ee("held"), b)]) for (a, b) in ring_pts(26, 55, 21, 18)]
    sps = [x.mob("spider", [(C("horde"), a), (Ee("held"), b)]) for (a, b) in ring_pts(6, 52, 21, 19)]
    cps = [x.mob("creeper", [(C("horde"), a), (Ee("held"), b)]) for (a, b) in ring_pts(4, 50, 22, 20)]
    out.append(E.Shot("horde", E.cam_orbit(BC, 46.0, 30.0, -0.6, 0.1, C("horde"), Ee("horde"), fov=52), stage="P6", phase=night, post=redt,
                      actors=zs + sps + cps, music="panic",
                      titles=[x.title(Wd("horde", "Dozens"), Ee("horde"), "THE HORDE", color=R, emoji="🧟", size=130)],
                      sfx=[(T("horde", 0.1 + 0.15 * k), "groan", -10, dict(seed=100 + k)) for k in range(6)]))
    # 3 the walls held: zombies stuck in the moat
    moat_z = [x.mob("zombie", [(C("held"), (2.0 + 3.5 * i, -3.0)), (Ee("held"), (2.5 + 3.5 * i, -2.8))],
                    y_off=lambda t, i=i: -0.9 + 0.12 * np.sin(t * 3 + i)) for i in range(7)]
    piph = x.pip([(0, (8.5, GY + 7.0, -0.5))], skin="pip_dia", item="sword:ench", yaw=PI)
    bish = x.biscuit([(0, (10.5, GY + 7.0, -0.5))], yaw=PI, sit=0.0)
    out.append(E.Shot("held", E.cam_static((8.5, GY + 3.2, -11.5), (8.5, GY + 4.0, -1.0), fov=60), stage="P6", phase=night, post=redt,
                      actors=moat_z + [piph, bish] + zs[:8],
                      titles=[x.title(Wd("held", "held"), Ee("held"), "THE WALLS HELD", color=G, emoji="🛡️", size=100)]))
    # 4-6 spiders climb the east wall, fight on top
    wx = x1 + 1.0
    sp_t = [T("climb", 0.2 + 0.18 * i) for i in range(3)]
    tf = [Wd("fight", "One"), Wd("fight", "two"), Wd("fight", "three")]
    climbers = []
    for i in range(3):
        zz = 11.5 + 2.6 * i
        t_up = sp_t[i] + 1.6
        climbers.append(x.mob("spider", [(C("climb"), (wx + 4.0, zz)), (sp_t[i], (wx + 0.3, GY + 1.0, zz)), (t_up, (wx + 0.3, GY + 7.2, zz)),
                                         (t_up + 0.6, (x1 + 0.0, GY + 7.6, zz)), (tf[i] - 0.4, (x1 - 0.5, GY + 7.0, zz - 0.2))],
                              climb=lambda t, a=sp_t[i], b=t_up: E.smooth(E.seg(t, a - 0.2, a + 0.2)) * (1 - E.smooth(E.seg(t, b, b + 0.4))),
                              hits=[tf[i]], death=tf[i] + 0.25, yaw=lambda t, a=sp_t[i], b=t_up: -PI / 2 if t < b + 0.4 else None))
    out.append(E.Shot("climb", E.cam_static((wx + 9.0, GY + 3.0, 6.0), (wx, GY + 4.5, 14.0), fov=55), stage="P6", phase=night, post=redt,
                      actors=climbers + zs[8:14],
                      titles=[x.title(Wd("climb", "climb"), Ee("climb"), "SPIDERS CLIMB!", color=R, emoji="🕷️", size=100)],
                      sfx=[(t, "hiss", -12, dict(dur=0.8, seed=i)) for i, t in enumerate(sp_t)]))
    pipf = x.pip([(C("fight"), (x1 - 0.5, GY + 7.0, 9.0)), (Ee("biscuithurt"), (x1 - 0.5, GY + 7.0, 12.0))], skin="pip_dia", item="sword:ench",
                 acts=[t - 0.1 for t in tf], yaw=0.0)
    tj = Wd("biscuithurt", "jumped")
    tko = Wd("biscuithurt", "knocked")
    jumper = x.mob("spider", [(C("biscuithurt"), (x1 + 0.3, GY + 7.2, 17.5)), (tj, (x1 - 0.5, GY + 7.6, 15.6)), (tj + 0.3, (x1 - 0.5, GY + 7.0, 14.6)),
                              (tko, (x1 - 0.5, GY + 7.0, 14.6)), (tko + 0.5, (x1 + 2.5, GY + 4.0, 16.5))],
                   hits=[tko], death=tko + 0.5, y_off=lambda t: 0.8 * max(0.0, np.sin(PI * E.seg(t, tj - 0.1, tj + 0.3))))
    bisf = x.biscuit([(0, (x1 - 0.5, GY + 7.0, 13.6))], yaw=0.0, flash=lambda t: 0.5 if tj + 0.2 <= t < tj + 0.8 and (t * 8) % 1 < 0.5 else 0.0)
    out.append(E.Shot("fight", E.cam_static((x1 - 4.0, GY + 9.6, 7.5), (x1 - 0.5, GY + 7.5, 14.0), fov=55), stage="P6", phase=night, post=redt,
                      actors=[pipf, bisf] + climbers,
                      titles=[x.title(tf[0], tf[1], "1", color=Wh, size=140), x.title(tf[1], tf[2], "2", color=Wh, size=140),
                              x.title(tf[2], Ee("fight"), "3", color=Wh, size=140)]))
    out.append(E.Shot("biscuithurt", E.cam_static((x1 - 4.5, GY + 9.2, 10.5), (x1 - 0.5, GY + 7.6, 15.0), fov=52), stage="P6", phase=night,
                      post=redt, actors=[pipf, bisf, jumper],
                      titles=[x.title(tj + 0.2, Ee("biscuithurt"), "BISCUIT!", color=R, emoji="😱", size=110)],
                      sfx=[(tj + 0.2, "hurt", -8, dict(seed=7)), (tko, "impact", -6, {})]))
    # 7 the breach
    cbx, cbz = x1 + 1.3, 15.5
    tb = T("breach", 0.55)
    crp = x.mob("creeper", [(C("breach"), (x1 + 1.3, 23.0)), (T("breach", 0.3), (cbx, cbz))], swell=[(T("breach", 0.32), tb)], death=tb)
    bcen = np.array([x1 + 0.5, GY + 2.5, 15.5])
    out.append(E.Shot("breach", E.cam_static((x1 + 9.5, GY + 3.0, 6.5), (x1 + 0.5, GY + 2.5, 15.0), fov=55), stage="P6", phase=night, post=redt,
                      actors=[crp] + zs[14:20], fx=explosion_fx(tb, bcen, 1800, 1.5), shake=[(tb, 1.4, 0.6)], flash=[(tb, 0.9, 0.5)],
                      titles=[x.title(tb, Ee("breach"), "BOOM", color=O, emoji="💥", size=160)],
                      sfx=[(T("breach", 0.32), "hiss", -5, dict(dur=tb - T("breach", 0.32))), (tb, "explosion", -1, {})]))
    # 8-10 zombies pour through the gap; last stand
    gap = (x1 - 0.5, 15.5)
    pour = []
    for i in range(6):
        t_in = T("pour", 0.1) + 0.35 * i
        pour.append(x.mob("zombie", [(C("pour"), (x1 + 4.5 + 1.5 * (i % 2), 12.0 + 1.6 * i)), (t_in + 0.8, (x1 + 1.5, 15.0 + 0.3 * (i % 3))),
                                     (t_in + 2.2, (x1 - 3.5 - 1.2 * (i % 3), 14.0 + 1.0 * (i % 2)))]))
    out.append(E.Shot("pour", E.cam_static((x1 - 8.0, GY + 3.5, 11.0), (x1, GY + 2.0, 15.5), fov=55), stage="P7", phase=night, post=redt,
                      actors=pour, music="panic",
                      titles=[x.title(T("pour", 0.3), Ee("pour"), "THEY'RE IN!", color=R, emoji="🧟", size=110)]))
    tls = x.every(T("laststand", 0.1), Ee("creeper3") - 0.4, 7)
    stand_p = (x1 - 1.2, 15.5)
    line = []
    for i, th in enumerate(tls[:6]):
        line.append(x.mob("zombie", [(C("laststand"), (x1 + 6.0 + 2.0 * i, 15.0 + 0.8 * np.sin(i))), (th - 0.3, (x1 + 0.4, 15.3))],
                          hits=[th - 0.15, th], death=th + 0.2))
    tcr = tls[6]
    crp3 = x.mob("creeper", [(C("creeper3"), (x1 + 6.0, 19.0)), (tcr - 0.2, (x1 + 0.6, 16.2)), (tcr + 0.45, (x1 + 3.6, GY + 3.0, 16.6)),
                             (tcr + 0.9, (x1 + 3.8, GY - 1.2, 16.8))], hits=[tcr], hidden=lambda t: t > tcr + 1.0)
    pipl = x.pip([(0, stand_p)], skin="pip_dia", item="sword:ench", yaw=PI / 2,
                 acts=[t - 0.25 for t in tls[:6]] + [t - 0.1 for t in tls[:6]] + [tcr - 0.1])
    bisl = x.biscuit([(0, (x1 - 2.4, 14.2))], yaw=PI / 2)
    out.append(E.Shot("laststand", E.cam_static((x1 - 5.5, GY + 2.6, 13.0), (x1 + 2.0, GY + 1.5, 15.8), fov=55), stage="P7", phase=night,
                      post=redt, actors=[pipl, bisl] + line + [crp3],
                      titles=[x.title(Wd("laststand", "Swing"), Ee("laststand"), "SWING! SWING! SWING!", color=Y, emoji="⚔️", size=90)]))
    out.append(E.Shot("creeper3", E.cam_static((x1 - 2.0, GY + 3.2, 21.5), (x1 + 2.0, GY + 1.0, 16.0), fov=55), stage="P7", phase=night,
                      post=redt, actors=[pipl, bisl] + line + [crp3],
                      fx=[E.Burst(tcr + 0.9, (x1 + 3.8, GY, 16.8), 30, speed=(1, 4), life=(0.5, 1.0), size=(0.08, 0.16),
                                  colors=((0.6, 0.75, 1, 0.9),), gravity=-9, drag=1, up_bias=1.0, seed=1810)],
                      titles=[x.title(tcr + 0.5, Ee("creeper3"), "INTO THE MOAT!", color=B, emoji="🌊", size=96)],
                      sfx=[(tcr, "impact", -6, {}), (tcr + 0.9, "splash", -6, {})]))
    # 11 the sun burns them all
    tb11 = T("burnall", 0.3)
    burn = [x.mob("zombie", [(C("burnall"), a), (Ee("burnall"), b)], burn=(tb11 + 0.08 * k, tb11 + 1.8 + 0.1 * k), death=tb11 + 1.8 + 0.1 * k)
            for k, (a, b) in enumerate(ring_pts(10, 24, 21, 33))]
    out.append(E.Shot("burnall", E.cam_orbit(BC, 34.0, 16.0, 1.6, 2.1, C("burnall"), Ee("burnall"), fov=55), stage="P7",
                      phase=lambda t: 2.0 + 1.0 * E.smooth(E.seg(t, C("burnall"), tb11 + 0.4)), actors=burn, music="resolve",
                      titles=[x.title(Wd("burnall", "burned"), Ee("burnall"), "SUNRISE", color=O, emoji="🔥", size=120)],
                      sfx=[(tb11, "crackle", -8, dict(dur=3.0))]))
    # 12 survived
    out.append(E.Shot("survived", E.cam_orbit(BC, 44.0, 22.0, 2.2, 2.6, C("survived"), Ee("survived"), fov=50), stage="P7", phase=3.8,
                      titles=[x.title(Wd("survived", "survived"), Ee("survived"), "WE SURVIVED", color=G, emoji="✅", size=120)]))
    return out


def ch19(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    x0, x1, z0, z1 = WD.CASTLE
    # 1 repair
    full = WD.stage_world("P6").blocks
    br_ = WD.stage_world("P7").blocks
    patch = sorted([(p, b) for p, b in full.items() if br_.get(p) != b and b != "water"], key=lambda kv: (kv[0][1], kv[0][2]))
    t0, t1 = T("repair", 0.1), Ee("repair") - 0.3
    rv = E.Reveal("repair", patch, t0, t1, K=20)
    pipr = x.pip([(0, (x1 - 1.6, 15.5))], skin="pip_dia", item="stone_bricks", yaw=PI / 2, acts=x.every(t0, t1, 12))
    out.append(E.Shot("repair", E.cam_static((x1 - 7.0, GY + 4.0, 10.0), (x1, GY + 2.5, 15.5), fov=52), stage="P7",
                      phase=timelapse(t0, t1, 1.0, 0.0), actors=[pipr], props=[rv], music="build",
                      titles=[x.title(T("repair", 0.3), Ee("repair"), "REPAIRS", color=(210, 210, 210), emoji="🧱")]))
    # 2 garden: trees grow, baby cows
    g8 = WD.stage_world("P8").blocks
    g6 = full
    trees = sorted([(p, b) for p, b in g8.items() if g6.get(p) != b], key=lambda kv: kv[0][1])
    tg0, tg1 = T("garden", 0.1), T("garden", 0.6)
    rtree = E.Reveal("garden", trees, tg0, tg1, K=16)
    cows = [x.mob("cow", [(0, (1.0 + 1.6 * i, 22.0 + 1.5 * (i % 2)))], yaw=0.7 * i) for i in range(3)]
    babies = [x.mob("baby_cow", [(T("garden", 0.75) + 0.3 * i, (2.0 + 1.4 * i, 24.5))], spawn=T("garden", 0.75) + 0.3 * i, yaw=0.4 * i) for i in range(2)]
    out.append(E.Shot("garden", E.cam_static((9.5, GY + 9.0, 1.0), (2.0, GY + 3.0, 14.0), fov=58), stage="P6", phase=0.0,
                      actors=cows + babies, props=[rtree],
                      fx=[E.poof(T("garden", 0.75) + 0.3 * i, (2.0 + 1.4 * i, GY + 1.5, 24.5), 1900 + i) for i in range(2)],
                      titles=[x.title(Wd("garden", "trees"), Wd("garden", "cows"), "TREES", color=G, emoji="🌳"),
                              x.title(Wd("garden", "cows"), Ee("garden"), "BABY COWS", color=Wh, emoji="🐮")]))
    # 3 visit the village
    vc = np.array([65.5, GY + 1.0, 144.0])
    vill = [x.mob("villager", [(0, (64.0 + 1.6 * i, 141.0))], yaw=PI, pose=lambda t, i=i: {"arms_up": 0.35 + 0.1 * np.sin(t * 6 + i), "flail_phase": t * 4})
            for i in range(4)]
    pipv = x.pip([(C("visit"), (66.0, 136.0)), (Ee("visit"), (66.0, 139.0))], skin="pip_dia", pose=lambda t: {"wave": 1.0, "wave_phase": t * 10})
    out.append(E.Shot("visit", E.cam_static((61.5, GY + 2.8, 134.5), (66.0, GY + 1.5, 140.0), fov=52), stage="P8", phase=0.0, actors=vill + [pipv],
                      titles=[x.title(Wd("visit", "remembered"), Ee("visit"), "THEY REMEMBERED!", color=Y, emoji="👋", size=96)]))
    # 4 walk back to spawn
    pipw = x.pip([(C("walkback"), (-9.0, 8.0)), (Ee("walkback") + 1.5, (S[0] + 5.0, S[2] - 0.5))], skin="pip_dia", item="sword:ench")
    bisw = x.follow(pipw, C("walkback"), Ee("walkback") + 1.5)
    out.append(E.Shot("walkback", E.cam_track(pipw, (3.5, 2.6, -4.5), look=(-1.0, 1.0, 1.0), fov=52), stage="P8", phase=0.95,
                      statics=["hut_open"], actors=[pipw, bisw], music="resolve",
                      titles=[x.title(Wd("walkback", "spawned"), Ee("walkback"), "WHERE IT ALL BEGAN", color=Y, emoji="📍", size=90)]))
    # 5 the old hut
    pipo = x.pip([(0, (S[0] + 4.2, S[2] - 0.5))], skin="pip_dia", item="sword:ench", yaw=-PI / 2)
    biso = x.biscuit([(0, (S[0] + 4.6, S[2] + 0.9))], yaw=-PI / 2, sit=1.0)
    out.append(E.Shot("oldhut", E.cam_static((S[0] + 8.0, GY + 2.4, S[2] - 4.0), (S[0] + 0.5, GY + 2.0, S[2] + 0.5), fov=50), stage="P8",
                      phase=1.0, statics=["hut_open"], actors=[pipo, biso],
                      titles=[x.title(Wd("oldhut", "hut"), Ee("oldhut"), "THE DIRT HUT", color=(200, 150, 100), emoji="🟫")]))
    # 6 memory: flashback to day 1
    x0h, x1h, z0h, z1h = WD.HUT
    eye = np.array([x0h + 2.5, GY + 2.55, z0h + 1.05])
    zf = x.mob("zombie", [(C("memory"), (x0h + 4.0, z0h - 4.0)), (Ee("memory"), (x0h + 1.0, z0h - 3.0))])
    out.append(E.Shot("memory", E.cam_static(eye, eye + (0.0, -0.25, -6.0), fov=58), stage="P0", phase=2.0, statics=["hut"], actors=[zf],
                      post=dict(saturation=0.25, tint=(1.12, 1.0, 0.82), vignette=0.6), hud=False, clouds=False,
                      titles=[x.title(T("memory", 0.1), Ee("memory"), "DAY 1", color=(230, 210, 170), emoji="🕰️", size=110)],
                      sfx=[(T("memory", 0.5), "groan", -14, dict(seed=190))]))
    # 7 sat inside
    pips = x.pip([(0, (S[0] - 0.2, GY + 0.45, S[2] + 0.6))], skin="pip_dia", yaw=PI * 0.75,
                 pose=lambda t: {"body_pitch": 0.0, "head_pitch": -0.15})
    biss = x.biscuit([(0, (S[0] + 1.0, S[2] - 0.6))], yaw=PI, sit=1.0)
    out.append(E.Shot("satinside", E.cam_static((x0h + 1.3, GY + 2.5, z1h - 1.3), (S[0] + 0.4, GY + 1.2, S[2] - 0.2), fov=64), stage="P8",
                      phase=1.0, statics=["hut_open"], actors=[pips, biss], clouds=False,
                      light=dict(ambient=(0.35, 0.32, 0.3), sun=(0.1, 0.08, 0.06)), fog=dict(color=(0.02, 0.015, 0.01), start=3.0, end=16.0),
                      titles=[x.title(Wd("satinside", "small"), Ee("satinside"), "SO SMALL NOW", color=Wh, emoji="🥲", size=90)]))
    # 8 leave it
    pipl = x.pip([(C("leaveit"), (S[0] + 4.2, S[2] - 0.5)), (Ee("leaveit") + 2.0, (S[0] + 16.0, S[2] + 1.0))], skin="pip_dia", item="sword:ench")
    bisl = x.follow(pipl, C("leaveit"), Ee("leaveit") + 2.0)
    out.append(E.Shot("leaveit", E.cam_static((S[0] - 3.5, GY + 3.5, S[2] - 5.5), (S[0] + 8.0, GY + 1.5, S[2] + 1.0), fov=55), stage="P8",
                      phase=1.1, statics=["hut_open"], actors=[pipl, bisl],
                      titles=[x.title(Wd("leaveit", "reminder"), Ee("leaveit"), "WHERE IT STARTED", color=Y, emoji="❤️", size=90)]))
    # 9 day 99
    out.append(E.Shot("day99", E.cam_orbit(BC, 40.0, 16.0, -2.0, -1.5, C("day99"), Ee("day99"), fov=52), stage="P8",
                      phase=lambda t: 1.3 + 0.7 * E.seg(t, C("day99"), Ee("day99")), music="suspense",
                      titles=[x.title(Wd("day99", "ninety-nine"), Ee("day99"), "ONE MORE NIGHT", color=Wh, emoji="🌙", size=100)]))
    return out


def ch20(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    kx0, kx1, kz0, kz1 = WD.KEEP
    ry = GY + 13.0                       # keep roof walk surface
    kp = (12.0, ry, kz0 + 1.6)
    pipk = x.pip([(0, kp)], skin="pip_dia", item=None, yaw=PI)
    bisk = x.biscuit([(0, (13.3, ry, kz0 + 1.7))], yaw=PI, sit=1.0)
    # 1 last night on the keep
    out.append(E.Shot("lastnight", E.cam_static((11.0, ry + 2.2, kz0 + 5.5), (13.0, ry - 3.0, -20.0), fov=58), stage="P8", phase=2.0,
                      actors=[pipk, bisk], music="resolve",
                      titles=[x.title(Wd("lastnight", "quiet"), Ee("lastnight"), "THE LAST NIGHT", color=Wh, emoji="🌌", size=96)]))
    # 2 sunrise on day 100
    tsun = Wd("sunrise100", "hundred")
    jump = lambda t: 0.7 * max(0.0, np.sin(PI * E.seg(t, tsun + 0.2, tsun + 0.6))) + 0.7 * max(0.0, np.sin(PI * E.seg(t, tsun + 0.7, tsun + 1.1)))
    pip2 = x.pip([(0, kp)], skin="pip_dia", yaw=PI, y_off=jump, pose=lambda t: {"arms_up": 0.9, "flail_phase": t * 5} if t > tsun else {})
    bis2 = x.biscuit([(0, (13.3, ry, kz0 + 1.7))], yaw=PI, y_off=lambda t: 0.5 * max(0.0, np.sin(PI * E.seg(t, tsun + 0.4, tsun + 0.8))))
    out.append(E.Shot("sunrise100", E.cam_static((12.5, ry + 1.6, kz0 - 3.2), (12.5, ry + 1.4, kz0 + 2.0), fov=55), stage="P8",
                      phase=lambda t: 2.0 + 1.9 * E.smooth(E.seg(t, C("sunrise100"), tsun + 0.5)), actors=[pip2, bis2],
                      fx=fireworks(tsun + 0.3, 8, (12.0, GY + 10.0, 10.0), seed=2000, spread=12),
                      titles=[x.title(tsun, Ee("sunrise100"), "DAY 100", color=Y, emoji="🎉", size=190)],
                      sfx=[(tsun, "impact", -5, {}), (tsun, "shimmer", -6, {})] + [(tsun + 0.3 + 0.45 * k, "explosion", -18, dict(seed=k, dur=0.8)) for k in range(8)]))
    # 3 look out over everything
    out.append(E.Shot("look100", E.cam_orbit(np.array([12.0, ry, kz0 + 1.6]), 7.0, 3.0, PI + 0.4, PI - 1.6, C("look100"), Ee("look100"), fov=62,
                                             look_y=-2.0, r1=11.0), stage="P8", phase=3.9, statics=["lighthouse_glow"], actors=[pipk, bisk],
                      titles=[x.title(Wd("look100", "castle"), Wd("look100", "farm"), "CASTLE", color=Y, emoji="🏰", size=80),
                              x.title(Wd("look100", "farm"), Wd("look100", "lighthouse"), "FARM", color=(230, 200, 90), emoji="🌾", size=80),
                              x.title(Wd("look100", "lighthouse"), Wd("look100", "river"), "LIGHTHOUSE", color=Y, emoji="🗼", size=80),
                              x.title(Wd("look100", "river"), Ee("look100"), "RIVER", color=B, emoji="🌊", size=80)]))
    # 4 recap montage, one cut per milestone word
    words = [("dirt", "hut"), ("house", "house"), ("stone", "stone"), ("village", "village"), ("Nether", "nether"), ("castle", "castle")]
    tws = [Wd("recap", w) - 0.15 for w, _ in words]
    starts = [C("recap")] + tws[1:]
    for i, (w, kind) in enumerate(words):
        st = starts[i]
        en = starts[i + 1] if i + 1 < len(words) else Ee("recap")
        if kind == "hut":
            sh = E.Shot(st, E.cam_dolly(st, en, (S[0] + 6.0, GY + 3.0, S[2] - 5.0), (S[0] + 4.5, GY + 2.5, S[2] - 4.0), (S[0], GY + 2.0, S[2]),
                                        (S[0], GY + 2.0, S[2]), fov=50), stage="P0", phase=1.8, statics=["hut"])
        elif kind == "house":
            sh = E.Shot(st, E.cam_orbit(HC, 16.0, 7.0, -0.4, 0.0, st, en, fov=50), stage="P2", phase=0.0)
        elif kind == "stone":
            sh = E.Shot(st, E.cam_orbit(BC, 34.0, 18.0, -2.4, -2.1, st, en, fov=50), stage="P4", phase=1.0)
        elif kind == "village":
            sh = E.Shot(st, E.cam_orbit(np.array([64.0, GY + 1.0, 138.0]), 30.0, 18.0, -1.0, -0.7, st, en, fov=50), stage="P4", phase=0.0)
        elif kind == "nether":
            sh = E.Shot(st, E.cam_dolly(st, en, x.ng(22, 34, 30), x.ng(18, 30, 24), x.ng(0, 16, 0), x.ng(0, 18, 0), fov=60), loc="nether")
        else:
            sh = E.Shot(st, E.cam_orbit(BC, 44.0, 22.0, -2.6, -2.2, st, en, fov=50), stage="P8", phase=1.0, statics=["lighthouse_glow"])
        sh.titles = [x.title(st + 0.1, en, {"hut": "DIRT HUT", "house": "HOUSE", "stone": "STONE BASE", "village": "VILLAGE SAVED",
                                            "nether": "2x NETHER", "castle": "CASTLE"}[kind], color=Y, size=110)]
        sh.sfx_list = [(st, "whoosh", -10, dict(dur=0.4))]
        out.append(sh)
    # 5 numbers
    lines = [(Wd("numbers", "Thousands"), "🧱", "1000s OF BLOCKS", Wh), (Wd("numbers", "thirty"), "💎", "30 DIAMONDS", B),
             (Wd("numbers", "ghast"), "👻", "1 GHAST", (240, 240, 240)), (Wd("numbers", "zero"), "💀", "0 DEATHS", R)]
    out.append(E.Shot("numbers", E.cam_orbit(BC, 46.0, 24.0, -2.2, -1.9, C("numbers"), Ee("numbers"), fov=50), stage="P8", phase=0.0,
                      statics=["lighthouse_glow"], overlay=stats_panel(lines)))
    # 6-7 best part / thanks Biscuit
    pip6 = x.pip([(0, kp)], skin="pip_dia", yaw=PI / 2 + 0.3, pose=lambda t: {"head_pitch": -0.25})
    bis6 = x.biscuit([(0, (13.3, ry, kz0 + 1.7))], yaw=-PI / 2 - 0.3, sit=1.0, pose=lambda t: {"head_pitch": 0.25})
    out.append(E.Shot("bestpart", E.cam_static((12.6, ry + 1.5, kz0 - 1.2), (12.7, ry + 1.0, kz0 + 1.7), fov=48), stage="P8", phase=3.95,
                      actors=[pip6, bis6], titles=[x.title(Wd("bestpart", "alone"), Ee("bestpart"), "NOT ALONE", color=(255, 200, 120), emoji="❤️")]))
    tp = Wd("thanks", "Biscuit")
    pip7 = x.pip([(0, kp)], skin="pip_dia", yaw=PI / 2 + 0.3, acts=[tp, tp + 0.5], pose=lambda t: {"head_pitch": -0.3})
    out.append(E.Shot("thanks", E.cam_static((14.9, ry + 1.4, kz0 + 3.6), (12.8, ry + 0.8, kz0 + 1.6), fov=50), stage="P8",
                      phase=3.95, actors=[pip7, bis6], fx=[E.hearts_burst(tp + 0.2, (13.3, ry + 1.0, kz0 + 1.7), 2010),
                                                          E.hearts_burst(tp + 0.7, (13.3, ry + 1.0, kz0 + 1.7), 2011)],
                      titles=[x.title(tp, Ee("thanks"), "THANKS, BISCUIT", color=(255, 200, 120), emoji="🐶", size=110)]))
    # 8 outro: wave and pull back over the whole world
    pip8 = x.pip([(0, kp)], skin="pip_dia", yaw=PI, pose=lambda t: {"wave": 1.0, "wave_phase": t * 10})
    bis8 = x.biscuit([(0, (13.3, ry, kz0 + 1.7))], yaw=PI, sit=1.0)
    def endcard(layer, t, t0=T("outro", 0.55)):
        if t < t0:
            return
        a = min(1.0, (t - t0) / 0.4)
        ov.title_pop(layer, "200 DAYS?", t, t0, 1e9, (960, 330), 170, fill=(255, 225, 40), emoji="🤔")
        if t > t0 + 0.6:
            ov.title_pop(layer, "COMMENT BELOW!", t, t0 + 0.6, 1e9, (960, 520), 90, fill=(255, 255, 255), emoji="💬")
    out.append(E.Shot("outro", E.cam_dolly(C("outro"), Ee("outro"), (12.5, ry + 1.8, kz0 - 3.5), (40.0, ry + 40.0, -60.0), (12.5, ry + 1.2, kz0 + 1.6),
                                           (12.0, GY + 5.0, 14.0), fov=52, fov1=55), stage="P8", phase=0.0, statics=["lighthouse_glow"],
                      actors=[pip8, bis8], overlay=endcard, hud=False, music="resolve"))
    return out
