"""Chapters 6-10: Biscuit, disaster, stone, diamonds, the village."""
import numpy as np

import engine as E
import world as WD
from shots import PI, GY, Y0, U, FLOOR, G, B, R, O, Y, Wh, PU, TOP
from ch_a import timelapse, circle_keys, ug, CAVE_TORCHES, HC
from mcengine.models import block_entity_vertices
from mcengine import models100 as M

F = FLOOR
INF = 1e9


def cave_torches():
    return E.Torches([(0, ug(31.5, F, 3.6))] + [(0, ug(tx_, F, tz)) for (tx_, tz) in CAVE_TORCHES])


def orange_overlay(f, t0, t1, peak=0.55, col="orange"):
    from PIL import Image

    def fn(layer, t):
        if t0 <= t < t1:
            k = (0.6 + 0.25 * np.sin(t * 14)) * (1 - E.seg(t, t1 - 0.6, t1))
            a = (f._orange if col == "orange" else f._red).copy()
            a[..., 3] = (a[..., 3] * k * peak).astype(np.uint8)
            layer.alpha_composite(Image.fromarray(a))
    return fn


def ch06(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    # 1 bones in the cave
    bp = [ug(55.0, F + 0.05, -1.0), ug(56.2, F + 0.05, 0.4), ug(54.6, F + 0.05, 1.2)]
    tp = x.every(T("bones", 0.35), Ee("bones") - 0.3, 3)
    pip1 = x.pip([(C("bones"), ug(51.0, F, 1.0)), (T("bones", 0.4), ug(54.0, F, 0.0))], skin="pip_iron", item="sword:iron",
                 pose=lambda t: {"head_pitch": -0.35})
    def bones(t):
        from mcengine.geometry import translate, rot_y, box_vertices
        v = [box_vertices((0.45, 0.04, 0.45), "bone", matrix=translate(*p) @ rot_y(0.7 * i)) for i, p in enumerate(bp) if t < tp[i]]
        return ([np.concatenate(v)] if v else []), [], []
    fly = E.Fly([(tp[i], tp[i] + 0.6, bp[i], pip1.pos(tp[i]) + (0, 1.0, 0), "bone") for i in range(3)])
    out.append(E.Shot("bones", E.cam_static(ug(57.5, F + 2.4, 2.8), ug(54.6, F + 0.4, -0.2), fov=52), loc="under", actors=[pip1],
                      props=[E.Custom(bones), fly, cave_torches()], music="adventure",
                      titles=[x.title(T("bones", 0.4), Ee("bones"), "BONES x3", color=Wh, emoji="🦴")],
                      sfx=[(t + 0.6, "pop", -12, dict(pitch=1.2)) for t in tp]))
    wx, wz = WD.WOLF_SPOT
    wolf_p = (wx, wz)
    # 2 the wolf
    tame_t = Wd("hearts", "hearts")
    wolf = x.mob("wolf", [(0, wolf_p)], yaw=PI / 2 + 0.3, tame=[(0, False), (tame_t, True)],
                 pose=lambda t: {"head_yaw": 0.3 * np.sin(t * 0.7)}, sit=[(0, 0.0)])
    pip2 = x.pip([(C("wolf"), (wx + 12, wz - 2)), (Ee("wolf"), (wx + 3.2, wz - 0.4))], skin="pip_iron", item="bone",
                 pose=lambda t: {"lean": 0.25})
    out.append(E.Shot("wolf", E.cam_static((wx - 2.5, GY + 1.5, wz + 3.0), (wx + 4.0, GY + 1.3, wz - 1.0), fov=50), stage="P2",
                      phase=0.0, actors=[wolf, pip2],
                      titles=[x.title(Wd("wolf", "wolf"), Ee("wolf"), "A WOLF", color=Wh, emoji="🐺")]))
    # 3 tame attempts
    pp = (wx + 2.2, wz - 0.2)
    f1, f2 = Wd("tame", "First") + 0.5, Wd("tame", "Second") + 0.5
    pip3 = x.pip([(0, pp)], skin="pip_iron", item="bone", yaw=-PI / 2 - 0.1, acts=[f1, f2])
    smoke = lambda tt, sd: E.Burst(tt + 0.2, (wx + 0.3, GY + 1.9, wz), 10, speed=(0.3, 1.0), life=(0.6, 1.0), size=(0.12, 0.22),
                                   colors=((0.35, 0.35, 0.35, 0.9),), gravity=1.0, drag=2, up_bias=0.8, seed=sd)
    out.append(E.Shot("tame", E.cam_static((wx - 1.4, GY + 1.9, wz - 3.6), (wx + 1.2, GY + 1.0, wz - 0.1), fov=50), stage="P2", phase=0.0,
                      actors=[wolf, pip3], fx=[smoke(f1, 1), smoke(f2, 2)],
                      titles=[x.title(f1 + 0.2, f2 - 0.3, "NOTHING", color=R, emoji="❌", size=90),
                              x.title(f2 + 0.2, Ee("tame"), "NOTHING", color=R, emoji="❌", size=90)]))
    # 4 hearts
    f3 = Wd("hearts", "Third") + 0.4
    pip4 = x.pip([(0, pp)], skin="pip_iron", item=[(0, "bone"), (f3 + 0.2, None)], yaw=-PI / 2 - 0.1, acts=[f3])
    out.append(E.Shot("hearts", E.cam_static((wx - 1.5, GY + 1.6, wz - 2.6), (wx + 0.8, GY + 1.0, wz), fov=46), stage="P2", phase=0.0,
                      actors=[wolf, pip4], fx=[E.hearts_burst(tame_t, (wx + 0.2, GY + 1.6, wz), 4),
                                              E.hearts_burst(tame_t + 0.4, (wx + 0.2, GY + 1.6, wz), 5)],
                      titles=[x.title(tame_t, Ee("hearts"), "TAMED!", color=(255, 120, 140), emoji="❤️")],
                      sfx=[(tame_t, "shimmer", -8, {}), (tame_t + 0.1, "ding", -9, {})]))
    # 5 name
    out.append(E.Shot("name", E.cam_front(wolf, dist=1.9, up=0.4, side=0.4, look=(0, 0.75, 0), fov=44, yaw=PI / 2 + 0.3),
                      stage="P2", phase=0.0, actors=[wolf],
                      titles=[x.title(Wd("name", "Biscuit"), Ee("name"), "BISCUIT", color=(255, 200, 120), emoji="🐶", size=140)],
                      sfx=[(Wd("name", "Biscuit"), "pop", -8, {})]))
    # 6 sit, follow, tilt
    ts = Wd("sit", "sat")
    tl = Wd("sit", "tilted")
    wolf6 = x.mob("wolf", [(C("sit"), wolf_p), (Wd("sit", "followed"), wolf_p), (Wd("sit", "followed") + 1.0, (wx + 1.2, wz - 0.3)),
                           (Ee("sit"), (wx + 1.2, wz - 0.3))], tame=True, yaw=PI / 2 + 0.2,
                  sit=lambda t: 1.0 if (ts <= t < Wd("sit", "followed")) or t > tl - 0.3 else 0.0,
                  pose=lambda t: {"head_yaw": 0.0, "head_pitch": 0.0} if t < tl else {"head_yaw": 0.0, "head_pitch": -0.1})
    from mcengine.geometry import rot_z
    pip6 = x.pip([(0, pp)], skin="pip_iron", yaw=-PI / 2 - 0.1)
    out.append(E.Shot("sit", E.cam_static((wx + 1.0, GY + 1.5, wz + 3.2), (wx + 1.0, GY + 0.8, wz - 0.2), fov=50), stage="P2",
                      phase=0.0, actors=[wolf6, pip6],
                      titles=[x.title(ts, Wd("sit", "followed"), "SIT", color=Wh, emoji="🐕", size=90),
                              x.title(tl, Ee("sit"), "*HEAD TILT*", color=Y, emoji="❓", size=90)]))
    # 7 follow home
    pip7 = x.pip([(C("follow"), (-17.0, 19.0)), (Ee("follow") + 2.0, (3.0, 9.0))], skin="pip_iron", item="sword:iron")
    bis7 = x.follow(pip7, C("follow"), Ee("follow") + 2.0)
    out.append(E.Shot("follow", E.cam_track(pip7, (2.0, 2.2, 6.5), look=(0.5, 0.9, 0.0), fov=50), stage="P2", phase=0.0,
                      actors=[pip7, bis7],
                      titles=[x.title(T("follow", 0.2), T("follow", 0.6), "FOLLOWS ME", color=G, emoji="🐾", size=90),
                              x.title(Wd("follow", "attack"), Ee("follow"), "PROTECTS ME", color=R, emoji="⚔️", size=90)]))
    # 8 a corner next to the bed
    bis8 = x.biscuit([(0, (8.6, 13.3))], yaw=PI * 0.75, sit=1.0, pose=lambda t: {"head_yaw": 0.25 * np.sin(t * 0.8)})
    pip8 = x.pip([(0, (12.2, 12.9))], skin="pip_iron", yaw=-PI * 0.55)
    out.append(E.Shot("corner", E.cam_static((12.6, GY + 2.9, 9.4), (8.2, GY + 0.6, 12.8), fov=52), stage="P2", phase=0.0,
                      actors=[bis8, pip8], clouds=False,
                      titles=[x.title(T("corner", 0.3), Ee("corner"), "BISCUIT'S CORNER", color=(255, 200, 120), emoji="🏠", size=86)]))
    # 9 protect: a zombie at the farm, Biscuit gets it
    fx0, fx1, fz0, fz1 = WD.FARM
    tz0 = C("protect")
    th = [T("protect", 0.55), T("protect", 0.68), T("protect", 0.8)]
    zom = x.mob("zombie", [(tz0, (fx1 + 6.0, fz1 + 8.0)), (th[0], (fx1 + 1.0, fz1 + 1.5))], hits=th, death=th[-1] + 0.15)
    bis9 = x.biscuit([(tz0, (fx0 - 1.0, fz1 - 3.0)), (T("protect", 0.4), (fx0 + 1.0, fz1 - 1.0)), (th[0] - 0.1, (fx1 - 0.2, fz1 + 0.5)),
                      (Ee("protect"), (fx1 - 0.5, fz1 + 0.2))],
                     y_off=lambda t: 0.6 * max(0.0, np.sin(PI * E.seg(t, th[0] - 0.35, th[0] + 0.05))),
                     face_target=lambda t: zom.pos(t))
    pip9 = x.pip([(0, (fx0 + 3.5, fz1 - 4.0))], skin="pip_iron", item=[(0, None), (th[1], "sword:iron")],
                 acts=x.every(tz0, T("protect", 0.45), 5), yaw=lambda t: 0.0 if t < T("protect", 0.5) else 0.9)
    out.append(E.Shot("protect", E.cam_static((fx0 - 2.0, GY + 3.5, fz1 - 9.0), (fx1 - 2.0, GY + 1.0, fz1 + 0.5), fov=52), stage="P2",
                      phase=1.15, actors=[zom, bis9, pip9], music="panic",
                      titles=[x.title(th[0], Ee("protect"), "GO BISCUIT!", color=(255, 200, 120), emoji="🐶", size=100)],
                      sfx=[(T("protect", 0.3), "groan", -8, dict(seed=60))]))
    # 10 best friend: sunset by the river
    rx = WD.river_x(10.0) - 5.2
    pip10 = x.pip([(0, (rx, 10.0))], skin="pip_iron", yaw=PI / 2)
    bis10 = x.biscuit([(0, (rx - 0.2, 11.4))], yaw=PI / 2, sit=1.0)
    out.append(E.Shot("best", E.cam_static((rx - 5.0, GY + 2.4, 11.8), (rx + 6.0, GY + 1.6, 9.0), fov=55), stage="P2", phase=1.0,
                      actors=[pip10, bis10], music="resolve",
                      titles=[x.title(T("best", 0.2), Ee("best"), "BEST FRIEND", color=(255, 120, 140), emoji="❤️", size=110)]))
    return out


def ch07(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    fx0, fx1, fz0, fz1 = WD.FARM
    cows = [x.mob("cow", [(0, (7.0 + 2.2 * i, 21.0 + 1.3 * (i % 2)))], yaw=0.5 * i) for i in range(3)] + \
           [x.mob("baby_cow", [(0, (12.5, 23.5))], yaw=2.0)]
    bis_sleep = x.biscuit([(0, (10.5, 6.6))], yaw=0.2, sit=1.0)
    # 1 calm
    out.append(E.Shot("calm", E.cam_orbit(np.array([14.0, GY + 1.0, 12.0]), 24.0, 12.0, 2.6, 3.2, C("calm"), Ee("calm"), fov=48),
                      stage="P2", phase=0.0, actors=cows + [bis_sleep], music="adventure",
                      titles=[x.title(Wd("calm", "peaceful"), Wd("calm", "Too"), "PEACEFUL", color=G, emoji="☮️"),
                              x.title(Wd("calm", "Too"), Ee("calm"), "TOO PEACEFUL", color=O, emoji="🤨")]))
    # 2 harvesting, Biscuit asleep
    hv = x.every(T("harvest", 0.1), Ee("harvest"), 7)
    pip2 = x.pip([(C("harvest"), (fx0 + 0.5, fz0 + 4.0)), (Ee("harvest") + 4, (fx0 + 0.5, fz1 - 2.0))], skin="pip_iron", acts=hv, yaw=PI / 2,
                 pose=lambda t: {"head_pitch": -0.3})
    out.append(E.Shot("harvest", E.cam_static((fx0 - 7.0, GY + 3.5, fz0 - 6.0), (fx0 - 1.5, GY + 1.0, fz0 + 3.5), fov=55), stage="P2",
                      phase=0.0, actors=[pip2, bis_sleep] + cows,
                      titles=[x.title(Wd("harvest", "sleeping"), Ee("harvest"), "Zzz", color=Wh, emoji="💤", pos=(560, 400), size=80)],
                      music="suspense"))
    # 3 hiss: the creeper at the house wall
    tb = T("boom", 0.05)
    cp = (WD.HOUSE[1] + 0.7, 9.6)
    crp = x.mob("creeper", [(C("hiss"), (cp[0] + 5.0, cp[1] + 7.0)), (T("hiss", 0.5), cp)], swell=[(T("hiss", 0.55), tb)], death=tb)
    out.append(E.Shot("hiss", E.cam_static((cp[0] + 3.0, GY + 1.9, cp[1] - 3.2), (cp[0] - 0.3, GY + 1.3, cp[1] + 0.5), fov=48), stage="P2",
                      phase=0.0, actors=[crp], music="silence",
                      titles=[x.title(T("hiss", 0.55), Ee("hiss"), "TSSSS...", color=G, emoji="💚")],
                      sfx=[(T("hiss", 0.55), "hiss", -4, dict(dur=tb - T("hiss", 0.55)))]))
    # 4 BOOM
    bc = np.array([13.5, GY + 2.0, 9.0])
    rng = np.random.default_rng(7)
    flying = []
    for i in range(16):
        d = rng.normal(size=3)
        d[1] = abs(d[1]) + 0.8
        d /= np.linalg.norm(d)
        flying.append((d * rng.uniform(5, 11), ["planks", "log", "planks", "dirt"][i % 4], rng.uniform(0, 6)))
    def blocks(t):
        if t < tb:
            return None
        a = t - tb
        v = []
        for vel, b, sp in flying:
            p = bc + vel * a + np.array([0, -9.0 * a * a / 2, 0])
            if p[1] > GY - 1:
                v.append(block_entity_vertices(b, p, 0.9, (a * sp, a * sp * 0.7, 0)))
        return [np.concatenate(v)] if v else None
    boom_fx = [E.Burst(tb, bc, 70, speed=(3, 9), life=(0.6, 1.6), size=(0.3, 0.8), colors=((1, 0.85, 0.5, 0.9), (0.6, 0.6, 0.6, 0.9),
                                                                                          (0.3, 0.3, 0.3, 0.9)),
                       gravity=1.0, drag=2.2, seed=9, grow=2.0),
               E.Burst(tb, bc, 30, speed=(1, 4), life=(0.2, 0.5), size=(0.6, 1.2), colors=((1, 0.8, 0.3, 1),), gravity=0, drag=3,
                       seed=10, additive=True)]
    out.append(E.Shot("boom", E.cam_static((fx0 - 4.0, GY + 4.0, fz0 - 9.0), bc + (0, 0.5, 0), fov=58), stage="P3", phase=0.0,
                      actors=[pip2] + cows, props=[E.Custom(blocks)], fx=boom_fx, shake=[(tb, 1.4, 0.6)], flash=[(tb, 0.9, 0.35)],
                      titles=[x.title(tb + 0.05, Ee("boom"), "BOOM", color=O, emoji="💥", size=180)], music="panic",
                      sfx=[(tb, "explosion", -1, {})]))
    smoke = [E.Burst(C("crater") + 0.3 * k, bc + (0, -0.5, 0), 4, speed=(0.2, 0.6), life=(2.0, 3.0), size=(0.5, 1.0),
                     colors=((0.45, 0.45, 0.45, 0.5),), gravity=0.8, drag=1.0, up_bias=1.0, seed=20 + k, box=1.5, grow=1.5)
             for k in range(int((Ee("crater") - C("crater")) / 0.3))]
    # 5 crater
    out.append(E.Shot("crater", E.cam_orbit(bc, 12.0, 6.0, 0.8, 1.8, C("crater"), Ee("crater"), fov=50, look_y=-1.0), stage="P3",
                      phase=0.0, fx=smoke,
                      titles=[x.title(Wd("crater", "half"), Ee("crater"), "HALF MY HOUSE", color=R, emoji="💥")]))
    # 6 creepers don't burn
    crp6 = x.mob("creeper", [(C("sneak"), (-4.0, 40.0)), (Ee("sneak"), (-1.0, 33.0))])
    out.append(E.Shot("sneak", E.cam_static((-1.0, GY + 1.6, 30.5), (-3.0, GY + 1.5, 37.0), fov=48), stage="P3", phase=0.0, actors=[crp6],
                      titles=[x.title(T("sneak", 0.2), Ee("sneak"), "DON'T BURN IN SUNLIGHT", color=G, emoji="☀️", size=86)]))
    # 7 gone
    pip7 = x.pip([(C("gone"), (14.0, 4.0)), (T("gone", 0.3), (12.6, 8.2))], skin="pip_iron",
                 pose=lambda t: {"head_yaw": 0.8 * np.sin((t - C("gone")) * 1.4)})
    out.append(E.Shot("gone", E.cam_static((17.5, GY + 4.0, 3.0), (12.0, GY + 1.0, 9.5), fov=52), stage="P3", phase=0.0, actors=[pip7],
                      fx=smoke[:3],
                      titles=[x.title(Wd("gone", "walls"), Ee("gone"), "WALLS", color=R, pos=(640, 160), size=76, emoji="❌"),
                              x.title(Wd("gone", "roof"), Ee("gone"), "ROOF", color=R, pos=(1040, 160), size=76, emoji="❌"),
                              x.title(Wd("gone", "chest"), Ee("gone"), "CHEST", color=R, pos=(1430, 160), size=76, emoji="❌")]))
    # 8 Biscuit is okay
    pip8 = x.pip([(0, (12.6, 8.2))], skin="pip_iron", yaw=-PI / 2)
    bis8 = x.biscuit([(C("okay"), (6.0, 4.5)), (T("okay", 0.4), (11.4, 8.0)), (Ee("okay"), (11.4, 8.0))], face_target=lambda t: pip8.pos(t))
    out.append(E.Shot("okay", E.cam_static((10.0, GY + 1.8, 4.6), (11.8, GY + 0.9, 8.2), fov=50), stage="P3", phase=0.0,
                      actors=[pip8, bis8], fx=[E.hearts_burst(T("okay", 0.5), (11.4, GY + 1.3, 8.0), 31)],
                      titles=[x.title(T("okay", 0.4), Ee("okay"), "HE'S OKAY", color=(255, 200, 120), emoji="🐶")], music="resolve"))
    # 9 lesson
    out.append(E.Shot("lesson", E.cam_static((20.0, GY + 6.0, 0.0), bc + (0, -1.0, 0), fov=50), stage="P3", phase=0.9,
                      titles=[x.title(Wd("lesson", "burns"), Ee("lesson"), "WOOD BURNS", color=O, emoji="🔥", pos=(560, 160), size=70),
                              x.title(Wd("lesson", "breaks"), Ee("lesson"), "BREAKS", color=O, emoji="💥", pos=(960, 160), size=70),
                              x.title(Wd("lesson", "stop"), Ee("lesson"), "NO CREEPER PROOF", color=R, emoji="❌", pos=(1050, 270), size=70)]))
    # 10 promise: night, torch in hand
    pip10 = x.pip([(0, (12.0, 4.6))], skin="pip_iron", item="torch", yaw=PI)
    bis10 = x.biscuit([(0, (13.3, 4.4))], yaw=PI, sit=1.0)
    out.append(E.Shot("promise", E.cam_front(pip10, dist=4.2, up=0.5, side=1.5, fov=48), stage="P3", phase=2.0, actors=[pip10, bis10],
                      props=[E.Custom(lambda t: ([], [], [[*(pip10.pos(t) + (0.35, 1.3, 0.4)), 0.8, 1.0, 0.75, 0.35, 0.3]]))],
                      titles=[x.title(Wd("promise", "stone"), Ee("promise"), "REBUILD IN STONE", color=(210, 210, 210), emoji="🪨", size=96)],
                      music="build"))
    return out


def stone_blocks():
    base = WD.stage_world("P3c").blocks
    tgt = WD.stage_world("P4").blocks
    house = WD.VoxelWorld()
    WD.stone_house(house)
    wall = WD.VoxelWorld()
    WD.stone_wall(wall)
    hb = [(p, b) for p, b in house.blocks.items() if p[1] > GY]
    hb.sort(key=lambda kv: (kv[0][1], np.arctan2(kv[0][2] - 11.5, kv[0][0] - 10.5)))
    wb = [(p, b) for p, b in wall.blocks.items() if p[1] > GY]
    wb.sort(key=lambda kv: (np.arctan2(kv[0][2] - 14.0, kv[0][0] - 13.0), kv[0][1]))
    return hb, wb


def ch08(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    cx, cz = WD.CAVE_MOUTH
    # 1 quarry: mining the tunnel walls
    wallp = [(cx - 2, GY + 1 + (k % 3), cz - 2 - k // 3) for k in range(9)]
    qt = x.every(T("quarry", 0.1), Ee("quarry") - 0.2, len(wallp))
    pip1 = x.pip([(C("quarry"), (cx - 0.4, Y0, cz - 1.0)), (Ee("quarry"), (cx - 0.4, Y0, cz - 3.5))], skin="pip_iron", item="pickaxe:iron",
                 acts=x.every(T("quarry", 0.05), Ee("quarry") - 0.2, 22), yaw=-PI / 2)
    br = E.Breaks([(qt[i], p, "stone") for i, p in enumerate(wallp)], dur=0.35, collect=lambda t: pip1.pos(t), drop="cobble")
    out.append(E.Shot("quarry", E.cam_static((cx + 1.0, GY + 2.6, cz + 2.5), (cx - 1.5, GY + 1.8, cz - 2.5), fov=55), stage="P3",
                      phase=0.0, actors=[pip1], props=[br], music="build",
                      titles=[x.title(Wd("quarry", "Hundreds"), Ee("quarry"), "COBBLESTONE x340", color=(210, 210, 210), emoji="🪨", size=90)]))
    hb, wb = stone_blocks()
    t0, t1 = T("rebuild", 0.05), Ee("rebuild") - 0.4
    rev_h = E.Reveal("stonehouse", hb, t0, t1, K=40)
    t2, t3 = T("wall", 0.05), Ee("gate") - 0.6
    rev_w = E.Reveal("stonewall", wb, t2, t3, K=40)
    rev_hd = E.Reveal("stonehouse", hb, t0 - 100, t0 - 99, K=40)    # finished house for the wall shots
    pipb = x.pip(circle_keys((10.5, 11.5), 6.5, C("rebuild"), Ee("rebuild"), turns=1.5, a0=-PI / 2), skin="pip_iron", item="cobble",
                 acts=x.every(t0, t1, int((t1 - t0) / 0.28)))
    out.append(E.Shot("rebuild", E.cam_orbit(HC, 16.0, 7.0, -0.6, 0.5, C("rebuild"), Ee("rebuild"), fov=48, look_y=0.5), stage="P3c",
                      phase=timelapse(t0, t1, 1.0, 0.0), actors=[pipb], props=[rev_h],
                      titles=[x.title(Wd("rebuild", "Stone"), Ee("rebuild"), "STONE HOUSE", color=(210, 210, 210), emoji="🏠")]))
    pipw = x.pip(circle_keys((13.0, 14.0), 17.0, C("wall"), Ee("gate"), turns=1.0, a0=-PI / 2), skin="pip_iron", item="cobble",
                 acts=x.every(t2, t3, int((t3 - t2) / 0.3)))
    out.append(E.Shot("wall", E.cam_orbit(np.array([13.0, GY + 1.0, 14.0]), 34.0, 20.0, 0.4, 1.2, C("wall"), Ee("wall"), fov=50),
                      stage="P3c", phase=timelapse(t2, t3, 1.0, 0.0), actors=[pipw], props=[rev_hd, rev_w],
                      titles=[x.title(Wd("wall", "Two"), Ee("wall"), "2 BLOCKS HIGH", color=(210, 210, 210), emoji="🧱")]))
    out.append(E.Shot("gate", E.cam_static((11.0, GY + 4.0, -9.0), (11.0, GY + 1.5, 2.0), fov=52), stage="P3c",
                      phase=timelapse(t2, t3, 1.0, 0.0), actors=[pipw], props=[rev_hd, rev_w],
                      titles=[x.title(Wd("gate", "gate"), Ee("gate"), "THE GATE", color=Y, emoji="🚪")]))
    # 5 torches pop on at dusk
    out.append(E.Shot("torches", E.cam_orbit(np.array([13.0, GY + 1.0, 14.0]), 30.0, 16.0, 1.4, 2.0, C("torches"), Ee("torches"), fov=50),
                      stage="P4", phase=1.6, torch_reveal=(T("torches", 0.1), Ee("torches") - 0.3), torches_on=True,
                      titles=[x.title(T("torches", 0.2), Ee("torches"), "TORCHES EVERYWHERE", color=Y, emoji="🔥", size=90)],
                      sfx=[(tt, "pop", -16, dict(pitch=0.8)) for tt in x.every(T("torches", 0.1), Ee("torches") - 0.3, 14)]))
    # 6 why: lit = safe; a zombie rises only outside in the dark
    tz = T("why", 0.4)
    zr = x.mob("zombie", [(tz, (-9.0, 8.0)), (Ee("why"), (-6.5, 8.5))], spawn=tz, y_off=lambda t: -1.9 * (1 - E.smooth(E.seg(t, tz, tz + 1.0))))
    out.append(E.Shot("why", E.cam_static((-14.0, GY + 6.0, 0.0), (0.0, GY + 1.0, 12.0), fov=55), stage="P4", phase=2.0, actors=[zr],
                      titles=[x.title(T("why", 0.1), Ee("why"), "LIGHT = NO SPAWNS", color=Y, emoji="💡", size=96)],
                      fx=[E.debris(tz, (-9.0, GY + 1.2, 8.0), "dirt", 3)], sfx=[(tz + 0.3, "groan", -12, dict(seed=70))]))
    # 7 night 26 from the roof
    zs = [x.mob("zombie", [(C("night26"), (-6.0, -4.0 + 6 * i)), (Ee("night26"), (-5.5 + 0.4 * i, 1.0 + 6 * i))]) for i in range(3)]
    zs += [x.mob("zombie", [(C("night26"), (8.0 + 6 * i, -5.0)), (Ee("night26"), (12.0 + 6 * i, -4.5))]) for i in range(2)]
    pip7 = x.pip([(0, (9.0, GY + 7.0, 9.5))], skin="pip_iron", item="sword:iron", yaw=-PI * 0.75)
    bis7 = x.biscuit([(0, (10.2, GY + 7.0, 9.2))], yaw=-PI * 0.75, sit=1.0)
    out.append(E.Shot("night26", E.cam_static((13.0, GY + 9.6, 13.5), (0.0, GY + 1.5, 2.0), fov=55), stage="P4", phase=2.0,
                      actors=zs + [pip7, bis7], music="suspense",
                      titles=[x.title(Wd("night26", "couldn't"), Ee("night26"), "NO WAY IN", color=G, emoji="🛡️")],
                      sfx=[(T("night26", 0.3), "groan", -16, dict(seed=71)), (T("night26", 0.8), "groan", -17, dict(seed=72))]))
    # 8-9 creeper vs wall
    th = [T("twohits", 0.15), T("twohits", 0.45)]
    crp = x.mob("creeper", [(C("creeper2"), (4.5, -9.0)), (T("creeper2", 0.55), (4.5, -2.25)), (Ee("twohits"), (4.5, -2.25))],
                hits=th, death=th[1] + 0.2, pose=lambda t: {"head_pitch": 0.3 if t > T("creeper2", 0.6) else 0.0})
    pipw2 = x.pip([(C("creeper2"), (6.5, GY + 3.0, -1.0 + 0.5)), (T("twohits", 0.1), (4.5, GY + 3.0, -0.5))], skin="pip_iron",
                  item="sword:iron", yaw=PI, acts=[t - 0.05 for t in th])
    out.append(E.Shot("creeper2", E.cam_static((9.0, GY + 1.8, -7.5), (4.5, GY + 1.6, -1.5), fov=50), stage="P4", phase=0.0,
                      actors=[crp, pipw2], music="adventure",
                      titles=[x.title(Wd("creeper2", "stood"), Ee("creeper2"), "...STUCK", color=G, emoji="🧱")]))
    out.append(E.Shot("twohits", E.cam_static((1.0, GY + 2.6, -5.5), (4.5, GY + 1.8, -1.2), fov=50), stage="P4", phase=0.0,
                      actors=[crp, pipw2],
                      titles=[x.title(th[1] + 0.2, Ee("twohits"), "BYE BYE", color=Y, emoji="👋")]))
    # 10 day 30
    out.append(E.Shot("day30", E.cam_orbit(np.array([13.0, GY + 1.0, 14.0]), 34.0, 18.0, -2.6, -1.9, C("day30"), Ee("day30"), fov=48),
                      stage="P4", phase=1.0, actors=[x.mob("cow", [(0, (7.0 + 2.2 * i, 21.5))], yaw=0.6 * i) for i in range(3)],
                      music="resolve", titles=[x.title(Wd("day30", "safe"), Ee("day30"), "SAFE", color=G, emoji="✅", size=120)]))
    return out


POURS = [(92, 95), (95, 98), (98, 101), (101, 104), (104, 106)]
ORES9 = [(116, -32, 3), (116, -32, 4), (116, -31, 3), (116, -31, 4), (116, -32, 5), (116, -31, 5)]


def lava_strip(pour_t):
    """The bridge strip (z 2..4) of the lava lake: lava until poured, obsidian after."""
    items_l, items_o = [], []
    for i, (a, b) in enumerate(POURS):
        for xx in range(a, b):
            for zz in range(2, 5):
                items_l.append((xx, zz, pour_t[i]))
    def fn(t):
        v, vt = [], []
        for (xx, zz, tp) in items_l:
            c = ug(xx + 0.5, F - 0.5, zz + 0.5)
            if t < tp + 0.25:
                v.append(block_entity_vertices("lava", c, 1.0))
            else:
                s = E.ease_out_back(E.seg(t, tp + 0.25, tp + 0.45))
                v.append(block_entity_vertices("obsidian", c, max(0.05, min(1.0, s))))
        return [np.concatenate(v)], vt, []
    return E.Custom(fn)


def ch09(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    # 1 deeper
    pip1 = x.pip([(0, (10.5, 5.0))], skin="pip_iron", item="pickaxe:iron", yaw=PI, pose=lambda t: {"head_pitch": -0.3})
    out.append(E.Shot("deeper", E.cam_front(pip1, dist=3.4, up=0.5, side=-1.0, fov=46), stage="P4", phase=0.0, actors=[pip1],
                      music="suspense", titles=[x.title(Wd("deeper", "diamonds"), Ee("deeper"), "DIAMONDS", color=B, emoji="💎", size=120)]))
    # 2 the staircase
    t0, t1 = C("stairs") + 0.3, Ee("deepslate") - 0.5
    ks = [(C("stairs"), ug(-3.0, 0, 1.5))]
    n = 30
    for k in range(n + 1):
        ks.append((t0 + (t1 - t0) * k / n, ug(k + 0.4, -k, 1.5)))
    pip2 = x.pip(ks, skin="pip_iron", item="pickaxe:iron", acts=[t0 + (t1 - t0) * (k + 0.5) / n for k in range(n)])
    tor = E.Torches([(t0 + (t1 - t0) * k / n, ug(k + 0.5, -k, 2.6)) for k in range(4, n + 1, 4)])
    chips = [E.debris(t0 + (t1 - t0) * (k + 0.5) / n, ug(k + 1.3, -k + 1.0, 1.5), "deepslate" if k > 17 else "stone", 300 + k, n=8)
             for k in range(n)]
    out.append(E.Shot("stairs", E.cam_follow(pip2, back=3.0, up=4.0, side=0.0, look=(0, 0.6, 0), fov=64, smooth_t=0.4, look_ahead=2.0),
                      loc="under", actors=[pip2], props=[tor], fx=chips,
                      titles=[x.title(Wd("stairs", "down"), T("stairs", 0.9), "DOWN", color=Wh, emoji="⬇️", size=90)],
                      sfx=[(t0 + (t1 - t0) * (k + 0.5) / n, "dig", -12, dict(seed=k)) for k in range(n)], cave=0.85))
    out.append(E.Shot("deepslate", E.cam_track(pip2, (-1.6, 3.2, -0.6), look=(1.4, 0.7, 0.6), fov=62), loc="under", actors=[pip2],
                      props=[tor], fx=chips, cave=0.85,
                      titles=[x.title(Wd("deepslate", "Deepslate"), Ee("deepslate"), "DEEPSLATE", color=(150, 150, 170), emoji="⬛")]))
    # 4 lava lake reveal
    pour_t = x.every(T("obsidian", 0.3), Ee("obsidian") - 0.3, 5)
    strip = lava_strip(pour_t)
    pip4 = x.pip([(C("lava"), ug(84.0, F, 3.5)), (T("lava", 0.6), ug(90.4, F, 3.5))], skin="pip_iron", item="pickaxe:iron")
    out.append(E.Shot("lava", E.cam_dolly(C("lava"), Ee("lava"), ug(88.0, F + 3.0, 8.5), ug(89.5, F + 5.5, 9.5), ug(100.0, F, 3.0),
                                          ug(104.0, F, 3.0), fov=60), loc="under", actors=[pip4], props=[strip], music="suspense",
                      titles=[x.title(Wd("lava", "lava"), Ee("lava"), "LAVA", color=O, emoji="🔥", size=130)],
                      sfx=[(C("lava"), "crackle", -11, dict(dur=6.0))]))
    # 5 bucket
    pip5 = x.pip([(0, ug(90.4, F, 3.5))], skin="pip_iron", item="water_bucket", yaw=PI / 2)
    out.append(E.Shot("bucket", E.cam_front(pip5, dist=3.0, up=0.4, side=-1.2, fov=48), loc="under", actors=[pip5], props=[strip],
                      titles=[x.title(Wd("bucket", "bucket"), Ee("bucket"), "WATER BUCKET", color=B, emoji="🪣")]))
    # 6 obsidian bridge
    ks = [(C("obsidian"), ug(90.4, F, 3.5))]
    for i, (a, b) in enumerate(POURS):
        ks += [(pour_t[i] - 0.1, ug(a - 0.6, F, 3.5)), (pour_t[i] + 0.7, ug(a - 0.6, F, 3.5))]
    ks.append((Ee("obsidian") + 4, ug(105.2, F, 3.5)))
    item6 = [(0, "water_bucket")]
    for tp in pour_t:
        item6 += [(tp, "bucket"), (tp + 0.6, "water_bucket")]
    pip6 = x.pip(ks, skin="pip_iron", item=item6, yaw=PI / 2,
                 pose=lambda t: {"reach": max([np.sin(PI * E.seg(t, tp - 0.2, tp + 0.5)) for tp in pour_t] + [0.0])})
    flows = []
    def water(t):
        from mcengine.geometry import translate, box_vertices
        vt = []
        for i, tp in enumerate(pour_t):
            if tp <= t < tp + 0.45:
                a, b = POURS[i]
                s = E.seg(t, tp, tp + 0.45)
                for xx in range(a, b):
                    vt.append(box_vertices((1, 0.25 * (1 - s) + 0.05, 3), "water_flow", matrix=translate(*ug(xx + 0.5, F + 0.1, 3.5)), alpha=0.5))
        return [], vt, []
    steam = [E.Burst(tp + 0.2, ug((POURS[i][0] + POURS[i][1]) / 2, F + 0.3, 3.5), 25, speed=(0.5, 2), life=(0.8, 1.5), size=(0.3, 0.6),
                     colors=((0.85, 0.85, 0.85, 0.6),), gravity=1.5, drag=1.5, up_bias=1.2, seed=400 + i, box=1.0, grow=1.5)
             for i, tp in enumerate(pour_t)]
    out.append(E.Shot("obsidian", E.cam_track(pip6, (-1.5, 3.4, 5.5), look=(2.5, 0.0, -0.5), fov=55), loc="under", actors=[pip6],
                      props=[strip, E.Custom(water)], fx=steam,
                      titles=[x.title(Wd("obsidian", "obsidian"), Ee("obsidian"), "WATER + LAVA = OBSIDIAN", color=PU, size=80)],
                      sfx=[(tp, "water", -9, dict(dur=0.8)) for tp in pour_t] + [(tp + 0.25, "hiss", -10, dict(dur=1.0, seed=i))
                                                                                 for i, tp in enumerate(pour_t)]))
    # 7 slip into the lava
    ts = T("slip", 0.3)
    tj = T("slip", 0.62)
    pip7 = x.pip([(C("slip"), ug(101.0, F, 3.5)), (ts - 0.2, ug(101.2, F, 4.6)), (ts, ug(101.3, F - 0.35, 5.6)),
                  (tj, ug(101.3, F - 0.35, 5.6)), (tj + 0.35, ug(101.0, F, 3.8)), (Ee("slip"), ug(101.0, F, 3.6))], skin="pip_iron",
                 item="water_bucket", flash=lambda t: 0.45 if ts <= t < tj + 0.6 and (t * 6) % 1 < 0.5 else 0.0,
                 pose=lambda t: {"arms_up": 0.8, "flail_phase": t * 9} if ts <= t < tj + 0.3 else {},
                 y_off=lambda t: 1.0 * np.sin(PI * E.seg(t, tj, tj + 0.35)))
    out.append(E.Shot("slip", E.cam_static(ug(98.0, F + 2.4, 0.4), ug(101.2, F + 0.6, 4.8), fov=52), loc="under", actors=[pip7],
                      props=[strip], fx=E.fire_burst(ts, ug(101.3, F + 0.6, 5.0), tj + 1.5 - ts, seed=500), music="panic",
                      hp=lambda t: 10.0 - 8.0 * E.seg(t, ts, tj + 0.4),
                      overlay=orange_overlay(f, ts, tj + 1.6),
                      titles=[x.title(Wd("slip", "hearts"), Ee("slip"), "2 HEARTS", color=R, emoji="💔")],
                      sfx=[(ts, "crackle", -5, dict(dur=2.0)), (ts + 0.1, "hurt", -6, {}), (ts + 0.6, "hurt", -7, dict(seed=1)),
                           (tj, "hurt", -7, dict(seed=2))]))
    # 8 diamonds
    dt_ = x.every(T("diamonds", 0.3), Ee("diamonds") - 0.3, 6)
    pip8 = x.pip([(C("diamonds"), ug(106.0, F, 3.8)), (T("diamonds", 0.25), ug(114.6, F, 4.0))], skin="pip_iron", item="pickaxe:iron",
              acts=x.every(T("diamonds", 0.25), Ee("diamonds") - 0.3, 18), yaw=lambda t: PI / 2 if t > T("diamonds", 0.25) else None)
    ores = E.Breaks([(dt_[i], ug(*p), "deep_diamond") for i, p in enumerate(ORES9)], dur=0.45, collect=lambda t: pip8.pos(t),
                    drop="diamond", overlay=False)
    def counter(layer, t):
        n = sum(1 for tt in dt_ if t >= tt + 0.5)
        if n > 0:
            from mcengine import overlay as ov
            ov.title_pop(layer, f"x{n}", t, dt_[0] + 0.5, Ee("diamonds") + 2, (1500, 330), 120, fill=B, emoji="💎", emoji_side="left")
    out.append(E.Shot("diamonds", E.cam_static(ug(111.0, F + 2.0, 7.6), ug(115.5, F + 0.9, 3.8), fov=52), loc="under", actors=[pip8],
                      props=[ores, strip], overlay=counter, hp=2.0, music="adventure",
                      sfx=[(tt + 0.05, "shimmer", -11, {}) for tt in dt_]))
    # 9 diamond pickaxe
    tdp = Wd("dpick", "diamond")
    pip9 = x.pip([(0, ug(114.6, F, 4.0))], skin="pip_iron", item=[(0, None), (tdp, "pickaxe:diamond")], yaw=-PI / 2,
                 pose=lambda t: {"arms_up": 0.85 * E.smooth(E.seg(t, tdp, tdp + 0.3)), "flail_phase": 0.0})
    out.append(E.Shot("dpick", E.cam_front(pip9, dist=3.0, up=0.2, side=0.9, fov=48, look=(0, 1.5, 0)), loc="under", actors=[pip9],
                      props=[strip], hp=lambda t: 2.0 + 8.0 * E.seg(t, C("dpick"), Ee("dpick")),
                      crafts=[(T("dpick", 0.0), tdp + 0.3, ["e:💎", "e:💎", "e:💎", None, "t:pick_handle", None, None, "t:pick_handle", None], "e:⛏️")],
                      fx=[E.Burst(tdp, ug(114.6, F + 2.6, 4.0), 40, speed=(1, 3), life=(0.6, 1.2), size=(0.06, 0.14),
                                  colors=((0.5, 1, 1, 1), (1, 1, 1, 1)), gravity=-1, drag=2, up_bias=0.6, seed=600, additive=True)],
                      titles=[x.title(tdp + 0.3, Ee("dpick"), "DIAMOND PICKAXE", color=B, emoji="⛏️")],
                      sfx=[(tdp, "shimmer", -6, {}), (tdp, "impact", -10, {})]))
    # 10 saved
    pip10 = x.pip([(0, (12.2, 12.4))], skin="pip_iron", item="pickaxe:diamond", yaw=PI / 2 + 0.3, acts=[T("saved", 0.3)])
    out.append(E.Shot("saved", E.cam_static((9.0, GY + 2.7, 10.0), (13.5, GY + 1.0, 13.4), fov=52), stage="P4", phase=0.0,
                      actors=[pip10], clouds=False,
                      titles=[x.title(T("saved", 0.3), Ee("saved"), "3 SAVED...", color=B, emoji="🤫")]))
    # 11 what's out there: the hill top at sunrise
    hx, hz = WD.HILL_C
    top = f.ground(hx, hz)
    pip11 = x.pip([(0, (hx, top, hz + 2.0))], skin="pip_iron", item="pickaxe:diamond", yaw=0.15)
    bis11 = x.biscuit([(0, (hx + 1.2, top, hz + 2.4))], yaw=0.15, sit=1.0)
    out.append(E.Shot("day35", E.cam_static((hx - 1.5, top + 2.2, hz - 2.5), (hx + 6.0, top - 2.0, hz + 30.0), fov=58), stage="P4",
                      phase=3.6, actors=[pip11, bis11], music="adventure",
                      titles=[x.title(Wd("day35", "out"), Ee("day35"), "WHAT'S OUT THERE?", color=Y, emoji="🗺️", size=96)]))
    return out


def river_pt(z, off=0.0):
    return (WD.river_x(z) + off, z)


def boat_keys(z0, z1, t0, t1, off=0.0, n=30):
    return [(t0 + (t1 - t0) * k / n, (WD.river_x(z0 + (z1 - z0) * k / n) + off, GY + 0.45, z0 + (z1 - z0) * k / n)) for k in range(n + 1)]


VILLAGERS = [(60.0, 138.5), (66.0, 128.0), (72.5, 140.5), (63.5, 146.0), (58.0, 130.0), (69.5, 136.0)]


def ch10(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    # 1-4: by boat down the river
    tb0, tb1 = T("boat", 0.45), Ee("cliffs")
    pipb = x.pip([(C("boat"), (WD.river_x(12.0) - 6.0, 12.0)), (tb0 - 0.4, (WD.river_x(12.0) - 0.3, GY + 0.45, 12.0))] +
                 boat_keys(12.0, 112.0, tb0, tb1), skin="pip_iron", item="pickaxe:diamond",
                 boat=lambda t: t > tb0 - 0.4)
    bisb = x.biscuit([(C("boat"), (WD.river_x(12.0) - 7.0, 13.0)), (tb0 - 0.4, (WD.river_x(12.0) - 0.3, GY + 0.45, 12.9))] +
                     boat_keys(12.9, 112.9, tb0, tb1), sit=lambda t: 1.0 if t > tb0 else 0.0)
    out.append(E.Shot("boat", E.cam_static((WD.river_x(4.0) - 5.0, GY + 3.0, 4.0), (WD.river_x(14.0), GY + 0.8, 14.0), fov=52),
                      stage="P4", phase=0.0, actors=[pipb, bisb], music="adventure",
                      crafts=[(T("boat", 0.0), T("boat", 0.4), [None, None, None, "planks", None, "planks", "planks", "planks", "planks"], "e:🛶")],
                      titles=[x.title(T("boat", 0.5), Ee("boat"), "BOAT TRIP", color=B, emoji="🛶")],
                      sfx=[(tb0 - 0.4, "splash", -10, {})]))
    out.append(E.Shot("biscuitboat", E.cam_track(pipb, (2.6, 1.6, 3.4), look=(0, 0.6, 0.6), fov=50), stage="P4", phase=0.0,
                      actors=[pipb, bisb],
                      titles=[x.title(Wd("biscuitboat", "dogs"), Ee("biscuitboat"), "DOGS LOVE BOATS", color=(255, 200, 120), emoji="🐶")]))
    out.append(E.Shot("faster", E.cam_track(pipb, (-6.0, 4.5, -7.0), look=(0, 0.5, 4.0), fov=55), stage="P4", phase=0.0,
                      actors=[pipb, bisb],
                      titles=[x.title(T("faster", 0.1), T("faster", 0.55), "FASTER", color=Y, emoji="⚡"),
                              x.title(Wd("faster", "mobs"), Ee("faster"), "SAFE ON WATER", color=B, emoji="🌊")]))
    out.append(E.Shot("cliffs", E.cam_track(pipb, (18.0, 22.0, -14.0), look=(0, 0, 8.0), fov=55), stage="P4", phase=0.0,
                      actors=[pipb, bisb]))
    # 5 roofs ahead
    tr0, tr1 = C("roofs"), Ee("roofs")
    pipr = x.pip(boat_keys(112.0, 134.0, tr0, tr1), skin="pip_iron", item="pickaxe:diamond", boat=True)
    bisr = x.biscuit(boat_keys(112.9, 134.9, tr0, tr1), sit=1.0)
    out.append(E.Shot("roofs", E.cam_track(pipr, (-1.5, 2.6, -4.5), look=(6.0, 1.0, 14.0), fov=55), stage="P4", phase=0.0,
                      actors=[pipr, bisr]))
    vc = np.array([64.0, GY + 1.0, 138.0])
    vill = [x.mob("villager", [(C("village"), p), (Ee("helping"), (p[0] + 2.0 * np.sin(i), p[1] + 1.5 * np.cos(i)))]) for i, p in enumerate(VILLAGERS)]
    kids = [x.mob("kid", [(C("village"), (61.0 + 3 * i, 141.5)), (C("village") + 2.0, (63.0 + 3 * i, 142.5)),
                          (C("village") + 4.0, (61.0 + 3 * i, 141.5)), (C("village") + 6.0, (63.0 + 3 * i, 142.5)),
                          (Ee("helping"), (61.0 + 3 * i, 141.5))]) for i in range(3)]
    for k in kids:
        k.path = E.PathTrack([(tt, p) for tt, p in zip(k.path.t, k.path.p)])
    # 6 village reveal
    out.append(E.Shot("village", E.cam_orbit(vc, 30.0, 20.0, -1.2, -0.4, C("village"), Ee("village"), fov=50), stage="P4", phase=0.0,
                      actors=vill + kids, music="build",
                      titles=[x.title(Wd("village", "village"), Ee("village"), "A VILLAGE!", color=Y, emoji="🏘️", size=120)]))
    # 7 who lives here
    roles = [("farmer", "FARMER", "🌾", VILLAGERS[0]), ("librarian", "LIBRARIAN", "📚", VILLAGERS[1]),
             ("blacksmith", "BLACKSMITH", "⚒️", VILLAGERS[2]), ("kids", "KIDS", "🧒", (63.0, 142.0))]
    pts = [np.array([p[0], GY + 1.0, p[1]]) for (_, _, _, p) in roles]
    tws = [Wd("kids", w) for (w, _, _, _) in roles]
    def kcam(t):
        i = int(np.searchsorted(tws, t, side="right")) - 1
        i = max(0, min(i, 3))
        p = pts[i]
        d = np.array([65.5 - p[0], 0.0, 139.5 - p[2]])
        d = d / (np.linalg.norm(d) + 1e-6) if np.linalg.norm(d) > 1 else np.array([0.6, 0, -0.8])
        side = np.array([-d[2], 0, d[0]])
        return dict(eye=(p + d * 4.2 + side * 1.2 + (0, 2.2, 0)).astype(np.float32), target=(p + (0, 1.0, 0)).astype(np.float32), fov=46, roll=0.0)
    out.append(E.Shot("kids", kcam, stage="P4", phase=0.0, actors=vill + kids,
                      titles=[x.title(tws[i], tws[i + 1] if i < 3 else Ee("kids"), r[1], color=Y, emoji=r[2]) for i, r in enumerate(roles)]))
    # 8 trade
    farmer = x.mob("villager", [(0, (60.0, 138.5))], yaw=PI / 2)
    pipt = x.pip([(0, (62.2, 138.5))], skin="pip_iron", item="wheat", yaw=-PI / 2)
    def trade_panel(layer, t, t0=T("trade", 0.5), t1=Ee("emeralds"), sale=False):
        if not (t0 <= t < t1):
            return
        from mcengine import overlay as ov
        from PIL import ImageDraw
        a = min(1.0, (t - t0) / 0.25, (t1 - t) / 0.25)
        d = ImageDraw.Draw(layer)
        x0, y0 = 1180, 300
        d.rounded_rectangle([x0, y0, x0 + 600, y0 + 200], radius=20, fill=(198, 198, 198, int(235 * a)), outline=(40, 40, 40, int(255 * a)), width=6)
        ov.draw_text(layer, "5" if sale else "20", (x0 + 90, y0 + 100), 70, fill=(255, 255, 255), alpha=a)
        ov.paste_emoji(layer, "🌾", (x0 + 195, y0 + 100), 88, alpha=a)
        ov.draw_text(layer, "→", (x0 + 300, y0 + 100), 90, fill=(80, 80, 80), alpha=a, shadow=False)
        ov.paste_emoji(layer, "💚", (x0 + 445, y0 + 100), 88, alpha=a)
        ov.draw_text(layer, "1", (x0 + 540, y0 + 100), 70, fill=(255, 255, 255), alpha=a)
        if sale:
            ov.draw_text(layer, "HERO DISCOUNT", (x0 + 300, y0 - 40), 64, fill=(255, 225, 40), rot=4, alpha=a)
    out.append(E.Shot("trade", E.cam_static((61.2, GY + 2.4, 134.8), (61.0, GY + 1.5, 138.6), fov=50), stage="P4", phase=0.0,
                      actors=[farmer, pipt] + vill[1:], overlay=trade_panel,
                      titles=[x.title(Wd("trade", "Wheat"), T("trade", 0.5) + 0.0, "WHEAT!", color=(230, 200, 90), emoji="🌾")]))
    em = [(T("emeralds", 0.1) + 0.45 * i, T("emeralds", 0.1) + 0.45 * i + 0.6, (60.6, GY + 2.2, 138.5), (62.0, GY + 2.0, 138.5), "emerald")
          for i in range(int((Ee("emeralds") - T("emeralds", 0.1)) / 0.45))]
    out.append(E.Shot("emeralds", E.cam_static((63.6, GY + 2.2, 135.4), (61.0, GY + 1.6, 138.6), fov=48), stage="P4", phase=0.0,
                      actors=[farmer, pipt] + vill[1:], props=[E.Fly(em)], overlay=trade_panel,
                      titles=[x.title(Wd("emeralds", "stack"), Ee("emeralds"), "x64", color=(90, 230, 120), emoji="💚", pos=(700, 160))],
                      sfx=[(e[1], "pop", -12, dict(pitch=1.3)) for e in em]))
    # 10 helping: fixing the farm (time-lapse)
    fx_, fz_, fw, fd = WD.VILLAGE_FARM
    cells = [(xx, zz) for xx in range(fx_ + fw, fx_ + fw + 4) for zz in range(fz_ + 1, fz_ + fd - 1)]
    th = x.every(T("helping", 0.15), Ee("helping") - 0.3, len(cells))
    farm = E.Pops([(th[i], (c[0], GY, c[1]), "farmland") for i, c in enumerate(cells)], sound=False, scale=1.012)
    def crops(t):
        v = [M.crop_vertices((c[0], GY + 1, c[1]), "crop_gold", 0.8) for i, c in enumerate(cells) if t > th[i] + 0.5]
        return ([np.concatenate(v)] if v else []), [], []
    piph = x.pip([(th[i], (c[0] + 0.5, c[1] - 0.7)) for i, c in enumerate(cells) if i % 3 == 0], skin="pip_iron", item="hoe",
                 acts=th[::2])
    out.append(E.Shot("helping", E.cam_static((76.0, GY + 5.0, 143.5), (70.0, GY + 1.0, 137.0), fov=55), stage="P4",
                      phase=timelapse(C("helping"), Ee("helping"), 1.0, 0.0), actors=[piph] + vill, props=[farm, E.Custom(crops)],
                      titles=[x.title(Wd("helping", "farm"), Ee("helping"), "FIXING THE FARM", color=(230, 200, 90), emoji="🌾", size=90)]))
    # 11 day 40: neighbours
    pip11 = x.pip([(0, (66.0, 139.5))], skin="pip_iron", yaw=PI, pose=lambda t: {"wave": 1.0, "wave_phase": t * 10})
    bis11 = x.biscuit([(0, (67.2, 139.2))], yaw=PI, sit=1.0)
    v11 = [x.mob("villager", [(0, (64.0 + 2.2 * i, 136.0 + (i % 2)))], yaw=0.0) for i in range(3)]
    out.append(E.Shot("day40", E.cam_static((66.5, GY + 3.0, 131.0), (66.0, GY + 1.6, 138.5), fov=50), stage="P4", phase=1.0,
                      actors=[pip11, bis11] + v11, music="resolve",
                      titles=[x.title(Wd("day40", "neighbors"), Ee("day40"), "NEIGHBORS", color=Y, emoji="🏡", size=110)]))
    return out
