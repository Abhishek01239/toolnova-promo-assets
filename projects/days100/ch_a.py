"""Chapters 1-5: spawn, first night, house, farm, mine."""
import numpy as np

import engine as E
import world as WD
from shots import PI, GY, Y0, U, FLOOR, G, B, R, O, Y, Wh, PU, TOP, look_pan

INF = 1e9
S = WD.SPAWN
TREE = WD.FIRST_TREE
TABLE = (-26, GY + 1, 7)


def trunk(x, breaks=(INF, INF, INF, INF, INF), collect=None):
    tx_, tz = TREE
    items = [(breaks[k], (tx_, GY + 1 + k, tz), "log") for k in range(5)]
    return E.Breaks(items, dur=0.7, collect=collect, drop="log", overlay=False)


def rock(x, breaks=None, collect=None):
    br = breaks or {}
    items = [(br.get(p, INF), p, "stone") for p in WD.ROCK]
    return E.Breaks(items, dur=0.6, collect=collect, drop="cobble", overlay=False)


def ch01(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    pp = (S[0], S[2])
    # log break times (shared by every shot of the chapter)
    lb = x.every(T("punch", 0.72), Ee("punch") - 0.4, 4)
    br = (lb[0], lb[1], lb[2], lb[3], INF)
    stone_t = x.every(T("stone", 0.35), T("stone", 0.7), 3)
    rk = {(-15, GY + 1, -3): stone_t[0], (-16, GY + 1, -3): stone_t[1], (-15, GY + 2, -4): stone_t[2]}

    # 1 spawn: crane down onto Pip popping into the world
    t_sp = 0.7
    pip = x.pip([(0, pp)], spawn=t_sp, yaw=0.6,
                pose=lambda t: {"head_yaw": 0.7 * np.sin((t - t_sp) * 1.6) * E.seg(t, t_sp + 0.5, t_sp + 1.0)})
    cam = E.cam_dolly(0.0, T("spawn", 0.7), S + (26, 30, -30), S + (3.6, 2.0, 4.0), S + (0, 0, 0), S + (0, 1.3, 0), fov=55, fov1=48)
    out.append(E.Shot("spawn", cam, stage="P0", phase=0.0, actors=[pip], props=[trunk(x, br), rock(x, rk)], music="adventure",
                      fx=[E.Burst(t_sp, S + (0, 1, 0), 30, speed=(1, 3), life=(0.5, 1.0), size=(0.1, 0.2),
                                  colors=((1, 1, 0.7, 1), (0.8, 1, 1, 1)), gravity=-1, drag=2, up_bias=0.6, seed=1, additive=True)],
                      sfx=[(t_sp, "shimmer", -8, {}), (t_sp, "pop", -10, {})],
                      titles=[x.title(T("spawn", 0.05), C("goal") + 0.2, "DAY 1", color=Y, emoji="☀️", size=140)],
                      hotbar=[None] * 9))
    # 2 goal: close front
    pip2 = x.pip([(0, pp)], yaw=PI / 2, pose=lambda t: {"head_yaw": 0.15 * np.sin(t * 1.3)})
    out.append(E.Shot("goal", E.cam_front(pip2, dist=3.4, up=0.25, side=0.9, fov=40), stage="P0", phase=0.0,
                      actors=[pip2], props=[trunk(x, br), rock(x, rk)], hotbar=[None] * 9,
                      titles=[x.title(T("goal", 0.25), Ee("goal"), "SURVIVE 100 DAYS", color=Y, emoji="💯", size=110)]))
    # 3 inventory: over the shoulder, empty hotbar
    pip3 = x.pip([(0, pp)], yaw=PI * 0.25, pose=lambda t: {"head_pitch": -0.35, "head_yaw": 0.35 * np.sin(t * 3.0)})
    out.append(E.Shot("inventory", E.cam_track(pip3, (-1.6, 2.4, -2.2), look=(0.6, 1.0, 1.2), fov=46), stage="P0",
                      actors=[pip3], props=[trunk(x, br), rock(x, rk)], hotbar=[None] * 9,
                      titles=[x.title(T("inventory", 0.3), Ee("inventory"), "0 ITEMS", color=R, emoji="❌", size=110)]))
    # 4 look: POV pan forest -> river -> hill
    keys = [(C("look"), -1.62), (T("look", 0.28), -1.45), (T("look", 0.5), 1.45), (T("look", 0.66), 1.62), (Ee("look") - 0.3, 2.5)]
    def pan(t, keys=keys):
        a = float(np.interp(t, [k[0] for k in keys], [k[1] for k in keys]))
        pitch = 0.04 + 0.1 * E.seg(t, T("look", 0.66), Ee("look") - 0.3)
        hgt = 1.8 + 16.0 * E.smooth(E.seg(t, T("look", 0.2), T("look", 0.55)))
        eye = S + (0, hgt, 0)
        pitch = pitch - 0.32 * E.smooth(E.seg(t, T("look", 0.2), T("look", 0.55)))
        d = np.array([np.sin(a), pitch, np.cos(a)], np.float32)
        return dict(eye=eye.astype(np.float32), target=(eye + d * 10).astype(np.float32), fov=60, roll=0.0)
    out.append(E.Shot("look", pan, stage="P0", props=[trunk(x, br), rock(x, rk)], hotbar=[None] * 9,
                      titles=[x.title(Wd("look", "forest"), T("look", 0.42), "FOREST", color=G, emoji="🌲"),
                              x.title(Wd("look", "river"), T("look", 0.75), "RIVER", color=B, emoji="🌊"),
                              x.title(Wd("look", "hill"), Ee("look"), "THE HILL", color=Wh, emoji="⛰️")]))
    # 5 punch the tree
    at = (TREE[0] + 0.5, TREE[1] - 1.45)
    acts = []
    tt = T("punch", 0.55)
    while tt < lb[-1]:
        acts.append(tt)
        tt += 0.24
    pip5 = x.pip([(C("punch"), pp), (T("punch", 0.5), at)], acts=acts, yaw=lambda t: 0.0 if t > T("punch", 0.5) else None,
                 pose=lambda t: {"head_pitch": 0.15 + 0.22 * sum(1 for b in lb if t > b)})
    cam = E.cam_dolly(C("punch"), T("punch", 0.6), (-20.2, 12.9, 5.0), (-21.6, 12.7, 6.7), (-23.5, 12.2, 7.0),
                      (-24.5, 12.8, 8.6), fov=52)
    out.append(E.Shot("punch", cam, stage="P0", actors=[pip5],
                      props=[trunk(x, br, collect=lambda t: pip5.pos(t)), rock(x, rk)],
                      hotbar=lambda t: ["log" if t > lb[0] + 0.8 else None] + [None] * 8,
                      titles=[x.title(lb[0] + 0.85, Ee("punch"), "+1 LOG", color=Y, emoji="🪵", pos=(1450, 330), size=80)]))
    # 6 logs -> planks -> crafting table
    tpl = Wd("logs", "table") + 0.4
    pip6 = x.pip([(0, at)], yaw=lambda t: -PI / 2 if t > tpl - 0.6 else 0.0, acts=[tpl - 0.15])
    tb = E.Pops([(tpl, TABLE, "crafting")])
    out.append(E.Shot("logs", E.cam_static((-21.0, 12.8, 5.2), (-24.6, 11.9, 7.9), fov=50), stage="P0", actors=[pip6],
                      props=[trunk(x, br), rock(x, rk), tb],
                      crafts=[(T("logs", 0.0), T("logs", 0.45), [None, None, None, None, "log", None, None, None, None], "planks"),
                              (T("logs", 0.5), Ee("logs"), ["planks", "planks", None, "planks", "planks", None, None, None, None], "crafting")],
                      hotbar=lambda t: (["planks", "crafting" if t < tpl else None] + [None] * 7, 1 if t < tpl else 0)))
    tbc = np.array(TABLE, float) + 0.5
    # 7 the crafting table: slow orbit
    pip7 = x.pip([(0, at)], yaw=-PI / 2, acts=[T("table", 0.3), T("table", 0.75)])
    out.append(E.Shot("table", E.cam_orbit(tbc, 3.8, 1.5, 1.3, 2.3, C("table"), Ee("table"), fov=44, look_y=0.3),
                      stage="P0", actors=[pip7], props=[trunk(x, br), tb],
                      titles=[x.title(T("table", 0.1), Ee("table"), "MOST IMPORTANT BLOCK", color=Y, emoji="🛠️", size=88)],
                      hotbar=["planks"] + [None] * 8))
    # 8 sticks
    out.append(E.Shot("sticks", E.cam_static((-22.4, 12.9, 4.4), (-24.8, 12.2, 7.6), fov=48), stage="P0", actors=[pip7],
                      props=[trunk(x, br), tb],
                      crafts=[(T("sticks", 0.0), Ee("sticks"), [None, "planks", None, None, "planks", None, None, None, None], "t:pick_handle")],
                      hotbar=["planks", "t:pick_handle"] + [None] * 7))
    # 9 wooden pickaxe
    tpk = Wd("pick", "pickaxe")
    pip9 = x.pip([(0, at)], yaw=PI * 0.75, item=[(tpk, "pickaxe")],
                 pose=lambda t: {"arms_up": 0.55 * E.smooth(E.seg(t, tpk, tpk + 0.3)) * (1 - E.seg(t, Ee("pick") - 0.6, Ee("pick")))})
    out.append(E.Shot("pick", E.cam_front(pip9, dist=3.2, up=0.6, side=-0.9, fov=46, look=(0, 1.45, 0)), stage="P0", actors=[pip9],
                      props=[trunk(x, br), tb],
                      crafts=[(T("pick", 0.0), tpk + 0.2, ["planks", "planks", "planks", None, "t:pick_handle", None, None,
                                                           "t:pick_handle", None], "e:⛏️")],
                      titles=[x.title(tpk + 0.1, Ee("pick"), "WOODEN PICKAXE", color=Y, emoji="⛏️")],
                      sfx=[(tpk, "ding", -9, {})], hotbar=(["e:⛏️", "planks"] + [None] * 7, 0)))
    # 10 stone
    rp = (-15.5, -1.4)
    acts = []
    tt = T("stone", 0.25)
    while tt < stone_t[-1]:
        acts.append(tt)
        tt += 0.25
    pip10 = x.pip([(C("stone"), (-19.5, 1.0)), (T("stone", 0.22), rp)], item="pickaxe", acts=acts,
                  yaw=lambda t: PI if t > T("stone", 0.22) else None,
                  pose=lambda t: {"head_pitch": 0.25 if t < stone_t[1] else -0.05})
    out.append(E.Shot("stone", E.cam_static((-12.6, 13.0, 0.6), (-16.0, 11.8, -2.6), fov=50), stage="P0", actors=[pip10],
                      props=[rock(x, rk, collect=lambda t: pip10.pos(t))],
                      crafts=[(T("stone", 0.74), Ee("stone"), ["cobble", "cobble", "cobble", None, "t:pick_handle", None, None,
                                                              "t:pick_handle", None], "e:⛏️")],
                      titles=[x.title(Wd("stone", "twice"), Ee("stone"), "2X FASTER", color=Y, emoji="⚡", pos=(700, 160))],
                      hotbar=lambda t: (["e:⛏️", "planks", "cobble" if t > stone_t[0] + 0.8 else None] + [None] * 6, 0)))
    # 11 stone sword
    pip11 = x.pip([(0, rp)], item="sword:iron", yaw=PI * 0.15, acts=[T("sword1", 0.4), T("sword1", 0.7)])
    out.append(E.Shot("sword1", E.cam_front(pip11, dist=2.8, up=-0.5, side=0.8, fov=50, look=(0, 1.5, 0)), stage="P0",
                      actors=[pip11], props=[rock(x, rk)],
                      titles=[x.title(T("sword1", 0.2), Ee("sword1"), "STONE SWORD", color=Wh, emoji="🗡️")],
                      hotbar=(["e:⛏️", "e:🗡️", "cobble", "planks"] + [None] * 5, 1)))
    # 12 sunset: time lapse, looking towards the low sun
    sp = (S[0] + 3, S[2] - 2)
    pip12 = x.pip([(C("sunset"), rp), (T("sunset", 0.6), sp)], item="sword:iron",
                  pose=lambda t: {"head_pitch": 0.35 * E.smooth(E.seg(t, T("sunset", 0.55), T("sunset", 0.8)))},
                  yaw=lambda t: 1.9 if t > T("sunset", 0.6) else None)
    cam = E.cam_dolly(C("sunset"), Ee("sunset"), (sp[0] - 4.5, 12.6, sp[1] + 2.5), (sp[0] - 3.2, 12.0, sp[1] + 1.8),
                      (sp[0] + 6, 13.5, sp[1] - 3.5), (sp[0] + 8, 16.5, sp[1] - 5.0), fov=55)
    out.append(E.Shot("sunset", cam, stage="P0", phase=lambda t: 0.25 + 0.75 * E.smooth(E.seg(t, C("sunset"), Ee("sunset"))),
                      actors=[pip12], props=[trunk(x, br)], music="suspense",
                      titles=[x.title(T("sunset", 0.6), Ee("sunset"), "SUNSET", color=O, emoji="🌅")]))
    # 13 the night rule: zombies rising at the forest edge
    zs = []
    for i, (zx, zz, tr) in enumerate([(-31.0, 0.0, 0.25), (-33.5, 6.5, 0.45), (-30.0, 12.0, 0.65)]):
        t0 = T("nightrule", tr)
        zs.append(x.mob("zombie", [(t0, (zx, zz)), (t0 + 1.2, (zx, zz)), (Ee("nightrule"), (zx + 1.8, zz - 0.4))], spawn=t0,
                        y_off=lambda t, t0=t0: -1.9 * (1 - E.smooth(E.seg(t, t0, t0 + 1.1)))))
    pip13 = x.pip([(0, sp)], item="sword:iron", yaw=-PI / 2 - 0.2)
    out.append(E.Shot("nightrule", E.cam_static((sp[0] + 3.5, 12.2, sp[1] + 2.5), (sp[0] - 8, 12.2, sp[1] + 3.0), fov=52),
                      stage="P0", phase=lambda t: 1.0 + 0.55 * E.seg(t, C("nightrule"), Ee("nightrule")),
                      actors=[pip13] + zs, props=[trunk(x, br)],
                      fx=[E.debris(z.spawn, z.path.pos(z.spawn) + (0, 0.2, 0), "dirt", 70 + i) for i, z in enumerate(zs)],
                      sfx=[(z.spawn + 0.3, "groan", -12, dict(seed=i)) for i, z in enumerate(zs)],
                      titles=[x.title(T("nightrule", 0.6), Ee("nightrule"), "NO SHELTER", color=R, emoji="⚠️")]))
    return out


def ch02(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    pp = (S[0], S[2])
    hc = np.array([S[0], GY + 2.5, S[2]])
    x0, x1, z0, z1 = WD.HUT
    blocks = WD.hut_blocks_ordered()
    t_b0, t_b1 = T("hut", 0.2), Ee("hut") - 0.5
    # 1 build the hut around Pip
    acts = x.every(t_b0, t_b1, 28)
    rev = E.Reveal("hut", [(p, b) for p, b in blocks], t_b0, t_b1, K=30)
    pip1 = x.pip([(C("hut"), (S[0] + 4, S[2] - 3)), (t_b0 - 0.1, pp)], item="dirt", acts=acts,
                 yaw=lambda t: (t - t_b0) * 2.6 if t > t_b0 else None)
    out.append(E.Shot("hut", E.cam_orbit(hc, 9.0, 3.5, 0.7, 2.2, C("hut"), Ee("hut"), fov=50, look_y=-0.6),
                      stage="P0", phase=lambda t: 1.55 + 0.3 * E.seg(t, C("hut"), Ee("hut")), actors=[pip1], props=[rev],
                      music="panic", titles=[x.title(T("hut", 0.6), Ee("hut"), "DIRT HUT", color=O, emoji="🟫")],
                      hotbar=(["dirt", "e:⛏️", "e:🗡️"] + [None] * 6, 0)))
    night = 2.0
    # 2 ugly
    out.append(E.Shot("ugly", E.cam_dolly(C("ugly"), Ee("ugly"), hc + (6.5, 1.2, -7.5), hc + (5.0, 0.6, -6.0), hc, hc + (0, -0.3, 0),
                                          fov=50), stage="P0", phase=night, statics=["hut"], music="suspense",
                      titles=[x.title(Wd("ugly", "ugly"), Wd("ugly", "Does"), "UGLY", color=O, emoji="✅"),
                              x.title(Wd("ugly", "work"), Ee("ugly"), "WORKS", color=G, emoji="✅")]))
    dark = dict(ambient=(0.24, 0.23, 0.3), sun=(0.03, 0.03, 0.04), exposure=1.0)
    # 3 dark inside
    pip3 = x.pip([(0, pp)], yaw=PI * 0.8, pose=lambda t: {"head_yaw": 0.6 * np.sin(t * 0.9)})
    out.append(E.Shot("dark", E.cam_static((x0 + 1.3, GY + 2.6, z1 - 1.4), (S[0], GY + 2.2, S[2] - 0.2), fov=62), stage="P0",
                      phase=night, statics=["hut"], actors=[pip3], light=dark,
                      fog=dict(color=(0.0, 0.0, 0.01), start=2.0, end=12.0), clouds=False,
                      sfx=[(T("dark", 0.3), "heartbeat", -10, {}), (T("dark", 0.7), "heartbeat", -11, dict(seed=1))]))
    # 4 groan: a zombie rises next to the hut
    tz = T("groan", 0.55)
    z1p = (x0 - 3.0, z0 - 1.5)
    zom = x.mob("zombie", [(tz, z1p), (Ee("zombie"), (x0 - 1.0, z0 + 1.5))], spawn=tz,
                y_off=lambda t: -1.9 * (1 - E.smooth(E.seg(t, tz, tz + 1.2))))
    out.append(E.Shot("groan", E.cam_static((x0 - 5.5, GY + 1.6, z0 - 6.0), (x0 - 1.5, GY + 1.8, z0 + 1.0), fov=50), stage="P0",
                      phase=night, statics=["hut"], actors=[zom], fx=[E.debris(tz, (z1p[0], GY + 1.2, z1p[1]), "dirt", 3)],
                      sfx=[(tz + 0.2, "dig", -10, {}), (tz + 0.5, "groan", -6, dict(seed=3))]))
    # 5 zombie bumps into the wall
    out.append(E.Shot("zombie", E.cam_static((x0 - 2.5, GY + 2.4, z0 - 6.5), (x0 - 0.5, GY + 2.0, z0 + 1.0), fov=46), stage="P0",
                      phase=night, statics=["hut"], actors=[zom],
                      titles=[x.title(T("zombie", 0.1), Ee("zombie"), "ZOMBIE", color=G, emoji="🧟")],
                      sfx=[(T("zombie", 0.5), "groan", -8, dict(seed=4)), (T("zombie", 0.8), "impact", -18, {})]))
    # 6 peek through the window (POV)
    wz = z0
    eye = np.array([x0 + 2.5, GY + 2.55, wz + 1.05])
    za = x.mob("zombie", [(C("peek"), (x0 + 4.0, wz - 4.0)), (Ee("peek"), (x0 + 1.0, wz - 3.0))])
    zb = x.mob("zombie", [(T("peek", 0.35), (x0 - 4.0, wz - 8.0)), (Ee("peek"), (x0 + 3.5, wz - 5.0))], spawn=T("peek", 0.35))
    spd = x.mob("spider", [(T("peek", 0.75), (x0 - 2.0, wz - 12.0)), (Ee("peek"), (x0 + 2.0, wz - 6.0))], spawn=T("peek", 0.75))
    out.append(E.Shot("peek", E.cam_static(eye, eye + (0.0, -0.25, -6.0), fov=58), stage="P0", phase=night, statics=["hut"],
                      actors=[za, zb, spd], clouds=False,
                      sfx=[(T("peek", 0.4), "groan", -12, dict(seed=5)), (T("peek", 0.8), "hiss", -16, dict(dur=0.8))]))
    # 7 spider climbs the hut
    tcl0, tcl1 = T("spider", 0.25), T("spider", 0.75)
    sp2 = x.mob("spider", [(C("spider"), (x0 + 2.5, wz - 4.0)), (tcl0, (x0 + 2.5, wz - 0.9)),
                           (tcl1, (x0 + 2.5, GY + 4.3, wz - 0.5)), (tcl1 + 0.6, (x0 + 2.5, GY + 5.0, wz + 1.5)),
                           (Ee("spider"), (x0 + 2.5, GY + 5.0, wz + 2.4))],
                climb=lambda t: E.smooth(E.seg(t, tcl0 - 0.2, tcl0 + 0.2)) * (1 - E.smooth(E.seg(t, tcl1, tcl1 + 0.4))))
    out.append(E.Shot("spider", E.cam_static((x0 + 7.0, GY + 3.2, wz - 6.5), (x0 + 2.5, GY + 3.0, wz), fov=52), stage="P0",
                      phase=night, statics=["hut"], actors=[sp2, za],
                      titles=[x.title(T("spider", 0.15), Ee("spider"), "SPIDERS CLIMB", color=R, emoji="🕷️")],
                      sfx=[(T("spider", 0.3), "hiss", -14, dict(dur=1.0))]))
    # 8 wait: slow orbit, mobs circling
    def circ(r, ph, sp, t0, t1, n=24):
        return [(t0 + (t1 - t0) * k / n, (S[0] + r * np.cos(ph + sp * k / n * 2 * PI), S[2] + r * np.sin(ph + sp * k / n * 2 * PI)))
                for k in range(n + 1)]
    m1 = x.mob("zombie", circ(4.6, 0.0, 0.5, C("wait"), Ee("wait")))
    m2 = x.mob("zombie", circ(5.4, 2.5, 0.4, C("wait"), Ee("wait")))
    sp3 = x.mob("spider", [(0, (S[0], GY + 5.0, S[2] - 1.0))], yaw=0.4)
    out.append(E.Shot("wait", E.cam_orbit(hc, 13.0, 6.0, 2.4, 3.6, C("wait"), Ee("wait"), fov=48, look_y=-1.0),
                      stage="P0", phase=night, statics=["hut"], actors=[m1, m2, sp3],
                      titles=[x.title(Wd("wait", "seven"), T("wait", 0.75), "7 MINUTES", color=Wh, emoji="⏳"),
                              x.title(Wd("wait", "hour"), Ee("wait"), "= 1 HOUR", color=O, emoji="😩")],
                      sfx=[(T("wait", 0.2), "groan", -16, dict(seed=8)), (T("wait", 0.7), "groan", -17, dict(seed=9))]))
    # 9 dawn
    out.append(E.Shot("dawn", E.cam_static(hc + (9.0, 1.5, 6.0), hc + (-6.0, 3.0, -2.0), fov=55), stage="P0",
                      phase=lambda t: 2.0 + E.smooth(E.seg(t, C("dawn"), Ee("dawn"))), statics=["hut"], actors=[m1, m2],
                      music="resolve", titles=[x.title(T("dawn", 0.5), Ee("dawn"), "SUNRISE", color=O, emoji="🌅")]))
    # 10 burn: zombies catch fire, Pip breaks out
    tb, td = T("burn", 0.25), T("burn", 0.85)
    m1b = x.mob("zombie", [(C("burn"), (S[0] + 3.6, S[2] + 2.6)), (Ee("burn"), (S[0] + 6.0, S[2] + 4.0))], burn=(tb, td), death=td)
    m2b = x.mob("zombie", [(C("burn"), (S[0] - 1.0, S[2] - 4.2)), (Ee("burn"), (S[0] - 3.0, S[2] - 6.0))], burn=(tb + 0.2, td + 0.2),
                death=td + 0.2)
    to = T("burn", 0.5)
    pip10 = x.pip([(C("burn"), pp), (to - 0.2, (S[0] + 0.6, S[2])), (to + 0.9, (S[0] + 3.5, S[2] + 0.2))], item="sword:iron",
                  acts=[to - 0.5, to - 0.3], hidden=lambda t: False)
    out.append(E.Shot("burn", E.cam_static(hc + (7.5, 0.5, -3.5), hc + (0.5, -0.8, 0.6), fov=50), stage="P0",
                      phase=lambda t: 3.0 + 0.6 * E.seg(t, C("burn"), Ee("burn")),
                      statics=["hut_open"], actors=[m1b, m2b, pip10],
                      fx=[E.debris(to - 0.3, (x1 - 0.5, GY + 1.5, z0 + 2.5), "dirt", 11)],
                      titles=[x.title(T("burn", 0.62), Ee("burn"), "NIGHT 1 SURVIVED", color=G, emoji="✅")],
                      sfx=[(tb, "crackle", -10, dict(dur=2.5)), (to - 0.3, "dig", -8, {})]))
    # 11 flesh
    pip11 = x.pip([(0, (S[0] + 3.5, S[2] + 0.2)), (Ee("flesh"), (S[0] + 5.2, S[2] + 2.8))], item="sword:iron",
                  pose=lambda t: {"head_pitch": -0.3})
    out.append(E.Shot("flesh", E.cam_track(pip11, (2.6, 1.8, -2.4), look=(0.6, 0.6, 0.8), fov=48), stage="P0", phase=3.65,
                      statics=["hut_open"], actors=[pip11],
                      props=[E.Fly([(C("flesh") + 0.3, T("flesh", 0.5), (S[0] + 6.0, GY + 1.3, S[2] + 4.0), (S[0] + 4.0, GY + 2.0, S[2] + 1.0), "flesh")])],
                      titles=[x.title(T("flesh", 0.15), Ee("flesh"), "ROTTEN FLESH", color=(190, 120, 90), emoji="🤢", size=92)]))
    # 12 ninety-nine more: walking off into the morning
    pip12 = x.pip([(C("ninetynine"), (S[0] + 5.2, S[2] + 2.8)), (Ee("ninetynine") + 2, (S[0] + 15.0, S[2] + 3.5))], item="sword:iron")
    out.append(E.Shot("ninetynine", E.cam_static((S[0] + 5.0, GY + 4.0, S[2] - 5.5), (S[0] + 13.0, GY + 1.5, S[2] + 3.0), fov=55),
                      stage="P0", phase=3.8, statics=["hut_open"], actors=[pip12],
                      titles=[x.title(T("ninetynine", 0.1), Ee("ninetynine"), "99 TO GO", color=Y, emoji="💪", size=120)]))
    return out


def timelapse(t0, t1, cycles=1.0, start=0.0):
    """Phase that runs full day/night cycles (0 day -> 1 golden -> 2 night -> 3 dawn -> 4=day)."""
    def fn(t):
        p = start + 4.0 * cycles * E.seg(t, t0, t1)
        p = p % 4.0
        return p
    return fn


def house_blocks():
    base = WD.stage_world("P0").blocks
    tgt = WD.stage_world("P1").blocks
    diff = [(p, b) for p, b in tgt.items() if base.get(p) != b and p[1] > GY]
    c = (10.5, 11.5)
    diff.sort(key=lambda kv: (kv[0][1], np.arctan2(kv[0][2] - c[1], kv[0][0] - c[0])))
    return diff


def circle_keys(c, r, t0, t1, turns=1.0, a0=0.0, n=40):
    return [(t0 + (t1 - t0) * k / n, (c[0] + r * np.cos(a0 + turns * 2 * PI * k / n), c[1] + r * np.sin(a0 + turns * 2 * PI * k / n)))
            for k in range(n + 1)]


HC = np.array([10.5, GY + 2.0, 11.5])      # house centre


def ch03(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    # 1 day 2: walking to the river site
    pip1 = x.pip([(C("day2"), (S[0] + 4, S[2] + 1)), (Ee("day2"), (8.0, 4.0))], item="pickaxe:stone")
    out.append(E.Shot("day2", E.cam_follow(pip1, back=9.0, up=6.0, side=2.0, fov=55, look_ahead=6.0), stage="P0", phase=0.0,
                      statics=["hut_open"], actors=[pip1], music="adventure",
                      titles=[x.title(Wd("day2", "river"), Ee("day2"), "BY THE RIVER", color=B, emoji="🌊")]))
    # 2-3 building the house (time-lapse)
    hb = house_blocks()
    t0, t1 = T("build", 0.05), Ee("windows") - 0.6
    rev = E.Reveal("house", hb, t0, t1, K=40)
    acts = x.every(t0, t1, int((t1 - t0) / 0.28))
    pipb = x.pip(circle_keys((10.5, 11.5), 6.0, C("build"), Ee("windows"), turns=1.6, a0=-PI / 2), item="planks", acts=acts,
                 yaw=lambda t: None)
    tl = timelapse(t0, t1, 1.5, 0.0)
    out.append(E.Shot("build", E.cam_orbit(HC, 17.0, 8.0, -0.5, 0.6, C("build"), Ee("build"), fov=48, look_y=0.5), stage="P0",
                      phase=tl, actors=[pipb], props=[rev], music="build", hotbar=(["planks", "log", "e:🪓", "e:⛏️"] + [None] * 5, 0),
                      titles=[x.title(Wd("build", "Logs"), Wd("build", "planks") + 0.1, "LOGS", color=(200, 160, 100), emoji="🪵", pos=(700, 160)),
                              x.title(Wd("build", "planks"), Wd("build", "roof"), "PLANKS", color=(230, 190, 120), emoji="🟫", pos=(960, 160)),
                              x.title(Wd("build", "roof"), Ee("build"), "ROOF", color=Y, emoji="🏠", pos=(1220, 160))]))
    out.append(E.Shot("windows", E.cam_orbit(HC, 11.0, 2.5, 0.8, 1.6, C("windows"), Ee("windows"), fov=50, look_y=0.6), stage="P0",
                      phase=tl, actors=[pipb], props=[rev], hotbar=(["glass", "planks", "log"] + [None] * 6, 0),
                      titles=[x.title(T("windows", 0.3), Ee("windows"), "WINDOWS", color=B, emoji="🪟")]))
    # 4 door at night
    dz = WD.DOOR
    zom = x.mob("zombie", [(C("door"), (dz[0] + 0.5 - 4, dz[1] - 6)), (T("door", 0.35), (dz[0] + 0.5, dz[1] - 0.8)),
                           (Ee("door"), (dz[0] + 0.6, dz[1] - 0.8))])
    out.append(E.Shot("door", E.cam_static((dz[0] + 4.5, GY + 2.6, dz[1] - 5.5), (dz[0] + 0.5, GY + 2.0, dz[1]), fov=50), stage="P1",
                      phase=2.0, actors=[zom], torches_on=False,
                      titles=[x.title(T("door", 0.1), Wd("door", "At"), "ZOMBIES CAN'T OPEN DOORS", color=G, emoji="🚪", size=80),
                              x.title(Wd("door", "normal"), Ee("door"), "(ON NORMAL)", color=Wh, size=70)],
                      sfx=[(T("door", 0.4), "groan", -9, dict(seed=21)), (T("door", 0.6), "impact", -16, {}),
                           (T("door", 0.8), "impact", -16, dict(seed=2))]))
    # 5 sheep in the meadow
    mx, mz = WD.MEADOW
    sheep = []
    for i, (dx, dz2, yw) in enumerate([(-2, 1, 0.3), (1.5, -1.5, 2.0), (3, 2, -1.0), (-1, -3.5, 1.2), (0.5, 3.5, 2.8)]):
        sheep.append(x.mob("sheep", [(C("sheep"), (mx + dx, mz + dz2)), (Ee("wool") + 1, (mx + dx + 0.6 * np.sin(i), mz + dz2 + 0.5))],
                           yaw=yw + 0.2 * i, pose=lambda t, i=i: {"head_pitch": 0.45 * (np.sin(t * 0.8 + i) > 0.2)}))
    pip5 = x.pip([(C("sheep"), (mx + 12, mz + 3)), (T("sheep", 0.8), (mx + 5.5, mz + 1))], item="sword:iron")
    out.append(E.Shot("sheep", E.cam_static((mx + 9, GY + 4.5, mz + 9), (mx + 1, GY + 1.5, mz), fov=50), stage="P0", phase=0.0,
                      actors=sheep + [pip5], titles=[x.title(Wd("sheep", "sheep"), Ee("sheep"), "SHEEP", color=Wh, emoji="🐑")]))
    pip6 = x.pip([(0, (mx + 5.5, mz + 1))], item=None, yaw=-PI / 2 - 0.3)
    out.append(E.Shot("wool", E.cam_static((mx + 9.5, GY + 2.8, mz + 4.2), (mx + 2.5, GY + 1.6, mz - 0.5), fov=46), stage="P0",
                      phase=0.0, actors=sheep + [pip6],
                      crafts=[(T("wool", 0.0), Ee("wool"), ["wool", "wool", "wool", "planks", "planks", "planks", None, None, None], "e:🛏️")],
                      titles=[x.title(Wd("wool", "cheat"), Ee("wool"), "CHEAT CODE", color=Y, emoji="🛏️", pos=(560, 160))]))
    # 7 sleep: inside, lying in bed, night skips to morning
    t_ly = T("sleep", 0.15)
    pipS = x.pip([(0, (7.5, GY + 0.72, 12.25))], yaw=0.0, pose=lambda t: {"body_pitch": -PI / 2})
    inside = lambda t: dict(ambient=tuple(np.array([0.55, 0.5, 0.48]) * (0.55 + 0.45 * (1 - min(1, abs(f.phase(t) - 2.0))))), exposure=1.1)
    out.append(E.Shot("sleep", E.cam_static((12.6, GY + 3.7, 13.5), (7.8, GY + 1.6, 12.0), fov=58), stage="P1",
                      phase=lambda t: 2.0 + 1.9 * E.smooth(E.seg(t, T("sleep", 0.45), Ee("sleep") - 0.4)), actors=[pipS],
                      titles=[x.title(T("sleep", 0.3), T("sleep", 0.75), "SKIP THE NIGHT", color=B, emoji="💤"),
                              x.title(Wd("sleep", "morning"), Ee("sleep"), "GOOD MORNING", color=Y, emoji="☀️")],
                      flash=[(Wd("sleep", "morning") - 0.3, 0.6, 0.5)], clouds=False, torches_on=False))
    # 8 spawn point
    pip8 = x.pip([(0, (9.6, 12.4))], yaw=-PI / 2 - 0.4, pose=lambda t: {"head_yaw": -0.3})
    out.append(E.Shot("spawnpoint", E.cam_static((12.8, GY + 2.9, 9.7), (8.3, GY + 1.0, 12.4), fov=55), stage="P1", phase=3.9,
                      actors=[pip8], clouds=False,
                      titles=[x.title(T("spawnpoint", 0.15), Ee("spawnpoint"), "SPAWN POINT SET", color=G, emoji="🏠", size=92)]))
    # 9 chest
    pip9 = x.pip([(0, (11.8, 12.6))], yaw=PI / 2 + 0.4, acts=[T("chest", 0.75)])
    out.append(E.Shot("chest", E.cam_static((9.0, GY + 2.7, 10.0), (13.5, GY + 1.0, 13.4), fov=52), stage="P1", phase=0.0,
                      actors=[pip9], clouds=False,
                      crafts=[(T("chest", 0.0), T("chest", 0.6), ["planks", "planks", "planks", "planks", None, "planks", "planks",
                                                                  "planks", "planks"], "e:📦")],
                      titles=[x.title(Wd("chest", "twenty-seven"), Ee("chest"), "27 SLOTS", color=Y, emoji="📦", pos=(560, 160))]))
    # 10 day 5: starving
    pip10 = x.pip([(0, (10.5, 6.2))], yaw=PI, pose=lambda t: {"head_pitch": -0.25, "lean": 0.15})
    out.append(E.Shot("day5", E.cam_front(pip10, dist=4.0, up=0.6, side=1.2, fov=48), stage="P1", phase=0.95, actors=[pip10],
                      titles=[x.title(Wd("day5", "house"), Wd("day5", "But"), "HOUSE ✅  BED ✅  CHEST ✅", color=G, size=74),
                              x.title(Wd("day5", "starving"), Ee("day5"), "STARVING", color=R, emoji="😩")],
                      hp=lambda t: 10.0 - 3.5 * E.seg(t, T("day5", 0.6), T("day5", 0.9)),
                      sfx=[(Wd("day5", "starving"), "groan", -14, dict(seed=40))]))
    return out


def ch04(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    fx0, fx1, fz0, fz1 = WD.FARM
    fc = np.array([(fx0 + fx1) / 2, GY + 1.0, (fz0 + fz1) / 2])
    # 1 hungry: an apple
    pip1 = x.pip([(0, (10.5, 6.2))], yaw=PI, pose=lambda t: {"punch": 0.35 + 0.15 * np.sin(t * 18) if T("hungry", 0.2) < t < T("hungry", 0.6) else 0.0,
                                                           "head_pitch": -0.15})
    out.append(E.Shot("hungry", E.cam_front(pip1, dist=3.2, up=0.4, side=-1.0, fov=46), stage="P1", phase=0.0, actors=[pip1],
                      music="adventure", hp=6.5,
                      titles=[x.title(Wd("hungry", "Apples"), Ee("hungry"), "NOT ENOUGH", color=R, emoji="🍎")],
                      sfx=[(T("hungry", 0.2) + 0.35 * j, "step", -12, dict(seed=50 + j)) for j in range(3)]))
    # 2 hoe: tilling the field row by row
    cells = [(xx, zz) for xx in range(fx0, fx1) for zz in range(fz0, fz1) if xx != WD.CANAL_X]
    cells.sort(key=lambda c: (c[0], c[1] if c[0] % 2 == 0 else -c[1]))
    t0, t1 = T("hoe", 0.35), Ee("hoe") - 0.3
    tt = np.linspace(t0, t1, len(cells))
    till = E.Pops([(tt[i], (c[0], GY, c[1]), "farmland") for i, c in enumerate(cells)], sound=False, scale=1.012)
    def seeds(t, tt=tt, cells=cells):
        from mcengine import models100 as M
        v = [M.crop_vertices((c[0], GY + 1, c[1]), "crop_young", 0.35) for i, c in enumerate(cells) if t > tt[i] + 0.4]
        return [np.concatenate(v)] if v else [], [], []
    hk = [(C("hoe"), (fx0 - 2.0, fz0 - 2.0))] + [(tt[i], (c[0] + 0.5, c[1] - 0.6 if c[0] % 2 == 0 else c[1] + 1.6)) for i, c in enumerate(cells) if i % 7 == 0]
    pip2 = x.pip(hk, item=[(0, None), (T("hoe", 0.3), "hoe")], acts=list(tt[::3]))
    out.append(E.Shot("hoe", E.cam_dolly(C("hoe"), Ee("hoe"), fc + (-10, 7, -12), fc + (-6, 6, -9), fc, fc, fov=50), stage="P1",
                      phase=0.0, actors=[pip2], props=[till, E.Custom(seeds)], hp=6.5,
                      crafts=[(T("hoe", 0.0), T("hoe", 0.3), ["cobble", "cobble", None, None, "t:pick_handle", None, None,
                                                              "t:pick_handle", None], "t:hoe_head")],
                      sfx=[(tt[i], "dig", -16, dict(seed=i % 9)) for i in range(0, len(tt), 4)]))
    # 3 water canal
    tw = np.linspace(T("water", 0.45), T("water", 0.95), fz1 - fz0)
    canal = E.Pops([(tw[i], (WD.CANAL_X, GY, fz0 + i), "water") for i in range(fz1 - fz0)], sound=False, scale=1.012)
    farm_done = E.Pops([(0, (c[0], GY, c[1]), "farmland") for c in cells], sound=False, scale=1.012)
    def seeds2(t):
        from mcengine import models100 as M
        return [np.concatenate([M.crop_vertices((c[0], GY + 1, c[1]), "crop_young", 0.35) for c in cells])], [], []
    pip3 = x.pip([(C("water"), (WD.CANAL_X + 0.5, fz0 - 1.5)), (T("water", 0.4), (WD.CANAL_X + 0.5, fz0 - 0.8)),
                  (Ee("water"), (WD.CANAL_X + 0.5, fz1 - 1.0))], item=[(0, "water_bucket"), (T("water", 0.45), "bucket")],
                 yaw=0.0, pose=lambda t: {"reach": 0.6 if t > T("water", 0.4) else 0.0})
    out.append(E.Shot("water", E.cam_static(fc + (9.0, 13.0, -9.0), fc + (-0.5, -1.0, 0.5), fov=52), stage="P1", phase=0.0,
                      actors=[pip3], props=[farm_done, canal, E.Custom(seeds2)], hp=6.5,
                      titles=[x.title(Wd("water", "four"), Ee("water"), "WATER = 4 BLOCKS", color=B, emoji="💧", size=90)],
                      sfx=[(T("water", 0.45), "water", -9, dict(dur=2.0))]))
    # 4 grow: time-lapse
    tg0, tg1 = C("grow"), Ee("grow")
    crops = [(tg0, "young"), (tg0 + (tg1 - tg0) * 0.35, "mid"), (tg0 + (tg1 - tg0) * 0.7, "gold")]
    out.append(E.Shot("grow", E.cam_orbit(fc, 12.5, 6.5, 0.7, 1.4, tg0, tg1, fov=50, look_y=-0.5), stage="P1b",
                      phase=timelapse(tg0, tg1, 1.5, 0.0), crops=crops, hp=6.5,
                      titles=[x.title(Wd("grow", "golden"), Ee("grow"), "GOLDEN WHEAT", color=Y, emoji="🌾")]))
    # 5 bread
    hv = x.every(T("bread2", 0.1), Ee("bread2") - 0.4, 8)
    pip5 = x.pip([(C("bread2"), (fx0 + 0.5, fz0 - 0.8)), (Ee("bread2"), (fx0 + 3.5, fz0 - 0.8))], acts=hv, yaw=0.0,
                 pose=lambda t: {"head_pitch": -0.3})
    out.append(E.Shot("bread2", E.cam_static((fx0 - 3.0, GY + 3.2, fz0 - 4.5), (fx0 + 2.5, GY + 1.2, fz0 + 1.0), fov=50), stage="P1b",
                      phase=0.0, crops="gold", actors=[pip5], hp=lambda t: 6.5 + 3.5 * E.seg(t, T("bread2", 0.7), Ee("bread2")),
                      crafts=[(T("bread2", 0.0), Ee("bread2"), ["t:wheat_item", "t:wheat_item", "t:wheat_item", None, None, None, None, None, None], "e:🍞")],
                      titles=[x.title(Wd("bread2", "twenty"), Ee("bread2"), "x20", color=Y, emoji="🍞", pos=(560, 160), size=120)],
                      sfx=[(h + 0.1, "dig", -14, dict(seed=60 + i)) for i, h in enumerate(hv)]))
    # 6 cows follow the wheat
    pip6 = x.pip([(C("cows"), (4.0, 60.0)), (Ee("cows"), (8.0, 42.0))], item="wheat")
    cows = [x.follow(pip6, C("cows"), Ee("cows"), kind="cow", lag=1.0 + 0.6 * i, side=(-1.2, 1.0, 0.0)[i], back=2.2 + 1.6 * i)
            for i in range(3)]
    out.append(E.Shot("cows", E.cam_track(pip6, (6.5, 3.0, 4.0), look=(0, 1.0, 3.0), fov=52), stage="P1b", phase=0.0, crops="gold",
                      actors=[pip6] + cows,
                      titles=[x.title(Wd("cows", "Cows"), Ee("cows"), "COWS LOVE WHEAT", color=Wh, emoji="🐄")],
                      sfx=[(T("cows", 0.4), "groan", -22, dict(seed=70))]))
    # 7 into the pen
    px0, px1, pz0, pz1 = WD.PEN
    pip7 = x.pip([(C("pen"), (9.5, 34.0)), (T("pen", 0.5), (9.8, 24.0)), (T("pen", 0.75), (9.8, 21.5)),
                  (Ee("pen"), (9.8, 16.6))], item="wheat")
    cows7 = [x.mob("cow", [(C("pen"), (8.0 + 1.5 * i, 37.0 + 2.5 * i)), (T("pen", 0.75), (8.5 + 1.2 * i, 26.0 - 1.0 * i)),
                           (Ee("pen"), (7.5 + 2.0 * i, 22.5 + 0.8 * i))]) for i in range(3)]
    out.append(E.Shot("pen", E.cam_static((17.5, GY + 5.0, 33.0), (9.5, GY + 1.0, 23.0), fov=52), stage="P2", phase=0.0,
                      actors=[pip7] + cows7, props=[E.Pops([(T("pen", 0.95), (9, GY + 1, pz0), "planks"), (T("pen", 0.98), (10, GY + 1, pz0), "planks")])],
                      titles=[x.title(T("pen", 0.6), Ee("pen"), "HOME", color=G, emoji="🏡")]))
    # 8 fence fact
    cowf = x.mob("cow", [(0, (8.0, 21.8))], yaw=PI)
    out.append(E.Shot("fence", E.cam_static((9.0, GY + 1.6, 16.8), (8.6, GY + 1.0, 19.5), fov=50), stage="P2", phase=0.0,
                      actors=[cowf] + cows7[1:],
                      titles=[x.title(Wd("fence", "one"), Ee("fence"), "1.5 BLOCKS TALL", color=Y, emoji="📏")]))
    # 9 breed
    tb = T("breed", 0.55)
    ca = x.mob("cow", [(0, (8.0, 22.5))], yaw=PI / 2)
    cb = x.mob("cow", [(0, (11.0, 22.5))], yaw=-PI / 2)
    baby = x.mob("baby_cow", [(tb, (9.5, 23.6)), (Ee("breed"), (9.7, 23.0))], spawn=tb, yaw=PI)
    pip9 = x.pip([(0, (9.6, 17.4))], item="wheat", yaw=0.0, acts=[T("breed", 0.15), T("breed", 0.3)])
    out.append(E.Shot("breed", E.cam_static((13.5, GY + 2.6, 18.6), (9.4, GY + 0.9, 22.8), fov=48), stage="P2", phase=0.0,
                      actors=[ca, cb, baby, pip9],
                      fx=[E.hearts_burst(T("breed", 0.3), (8.4, GY + 2.2, 22.5), 1), E.hearts_burst(T("breed", 0.35), (10.6, GY + 2.2, 22.5), 2),
                          E.poof(tb, (9.5, GY + 1.5, 23.6), 3)],
                      titles=[x.title(tb, Ee("breed"), "BABY COW!", color=Wh, emoji="🐮")], sfx=[(tb, "pop", -8, {}), (tb, "shimmer", -12, {})]))
    # 10 day 10: the farm at golden hour
    cows10 = [x.mob("cow", [(0, (7.0 + 2.2 * i, 21.0 + 1.3 * (i % 2)))], yaw=0.5 * i) for i in range(4)]
    out.append(E.Shot("day10", E.cam_orbit(np.array([14.0, GY + 1.0, 14.0]), 26.0, 14.0, -2.4, -1.7, C("day10"), Ee("day10"), fov=48),
                      stage="P2", phase=0.95, actors=cows10, music="resolve",
                      titles=[x.title(Wd("day10", "Unlimited"), Ee("day10"), "UNLIMITED FOOD", color=Y, emoji="♾️")]))
    return out


# ---------------------------------------------------------------- the mine (Day 2 cave set)
def ug(x_, y, z):
    return np.array([x_, y, z], float) + U


COAL = [(35, -31, -2), (36, -31, -2), (36, -30, -2), (37, -31, -2)]
IRON = [(47, -31, -12), (47, -30, -12), (47, -31, -13), (47, -30, -11)]
CAVE_TORCHES = [(33.5, 3.6), (38.5, 3.6), (43.5, 0.5), (48.5, -4.5), (49.5, -9.0), (49.5, -14.0)]


def ch05(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    cx, cz = WD.CAVE_MOUTH
    # 1 the cave mouth
    pip1 = x.pip([(C("cave"), (cx + 0.5, Y0, cz + 9.0)), (Ee("cave"), (cx + 0.5, Y0, cz - 5.0))], item="pickaxe:stone")
    out.append(E.Shot("cave", E.cam_static((cx + 3.5, GY + 2.6, cz + 12.0), (cx + 0.5, GY + 2.0, cz - 2.0), fov=50), stage="P2",
                      phase=0.0, actors=[pip1], music="suspense",
                      titles=[x.title(T("cave", 0.5), Ee("cave"), "THE CAVE", color=Wh, emoji="⛏️")]))
    F = FLOOR
    # 2 coal on the porch wall
    ct = x.every(T("coal", 0.2), Ee("coal") - 0.3, len(COAL))
    acts = x.every(T("coal", 0.15), Ee("coal") - 0.3, 14)
    pip2 = x.pip([(C("coal"), ug(31.0, F, 1.5)), (T("coal", 0.18), ug(36.0, F, -0.4))], item="pickaxe:stone", acts=acts,
                 yaw=lambda t: PI if t > T("coal", 0.18) else None)
    coal = E.Breaks([(ct[i], ug(*p), "coal_ore") for i, p in enumerate(COAL)], collect=lambda t: pip2.pos(t), drop="coal")
    tor0 = E.Torches([(0, ug(31.5, F, 3.6))])
    out.append(E.Shot("coal", E.cam_static(ug(38.8, F + 2.6, 3.0), ug(35.8, F + 1.0, -1.6), fov=55), loc="under", actors=[pip2],
                      props=[coal, tor0], titles=[x.title(T("coal", 0.05), Ee("coal"), "COAL!", color=(180, 180, 180), emoji="⚫")]))
    # 3 torches like breadcrumbs
    ttor = x.every(T("light", 0.1), Ee("light") - 0.2, len(CAVE_TORCHES))
    keys = [(C("light"), ug(33.0, F, 2.6))]
    for i, (tx_, tz) in enumerate(CAVE_TORCHES):
        keys.append((ttor[i] - 0.3, ug(tx_ + 0.5, F, tz - 0.9)))
    pip3 = x.pip(keys, item="torch")
    tor = E.Torches([(0, ug(31.5, F, 3.6))] + [(ttor[i], ug(tx_, F, tz)) for i, (tx_, tz) in enumerate(CAVE_TORCHES)])
    out.append(E.Shot("light", E.cam_follow(pip3, back=5.5, up=2.8, side=1.5, fov=56), loc="under", actors=[pip3], props=[tor],
                      titles=[x.title(Wd("light", "dark"), Wd("light", "So"), "MOBS SPAWN IN THE DARK", color=R, emoji="🌑", size=80),
                              x.title(Wd("light", "torches"), Ee("light"), "TORCHES = SAFE", color=Y, emoji="🔥", size=90)]))
    # 4 echo: slow push across the lake cavern
    pip4 = x.pip([(C("echo"), ug(50.0, F, -1.0)), (Ee("echo"), ug(58.0, F, 3.5))], item="torch")
    out.append(E.Shot("echo", E.cam_dolly(C("echo"), Ee("echo"), ug(44.0, F + 7.0, 14.0), ug(47.0, F + 5.0, 11.0), ug(58.0, F, 2.0),
                                          ug(58.0, F + 1.0, 4.0), fov=58), loc="under", actors=[pip4], props=[tor],
                      sfx=[(C("echo") + 0.8 * k, "ding", -26, dict(base=1500 + 200 * (k % 3))) for k in range(int((Ee("echo") - C("echo")) / 0.8))]))
    # 5 iron in the side tunnel
    it_ = x.every(T("iron", 0.55), Ee("iron") - 0.3, len(IRON))
    pip5 = x.pip([(C("iron"), ug(49.5, F, -6.0)), (T("iron", 0.3), ug(48.5, F, -12.0))], item="pickaxe:stone",
                 acts=x.every(T("iron", 0.5), Ee("iron") - 0.3, 12), yaw=lambda t: -PI / 2 if t > T("iron", 0.3) else None)
    iron = E.Breaks([(it_[i], ug(*p), "iron_ore") for i, p in enumerate(IRON)], collect=lambda t: pip5.pos(t), drop="rawiron")
    out.append(E.Shot("iron", E.cam_static(ug(50.6, F + 2.2, -9.0), ug(47.6, F + 1.2, -12.5), fov=55), loc="under", actors=[pip5],
                      props=[iron, tor], titles=[x.title(Wd("iron", "Iron"), Ee("iron"), "IRON!", color=(236, 190, 150), emoji="🟠")],
                      sfx=[(Wd("iron", "Iron"), "shimmer", -9, {})]))
    # 6 furnace + ingots
    fp = ug(51.5, F + 0.5, -10.5)
    tfu = T("furnace", 0.25)
    furn = E.Custom(lambda t: ([__import__("mcengine.models", fromlist=["x"]).block_entity_vertices(
        "furnace_lit", fp, max(0.05, E.ease_out_back(E.seg(t, tfu, tfu + 0.2))))], [],
        [[fp[0] - 0.6, fp[1], fp[2], 0.9, 1.0, 0.6, 0.2, 0.3]] if t > T("furnace", 0.5) else []) if t > tfu else None,
                    sfxlist=[(tfu, "place", -8, {}), (T("furnace", 0.5), "crackle", -10, dict(dur=4.0))])
    ing = E.Fly([(T("furnace", 0.75) + 0.25 * i, T("furnace", 0.75) + 0.25 * i + 0.8, fp, ug(49.5, F + 1.0, -10.5), "iron") for i in range(5)])
    pip6 = x.pip([(0, ug(49.3, F, -10.5))], yaw=PI / 2)
    out.append(E.Shot("furnace", E.cam_static(ug(48.2, F + 2.3, -7.2), ug(51.0, F + 0.8, -10.8), fov=52), loc="under", actors=[pip6],
                      props=[furn, ing, tor],
                      crafts=[(T("furnace", 0.0), tfu, ["cobble", "cobble", "cobble", "cobble", None, "cobble", "cobble", "cobble", "cobble"], "furnace")],
                      titles=[x.title(T("furnace", 0.75), Ee("furnace"), "IRON INGOTS", color=(230, 230, 236), emoji="✨", pos=(700, 160))],
                      sfx=[(T("furnace", 0.75) + 0.25 * i + 0.8, "pop", -11, dict(pitch=1.1 + 0.05 * i)) for i in range(5)]))
    # 7 iron pickaxe
    tpk = Wd("pick2", "pickaxe")
    pip7 = x.pip([(0, ug(49.3, F, -10.5))], yaw=PI * 0.2, item=[(0, None), (tpk, "pickaxe:iron")],
                 pose=lambda t: {"arms_up": 0.55 * E.smooth(E.seg(t, tpk, tpk + 0.3))})
    out.append(E.Shot("pick2", E.cam_front(pip7, dist=3.0, up=0.5, side=0.9, fov=48), loc="under", actors=[pip7], props=[furn, tor],
                      crafts=[(T("pick2", 0.0), Ee("pick2"), ["t:ingot", "t:ingot", "t:ingot", None, "t:pick_handle", None, None,
                                                             "t:pick_handle", None], "e:⛏️")],
                      sfx=[(tpk, "ding", -8, {})]))
    # 8 iron armor
    words = ["Helmet", "chestplate", "leggings", "boots"]
    tw = [Wd("armor", w) for w in words]
    pip8 = x.pip([(0, ug(49.3, F, -10.5))], yaw=PI * 0.2, item="pickaxe:iron", skin=[(0, "pip"), (tw[0], "pip_iron")])
    out.append(E.Shot("armor", E.cam_orbit(ug(49.3, F, -10.5), 3.6, 1.8, 0.2, 2.2, C("armor"), Ee("armor"), fov=46, look_y=1.1),
                      loc="under", actors=[pip8], props=[furn, tor],
                      fx=[E.Burst(tw[0], ug(49.3, F + 1.2, -10.5), 50, speed=(1, 3), life=(0.6, 1.2), size=(0.06, 0.14),
                                  colors=((1, 1, 1, 1), (0.85, 0.85, 0.9, 1)), gravity=-1, drag=2, up_bias=0.5, seed=5, additive=True)],
                      titles=[x.title(tw[i], tw[i + 1] if i < 3 else Ee("armor"), w.upper(), color=(230, 230, 236), emoji="🛡️") for i, w in enumerate(words)],
                      sfx=[(tw[i], "place", -8, dict(seed=i)) for i in range(4)] + [(tw[0], "shimmer", -8, {})]))
    # 9 cave zombie behind
    zt0 = C("cavezombie")
    zom = x.mob("zombie", [(zt0, ug(49.5, F, -18.0)), (T("hits", 0.3), ug(49.4, F, -12.6))],
                hits=[T("hits", 0.45), T("hits", 0.62), T("hits", 0.8)], death=T("hits", 0.85))
    pip9 = x.pip([(0, ug(49.3, F, -10.5))], item=[(0, "pickaxe:iron"), (T("hits", 0.3), "sword:iron")], skin="pip_iron",
                 yaw=lambda t: PI if t > T("cavezombie", 0.7) else PI * 0.2,
                 acts=[T("hits", 0.42), T("hits", 0.59), T("hits", 0.77)])
    out.append(E.Shot("cavezombie", E.cam_static(ug(48.5, F + 1.7, -7.8), ug(49.4, F + 1.4, -13.0), fov=52), loc="under",
                      actors=[pip9, zom], props=[tor], music="panic",
                      sfx=[(T("cavezombie", 0.4), "groan", -6, dict(seed=31))]))
    out.append(E.Shot("hits", E.cam_static(ug(50.7, F + 2.1, -8.2), ug(48.9, F + 1.1, -12.6), fov=54), loc="under",
                      actors=[pip9, zom], props=[tor], hp=lambda t: 10.0 - 0.5 * (t > T("hits", 0.2)),
                      titles=[x.title(T("hits", 0.85), Ee("hits"), "3 HITS", color=Wh, emoji="⚔️")]))
    # 11 day 15: out into the daylight
    pip11 = x.pip([(C("day15"), (cx + 0.5, Y0, cz - 6.0)), (Ee("day15"), (cx + 0.5, Y0, cz + 6.5))], skin="pip_iron", item="sword:iron")
    out.append(E.Shot("day15", E.cam_static((cx + 2.5, GY + 1.4, cz + 9.5), (cx + 0.5, GY + 2.2, cz - 1.0), fov=48), stage="P2",
                      phase=0.0, actors=[pip11], music="resolve",
                      titles=[x.title(T("day15", 0.5), Ee("day15"), "LEVEL UP", color=Y, emoji="😎")]))
    return out
