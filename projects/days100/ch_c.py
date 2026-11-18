"""Chapters 11-15: the siege, halfway, the portal, the Nether, payback."""
import numpy as np

import engine as E
import world as WD
from shots import PI, GY, Y0, U, FLOOR, G, B, R, O, Y, Wh, PU, TOP
from ch_a import timelapse, circle_keys, HC
from ch_b import VILLAGERS, orange_overlay
from mcengine.models import block_entity_vertices
from mcengine import models100 as M
from mcengine import overlay as ov

VC = np.array([65.5, GY + 1.0, 144.0])
DOORS = [(52, 126, "w"), (64, 120, "s"), (72, 129, "w"), (55, 151, "w"), (70, 148, "w")]


def door_front(i, d=1.0):
    x_, z_, side = DOORS[i]
    if side == "w":
        return (x_ - d + 0.5, z_ + 0.5)
    return (x_ + 0.5, z_ - d + 0.5)


def fireworks(t0, n, center, seed=0, spread=14.0):
    rng = np.random.default_rng(seed)
    out = []
    cols = [((1, 0.3, 0.3, 1), (1, 0.8, 0.3, 1)), ((0.4, 0.8, 1, 1), (1, 1, 1, 1)), ((0.5, 1, 0.4, 1), (1, 1, 0.5, 1)),
            ((1, 0.5, 1, 1), (1, 1, 1, 1))]
    for k in range(n):
        p = np.asarray(center, float) + (rng.uniform(-spread, spread), rng.uniform(8, 16), rng.uniform(-spread, spread))
        out.append(E.Burst(t0 + 0.45 * k, p, 60, speed=(5, 8), life=(0.8, 1.4), size=(0.15, 0.3), colors=cols[k % 4],
                           gravity=-3, drag=1.6, seed=seed + k, additive=True))
    return out


def ch11(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    # 1 stayed too long: sunset
    pip1 = x.pip([(0, (65.5, 144.0))], skin="pip_iron", item="sword:iron", yaw=PI / 2 + 0.6, pose=lambda t: {"head_pitch": 0.1})
    bis1 = x.biscuit([(0, (66.6, 144.9))], yaw=PI / 2 + 0.6, sit=1.0)
    out.append(E.Shot("toolong", E.cam_static((61.5, GY + 2.2, 147.0), (75.0, GY + 4.0, 139.5), fov=55), stage="P4",
                      phase=lambda t: 1.0 + 0.7 * E.seg(t, C("toolong"), Ee("toolong")), actors=[pip1, bis1], music="suspense",
                      titles=[x.title(T("toolong", 0.5), Ee("toolong"), "UH OH", color=O, emoji="😬")]))
    # zombies: ten of them out of the dark
    rng = np.random.default_rng(41)
    zstart = [(88.0 + rng.uniform(-6, 6), 160.0 + rng.uniform(-14, 14)) for _ in range(10)]
    night = 2.0
    # 2 siege
    zs = [x.mob("zombie", [(C("siege"), p), (Ee("doors"), (p[0] - 14.0, p[1] - 8.0 + rng.uniform(-3, 3)))]) for p in zstart]
    out.append(E.Shot("siege", E.cam_static((80.0, GY + 9.0, 138.0), (88.0, GY + 1.0, 156.0), fov=55), stage="P4", phase=night,
                      actors=zs, titles=[x.title(Wd("siege", "Ten"), Ee("siege"), "10 ZOMBIES", color=G, emoji="🧟", size=120)],
                      sfx=[(T("siege", 0.2), "groan", -10, dict(seed=80)), (T("siege", 0.6), "groan", -12, dict(seed=81))]))
    # 3 villagers run inside
    runs = []
    for i, p in enumerate(VILLAGERS[:5]):
        dp = door_front(i, -1.2)
        tin = T("doors", 0.35 + 0.1 * i)
        runs.append(x.mob("villager", [(C("doors"), p), (tin, door_front(i, 0.6)), (tin + 0.4, dp)], hidden=lambda t, tin=tin: t > tin + 0.4,
                          walk_scale=3.0))
    out.append(E.Shot("doors", E.cam_static((59.0, GY + 9.0, 141.0), (61.0, GY + 1.0, 127.0), fov=62), stage="P4", phase=night,
                      actors=runs + zs[:3], titles=[x.title(Wd("doors", "only"), Ee("doors"), "ALONE", color=R, emoji="😰")],
                      sfx=[(T("doors", 0.35 + 0.1 * i) + 0.4, "impact", -12, dict(seed=i)) for i in range(5)]))
    # 4-6 the fight in the square
    zp = [(VC[0] + 5.5 * np.cos(a), VC[2] + 5.5 * np.sin(a)) for a in np.linspace(0.2, 2.2, 6)]
    tw = [Wd("wave", "One"), Wd("wave", "Two"), Wd("wave", "Three")]
    fights = []
    for i in range(3):
        p0 = (VC[0] + 14 * np.cos(0.3 + 0.6 * i), VC[2] + 14 * np.sin(0.3 + 0.6 * i))
        hit = tw[i]
        fights.append(x.mob("zombie", [(C("stand"), p0), (hit - 0.5, (VC[0] + 1.6 * np.cos(0.3 + 0.6 * i), VC[2] + 1.6 * np.sin(0.3 + 0.6 * i)))],
                            hits=[hit - 0.2, hit + 0.05], death=hit + 0.2))
    more = [x.mob("zombie", [(C("stand"), (VC[0] + 16 * np.cos(a), VC[2] + 16 * np.sin(a))), (Ee("biscuitfight"), (VC[0] + 6 * np.cos(a), VC[2] + 6 * np.sin(a)))])
            for a in (0.0, 1.0, 1.9)]
    def face(t):
        best, bd = None, 99
        for z in fights + more:
            if z.alive(t):
                d = np.linalg.norm(z.pos(t) - VC)
                if d < bd:
                    best, bd = z, d
        p = best.pos(t) if best is not None else VC + (1, 0, 0)
        return float(np.arctan2(p[0] - VC[0], p[2] - VC[2]))
    acts = [t - 0.25 for t in tw] + [t for t in tw]
    tbf = [Wd("biscuitfight", "jumped"), Wd("biscuitfight", "other") - 0.3, Wd("biscuitfight", "other") + 0.2]
    acts += [tbf[1] - 0.2, tbf[2] - 0.2]
    pipf = x.pip([(0, (VC[0], VC[2]))], skin="pip_iron", item="sword:iron", yaw=face, acts=acts)
    zb = x.mob("zombie", [(C("biscuitfight"), (VC[0] - 6.0, VC[2] + 4.0)), (tbf[0] + 0.2, (VC[0] - 3.0, VC[2] + 2.2))],
               hits=[tbf[0] + 0.25, tbf[0] + 0.6], death=tbf[0] + 0.9)
    bisf = x.biscuit([(C("stand"), (VC[0] - 1.2, VC[2] + 0.4)), (tbf[0], (VC[0] - 1.2, VC[2] + 0.4)), (tbf[0] + 0.25, (VC[0] - 2.6, VC[2] + 1.9)),
                      (Ee("biscuitfight"), (VC[0] - 2.4, VC[2] + 1.6))],
                     y_off=lambda t: 0.8 * max(0.0, np.sin(PI * E.seg(t, tbf[0] - 0.1, tbf[0] + 0.35))), face_target=lambda t: zb.pos(t))
    m2 = [x.mob("zombie", [(C("biscuitfight"), (VC[0] + 5.0 * np.cos(a), VC[2] + 5.0 * np.sin(a))), (tbf[k + 1] - 0.3, (VC[0] + 1.5 * np.cos(a), VC[2] + 1.5 * np.sin(a)))],
                hits=[tbf[k + 1]], death=tbf[k + 1] + 0.25) for k, a in enumerate((0.5, 1.4))]
    out.append(E.Shot("stand", E.cam_static((VC[0] - 2.0, GY + 1.6, VC[2] - 3.0), VC + (0.5, 1.9, 1.0), fov=58), stage="P4", phase=night,
                      actors=[pipf, bisf] + fights + more, music="panic"))
    out.append(E.Shot("wave", E.cam_orbit(VC, 7.5, 3.5, -0.6, 0.6, C("wave"), Ee("wave"), fov=55, look_y=0.8), stage="P4", phase=night,
                      actors=[pipf, bisf] + fights + more,
                      titles=[x.title(tw[0] + 0.2, tw[1], "1", color=Wh, size=150), x.title(tw[1] + 0.2, tw[2], "2", color=Wh, size=150),
                              x.title(tw[2] + 0.2, Ee("wave"), "3", color=Wh, size=150)]))
    out.append(E.Shot("biscuitfight", E.cam_static((VC[0] + 3.0, GY + 2.6, VC[2] - 4.5), VC + (-1.5, 0.8, 1.2), fov=55), stage="P4",
                      phase=night, actors=[pipf, bisf, zb] + m2,
                      titles=[x.title(tbf[0] + 0.3, Ee("biscuitfight"), "TEAMWORK", color=(255, 200, 120), emoji="🐶")]))
    # 7-8 the door
    d_x, d_z = 64, 120
    tsave = [Wd("save", "One"), Wd("save", "two")]
    zd = x.mob("zombie", [(0, (d_x + 0.5, d_z - 0.9))], yaw=0.0, hits=tsave, death=tsave[1] + 0.3,
               pose=lambda t: {"punch": (0.5 + 0.5 * np.sin(t * 9)) if t < tsave[0] else 0.0})
    def door_crack(t):
        if t >= tsave[1] + 0.3:
            return None
        from mcengine.geometry import translate, box_vertices
        c = min(3, int(4 * E.seg(t, C("breakdoor"), T("save", 0.2))))
        return [box_vertices((1.0, 2.0, 0.22), f"crack{c}", matrix=translate(d_x + 0.5, GY + 2.0, d_z + 0.05), inflate=0.01)]
    vin = x.mob("villager", [(0, (d_x + 0.5, d_z + 2.0))], yaw=PI, pose=lambda t: {"arms_up": 0.4, "flail_phase": t * 6})
    out.append(E.Shot("breakdoor", E.cam_static((d_x + 3.5, GY + 2.2, d_z - 4.0), (d_x + 0.5, GY + 1.6, d_z), fov=50), stage="P4",
                      phase=night, actors=[zd, vin], props=[E.Custom(door_crack)],
                      titles=[x.title(Wd("breakdoor", "villager"), Ee("breakdoor"), "VILLAGER INSIDE!", color=R, emoji="😱", size=96)],
                      sfx=[(C("breakdoor") + 0.45 * k, "impact", -11, dict(seed=k)) for k in range(int((Ee("breakdoor") - C("breakdoor")) / 0.45))]))
    pips = x.pip([(C("save"), (d_x + 4.5, d_z - 6.0)), (tsave[0] - 0.25, (d_x + 0.9, d_z - 2.3))], skin="pip_iron", item="sword:iron",
                 acts=[t - 0.1 for t in tsave], face_target=lambda t: zd.pos(t))
    out.append(E.Shot("save", E.cam_static((d_x - 3.0, GY + 2.0, d_z - 4.2), (d_x + 0.8, GY + 1.4, d_z - 1.2), fov=52), stage="P4",
                      phase=night, actors=[zd, pips, vin], props=[E.Custom(door_crack)]))
    # 9 sunrise: the rest burn
    tb = T("sunrise", 0.35)
    burners = [x.mob("zombie", [(C("sunrise"), (VC[0] + 6 * np.cos(a), VC[2] + 6 * np.sin(a))), (Ee("sunrise"), (VC[0] + 7 * np.cos(a), VC[2] + 7 * np.sin(a)))],
                     burn=(tb + 0.1 * k, tb + 1.6 + 0.2 * k), death=tb + 1.6 + 0.2 * k) for k, a in enumerate((0.0, 1.0, 1.9, 3.0))]
    pipr = x.pip([(0, (VC[0], VC[2]))], skin="pip_iron", item="sword:iron", yaw=0.4)
    bisr = x.biscuit([(0, (VC[0] - 1.2, VC[2] + 0.4))], yaw=0.4, sit=1.0)
    out.append(E.Shot("sunrise", E.cam_static(VC + (-9.0, 4.0, -8.0), VC + (0, 1.0, 0), fov=52), stage="P4",
                      phase=lambda t: 2.0 + 1.0 * E.smooth(E.seg(t, C("sunrise"), tb + 0.3)), actors=burners + [pipr, bisr], music="resolve",
                      titles=[x.title(Wd("sunrise", "villager"), Ee("sunrise"), "EVERYONE SAFE", color=G, emoji="✅")],
                      sfx=[(tb, "crackle", -9, dict(dur=3.0))]))
    # 10 hero discount
    from ch_b import VILLAGERS as VV
    farmer = x.mob("villager", [(0, (60.0, 138.5))], yaw=PI / 2)
    crowd = [x.mob("villager", [(0, (61.5 + 1.3 * i, 136.4 - 0.2 * i))], yaw=0.2 * i) for i in range(3)]
    pipd = x.pip([(0, (62.2, 138.5))], skin="pip_iron", yaw=-PI / 2)
    def panel(layer, t, t0=T("discount", 0.3), t1=Ee("discount")):
        if not (t0 <= t < t1):
            return
        from PIL import ImageDraw
        a = min(1.0, (t - t0) / 0.25, (t1 - t) / 0.25)
        d = ImageDraw.Draw(layer)
        x0, y0 = 1180, 300
        d.rounded_rectangle([x0, y0, x0 + 600, y0 + 200], radius=20, fill=(198, 198, 198, int(235 * a)), outline=(40, 40, 40, int(255 * a)), width=6)
        ov.draw_text(layer, "20", (x0 + 90, y0 + 100), 60, fill=(150, 150, 150), alpha=a)
        ov.draw_text(layer, "5", (x0 + 90, y0 + 160), 60, fill=(120, 255, 120), alpha=a)
        ov.paste_emoji(layer, "🌾", (x0 + 195, y0 + 100), 88, alpha=a)
        ov.draw_text(layer, "→", (x0 + 300, y0 + 100), 90, fill=(80, 80, 80), alpha=a, shadow=False)
        ov.paste_emoji(layer, "💚", (x0 + 445, y0 + 100), 88, alpha=a)
        ov.draw_text(layer, "1", (x0 + 540, y0 + 100), 70, fill=(255, 255, 255), alpha=a)
        ov.draw_text(layer, "HERO DISCOUNT", (x0 + 300, y0 - 40), 64, fill=(255, 225, 40), rot=4, alpha=a)
    out.append(E.Shot("discount", E.cam_static((61.2, GY + 2.4, 134.0), (61.0, GY + 1.5, 138.6), fov=52), stage="P4", phase=3.8,
                      actors=[farmer, pipd] + crowd, overlay=panel,
                      fx=[E.hearts_burst(T("discount", 0.2) + 0.3 * i, (61.5 + 1.3 * i, GY + 2.4, 136.4), 90 + i) for i in range(3)]))
    # 11 worth it
    out.append(E.Shot("day45", E.cam_orbit(VC, 26.0, 14.0, 0.6, 1.1, C("day45"), Ee("day45"), fov=50), stage="P4", phase=0.0,
                      actors=[x.mob("villager", [(0, p)], yaw=i) for i, p in enumerate(VILLAGERS)],
                      titles=[x.title(Wd("day45", "Worth"), Ee("day45"), "WORTH IT", color=G, emoji="✅", size=110)]))
    return out


def lighthouse_blocks(f):
    w, top, y0 = WD.lighthouse_world(f.terrain, glow=False)
    top = top[1]
    bl = sorted(w.blocks.items(), key=lambda kv: (kv[0][1], np.arctan2(kv[0][2] - WD.HILL_C[1], kv[0][0] - WD.HILL_C[0])))
    return bl, top, y0


def stats_panel(lines, x0=1150, y0=230):
    """lines: [(t_appear, emoji, text, color)]"""
    def fn(layer, t):
        shown = [l for l in lines if t >= l[0]]
        if not shown:
            return
        from PIL import ImageDraw
        d = ImageDraw.Draw(layer)
        h = 30 + 92 * len(shown)
        d.rounded_rectangle([x0, y0, x0 + 690, y0 + h], radius=22, fill=(0, 0, 0, 165))
        for i, (ta, emo, txt, col) in enumerate(shown):
            pop = E.ease_out_back(E.seg(t, ta, ta + 0.25))
            yy = y0 + 62 + 92 * i
            ov.paste_emoji(layer, emo, (x0 + 60, yy), 62 * max(pop, 0.05))
            ov.draw_text(layer, txt, (x0 + 380, yy), 56, fill=col, scale=max(pop, 0.05), shadow=False)
    return fn


def ch12(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    hx, hz = WD.HILL_C
    # 1 home: on the wall, looking out
    pip1 = x.pip([(0, (20.0, GY + 3.0, -0.5))], skin="pip_iron", item="sword:iron", yaw=PI + 0.4, pose=lambda t: {"head_yaw": 0.5 * np.sin(t * 0.8)})
    out.append(E.Shot("home", E.cam_static((22.5, GY + 4.6, 3.5), (12.0, GY + 3.0, -25.0), fov=55), stage="P4", phase=0.0, actors=[pip1],
                      music="adventure",
                      titles=[x.title(Wd("home", "find"), Ee("home"), "FIND HOME?", color=Y, emoji="🧭")]))
    # 2 lighthouse build (time-lapse)
    bl, top, y0 = lighthouse_blocks(f)
    t0, t1 = T("lighthouse", 0.1), Ee("lighthouse") - 0.4
    rev = E.Reveal("lighthouse", bl, t0, t1, K=40)
    ks = [(t0 + (t1 - t0) * k / 40, (hx + 0.5 + 3.6 * np.cos(k * 0.6), y0 + (top - y0) * k / 40, hz + 0.5 + 3.6 * np.sin(k * 0.6))) for k in range(41)]
    pipl = x.pip([(C("lighthouse"), (hx + 0.5, y0, hz + 5.0))] + ks, skin="pip_iron", item="stone_bricks", acts=x.every(t0, t1, 40))
    lc = np.array([hx + 0.5, y0 + 10.0, hz + 0.5])
    out.append(E.Shot("lighthouse", E.cam_orbit(lc, 28.0, 4.0, 0.3, 1.3, C("lighthouse"), Ee("lighthouse"), fov=55, look_y=0.0, h1=12.0),
                      stage="P4", phase=timelapse(t0, t1, 1.0, 0.0), actors=[pipl], props=[rev], music="build",
                      titles=[x.title(Wd("lighthouse", "lighthouse"), Ee("lighthouse"), "LIGHTHOUSE", color=Y, emoji="🗼")]))
    # 3 inside: looking up the shaft
    pips = x.pip([(C("spiral"), (hx + 0.5, y0, hz + 1.5)), (Ee("spiral"), (hx + 0.5, y0 + 14.0, hz + 1.5))], skin="pip_iron", yaw=PI,
                 pose=lambda t: {"walk": 0.8, "walk_phase": t * 9})
    out.append(E.Shot("spiral", E.cam_static((hx + 0.5, y0 + 0.6, hz + 0.5), (hx + 0.55, y0 + 20.0, hz + 1.2), fov=70), stage="P4", phase=0.0,
                      statics=["lighthouse"], actors=[pips], light=dict(ambient=(0.5, 0.48, 0.46)),
                      titles=[x.title(T("spiral", 0.3), Ee("spiral"), "ALL THE WAY UP", color=Wh, emoji="⬆️")]))
    # 4 visible from the village (telephoto)
    out.append(E.Shot("tall", E.cam_static((64.0, GY + 22.0, 134.0), (hx + 0.5, y0 + 6.0, hz), fov=10), stage="P4", phase=0.0,
                      statics=["lighthouse"], fog=dict(start=150.0, end=700.0),
                      titles=[x.title(Wd("tall", "village"), Ee("tall"), "SEEN FROM THE VILLAGE", color=Y, emoji="👀", size=90)]))
    # 5 day 50
    bc = np.array([13.0, GY + 1.0, 14.0])
    out.append(E.Shot("day50", E.cam_orbit(bc, 40.0, 22.0, -1.0, -0.4, C("day50"), Ee("day50"), fov=52), stage="P4", phase=1.0,
                      statics=["lighthouse"], fx=fireworks(T("day50", 0.3), 6, bc, seed=500), music="resolve",
                      titles=[x.title(Wd("day50", "fifty"), Wd("day50", "Halfway"), "DAY 50", color=Y, emoji="🎉", size=160),
                              x.title(Wd("day50", "Halfway"), Ee("day50"), "HALFWAY!", color=Y, emoji="🏁", size=150)],
                      sfx=[(T("day50", 0.3) + 0.45 * k, "explosion", -18, dict(seed=k, dur=0.8)) for k in range(6)]))
    # 6-7 stats from the lighthouse top
    lines = [(Wd("stats", "Two"), "🏠", "2 HOUSES", Wh), (Wd("stats", "One"), "🐶", "1 WOLF", (255, 200, 120)),
             (Wd("stats", "Six"), "💎", "6 DIAMONDS", B), (Wd("stats", "four"), "🧟", "~400 ZOMBIES?", G),
             (Wd("deaths", "Zero"), "💀", "0 DEATHS", R)]
    pt = x.pip([(0, (hx + 0.5, top + 0.0, hz - 1.5))], skin="pip_iron", yaw=PI * 0.9)
    bt = x.biscuit([(0, (hx + 1.7, top + 0.0, hz - 1.6))], yaw=PI * 0.9, sit=1.0)
    for key in ("stats", "deaths"):
        out.append(E.Shot(key, E.cam_static((hx - 3.5, top + 3.0, hz + 5.5), (hx + 4.0, top - 6.0, hz - 18.0), fov=58), stage="P4",
                          phase=1.05, statics=["lighthouse"], actors=[pt, bt], overlay=stats_panel(lines)))
    # 8 next: something bigger
    pipn = x.pip([(0, (10.5, 5.0))], skin="pip_iron", item="pickaxe:diamond", yaw=PI)
    out.append(E.Shot("next", E.cam_front(pipn, dist=3.0, up=0.3, side=0.9, fov=44), stage="P4", phase=1.4, actors=[pipn],
                      titles=[x.title(Wd("next", "bigger"), Ee("next"), "SOMETHING BIGGER", color=O, emoji="👀")], music="suspense"))
    # 9 the nether (flash-forward)
    out.append(E.Shot("nether", E.cam_dolly(C("nether"), Ee("nether"), x.ng(30, 40, 40), x.ng(22, 34, 30), x.ng(0, 14, 0), x.ng(0, 18, 0), fov=60),
                      loc="nether", flash=[(C("nether"), 0.6, 0.5)],
                      titles=[x.title(T("nether", 0.0), Ee("nether"), "THE NETHER", color=(255, 90, 60), emoji="🔥", size=150)],
                      sfx=[(C("nether"), "impact", -6, {}), (C("nether"), "riser", -10, dict(dur=2.0))]))
    return out


PORTAL_SHAPE = [(1, 0), (2, 0), (0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (3, 3), (1, 4), (2, 4)]


def portal_diagram(t0, t1):
    def fn(layer, t):
        if not (t0 <= t < t1):
            return
        from PIL import ImageDraw
        a = min(1.0, (t - t0) / 0.3, (t1 - t) / 0.25)
        d = ImageDraw.Draw(layer)
        cs = 84
        x0, y0 = 1260, 240
        d.rounded_rectangle([x0 - 30, y0 - 30, x0 + 4 * cs + 30, y0 + 5 * cs + 30], radius=20, fill=(198, 198, 198, int(225 * a)), outline=(40, 40, 40, int(255 * a)), width=6)
        ic = ov.iso_block_icon("obsidian", cs - 6)
        for i, (a_, b_) in enumerate(PORTAL_SHAPE):
            if t >= t0 + 0.12 * i:
                layer.alpha_composite(ic, (int(x0 + a_ * cs + 3), int(y0 + (4 - b_) * cs + 3)))
        ov.draw_text(layer, "x10", (x0 + 2 * cs, y0 + 5 * cs + 70), 64, fill=(200, 140, 255), alpha=a)
    return fn


def ch13(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    px, pz = WD.PORTAL
    # 1 need obsidian
    pip1 = x.pip([(0, (3.0, 9.0))], skin="pip_iron", item="pickaxe:diamond", yaw=-PI * 0.75)
    out.append(E.Shot("need", E.cam_static((7.5, GY + 2.6, 4.5), (1.5, GY + 1.3, 11.5), fov=52), stage="P4", phase=0.0, actors=[pip1],
                      overlay=portal_diagram(T("need", 0.4), Ee("need")), music="adventure",
                      titles=[x.title(Wd("need", "portal"), T("need", 0.4), "NETHER PORTAL", color=PU, emoji="🟪")]))
    # 2 lava pool + water
    x0, x1, z0, z1 = WD.LAVA_POOL
    cells = [(xx, zz) for xx in range(x0, x1) for zz in range(z0, z1)]
    tp = T("pool", 0.6)
    obs = E.Pops([(tp + 0.25 + 0.04 * i, (c[0], GY, c[1]), "obsidian") for i, c in enumerate(cells)], sound=False, scale=1.01)
    pc = np.array([(x0 + x1) / 2, GY + 1.0, (z0 + z1) / 2])
    pip2 = x.pip([(C("pool"), (pc[0] - 4.0, pc[2] + 5.0)), (T("pool", 0.5), (pc[0] - 0.5, z1 + 0.8))], skin="pip_iron",
                 item=[(0, "water_bucket"), (tp, "bucket")], pose=lambda t: {"reach": np.sin(PI * E.seg(t, tp - 0.2, tp + 0.5))},
                 yaw=lambda t: PI if t > T("pool", 0.5) else None)
    steam = E.Burst(tp + 0.2, pc, 40, speed=(0.5, 2), life=(0.8, 1.6), size=(0.3, 0.7), colors=((0.85, 0.85, 0.85, 0.6),), gravity=1.5,
                    drag=1.5, up_bias=1.2, seed=700, box=1.5, grow=1.5)
    out.append(E.Shot("pool", E.cam_static(pc + (5.0, 3.5, 5.0), pc + (0, -0.5, 0), fov=50), stage="P4", phase=0.0, statics=["pool_lava"],
                      actors=[pip2], props=[obs], fx=[steam], sfx=[(tp, "water", -9, dict(dur=1.0)), (tp + 0.2, "hiss", -9, dict(dur=1.5))],
                      titles=[x.title(Wd("pool", "trick"), Ee("pool"), "WATER BUCKET TRICK", color=B, emoji="🪣", size=86)]))
    # 3 mining it with the diamond pickaxe
    tb = x.every(T("mineobs", 0.2), Ee("mineobs") - 0.3, len(cells))
    pip3 = x.pip([(0, (pc[0] - 0.5, z1 + 0.8))], skin="pip_iron", item="pickaxe:diamond", yaw=PI, acts=x.every(T("mineobs", 0.15), Ee("mineobs") - 0.3, 16),
                 pose=lambda t: {"head_pitch": -0.35})
    br = E.Breaks([(tb[i], (c[0], GY, c[1]), "obsidian") for i, c in enumerate(cells)], dur=0.35, collect=lambda t: pip3.pos(t), drop="cobble",
                  overlay=False)
    out.append(E.Shot("mineobs", E.cam_static(pc + (-4.5, 2.6, 4.5), pc + (0.3, -0.6, 0), fov=50), stage="P4", phase=0.0, statics=["pool_mined"],
                      actors=[pip3], props=[br],
                      titles=[x.title(Wd("mineobs", "diamond"), Ee("mineobs"), "DIAMOND PICKAXE ONLY", color=B, emoji="⛏️", size=86)]))
    # 4 the frame
    tf = x.every(T("frame", 0.1), Ee("frame") - 0.4, 10)
    order = sorted(PORTAL_SHAPE, key=lambda ab: (ab[1], ab[0]))
    frame = E.Pops([(tf[i], (px + a, GY + 1 + b, pz), "obsidian") for i, (a, b) in enumerate(order)])
    pc2 = np.array([px + 2.0, GY + 1.0, pz + 0.5])
    pip4 = x.pip([(0, (px + 2.0, pz - 2.2))], skin="pip_iron", item="obsidian", yaw=0.0, acts=[t - 0.1 for t in tf])
    out.append(E.Shot("frame", E.cam_static(pc2 + (4.5, 2.6, -6.0), pc2 + (0, 2.0, 0), fov=50), stage="P4", phase=0.0, actors=[pip4], props=[frame],
                      titles=[x.title(Wd("frame", "Four"), Ee("frame"), "4 x 5", color=PU, emoji="🟪")]))
    framed = E.Pops([(0, (px + a, GY + 1 + b, pz), "obsidian") for (a, b) in order], sound=False)
    # 5 flint and steel
    tcl = Wd("flint", "click")
    pip5 = x.pip([(0, (px + 2.0, pz - 1.6))], skin="pip_iron", item="flint", yaw=0.0, acts=[tcl])
    spark = E.Burst(tcl + 0.1, pc2 + (0, 0.3, 0), 18, speed=(1, 3), life=(0.2, 0.5), size=(0.05, 0.1), colors=((1, 0.8, 0.3, 1),), gravity=-6,
                    drag=1.5, seed=701, additive=True)
    out.append(E.Shot("flint", E.cam_static(pc2 + (-3.0, 1.8, -4.0), pc2 + (0, 1.5, 0), fov=50), stage="P4", phase=0.0, actors=[pip5],
                      props=[framed], fx=[spark], sfx=[(tcl + 0.1, "dig", -8, dict(seed=3))],
                      titles=[x.title(Wd("flint", "flint"), Ee("flint"), "FLINT & STEEL", color=O, emoji="🔥")]))
    # 6 lit!
    swirl = [E.Burst(C("lit") + 0.25 * k, pc2 + (0, 2.4, 0), 6, speed=(0.3, 1.0), life=(1.0, 1.8), size=(0.08, 0.16),
                     colors=((0.75, 0.35, 1, 1), (0.5, 0.2, 0.9, 1)), gravity=0.3, drag=1.0, seed=710 + k, box=1.0, additive=True)
             for k in range(int((Ee("hum") - C("lit")) / 0.25))]
    out.append(E.Shot("lit", E.cam_dolly(C("lit"), Ee("lit"), pc2 + (0.0, 2.0, -9.0), pc2 + (0.0, 2.2, -6.0), pc2 + (0, 2.2, 0), pc2 + (0, 2.4, 0),
                                         fov=52), stage="P5", phase=0.0, actors=[pip5], fx=swirl, flash=[(C("lit"), 0.5, 0.6)],
                      titles=[x.title(Wd("lit", "lit"), Ee("lit"), "IT'S LIT!", color=PU, emoji="🟪", size=130)],
                      sfx=[(C("lit"), "shimmer", -5, {}), (C("lit"), "impact", -8, {}), (C("lit") + 0.3, "riser", -14, dict(dur=3.0))]))
    out.append(E.Shot("hum", E.cam_dolly(C("hum"), Ee("hum"), pc2 + (2.5, 1.8, -4.0), pc2 + (1.0, 2.2, -2.6), pc2 + (0, 2.3, 0), pc2 + (0, 2.5, 0),
                                         fov=50), stage="P5", phase=0.0, fx=swirl,
                      titles=[x.title(Wd("hum", "creepy"), Ee("hum"), "*HUMMMM*", color=PU, emoji="🔊", size=100)]))
    # 8 stay
    bis8 = x.biscuit([(0, (px + 4.6, pz - 1.6))], yaw=-PI / 2 + 0.3, sit=1.0, pose=lambda t: {"head_yaw": 0.15 * np.sin(t)})
    pip8 = x.pip([(0, (px + 3.4, pz - 1.6))], skin="pip_iron", yaw=PI / 2, acts=[T("stay", 0.6)])
    out.append(E.Shot("stay", E.cam_static((px + 7.0, GY + 2.4, pz - 6.5), (px + 3.6, GY + 1.0, pz - 1.4), fov=50), stage="P5", phase=0.0,
                      actors=[bis8, pip8], fx=swirl[-6:],
                      titles=[x.title(Wd("stay", "stayed"), Ee("stay"), "STAY, BISCUIT", color=(255, 200, 120), emoji="🐶")]))
    # 9 step in
    tin = T("step", 0.7)
    pip9 = x.pip([(C("step"), (px + 2.0, pz - 2.0)), (tin, (px + 2.0, pz + 0.4))], skin="pip_iron", item="sword:iron", hidden=lambda t: t > tin + 0.3)
    out.append(E.Shot("step", E.cam_static((px + 2.0, GY + 2.0, pz - 5.5), pc2 + (0, 1.6, 0), fov=52), stage="P5", phase=0.0, actors=[pip9, bis8],
                      fx=swirl[-8:] + [E.Burst(tin, pc2 + (0, 1.5, 0), 60, speed=(1, 4), life=(0.5, 1.0), size=(0.1, 0.2),
                                                colors=((0.8, 0.4, 1, 1),), gravity=0, drag=2, seed=730, additive=True)],
                      flash=[(tin + 0.2, 0.8, 0.9)], music="silence", sfx=[(tin, "whoosh", -6, dict(dur=1.2)), (tin, "shimmer", -8, {})]))
    return out


def nether_ground(x, xx, zz):
    return x.ng(xx, x.f.ground(*x.ng(xx, 0, zz)[[0, 2]], "nether"), zz)


def fireball_prop(t0, t1, p0, p1, back=None, boom=True):
    """Fireball flying p0->p1 between t0 and t1; optional return trip `back`=(t1b, p_back)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)

    def fn(t):
        if t < t0:
            return None
        if t < t1:
            u = E.seg(t, t0, t1)
            p = p0 * (1 - u) + p1 * u
        elif back is not None and t < back[0]:
            u = E.seg(t, t1, back[0])
            p = p1 * (1 - u) + np.asarray(back[1], float) * u
        else:
            return None
        return [M.fireball_vertices(p, 1.0, t)], [], [[p[0], p[1], p[2], 2.2, 1.0, 0.55, 0.15, 0.45]]
    return E.Custom(fn)


def explosion_fx(t, p, seed, big=1.0):
    return [E.Burst(t, p, int(60 * big), speed=(3, 8), life=(0.5, 1.3), size=(0.3, 0.8), colors=((1, 0.8, 0.4, 0.9), (0.5, 0.45, 0.4, 0.9)),
                    gravity=1.0, drag=2.2, seed=seed, grow=2.0),
            E.Burst(t, p, int(24 * big), speed=(1, 4), life=(0.2, 0.5), size=(0.6, 1.2), colors=((1, 0.75, 0.3, 1),), gravity=0, drag=3,
                    seed=seed + 1, additive=True)]


def ch14(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    ng = lambda a, b: nether_ground(x, a, b)
    portal_c = x.ng(-0.5, 17.0, 0.5)
    pfx = [E.Burst(C("red") + 0.3 * k, x.ng(-0.0, 19.5, 0.5), 5, speed=(0.3, 1.0), life=(1.0, 1.8), size=(0.08, 0.16),
                   colors=((0.75, 0.35, 1, 1),), gravity=0.3, drag=1.0, seed=800 + k, box=1.0, additive=True) for k in range(12)]
    def portal_inner(t):
        vt = [block_entity_vertices("portal", x.ng(a + 0.5, b + 0.5, 0.5), 1.0) for (a, b, _) in WD.nether_portal_inner()]
        return [], vt, [[*x.ng(0.0, 19.5, 0.5), 2.0, 0.7, 0.3, 1.0, 0.3]]
    pportal = E.Custom(portal_inner)
    # 1 red: arriving
    pip1 = x.pip([(C("red"), x.ng(-0.5, 17.0, 0.6)), (T("red", 0.4), x.ng(-0.5, 17.0, -1.6))], skin="pip_iron", item="sword:iron",
                 flash=[(C("red"), 0.0)])
    out.append(E.Shot("red", E.cam_orbit(portal_c, 16.0, 7.0, -2.2, -0.9, C("red"), Ee("red"), fov=58, look_y=0.0, r1=26.0, h1=13.0), loc="nether",
                      actors=[pip1], props=[pportal], fx=pfx, flash=[(C("red"), 0.6, 0.8)], music="suspense",
                      titles=[x.title(T("red", 0.3), Ee("red"), "THE NETHER", color=(255, 90, 60), emoji="🔥", size=130)],
                      sfx=[(C("red"), "crackle", -14, dict(dur=6.0))]))
    # 2 netherrack burns forever
    nb = ng(4, 1)
    tfire = Wd("netherrack", "fire")
    pip2 = x.pip([(0, ng(2.6, 1))], skin="pip_iron", item="flint", yaw=PI / 2, acts=[tfire], pose=lambda t: {"head_pitch": -0.4})
    out.append(E.Shot("netherrack", E.cam_static(nb + (1.0, 2.0, -3.2), nb + (0.2, 0.2, 0), fov=50), loc="nether", actors=[pip2], props=[pportal],
                      fx=E.fire_burst(tfire, nb + (0.5, 0.2, 0.5), Ee("netherrack") + 2 - tfire, seed=810),
                      titles=[x.title(Wd("netherrack", "forever"), Ee("netherrack"), "BURNS FOREVER", color=O, emoji="🔥")],
                      sfx=[(tfire, "crackle", -8, dict(dur=4.0))]))
    # 3 no water / beds explode
    tw = Wd("nowater", "evaporates") - 0.6
    tbed = Wd("nowater", "explodes")
    pip3 = x.pip([(0, ng(5.5, 1))], skin="pip_iron", item=[(0, "water_bucket"), (tw, "bucket")], yaw=PI / 2,
                 pose=lambda t: {"reach": np.sin(PI * E.seg(t, tw - 0.2, tw + 0.5))})
    p3 = ng(7.0, 1)
    out.append(E.Shot("nowater", E.cam_static(p3 + (-1.5, 2.2, -3.5), p3 + (0, 0.5, 0), fov=50), loc="nether", actors=[pip3],
                      fx=[E.Burst(tw + 0.1, p3 + (0, 0.4, 0), 30, speed=(0.5, 2), life=(0.6, 1.2), size=(0.2, 0.5),
                                  colors=((0.9, 0.9, 0.9, 0.7),), gravity=2, drag=1.5, up_bias=1.5, seed=820, grow=1.5)] +
                         explosion_fx(tbed + 0.2, p3 + (1.5, 0.5, 1.5), 821, 0.6),
                      titles=[x.title(Wd("nowater", "water"), tbed - 0.2, "WATER EVAPORATES", color=B, emoji="💧", size=86),
                              x.title(tbed, Ee("nowater"), "BEDS EXPLODE", color=R, emoji="🛏️", size=96)],
                      sfx=[(tw + 0.1, "hiss", -9, dict(dur=1.0)), (tbed + 0.2, "explosion", -8, dict(seed=4, dur=2.0))]))
    # 4-5 the ghast
    gp0 = x.ng(26.0, 34.0, -18.0)
    gp1 = x.ng(18.0, 28.0, -10.0)
    ghast = x.mob("ghast", [(C("cry"), gp0), (Ee("hole") + 4, gp1)], yaw=-PI * 0.75,
                  open_=lambda t: (T("fireball", 0.0) - 0.6 < t < T("fireball", 0.0) + 0.3) or (C("ghast") < t < T("ghast", 0.6)))
    pipg = x.pip([(0, ng(9.5, 2))], skin="pip_iron", item="sword:iron", yaw=lambda t: 2.4, pose=lambda t: {"head_pitch": 0.5})
    out.append(E.Shot("cry", E.cam_static(ng(8.0, 4) + (0, 1.4, 0), gp0 + (0, 1.5, 0), fov=40), loc="nether", actors=[ghast],
                      sfx=[(T("cry", 0.6), "groan", -9, dict(seed=90)), (T("cry", 0.9), "groan", -10, dict(seed=91))]))
    out.append(E.Shot("ghast", E.cam_static(ng(7.5, 5) + (0, 2.0, 0), (gp0 + gp1) / 2 + (0, 0.5, 0), fov=55), loc="nether", actors=[ghast, pipg],
                      music="panic",
                      titles=[x.title(Wd("ghast", "ghast"), Ee("ghast"), "GHAST!", color=Wh, emoji="👻", size=140)],
                      sfx=[(T("ghast", 0.2), "groan", -6, dict(seed=92))]))
    # 6-7 fireball
    tf0 = T("fireball", 0.0)
    tf1 = tf0 + 1.0
    hit = ng(10.8, 3.5) + (0, 0.3, 0)
    pipb = x.pip([(C("fireball"), ng(9.5, 2)), (tf1, ng(9.5, 2)), (tf1 + 0.3, ng(8.2, 1.2)), (Ee("hole") + 2, ng(8.0, 1.0))], skin="pip_iron",
                 item="sword:iron", yaw=2.4, flash=lambda t: 0.5 if tf1 <= t < tf1 + 0.5 and (t * 8) % 1 < 0.5 else 0.0)
    fb = fireball_prop(tf0, tf1, gp0 + (0, 1.5, 1.0) + (gp1 - gp0) * 0.2, hit)
    hpfn = lambda t: 10.0 if t < tf1 else (5.0 if t < Wd("threehearts", "Three") else 3.0)
    out.append(E.Shot("fireball", E.cam_static(ng(6.0, -1.5) + (0, 2.4, 0), ng(10.0, 3.0) + (0, 1.5, 0), fov=56), loc="nether",
                      actors=[ghast, pipb], props=[fb], fx=explosion_fx(tf1, hit, 830), shake=[(tf1, 1.2, 0.5)], flash=[(tf1, 0.5, 0.4)],
                      hp=hpfn, sfx=[(tf0, "whoosh", -6, dict(dur=1.0)), (tf1, "explosion", -2, {}), (tf1 + 0.1, "hurt", -6, {})]))
    crater = [E.Burst(C("hole") + 0.35 * k, hit, 4, speed=(0.2, 0.6), life=(1.5, 2.5), size=(0.4, 0.8), colors=((0.3, 0.3, 0.3, 0.6),),
                      gravity=0.8, drag=1.0, up_bias=1.0, seed=840 + k, box=1.0, grow=1.5) for k in range(10)]
    out.append(E.Shot("hole", E.cam_static(hit + (-5.5, 5.5, -5.0), hit, fov=50), loc="nether", actors=[pipb], hp=hpfn,
                      fx=crater + E.fire_burst(C("hole"), hit, 4.0, seed=850),
                      titles=[x.title(T("hole", 0.4), Ee("hole"), "HUGE HOLE", color=O, emoji="💥")]))
    # 8 run back
    pipr = x.pip([(C("threehearts"), ng(8.0, 1.0)), (Ee("threehearts") - 0.2, ng(0.0, -1.0))], skin="pip_iron", item="sword:iron")
    out.append(E.Shot("threehearts", E.cam_track(pipr, (-3.0, 2.0, -3.5), look=(-1.0, 1.0, 0.0), fov=55), loc="nether", actors=[pipr, ghast],
                      props=[pportal, fireball_prop(T("threehearts", 0.3), T("threehearts", 0.3) + 1.2, gp1 + (0, 1.5, 1.0), ng(3.0, 2.5) + (0, 0.5, 0))],
                      fx=explosion_fx(T("threehearts", 0.3) + 1.2, ng(3.0, 2.5) + (0, 0.5, 0), 860, 0.7), hp=3.0,
                      overlay=orange_overlay(f, C("threehearts"), Ee("threehearts") + 0.5, peak=0.4, col="red"),
                      sfx=[(T("threehearts", 0.3) + 1.2, "explosion", -6, dict(seed=5)), (T("threehearts", 0.5), "heartbeat", -6, {})]))
    # 9 escape: back home
    px, pz = WD.PORTAL
    pip9 = x.pip([(C("escape"), (px + 2.0, pz + 0.3)), (T("escape", 0.4), (px + 2.0, pz - 2.8))], skin="pip_iron", item="sword:iron")
    bis9 = x.biscuit([(C("escape"), (px + 4.6, pz - 1.6)), (T("escape", 0.5), (px + 4.6, pz - 1.6)), (Ee("escape"), (px + 3.4, pz - 2.9))],
                     sit=lambda t: 1.0 if t < T("escape", 0.5) else 0.0, face_target=lambda t: pip9.pos(t))
    out.append(E.Shot("escape", E.cam_static((px + 2.5, GY + 2.2, pz - 7.0), (px + 2.5, GY + 1.6, pz), fov=52), stage="P5", phase=0.0,
                      actors=[pip9, bis9], flash=[(C("escape"), 0.6, 0.8)], hp=3.0, music="resolve",
                      fx=[E.hearts_burst(T("escape", 0.75), (px + 3.4, GY + 1.6, pz - 2.9), 870)],
                      titles=[x.title(T("escape", 0.6), Ee("escape"), "SAFE...", color=G, emoji="😮‍💨")]))
    # 10 round one
    pip10 = x.pip([(0, (px + 2.0, pz - 2.8))], skin="pip_iron", item="sword:iron", yaw=0.0)
    out.append(E.Shot("day60", E.cam_static((px + 0.0, GY + 1.2, pz - 6.5), (px + 2.0, GY + 2.2, pz), fov=48), stage="P5", phase=1.0,
                      actors=[pip10, bis9], hp=lambda t: 3.0 + 7.0 * E.seg(t, C("day60"), Ee("day60")),
                      titles=[x.title(Wd("day60", "Round"), Wd("day60", "But"), "ROUND 1: NETHER", color=R, emoji="🏆", size=96),
                              x.title(Wd("day60", "back"), Ee("day60"), "I'LL BE BACK", color=Y, emoji="😤", size=110)]))
    return out


def ch15(x):
    T, C, Wd, Ee, f = x.T, x.C, x.Wd, x.E, x.f
    out = []
    ng = lambda a, b: nether_ground(x, a, b)
    # 1 prepared
    pip1 = x.pip([(0, (12.2, 12.4))], skin="pip_iron", item="sword:iron", yaw=PI / 2 + 0.3, acts=[T("prepared", 0.3), T("prepared", 0.6)])
    out.append(E.Shot("prepared", E.cam_static((9.0, GY + 2.7, 10.0), (13.5, GY + 1.0, 13.4), fov=52), stage="P5", phase=0.0, actors=[pip1],
                      clouds=False, music="adventure",
                      hotbar=(["e:🗡️", "e:⛏️", "cobble", "cobble", "cobble", "e:🍞", "e:🍞", "e:🥩", "e:🏹"], 2),
                      titles=[x.title(Wd("prepared", "Blocks"), Wd("prepared", "food"), "BLOCKS", color=Wh, emoji="🧱", pos=(560, 160), size=80),
                              x.title(Wd("prepared", "food"), Wd("prepared", "plan"), "FOOD", color=O, emoji="🍞", pos=(900, 160), size=80),
                              x.title(Wd("prepared", "plan"), Ee("prepared"), "A PLAN", color=Y, emoji="📝", pos=(1240, 160), size=80)]))
    gp = x.ng(20.0, 27.0, -12.0)
    ghast = x.mob("ghast", [(C("secret"), gp), (Ee("direct"), gp + (0, 1.0, 0))], yaw=-PI * 0.7,
                  open_=lambda t: T("timing", 0.3) < t < T("timing", 0.5), death=Wd("direct", "down") + 0.1,
                  flash=lambda t: 0.6 if Wd("direct", "Direct") <= t < Wd("direct", "Direct") + 0.5 else 0.0)
    pp = ng(10.0, 3.0)
    pip2 = x.pip([(0, pp)], skin="pip_iron", item="sword:iron", yaw=2.3, pose=lambda t: {"head_pitch": 0.35})
    # 2 secret
    def arrow(layer, t, t0=Wd("secret", "hit"), t1=Ee("secret")):
        if t0 <= t < t1:
            ov.draw_text(layer, "↩", (1500, 420), 200, fill=(255, 225, 40), alpha=min(1, (t - t0) / 0.3))
    out.append(E.Shot("secret", E.cam_static(pp + (-1.9, 3.2, 3.0), gp + (0, 0.0, 0), fov=58), loc="nether", actors=[ghast, pip2], overlay=arrow,
                      music="suspense",
                      titles=[x.title(Wd("secret", "hit"), Ee("secret"), "HIT IT BACK!", color=Y, emoji="⚾", size=110)]))
    # 3-4 the swing
    tl = T("timing", 0.45)
    tsw = Wd("timing", "swung")
    tback = Wd("direct", "Direct")
    fb = fireball_prop(tl, tsw, gp + (0, 1.6, 1.5), pp + (0.6, 1.4, -0.4), back=(tback, gp + (0, 2.0, 0)))
    pip3 = x.pip([(0, pp)], skin="pip_iron", item="sword:iron", yaw=2.3, acts=[tsw - 0.08])
    def slowmo_cam(t):
        u = E.smooth(E.seg(t, C("timing"), tsw))
        e = pp + np.array([-1.9, 3.2, 3.0]) * (1 - u) + np.array([-1.2, 2.6, 2.2]) * u
        return dict(eye=e.astype(np.float32), target=(pp + (gp - pp) * 0.22 + np.array([0, 0.6, 0])).astype(np.float32), fov=58 - 6 * u, roll=0.0)
    out.append(E.Shot("timing", E.cam_track(pip3, (-2.2, 2.3, 2.4), look=(2.2, 1.6, -2.6), fov=58), loc="nether", actors=[ghast, pip3], props=[fb],
                      titles=[x.title(Wd("timing", "waited"), Wd("timing", "timed"), "WAIT...", color=Wh, size=90),
                              x.title(Wd("timing", "timed"), tsw, "WAIT...", color=Wh, size=90),
                              x.title(tsw, Ee("timing"), "SWING!", color=Y, emoji="⚔️", size=120)],
                      sfx=[(tl, "whoosh", -8, dict(dur=1.4)), (tsw, "impact", -4, {}), (tsw, "whoosh", -6, dict(dur=0.6))]))
    out.append(E.Shot("direct", E.cam_static(pp + (-1.5, 3.4, 2.6), gp + (0, 2.0, 0), fov=52), loc="nether", actors=[ghast, pip3], props=[fb],
                      fx=explosion_fx(tback, gp + (0, 2.0, 0), 900, 1.4) + [E.poof(Wd("direct", "down"), gp + (0, 2.0, 0), 901, n=40)],
                      shake=[(tback, 1.0, 0.4)], flash=[(tback, 0.4, 0.4)],
                      titles=[x.title(tback, Ee("direct"), "DIRECT HIT!", color=Y, emoji="🎯", size=130)],
                      sfx=[(tback, "explosion", -3, {}), (Wd("direct", "down"), "pop", -6, dict(pitch=0.5))]))
    # 5 ghast tear
    pip5 = x.pip([(0, pp)], skin="pip_iron", item="sword:iron", yaw=2.3, pose=lambda t: {"head_pitch": 0.2})
    out.append(E.Shot("tear", E.cam_front(pip5, dist=3.0, up=0.5, side=-1.0, fov=48), loc="nether", actors=[pip5],
                      props=[E.Fly([(C("tear"), T("tear", 0.6), gp + (0, 2.0, 0), pp + (0, 1.2, 0), "diamond")])],
                      titles=[x.title(Wd("tear", "tear"), Ee("tear"), "GHAST TEAR", color=(220, 240, 255), emoji="💧"),
                              x.title(Wd("tear", "rare"), Ee("tear"), "SUPER RARE", color=Y, pos=(1050, 270), size=70)]))
    # 6 glowstone
    gs_base = ng(14.0, 4.0)
    gsp = [(14, 0, 6), (15, 0, 6), (14, 1, 6), (15, 1, 6), (14, 2, 6), (15, 0, 7), (14, 0, 7), (15, 1, 7)]
    by = int(gs_base[1])
    cells = [x.ng(a, by + b, c) - 0.5 for (a, b, c) in gsp]
    tg = x.every(T("glowstone", 0.3), Ee("glowstone") - 0.3, len(cells))
    pip6 = x.pip([(C("glowstone"), ng(11.5, 3.0)), (T("glowstone", 0.25), ng(14.5, 4.6))], skin="pip_iron", item="pickaxe:diamond",
                 acts=x.every(T("glowstone", 0.25), Ee("glowstone") - 0.3, 16), yaw=lambda t: 0.0 if t > T("glowstone", 0.25) else None)
    br = E.Breaks([(tg[i], tuple(cells[i]), "glowstone") for i in range(len(cells))], dur=0.4, collect=lambda t: pip6.pos(t), drop="tnt",
                  overlay=False)
    br.drop = None
    fly = E.Fly([(tg[i], tg[i] + 0.7, cells[i] + 0.5, pip6.pos(tg[i]) + (0, 1.0, 0), "glowstone") for i in range(len(cells))])
    gl = E.Custom(lambda t: ([], [], [[c[0] + 0.5, c[1] + 0.5, c[2] + 0.5, 1.2, 1.0, 0.85, 0.5, 0.25] for i, c in enumerate(cells) if t < tg[i]]))
    out.append(E.Shot("glowstone", E.cam_static(ng(17.0, 2.0) + (0, 2.2, 0), ng(14.6, 6.0) + (0, 1.0, 0), fov=52), loc="nether", actors=[pip6],
                      props=[br, fly, gl], music="adventure",
                      titles=[x.title(Wd("glowstone", "Glowstone"), Ee("glowstone"), "GLOWSTONE", color=Y, emoji="✨", size=120)]))
    # 7 upgrade: full inventory
    pip7 = x.pip([(0, ng(14.5, 4.6))], skin="pip_iron", item="glowstone", yaw=PI * 0.9)
    out.append(E.Shot("upgrade", E.cam_track(pip7, (2.4, 2.8, 2.6), look=(0, 1.2, 0), fov=50), loc="nether", actors=[pip7],
                      hotbar=(["glowstone"] * 9, 4),
                      titles=[x.title(T("upgrade", 0.2), Ee("upgrade"), "x64 GLOWSTONE", color=Y, emoji="✨")]))
    # 8 the glowing lighthouse
    w, top, y0 = WD.lighthouse_world(f.terrain, glow=True)
    top = top[1]
    hx, hz = WD.HILL_C
    tg8 = T("glowtop", 0.35)
    def glow(t):
        k = E.smooth(E.seg(t, tg8, tg8 + 0.6))
        if k <= 0:
            return None
        return [], [], [[hx + 0.5 + dx, top + 3.4 + dy, hz + 0.5 + dz, 2.6 * k, 1.0, 0.85, 0.45, 0.4] for dx, dy, dz in ((0, 0, 0), (1.2, -0.6, 0.8), (-1.2, -0.6, -0.8), (0.8, 0.5, -1.0), (-0.8, 0.5, 1.0))]
    out.append(E.Shot("glowtop", E.cam_static((14.0, GY + 4.0, 6.0), (hx + 0.5, top - 2.0, hz), fov=34), stage="P5", phase=2.0,
                      statics=["lighthouse_glow"], props=[E.Custom(glow)], music="resolve",
                      titles=[x.title(T("glowtop", 0.5), Ee("glowtop"), "LIGHTHOUSE 2.0", color=Y, emoji="✨", size=110)],
                      sfx=[(tg8, "shimmer", -8, {})]))
    # 9 day 65
    pip9 = x.pip([(0, (10.5, 5.0))], skin="pip_iron", item="sword:iron", yaw=PI, pose=lambda t: {"wave": 1.0, "wave_phase": t * 8})
    out.append(E.Shot("day65", E.cam_front(pip9, dist=3.6, up=0.5, side=1.0, fov=48), stage="P5", phase=0.0, actors=[pip9],
                      titles=[x.title(Wd("day65", "Nether"), Wd("day65", "Kind"), "NETHER: DEFEATED", color=G, emoji="✅", size=96),
                              x.title(Wd("day65", "Kind"), Ee("day65"), "(KIND OF)", color=Wh, emoji="😅", size=96)]))
    return out
