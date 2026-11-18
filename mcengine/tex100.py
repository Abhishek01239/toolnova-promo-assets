"""Tiles for '100 Days' (appended after all older tiles so existing cached meshes keep their UVs):
animals, Biscuit the wolf, spider, ghast, villagers, armoured Pip, nether blocks, farm and items."""
import numpy as np


def make(add, _noise_tile, _rng, T):
    def border(img, col):
        img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = col
        return img

    # ---------------------------------------------------------------- blocks
    wool = _noise_tile("wool", [232, 232, 228], 8)
    r = _rng("wool_curls")
    for _ in range(30):
        y, x = r.integers(0, T, 2)
        wool[y, x] = [205, 205, 200]
    add("wool", wool)
    sb = _noise_tile("stone_bricks", [122, 122, 122], 8)
    for y in (0, 7, 8, 15):
        sb[y, :] = (sb[y, :] * 0.7).astype(np.uint8)
    sb[0:8, 0] = sb[0:8, 15] = (sb[0:8, 0] * 0.7).astype(np.uint8)
    sb[8:16, 7] = sb[8:16, 8] = (sb[8:16, 7] * 0.7).astype(np.uint8)
    add("stone_bricks", sb)
    gs = _noise_tile("glowstone", [222, 180, 90], 30)
    r = _rng("glow_spots")
    for _ in range(14):
        y, x = r.integers(0, 15, 2)
        gs[y:y + 2, x:x + 2] = [255, 238, 160]
    add("glowstone", gs)
    nr = _noise_tile("netherrack", [112, 44, 44], 22)
    r = _rng("netherrack_spots")
    for _ in range(26):
        y, x = r.integers(0, T, 2)
        nr[y, x] = [70, 22, 26]
    for _ in range(10):
        y, x = r.integers(0, T, 2)
        nr[y, x] = [150, 70, 64]
    add("netherrack", nr)
    pt = np.zeros((T, T, 4), np.uint8)
    yy, xx = np.mgrid[0:T, 0:T]
    v = 0.5 + 0.5 * np.sin(xx * 0.9 + np.sin(yy * 0.7) * 2.2)
    pt[..., 0] = (110 + 90 * v).astype(np.uint8)
    pt[..., 1] = (30 + 40 * v).astype(np.uint8)
    pt[..., 2] = (190 + 60 * v).astype(np.uint8)
    pt[..., 3] = 200
    add("portal", pt)
    pa = _noise_tile("path_top", [148, 122, 66], 12)
    add("path_top", pa)
    hay = _noise_tile("hay_side", [200, 170, 60], 16)
    hay[3, :] = hay[12, :] = [140, 70, 30]
    add("hay_side", hay)
    add("hay_top", _noise_tile("hay_top", [196, 168, 64], 20))
    et = _noise_tile("enchant_top", [160, 30, 40], 10)
    border(et, [40, 30, 40])
    et[4:12, 4:12] = [230, 230, 220]
    et[4:12, 7:9] = [120, 40, 40]
    add("enchant_top", et)
    es = _noise_tile("enchant_side", [30, 26, 40], 8)
    es[0:4, :] = [160, 30, 40]
    es[5, 2] = es[9, 11] = es[12, 5] = [120, 240, 230]
    add("enchant_side", es)
    dr = _noise_tile("door", [150, 108, 62], 8)
    border(dr, [96, 66, 36])
    dr[2:7, 3:13] = [190, 220, 235]
    dr[4, :] = [96, 66, 36]
    dr[9, 11:13] = [60, 60, 60]
    add("door", dr)
    bt = _noise_tile("bed_top", [190, 30, 36], 8)
    bt[0:5, :] = [236, 236, 236]
    add("bed_top", bt)
    add("bed_side", _noise_tile("bed_side", [150, 108, 62], 8))
    lt = _noise_tile("leaves_dark", [46, 110, 40], 22)
    add("leaves_dark", lt)
    cy = np.zeros((T, T, 4), np.uint8)
    r = _rng("crop_young")
    for x in range(1, 15, 3):
        h = r.integers(3, 7)
        cy[T - h:, x] = [80, 180, 60, 255]
    add("crop_young", cy)
    cg = np.zeros((T, T, 4), np.uint8)
    r = _rng("crop_gold")
    for x in range(1, 15, 2):
        h = r.integers(10, 16)
        cg[T - h:, x] = [180, 160, 60, 255]
        cg[T - h:T - h + 5, x] = [230, 200, 80, 255]
    add("crop_gold", cg)
    add("fire", _noise_tile("fire", [255, 140, 30], 40))

    # ---------------------------------------------------------------- items (flat icons)
    def icon(name, pts, col, bg=(0, 0, 0, 0)):
        a = np.zeros((T, T, 4), np.uint8)
        a[:] = bg
        for (y, x) in pts:
            a[y, x] = list(col) + [255]
        return a
    bone = np.zeros((T, T, 4), np.uint8)
    for i in range(3, 13):
        bone[i, i] = bone[i, i - 1] = [235, 232, 220, 255]
    for (y, x) in ((2, 2), (2, 3), (3, 1), (13, 13), (13, 12), (12, 14), (2, 4), (4, 1)):
        bone[y, x] = [235, 232, 220, 255]
    add("bone", bone)
    em = np.zeros((T, T, 4), np.uint8)
    for y in range(2, 14):
        w = min(y - 1, 14 - y, 5)
        em[y, 8 - w:8 + w] = [40, 200, 90, 255]
        em[y, 8 - w:8 - w + 2] = [140, 250, 170, 255]
    add("emerald", em)
    wh = np.zeros((T, T, 4), np.uint8)
    for i in range(2, 14):
        wh[i, 14 - i] = [200, 170, 60, 255]
        if i < 9:
            wh[i, 15 - i] = wh[i, 13 - i] = [230, 200, 90, 255]
    add("wheat_item", wh)
    fs = np.zeros((T, T, 4), np.uint8)
    fs[3:9, 3:5] = [200, 200, 205, 255]
    fs[3:5, 3:9] = [200, 200, 205, 255]
    fs[8:14, 8:13] = [60, 60, 66, 255]
    add("flint_steel", fs)
    add("blade_iron", _noise_tile("blade_iron", [220, 220, 228], 10, (2, 10)))
    add("blade_ench", _noise_tile("blade_ench", [180, 120, 255], 20, (2, 10)))
    add("hoe_head", _noise_tile("hoe_head", [128, 128, 128], 10, (6, 2)))
    add("boat_top", _noise_tile("boat_top", [150, 108, 62], 10))
    fb = _noise_tile("fireball", [255, 150, 40], 40)
    fb[4:12, 4:12] = [255, 220, 120]
    add("fireball", fb)
    add("book", _noise_tile("book", [120, 60, 40], 10))

    # ---------------------------------------------------------------- cow
    def cow_tile(name, size):
        img = _noise_tile(name, [80, 56, 40], 10, size)
        r = _rng(name + "_spots")
        h, w = img.shape[:2]
        for _ in range(max(1, (h * w) // 18)):
            y, x = r.integers(0, h), r.integers(0, w)
            img[y:y + 2, x:x + 3] = [228, 228, 228]
        return img
    add("cow_side", cow_tile("cow_side", (12, 10)))
    add("cow_top", cow_tile("cow_top", (12, 18)))
    add("cow_end", cow_tile("cow_end", (12, 10)))
    cf = _noise_tile("cow_face", [80, 56, 40], 8, (8, 8))
    cf[1:3, 0:3] = [228, 228, 228]
    cf[2, 1] = cf[2, 6] = [20, 20, 20]
    cf[5:8, 2:6] = [210, 160, 150]
    cf[6, 3] = cf[6, 4] = [90, 60, 60]
    add("cow_face", cf)
    add("cow_head", _noise_tile("cow_head", [80, 56, 40], 8, (8, 8)))
    cl = _noise_tile("cow_leg", [80, 56, 40], 8, (4, 12))
    cl[0:6, :] = [228, 228, 228]
    cl[10:, :] = [60, 50, 40]
    add("cow_leg", cl)
    add("horn", _noise_tile("horn", [220, 214, 196], 6, (2, 2)))

    # ---------------------------------------------------------------- sheep
    sf = _noise_tile("sheep_face", [222, 190, 170], 8, (6, 6))
    sf[2, 0] = sf[2, 5] = [20, 20, 20]
    sf[2, 1] = sf[2, 4] = [240, 240, 240]
    sf[4, 2:4] = [230, 160, 160]
    add("sheep_face", sf)
    add("sheep_head", _noise_tile("sheep_head", [222, 190, 170], 8, (6, 6)))
    sl = _noise_tile("sheep_leg", [232, 232, 228], 8, (4, 12))
    sl[6:, :] = [222, 190, 170]
    add("sheep_leg", sl)
    add("sheep_shorn", _noise_tile("sheep_shorn", [222, 190, 170], 8))

    # ---------------------------------------------------------------- wolf (Biscuit)
    grey = [214, 210, 206]
    add("wolf_body", _noise_tile("wolf_body", grey, 10, (6, 9)))
    add("wolf_mane", _noise_tile("wolf_mane", [200, 196, 192], 12, (8, 6)))
    wf = _noise_tile("wolf_face", grey, 8, (6, 6))
    wf[2, 1] = wf[2, 4] = [30, 30, 30]
    wf[1, 1] = wf[1, 4] = [150, 140, 130]
    add("wolf_face", wf)
    add("wolf_head", _noise_tile("wolf_head", grey, 8, (6, 6)))
    sn = _noise_tile("wolf_snout", grey, 6, (3, 3))
    sn[0, 1] = [30, 30, 30]
    add("wolf_snout", sn)
    add("wolf_leg", _noise_tile("wolf_leg", grey, 8, (2, 8)))
    add("collar", _noise_tile("collar", [200, 30, 36], 8, (4, 2)))

    # ---------------------------------------------------------------- spider
    sp = _noise_tile("spider_body", [52, 44, 40], 10, (10, 12))
    r = _rng("spider_hair")
    for _ in range(20):
        y, x = r.integers(0, 12), r.integers(0, 10)
        sp[y, x] = [80, 70, 62]
    add("spider_body", sp)
    sface = _noise_tile("spider_face", [46, 40, 36], 8, (8, 8))
    for (y, x) in ((2, 1), (2, 6), (3, 2), (3, 5), (2, 3), (2, 4)):
        sface[y, x] = [220, 30, 30]
    sface[3, 3] = sface[3, 4] = [255, 70, 50]
    add("spider_face", sface)
    add("spider_head", _noise_tile("spider_head", [46, 40, 36], 8, (8, 8)))
    add("spider_leg", _noise_tile("spider_leg", [40, 34, 30], 6, (16, 2)))

    # ---------------------------------------------------------------- ghast
    gh = _noise_tile("ghast_body", [236, 236, 236], 6, (16, 16))
    add("ghast_body", gh)
    gf = gh.copy()
    gf[5:7, 3:6] = [60, 60, 60]
    gf[5:7, 10:13] = [60, 60, 60]
    gf[10:12, 6:10] = [60, 60, 60]
    add("ghast_face", gf)
    go = gh.copy()
    go[4:8, 3:6] = [30, 30, 30]
    go[4:8, 10:13] = [30, 30, 30]
    go[7, 4] = go[7, 11] = [180, 40, 40]
    go[9:14, 5:11] = [30, 30, 30]
    add("ghast_face_open", go)
    add("ghast_tentacle", _noise_tile("ghast_tentacle", [226, 226, 226], 6, (2, 10)))

    # ---------------------------------------------------------------- villager
    vs = np.array([196, 140, 108])
    robe = np.array([110, 74, 46])
    vface = _noise_tile("villager_face", vs, 6, (8, 8))
    vface[0:2, :] = [70, 50, 30]
    vface[3, 1:7] = [70, 50, 30]                 # unibrow
    vface[4, 1:3] = [255, 255, 255]
    vface[4, 5:7] = [255, 255, 255]
    vface[4, 2] = vface[4, 5] = [40, 140, 60]
    vface[4:8, 3:5] = (vs * 0.8).astype(np.uint8)   # big nose
    vface[7, 2:6] = [120, 80, 60]
    add("villager_face", vface)
    hs = _noise_tile("villager_head_side", vs, 6, (8, 8))
    hs[0:2, :] = [70, 50, 30]
    add("villager_head_side", hs)
    add("villager_head_side_m", hs[:, ::-1].copy())
    add("villager_head_back", _noise_tile("villager_head_back", vs, 6, (8, 8)))
    add("villager_hair_top", _noise_tile("villager_hair_top", [70, 50, 30], 6, (8, 8)))
    add("villager_chin", _noise_tile("villager_chin", vs, 6, (8, 8)))
    vb = _noise_tile("villager_body_front", robe, 8, (8, 12))
    vb[0:3, 1:7] = [60, 120, 50]              # green collar
    vb[6, :] = [70, 46, 30]                   # belt
    add("villager_body_front", vb)
    add("villager_body_back", _noise_tile("villager_body_back", robe, 8, (8, 12)))
    add("villager_body_side", _noise_tile("villager_body_side", robe, 8, (4, 12)))
    add("villager_body_top", _noise_tile("villager_body_top", robe, 8, (8, 4)))
    va = _noise_tile("villager_arm", robe, 8, (4, 12))
    va[9:12, :] = vs
    add("villager_arm", va)
    add("villager_arm_top", _noise_tile("villager_arm_top", robe, 8, (4, 4)))
    add("villager_hand", _noise_tile("villager_hand", vs, 5, (4, 4)))
    vl = _noise_tile("villager_leg", robe, 8, (4, 12))
    vl[10:, :] = [60, 44, 30]
    add("villager_leg", vl)
    add("villager_shoe", _noise_tile("villager_shoe", [60, 44, 30], 5, (4, 4)))


def make_armor(add, tiles, _noise_tile, T):
    """pip_iron_* and pip_dia_*: Pip's tiles with an armour set painted over them."""
    from PIL import Image

    def native(name, size):
        a = tiles[name]
        return np.array(Image.fromarray(a, "RGBA").resize(size, Image.NEAREST))

    for skin, col in (("pip_iron", (206, 206, 212)), ("pip_dia", (84, 224, 214))):
        c = np.array(col)
        dark = (c * 0.72).astype(np.uint8)

        def metal(name, size):
            m = _noise_tile(skin + name, c, 10, size)
            m = np.concatenate([m, np.full(m.shape[:2] + (1,), 255, np.uint8)], 2)
            return m

        def paint(dst, src, mask):
            dst[mask] = src[mask]
            return dst
        # head: helmet over hair and sides, face visible
        f = native("pip_face", (8, 8))
        m = metal("_hf", (8, 8))
        f[0:3, :] = m[0:3, :]
        f[3:8, 0] = m[3:8, 0]
        f[3:8, 7] = m[3:8, 7]
        f[2, :, :3] = dark
        add(skin + "_face", f[..., :3])
        hs = metal("_hs", (8, 8))
        hs[7, :, :3] = dark
        add(skin + "_head_side", hs[..., :3])
        add(skin + "_head_side_m", hs[:, ::-1, :3].copy())
        add(skin + "_head_back", metal("_hb", (8, 8))[..., :3])
        add(skin + "_hair_top", metal("_ht", (8, 8))[..., :3])
        add(skin + "_chin", native("pip_chin", (8, 8))[..., :3])
        bf = metal("_bf", (8, 12))
        bf[0, 2:6, :3] = dark
        bf[5, :, :3] = dark
        bf[10:12, :, :3] = dark
        add(skin + "_body_front", bf[..., :3])
        bb = metal("_bb", (8, 12))
        bb[10:12, :, :3] = dark
        add(skin + "_body_back", bb[..., :3])
        add(skin + "_body_side", metal("_bs", (4, 12))[..., :3])
        add(skin + "_body_top", metal("_bt", (8, 4))[..., :3])
        arm = native("pip_arm", (4, 12))
        am = metal("_ar", (4, 12))
        arm[0:5] = am[0:5]
        arm[5, :, :3] = dark
        add(skin + "_arm", arm[..., :3])
        add(skin + "_arm_top", metal("_at", (4, 4))[..., :3])
        add(skin + "_hand", native("pip_hand", (4, 4))[..., :3])
        lg = metal("_lg", (4, 12))
        lg[8, :, :3] = dark
        add(skin + "_leg", lg[..., :3])
        add(skin + "_shoe", metal("_sh", (4, 4))[..., :3])


def make_more(add, _noise_tile, _rng, T):
    fl = np.zeros((T, T, 4), np.uint8)
    for y in range(3, 13):
        w = 5 - abs(y - 8) // 2
        fl[y, 8 - w:8 + w] = [150, 80, 60, 255]
        fl[y, 8 - w] = [110, 140, 70, 255]
    add("flesh", fl)
