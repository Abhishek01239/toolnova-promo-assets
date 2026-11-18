"""Procedural Minecraft-style 16x16 pixel textures packed into one atlas.

Every texture is generated from code (no copyrighted game assets). Tiles are
addressed by name; `TILE_UV[name]` gives the (u0, v0, u1, v1) rectangle.
Character skin parts are drawn at their native pixel size (e.g. 8x12 for a
torso front) and scaled to 16x16 with nearest-neighbour so they look correct
once mapped onto a box face with the same proportions.
"""
import numpy as np
from PIL import Image

T = 16              # tile size in pixels
GRID = 16           # atlas is GRID x GRID tiles
ATLAS_PX = T * GRID

_tiles = {}
TILE_INDEX = {}


def _rng(name):
    # deterministic per-name seed (Python's hash() is salted per process)
    seed = 0
    for ch in name:
        seed = (seed * 131 + ord(ch)) % (2**32)
    return np.random.default_rng(seed)


def _noise_tile(name, base, var=18, size=(T, T)):
    r = _rng(name)
    h, w = size[1], size[0]
    n = r.integers(-var, var + 1, (h, w, 1))
    img = np.clip(np.array(base, dtype=int)[None, None, :] + n, 0, 255)
    return img.astype(np.uint8)


def _rgba(img, a=255):
    if img.shape[2] == 4:
        return img
    alpha = np.full(img.shape[:2] + (1,), a, np.uint8)
    return np.concatenate([img, alpha], 2)


def _scale_to_tile(img):
    """Nearest-neighbour scale of an arbitrary w x h pixel image to 16x16."""
    pil = Image.fromarray(_rgba(img), "RGBA").resize((T, T), Image.NEAREST)
    return np.array(pil)


def add(name, img):
    if img.shape[0] != T or img.shape[1] != T:
        img = _scale_to_tile(img)
    _tiles[name] = _rgba(img)


# --------------------------------------------------------------- blocks
def _make_blocks():
    grass = np.array([95, 159, 53])
    dirt = np.array([134, 96, 67])

    add("grass_top", _noise_tile("grass_top", grass, 22))

    side = _noise_tile("grass_side", dirt, 16)
    r = _rng("grass_side_edge")
    g = _noise_tile("gs2", grass, 20)
    for x in range(T):
        depth = 3 + r.integers(0, 3)
        side[:depth, x] = g[:depth, x]
    add("grass_side", side)

    d = _noise_tile("dirt", dirt, 18)
    r = _rng("dirt_specks")
    for _ in range(14):
        y, x = r.integers(0, T, 2)
        d[y, x] = (dirt * 0.7).astype(int)
    add("dirt", d)

    add("stone", _noise_tile("stone", [125, 125, 125], 14))

    c = _noise_tile("cobble", [118, 118, 118], 12)
    r = _rng("cobble_cracks")
    for _ in range(40):
        y, x = r.integers(0, T, 2)
        c[y, x] = [80, 80, 80]
    for y in (4, 9, 14):
        c[y, :] = (c[y, :] * 0.65).astype(np.uint8)
    for (y0, y1, x) in [(0, 4, 6), (4, 9, 2), (4, 9, 11), (9, 14, 7), (14, 16, 3), (14, 16, 12)]:
        c[y0:y1, x] = (c[y0:y1, x] * 0.6).astype(np.uint8)
    add("cobble", c)

    log = _noise_tile("log_side", [104, 83, 50], 10)
    for x in range(0, T, 3):
        log[:, x] = (log[:, x] * 0.72).astype(np.uint8)
    add("log_side", log)

    lt = _noise_tile("log_top", [176, 142, 88], 8)
    yy, xx = np.mgrid[0:T, 0:T]
    rr = np.maximum(abs(yy - 7.5), abs(xx - 7.5))
    lt[(rr.astype(int) % 3) == 0] = [140, 108, 64]
    lt[rr > 6.6] = [104, 83, 50]
    add("log_top", lt)

    pl = _noise_tile("planks", [168, 132, 82], 8)
    for y in (3, 7, 11, 15):
        pl[y, :] = [110, 84, 50]
    for (y0, y1, x) in [(0, 3, 9), (4, 7, 3), (8, 11, 12), (12, 15, 6)]:
        pl[y0:y1, x] = [120, 92, 56]
    add("planks", pl)

    lv = _noise_tile("leaves", [62, 128, 40], 30)
    r = _rng("leaf_holes")
    a = np.full((T, T, 1), 255, np.uint8)
    holes = r.random((T, T)) < 0.13
    a[holes] = 0
    add("leaves", np.concatenate([lv, a], 2))

    add("sand", _noise_tile("sand", [219, 207, 160], 10))

    w = _noise_tile("water", [52, 92, 210], 12)
    add("water", np.concatenate([w, np.full((T, T, 1), 175, np.uint8)], 2))

    gl = np.zeros((T, T, 4), np.uint8)
    gl[[0, -1], :] = [220, 240, 245, 255]
    gl[:, [0, -1]] = [220, 240, 245, 255]
    for i in range(3, 8):
        gl[i, i + 2] = [255, 255, 255, 220]
    add("glass", gl)

    tnt = np.zeros((T, T, 3), np.uint8)
    tnt[:] = [200, 40, 30]
    tnt[5:11, :] = [235, 235, 235]
    tnt[6:10, 3:13:2] = [30, 30, 30]
    add("tnt_side", tnt)
    tt = _noise_tile("tnt_top", [200, 40, 30], 10)
    tt[6:10, 6:10] = [40, 40, 40]
    add("tnt_top", tt)

    add("cloud", np.full((T, T, 3), 250, np.uint8))

    ob = _noise_tile("obsidian", [24, 16, 38], 8)
    add("obsidian", ob)

    # "lucky" one-block: golden frame with a glowing core (original design)
    lb = _noise_tile("lucky", [236, 190, 52], 14)
    lb[[0, 1, -1, -2], :] = [150, 100, 20]
    lb[:, [0, 1, -1, -2]] = [150, 100, 20]
    lb[5:11, 5:11] = [255, 240, 140]
    lb[7:9, 7:9] = [255, 255, 255]
    add("lucky", lb)

    # breaking-crack overlay stages
    for s in range(4):
        cr = np.zeros((T, T, 4), np.uint8)
        r = _rng(f"crack{s}")
        for _ in range(10 + s * 18):
            y, x = r.integers(0, T, 2)
            cr[y, x] = [20, 20, 20, 230]
        add(f"crack{s}", cr)

    add("white", np.full((T, T, 3), 255, np.uint8))
    add("black", np.zeros((T, T, 3), np.uint8))
    add("smoke", np.full((T, T, 3), 90, np.uint8))

    # flowing water (lighter)
    wf = _noise_tile("water_flow", [70, 120, 230], 14)
    add("water_flow", np.concatenate([wf, np.full((T, T, 1), 160, np.uint8)], 2))

    iron = _noise_tile("iron", [190, 190, 196], 8)
    add("bucket_side", iron)
    bt = _noise_tile("bucket_top", [52, 92, 210], 10)
    bt[0, :] = bt[-1, :] = bt[:, 0] = bt[:, -1] = [170, 170, 176]
    add("bucket_top", bt)


# --------------------------------------------------------------- characters
def _make_pip():
    """Pip: the channel's original hero. Orange hoodie, dark-brown hair, teal pants."""
    skin = np.array([228, 176, 132])
    hair = np.array([66, 42, 26])
    hood = np.array([238, 120, 30])
    pants = np.array([30, 128, 138])
    shoe = np.array([50, 50, 58])

    face = _noise_tile("pip_face", skin, 5, (8, 8))
    face[0:2, :] = hair
    face[2, 0] = face[2, 7] = hair
    face[4, 1:3] = [255, 255, 255]
    face[4, 5:7] = [255, 255, 255]
    face[4, 2] = [40, 90, 200]
    face[4, 5] = [40, 90, 200]
    face[3, 1:3] = hair * 0.9
    face[3, 5:7] = hair * 0.9
    face[6, 3:5] = [150, 90, 70]
    face[5, 3:5] = skin * 0.9
    add("pip_face", face)

    hs = _noise_tile("pip_head_side", skin, 5, (8, 8))
    hs[0:3, :] = hair
    hs[3:5, 5:8] = hair
    add("pip_head_side", hs)
    add("pip_head_side_m", hs[:, ::-1].copy())
    add("pip_head_back", _noise_tile("pip_head_back", hair, 8, (8, 8)))
    add("pip_hair_top", _noise_tile("pip_hair_top", hair, 8, (8, 8)))
    add("pip_chin", _noise_tile("pip_chin", skin, 5, (8, 8)))

    bf = _noise_tile("pip_body_front", hood, 10, (8, 12))
    bf[0, 2:6] = skin                       # neck
    bf[1, 3:5] = hood * 0.7                 # hood strings
    bf[1:6, 2] = [245, 245, 245]
    bf[1:6, 5] = [245, 245, 245]
    bf[6:9, 2:6] = hood * 0.78              # front pocket
    bf[4:6, 3:5] = [255, 220, 60]           # little star logo
    bf[10:12, :] = pants
    add("pip_body_front", bf)
    bb = _noise_tile("pip_body_back", hood, 10, (8, 12))
    bb[0:3, 1:7] = hood * 0.8               # hood
    bb[10:12, :] = pants
    add("pip_body_back", bb)
    bs = _noise_tile("pip_body_side", hood, 10, (4, 12))
    bs[10:12, :] = pants
    add("pip_body_side", bs)
    add("pip_body_top", _noise_tile("pip_body_top", hood, 10, (8, 4)))

    arm = _noise_tile("pip_arm", hood, 10, (4, 12))
    arm[8, :] = hood * 0.7                  # cuff
    arm[9:12, :] = skin
    add("pip_arm", arm)
    add("pip_arm_top", _noise_tile("pip_arm_top", hood, 10, (4, 4)))
    add("pip_hand", _noise_tile("pip_hand", skin, 5, (4, 4)))

    leg = _noise_tile("pip_leg", pants, 9, (4, 12))
    leg[10:12, :] = shoe
    leg[9, :] = [235, 235, 235]
    add("pip_leg", leg)
    add("pip_shoe", _noise_tile("pip_shoe", shoe, 5, (4, 4)))


def _make_creeper():
    def green(name, size):
        r = _rng(name)
        img = _noise_tile(name, [80, 170, 70], 26, size)
        spots = r.random(img.shape[:2]) < 0.25
        img[spots] = [150, 210, 140]
        dark = r.random(img.shape[:2]) < 0.1
        img[dark] = [40, 100, 40]
        return img

    face = green("creeper_face", (8, 8))
    k = [20, 20, 20]
    face[2:4, 1:3] = k
    face[2:4, 5:7] = k
    face[4:6, 3:5] = k
    face[5:8, 2] = k
    face[5:8, 5] = k
    face[6:8, 3:5] = k
    add("creeper_face", face)
    add("creeper_head", green("creeper_head", (8, 8)))
    add("creeper_body", green("creeper_body", (8, 12)))
    add("creeper_body_side", green("creeper_body_side", (4, 12)))
    lg = green("creeper_leg", (4, 6))
    lg[5, :] = [40, 70, 40]
    add("creeper_leg", lg)
    add("creeper_top", green("creeper_top", (8, 4)))


def _make_zombie():
    """An original zombie: grey-green skin, ragged brown shirt, dark trousers."""
    skin = np.array([104, 138, 96])
    hair = np.array([58, 74, 52])
    shirt = np.array([112, 78, 52])
    pants = np.array([58, 58, 70])
    face = _noise_tile("zombie_face", skin, 10, (8, 8))
    face[0:2, :] = hair
    face[3:5, 1:3] = [20, 24, 20]
    face[3:5, 5:7] = [20, 24, 20]
    face[3, 2] = face[3, 5] = [170, 40, 30]          # glowing red pupils
    face[6, 2:6] = [50, 40, 36]
    add("zombie_face", face)
    hs = _noise_tile("zombie_head_side", skin, 10, (8, 8))
    hs[0:2, :] = hair
    add("zombie_head_side", hs)
    add("zombie_head_side_m", hs[:, ::-1].copy())
    add("zombie_head_back", _noise_tile("zombie_head_back", hair, 10, (8, 8)))
    add("zombie_hair_top", _noise_tile("zombie_hair_top", hair, 10, (8, 8)))
    add("zombie_chin", _noise_tile("zombie_chin", skin, 8, (8, 8)))
    bf = _noise_tile("zombie_body_front", shirt, 14, (8, 12))
    r = _rng("zombie_rips")
    for _ in range(9):
        y, x = r.integers(1, 10), r.integers(0, 8)
        bf[y, x] = skin
    bf[10:12, :] = pants
    add("zombie_body_front", bf)
    bb = _noise_tile("zombie_body_back", shirt, 14, (8, 12))
    bb[10:12, :] = pants
    add("zombie_body_back", bb)
    bs = _noise_tile("zombie_body_side", shirt, 14, (4, 12))
    bs[10:12, :] = pants
    add("zombie_body_side", bs)
    add("zombie_body_top", _noise_tile("zombie_body_top", shirt, 12, (8, 4)))
    arm = _noise_tile("zombie_arm", skin, 10, (4, 12))
    arm[0:4, :] = shirt
    add("zombie_arm", arm)
    add("zombie_arm_top", _noise_tile("zombie_arm_top", shirt, 10, (4, 4)))
    add("zombie_hand", _noise_tile("zombie_hand", skin, 8, (4, 4)))
    leg = _noise_tile("zombie_leg", pants, 10, (4, 12))
    leg[10:12, :] = [40, 40, 46]
    add("zombie_leg", leg)
    add("zombie_shoe", _noise_tile("zombie_shoe", [40, 40, 46], 6, (4, 4)))


def _make_extras():
    st = _noise_tile("coal_ore", [125, 125, 125], 14)
    r = _rng("coal_specks")
    for _ in range(7):
        y, x = r.integers(1, 14, 2)
        st[y:y + 2, x:x + 2] = [30, 30, 32]
    add("coal_ore", st)
    dd = _noise_tile("diamond_ore", [125, 125, 125], 14)
    r = _rng("diamond_specks")
    for _ in range(6):
        y, x = r.integers(1, 14, 2)
        dd[y:y + 2, x:x + 2] = [90, 230, 225]
        dd[y, x] = [220, 255, 255]
    add("diamond_ore", dd)
    cf = _noise_tile("chest_front", [150, 104, 50], 10)
    cf[0, :] = cf[-1, :] = cf[:, 0] = cf[:, -1] = [80, 54, 24]
    cf[6, :] = [80, 54, 24]
    cf[5:9, 7:9] = [200, 200, 205]
    add("chest_front", cf)
    cs = _noise_tile("chest_side", [150, 104, 50], 10)
    cs[0, :] = cs[-1, :] = cs[:, 0] = cs[:, -1] = [80, 54, 24]
    cs[6, :] = [80, 54, 24]
    add("chest_side", cs)
    ct = _noise_tile("chest_top", [158, 110, 54], 10)
    ct[0, :] = ct[-1, :] = ct[:, 0] = ct[:, -1] = [80, 54, 24]
    add("chest_top", ct)
    fl = _noise_tile("farmland", [96, 62, 38], 10)
    for y in (2, 6, 10, 14):
        fl[y, :] = [66, 42, 24]
    add("farmland", fl)
    cr = np.zeros((T, T, 4), np.uint8)
    r = _rng("crop")
    for x in range(1, 15, 3):
        h = r.integers(8, 15)
        cr[T - h:, x] = [90, 170, 50, 255]
        cr[T - h:T - h + 3, x] = [200, 190, 70, 255]
    add("crop", cr)
    add("torch_stick", _noise_tile("torch_stick", [120, 86, 48], 8, (2, 8)))
    fl2 = np.zeros((4, 4, 3), np.uint8)
    fl2[:] = [255, 200, 80]
    fl2[0:2, 1:3] = [255, 245, 190]
    add("torch_flame", fl2)
    add("pick_handle", _noise_tile("pick_handle", [120, 86, 48], 8, (2, 10)))
    add("pick_head", _noise_tile("pick_head", [150, 112, 66], 8, (10, 2)))
    add("diamond", _noise_tile("diamond_gem", [100, 235, 230], 18, (4, 4)))
    add("bridge", _noise_tile("bridge", [150, 112, 66], 8))
    _make_day2()


def _make_day2():
    """Underground / tool tiles (iron, furnace, lava, deepslate, gravel, tool materials)."""
    io = _noise_tile("iron_ore", [125, 125, 125], 14)
    r = _rng("iron_specks")
    for _ in range(6):
        y, x = r.integers(1, 14, 2)
        io[y:y + 2, x:x + 2] = [216, 160, 112]
        io[y, x] = [236, 200, 160]
    add("iron_ore", io)
    ff = _noise_tile("furnace_front", [110, 110, 110], 12)
    ff[0, :] = ff[-1, :] = ff[:, 0] = ff[:, -1] = [70, 70, 70]
    ff[9:14, 4:12] = [30, 30, 30]
    ff[11:14, 5:11] = [255, 140, 40]
    ff[12:14, 6:10] = [255, 220, 110]
    add("furnace_front", ff)
    fs = _noise_tile("furnace_side", [118, 118, 118], 12)
    fs[0, :] = fs[-1, :] = fs[:, 0] = fs[:, -1] = [70, 70, 70]
    add("furnace_side", fs)
    lv = _noise_tile("lava", [205, 78, 14], 26)
    r = _rng("lava_spots")
    for _ in range(10):
        y, x = r.integers(0, 15, 2)
        lv[y:y + 2, x:x + 2] = [250, 160, 40]
    add("lava", lv)
    ds = _noise_tile("deepslate", [72, 72, 78], 10)
    for y in range(0, T, 4):
        ds[y, :] = (ds[y, :] * 0.8).astype(np.uint8)
    add("deepslate", ds)
    dd = _noise_tile("deep_diamond", [72, 72, 78], 10)
    r = _rng("deep_diamond_specks")
    for _ in range(7):
        y, x = r.integers(1, 14, 2)
        dd[y:y + 2, x:x + 2] = [90, 230, 225]
        dd[y, x] = [220, 255, 255]
    add("deep_diamond", dd)
    gv = _noise_tile("gravel", [130, 124, 120], 34)
    add("gravel", gv)
    add("pick_head_stone", _noise_tile("pick_head_stone", [128, 128, 128], 10, (10, 2)))
    add("pick_head_iron", _noise_tile("pick_head_iron", [216, 216, 222], 8, (10, 2)))
    add("pick_head_diamond", _noise_tile("pick_head_diamond", [80, 225, 220], 12, (10, 2)))
    add("blade_diamond", _noise_tile("blade_diamond", [110, 235, 230], 12, (2, 10)))
    add("ingot", _noise_tile("ingot", [222, 222, 228], 10, (4, 4)))
    ct = _noise_tile("crafting_top", [160, 118, 70], 10)
    ct[0, :] = ct[-1, :] = ct[:, 0] = ct[:, -1] = [96, 66, 36]
    ct[5, 2:14] = ct[10, 2:14] = [96, 66, 36]
    ct[2:14, 5] = ct[2:14, 10] = [96, 66, 36]
    add("crafting_top", ct)
    cs = _noise_tile("crafting_side", [150, 108, 62], 10)
    cs[0, :] = cs[-1, :] = [96, 66, 36]
    cs[4:10, 3:6] = [130, 130, 136]
    cs[4:10, 10:13] = [110, 80, 46]
    add("crafting_side", cs)


def build_atlas():
    _make_blocks()
    _make_pip()
    _make_creeper()
    _make_zombie()
    _make_extras()
    from . import tex100
    tex100.make(add, _noise_tile, _rng, T)
    tex100.make_armor(add, _tiles, _noise_tile, T)
    tex100.make_more(add, _noise_tile, _rng, T)
    atlas = np.zeros((ATLAS_PX, ATLAS_PX, 4), np.uint8)
    for i, (name, img) in enumerate(_tiles.items()):
        gx, gy = i % GRID, i // GRID
        atlas[gy * T:(gy + 1) * T, gx * T:(gx + 1) * T] = img
        TILE_INDEX[name] = i
    return atlas


_ATLAS = None


def get_atlas():
    """Build the atlas once and reuse it."""
    global _ATLAS
    if _ATLAS is None:
        _ATLAS = build_atlas()
    return _ATLAS


def tile_image(name, px=16):
    from PIL import Image
    a = get_atlas()
    i = TILE_INDEX[name]
    gx, gy = i % GRID, i // GRID
    return Image.fromarray(a[gy * T:(gy + 1) * T, gx * T:(gx + 1) * T]).resize((px, px), Image.NEAREST)


def tile_uv(name):
    get_atlas()
    i = TILE_INDEX[name]
    gx, gy = i % GRID, i // GRID
    # half-texel inset avoids bleeding between neighbouring tiles
    e = 0.02 / GRID
    u0, v0 = gx / GRID + e, gy / GRID + e
    u1, v1 = (gx + 1) / GRID - e, (gy + 1) / GRID - e
    return u0, v0, u1, v1


# Block registry: name -> (top, side, bottom, flags)
BLOCKS = {
    "grass": ("grass_top", "grass_side", "dirt", {}),
    "dirt": ("dirt", "dirt", "dirt", {}),
    "stone": ("stone", "stone", "stone", {}),
    "cobble": ("cobble", "cobble", "cobble", {}),
    "log": ("log_top", "log_side", "log_top", {}),
    "planks": ("planks", "planks", "planks", {}),
    "leaves": ("leaves", "leaves", "leaves", {"cutout": True}),
    "sand": ("sand", "sand", "sand", {}),
    "water": ("water", "water", "water", {"transparent": True}),
    "water_flow": ("water_flow", "water_flow", "water_flow", {"transparent": True}),
    "glass": ("glass", "glass", "glass", {"cutout": True}),
    "tnt": ("tnt_top", "tnt_side", "tnt_top", {}),
    "cloud": ("cloud", "cloud", "cloud", {"transparent": True}),
    "obsidian": ("obsidian", "obsidian", "obsidian", {}),
    "lucky": ("lucky", "lucky", "lucky", {"emissive": 0.35}),
    "coal_ore": ("coal_ore", "coal_ore", "coal_ore", {}),
    "diamond_ore": ("diamond_ore", "diamond_ore", "diamond_ore", {"emissive": 0.15}),
    "farmland": ("farmland", "dirt", "dirt", {}),
    "iron_ore": ("iron_ore", "iron_ore", "iron_ore", {}),
    "furnace": ("furnace_side", "furnace_side", "furnace_side", {}),
    "lava": ("lava", "lava", "lava", {"emissive": 0.7}),
    "deepslate": ("deepslate", "deepslate", "deepslate", {}),
    "deep_diamond": ("deep_diamond", "deep_diamond", "deep_diamond", {"emissive": 0.25}),
    "gravel": ("gravel", "gravel", "gravel", {}),
    "crafting": ("crafting_top", "crafting_side", "planks", {}),
    # 100 Days
    "wool": ("wool", "wool", "wool", {}),
    "stone_bricks": ("stone_bricks", "stone_bricks", "stone_bricks", {}),
    "glowstone": ("glowstone", "glowstone", "glowstone", {"emissive": 0.8}),
    "netherrack": ("netherrack", "netherrack", "netherrack", {}),
    "portal": ("portal", "portal", "portal", {"transparent": True, "emissive": 0.85}),
    "path": ("path_top", "dirt", "dirt", {}),
    "hay": ("hay_top", "hay_side", "hay_top", {}),
    "enchant": ("enchant_top", "enchant_side", "obsidian", {}),
    "leaves_dark": ("leaves_dark", "leaves_dark", "leaves_dark", {"cutout": True}),
    "furnace_lit": ("furnace_side", "furnace_front", "furnace_side", {}),
    "void": ("black", "black", "black", {}),
}


def block_faces(name):
    """Return tile names in face order: +x, -x, +y, -y, +z, -z."""
    top, side, bottom, _ = BLOCKS[name]
    return [side, side, top, bottom, side, side]


def is_transparent(name):
    f = BLOCKS[name][3]
    return f.get("transparent", False) or f.get("cutout", False)


def is_translucent(name):
    return BLOCKS[name][3].get("transparent", False)
