"""Sets for '100 Days'.

OVERWORLD terrain (static, cached): plains, forest (west), river (east, flowing south), big hill (north)
with a cave mouth, a sheep meadow, a village down river, the spawn point with its dirt hut.
PAD: the square around the base (x PAD_X, z PAD_Z) is cut out of the terrain mesh and meshed per STAGE
(house -> farm -> crater -> stone base -> portal -> castle -> breach -> garden).
UNDER: the cave / lava chamber set from Day 2 (reused geometry, world offset U).
NETHER: netherrack cliffs over a lava ocean at offset NO.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mcengine.geometry import VoxelWorld, box_vertices, translate          # noqa: E402
from mcengine import models100 as M                                        # noqa: E402
from mcengine.models import torch_vertices, chest_vertices                 # noqa: E402

PI_ = np.pi
GY = 10                       # grass top block y; walk surface is GY + 1
X0, X1, Z0, Z1 = -84, 104, -96, 184
PAD_X, PAD_Z = (-14, 31), (-14, 36)
SPAWN = np.array([-21.5, GY + 1, 4.5])
HUT = (-24, -19, 2, 7)        # dirt hut x0, x1(excl), z0, z1(excl)
HILL_C, HILL_R, HILL_H = (16.0, -50.0), 21.0, 15.0
CAVE_MOUTH = (12, -31)        # (x, z) of the tunnel mouth on the hill's south face
LAVA_POOL = (0, 4, -27, -24)  # x0, x1, z0, z1 (excl), surface pool at the foot of the hill
MEADOW = (-62.0, -22.0)
WOLF_SPOT = (-52.0, 26.0)
VILLAGE_C = (64.0, 140.0)
NO = np.array([-700.0, 0.0, 0.0])          # nether offset

# base layout (pad)
HOUSE = (6, 15, 8, 15)        # x0, x1, z0, z1 (excl): 9 x 7
DOOR = (10, 8)                # door column (x, z) on the south wall
FARM = (18, 27, 3, 17)
CANAL_X = 22
PEN = (5, 15, 18, 26)
WALL = (-3, 29, -1, 30)       # stone wall ring (ch 8)
GATE_X = (10, 12)
PORTAL = (-1, 13)             # portal frame min corner (x, z); frame spans x..x+3, y GY+1..GY+5
CASTLE = (-3, 27, -1, 29)     # castle walls ring
KEEP = (7, 17, 9, 19)


def river_x(z):
    return 42 + 4 * np.sin(z / 30.0) + 2 * np.sin(z / 13.0 + 1.0)


def _noise2(x, z, seed=0):
    return (np.sin(x * 0.11 + seed) * np.cos(z * 0.09 - seed * 0.7) + 0.5 * np.sin(x * 0.23 + z * 0.17 + seed * 2)
            + 0.3 * np.cos(x * 0.05 - z * 0.07 + seed))


def in_pad(x, z):
    return PAD_X[0] <= x <= PAD_X[1] and PAD_Z[0] <= z <= PAD_Z[1]


# ------------------------------------------------------------------ heightmap
class Terrain:
    def __init__(self):
        xs = np.arange(X0, X1)
        zs = np.arange(Z0, Z1)
        self.xs, self.zs = xs, zs
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        h = GY + np.round(1.1 * _noise2(X, Z, 1.3)).astype(int)
        # flatten around the base, spawn and village
        def flat(cx, cz, r, k=0.0):
            d = np.hypot(X - cx, Z - cz)
            return np.clip((d - r) / 10.0, 0, 1)
        f = np.minimum.reduce([flat(9, 11, 30), flat(SPAWN[0], SPAWN[2], 9), flat(VILLAGE_C[0], VILLAGE_C[1], 24)])
        f = np.where((X >= PAD_X[0] - 2) & (X <= PAD_X[1] + 2) & (Z >= PAD_Z[0] - 2) & (Z <= PAD_Z[1] + 2), 0, f)
        h = np.round(GY + (h - GY) * f).astype(int)
        # hill
        d = np.hypot((X - HILL_C[0]) / 1.0, (Z - HILL_C[1]) / 0.85)
        hill = HILL_H * np.clip(1 - (d / HILL_R) ** 2, 0, 1) ** 0.8 + 1.5 * _noise2(X, Z, 4.1) * (d < HILL_R)
        h = np.maximum(h, np.round(GY + hill).astype(int))
        # river
        rx = river_x(Z)
        dr = np.abs(X - rx)
        self.water = dr < 3.6
        self.sand = (dr < 5.2) & ~self.water
        h = np.where(dr < 6.5, np.minimum(h, GY), h)
        h = np.where(self.water, GY - 3, h)
        self.h = h
        self.stone_top = (h >= GY + 6) & (_noise2(X * 1.7, Z * 1.7, 7) > -0.3)

    def top(self, x, z):
        i, j = int(np.floor(x)) - X0, int(np.floor(z)) - Z0
        if 0 <= i < self.h.shape[0] and 0 <= j < self.h.shape[1]:
            hh = self.h[i, j]
            if self.water[i, j]:
                return GY - 1          # water surface
            return hh
        return GY

    def ground(self, x, z):
        """Walk surface height (feet y) at x, z."""
        return float(self.top(x, z) + 1)


def tree(w, x, y, z, height=5, dark=False, rng=None):
    leaves = "leaves_dark" if dark else "leaves"
    for k in range(height):
        w.set((x, y + k, z), "log")
    top = y + height
    for dy in (-2, -1, 0, 1):
        r = 2 if dy < 0 else 1
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if abs(dx) == r and abs(dz) == r and (dy >= 0 or (rng is not None and rng.random() < 0.5)):
                    continue
                p = (x + dx, top + dy, z + dz)
                if w.get(p) is None:
                    w.set(p, leaves)


def tree_spots():
    rng = np.random.default_rng(11)
    spots = []
    for _ in range(2600):
        x = rng.uniform(X0 + 3, X1 - 3)
        z = rng.uniform(Z0 + 3, Z1 - 3)
        forest = x < -16 or (x > 60 and z < 80) or z < -80
        if not forest and rng.random() > 0.05:
            continue
        if forest and rng.random() > 0.55:
            continue
        if abs(x - river_x(z)) < 8:
            continue
        if PAD_X[0] - 4 <= x <= PAD_X[1] + 3 and PAD_Z[0] - 4 <= z <= PAD_Z[1] + 3:
            continue
        if np.hypot(x - SPAWN[0], z - SPAWN[2]) < 12.5 or np.hypot(x - VILLAGE_C[0], z - VILLAGE_C[1]) < 30:
            continue
        if np.hypot(x - MEADOW[0], z - MEADOW[1]) < 11 or np.hypot(x - WOLF_SPOT[0], z - WOLF_SPOT[1]) < 6:
            continue
        if np.hypot(x - HILL_C[0], (z - HILL_C[1]) / 0.85) < HILL_R + 2:
            continue
        if np.hypot(x - LAVA_POOL[0], z - LAVA_POOL[2]) < 7:
            continue
        # keep the spawn-side forest edge readable: one tree right next to spawn for punching
        spots.append((int(x), int(z), int(rng.integers(4, 7)), bool(rng.random() < 0.35)))
    return spots


FIRST_TREE = (-25, 9)          # its trunk is drawn by the shots (it gets punched)
ROCK = [(-15, 11, -4), (-15, 11, -3), (-16, 11, -4), (-15, 12, -4), (-16, 11, -3), (-16, 12, -4), (-17, 11, -4)]


# ------------------------------------------------------------------ village
VILLAGE_HOUSES = [  # x0, z0, w, d, door side
    (52, 124, 7, 6, "w"), (62, 120, 6, 6, "s"), (72, 126, 7, 7, "w"), (55, 148, 6, 7, "w"), (70, 146, 8, 6, "w"),
]
VILLAGE_FARM = (60, 134, 9, 7)


def build_house(w, x0, z0, wd, dp, y, door="s", wall="planks", corner="log", roof="planks", windows=True):
    x1, z1 = x0 + wd - 1, z0 + dp - 1
    hgt = 4
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            w.set((x, y - 1, z), "cobble" if wall != "planks" else "planks")
            edge = x in (x0, x1) or z in (z0, z1)
            if not edge:
                continue
            corner_col = x in (x0, x1) and z in (z0, z1)
            for k in range(hgt):
                b = corner if corner_col else wall
                if windows and not corner_col and k == 1 and ((x - x0) % 3 == 2 or (z - z0) % 3 == 2):
                    b = "glass"
                w.set((x, y + k, z), b)
    # door gap
    dx, dz = {"s": ((x0 + x1) // 2, z0), "n": ((x0 + x1) // 2, z1), "w": (x0, (z0 + z1) // 2), "e": (x1, (z0 + z1) // 2)}[door]
    w.set((dx, y, dz), None)
    w.set((dx, y + 1, dz), None)
    # stepped gable roof along x
    mid = (z0 + z1) / 2.0
    for k in range(0, (dp + 3) // 2 + 1):
        for x in range(x0 - 1, x1 + 2):
            for zz in (z0 - 1 + k, z1 + 1 - k):
                if z0 - 1 <= zz <= z1 + 1:
                    w.set((x, y + hgt + k, zz), roof)
        if z0 - 1 + k >= mid:
            break
    # gable ends
    for k in range(1, dp // 2 + 1):
        for zz in range(z0 + k, z1 - k + 1):
            for x in (x0, x1):
                w.set((x, y + hgt - 1 + k, zz), wall)
    return (dx, dz)


def build_village(w, terrain):
    y = GY + 1
    doors = []
    for (x0, z0, wd, dp, side) in VILLAGE_HOUSES:
        roof = "log" if (x0 % 2) else "planks"
        wall = "planks" if (z0 % 3) else "cobble"
        doors.append(build_house(w, x0, z0, wd, dp, y, door=side, wall=wall, corner="log", roof=roof))
    # paths (crossroads to the river)
    for x in range(int(river_x(140)) + 4, 80):
        for z in (139, 140):
            w.set((x, GY, z), "path")
    for z in range(118, 160):
        for x in (65, 66):
            w.set((x, GY, z), "path")
    # farm
    fx, fz, fw, fd = VILLAGE_FARM
    for x in range(fx, fx + fw):
        for z in range(fz, fz + fd):
            edge = x in (fx, fx + fw - 1) or z in (fz, fz + fd - 1)
            w.set((x, GY, z), "log" if edge else ("water" if x == fx + fw // 2 else "farmland"))
    # well
    for x in range(68, 72):
        for z in range(130, 134):
            w.set((x, GY, z), "cobble")
            if x in (68, 71) or z in (130, 133):
                w.set((x, GY + 1, z), "cobble")
            else:
                w.set((x, GY, z), "water")
    for (x, z) in ((68, 130), (71, 130), (68, 133), (71, 133)):
        w.set((x, GY + 2, z), "log")
        w.set((x, GY + 3, z), "log")
    for x in range(68, 72):
        for z in range(130, 134):
            w.set((x, GY + 4, z), "cobble")
    # hay bales
    for p in ((60, GY + 1, 131), (61, GY + 1, 131), (60, GY + 2, 131)):
        w.set(p, "hay")
    return doors


def village_crops():
    fx, fz, fw, fd = VILLAGE_FARM
    return [(x, GY + 1, z) for x in range(fx + 1, fx + fw - 1) for z in range(fz + 1, fz + fd - 1) if x != fx + fw // 2]


# ------------------------------------------------------------------ overworld terrain mesh
def build_terrain_world(terrain):
    w = VoxelWorld()
    h = terrain.h
    nx, nz = h.shape
    for i in range(nx):
        x = X0 + i
        for j in range(nz):
            z = Z0 + j
            if in_pad(x, z):
                continue
            hh = h[i, j]
            nb = [h[max(i - 1, 0), j], h[min(i + 1, nx - 1), j], h[i, max(j - 1, 0)], h[i, min(j + 1, nz - 1)]]
            lo = min(min(nb), hh) - 1
            if terrain.water[i, j]:
                w.set((x, GY - 1, z), "water")
                w.set((x, GY - 2, z), "water")
                w.set((x, hh, z), "sand")
                w.set((x, hh - 1, z), "dirt")
                continue
            for y in range(max(lo, hh - 3), hh + 1):
                if y == hh:
                    b = "sand" if terrain.sand[i, j] else ("stone" if terrain.stone_top[i, j] else "grass")
                else:
                    b = "stone" if (hh - y > 2 or hh >= GY + 6) else "dirt"
                w.set((x, y, z), b)
    rng = np.random.default_rng(5)
    for (x, z, ht, dark) in tree_spots():
        y = terrain.top(x, z) + 1
        tree(w, x, y, z, ht, dark, rng)
    # leaves of the first tree (trunk is a prop)
    fx, fz = FIRST_TREE
    w2 = VoxelWorld()
    tree(w2, fx, GY + 1, fz, 5, False, rng)
    for p, b in w2.blocks.items():
        if b != "log":
            w.set(p, b)
    # cave mouth in the hill: a 3x3 tunnel going north into darkness
    cx, cz = CAVE_MOUTH
    for z in range(cz - 12, cz + 3):
        for x in range(cx - 1, cx + 2):
            for y in range(GY + 1, GY + 4):
                w.set((x, y, z), None)
        for x in range(cx - 1, cx + 2):
            w.set((x, GY, z), "stone")
    for x in range(cx - 1, cx + 2):
        for y in range(GY + 1, GY + 4):
            w.set((x, y, cz - 12), "void")
    # dirt hut is a stage overlay, not part of the terrain
    build_village(w, terrain)
    return w


def far_ring():
    """Low distant hills around the map so the horizon is never empty."""
    w = VoxelWorld()
    for x in range(X0 - 60, X1 + 60, 2):
        for z in range(Z0 - 60, Z1 + 60, 2):
            if X0 <= x < X1 and Z0 <= z < Z1:
                continue
            hh = GY + int(round(2 + 3 * _noise2(x * 0.6, z * 0.6, 9) + 4 * np.clip(_noise2(x * 0.25, z * 0.25, 2), 0, 2)))
            for dx in (0, 1):
                for dz in (0, 1):
                    w.set((x + dx, hh, z + dz), "grass")
                    w.set((x + dx, hh - 1, z + dz), "dirt")
    return w


# ------------------------------------------------------------------ pad stages
def pad_ground(w):
    for x in range(PAD_X[0], PAD_X[1] + 1):
        for z in range(PAD_Z[0], PAD_Z[1] + 1):
            w.set((x, GY, z), "grass")
            w.set((x, GY - 1, z), "dirt")
            w.set((x, GY - 2, z), "dirt")


def wood_house(w):
    x0, x1, z0, z1 = HOUSE
    build_house(w, x0, z0, x1 - x0, z1 - z0, GY + 1, door="s", wall="planks", corner="log", roof="log")


def farm(w, castle=False):
    x0, x1, z0, z1 = FARM
    for x in range(x0, x1):
        for z in range(z0, z1):
            w.set((x, GY, z), "water" if x == CANAL_X else "farmland")
            w.set((x, GY - 1, z), "dirt")


def farm_crops():
    x0, x1, z0, z1 = FARM
    return [(x, GY + 1, z) for x in range(x0, x1) for z in range(z0, z1) if x != CANAL_X]


def pen_posts():
    x0, x1, z0, z1 = PEN
    posts = []
    for x in range(x0, x1):
        for z in range(z0, z1):
            if x in (x0, x1 - 1) or z in (z0, z1 - 1):
                if z == z0 and x in (9, 10):
                    continue                 # gate gap facing the house
                posts.append((x, z))
    return posts


def crater(w):
    """The creeper blast: a sphere removed from the south-east corner of the wooden house."""
    c = np.array([13.5, GY + 1.5, 9.0])
    for x in range(int(c[0]) - 5, int(c[0]) + 6):
        for y in range(GY - 3, GY + 10):
            for z in range(int(c[2]) - 5, int(c[2]) + 6):
                d = np.linalg.norm((np.array([x, y, z]) + 0.5 - c) * np.array([1, 1.15, 1]))
                if d < 4.2 + 0.6 * np.sin(x * 1.7 + z * 2.3 + y):
                    w.set((x, y, z), None)
    # scorched rim
    for x in range(int(c[0]) - 6, int(c[0]) + 7):
        for z in range(int(c[2]) - 6, int(c[2]) + 7):
            for y in range(GY - 3, GY + 1):
                if w.get((x, y, z)) in ("grass",) and w.get((x, y + 1, z)) is None and np.hypot(x - c[0], z - c[2]) < 6:
                    w.set((x, y, z), "dirt")


def stone_house(w):
    x0, x1, z0, z1 = HOUSE
    y = GY + 1
    for x in range(x0, x1):
        for z in range(z0, z1):
            w.set((x, y - 1, z), "stone")
            edge = x in (x0, x1 - 1) or z in (z0, z1 - 1)
            if not edge:
                continue
            corner = x in (x0, x1 - 1) and z in (z0, z1 - 1)
            for k in range(5):
                b = "log" if corner else "cobble"
                if not corner and k in (1, 2) and ((x - x0) % 3 != 0 if z in (z0, z1 - 1) else (z - z0) % 3 != 0):
                    b = "glass"
                if k == 4:
                    b = "log" if not corner else "log"
                w.set((x, y + k, z), b)
    # flat roof with a rim
    for x in range(x0 - 1, x1 + 1):
        for z in range(z0 - 1, z1 + 1):
            w.set((x, y + 5, z), "stone_bricks" if x in (x0 - 1, x1) or z in (z0 - 1, z1) else "cobble")
    for x in range(x0 - 1, x1 + 1):
        for z in range(z0 - 1, z1 + 1):
            if (x in (x0 - 1, x1) or z in (z0 - 1, z1)) and (x + z) % 2 == 0:
                w.set((x, y + 6, z), "cobble")
    dx, dz = DOOR
    w.set((dx, y, dz), None)
    w.set((dx, y + 1, dz), None)
    w.set((dx, y + 2, dz), "cobble")


def stone_wall(w):
    x0, x1, z0, z1 = WALL
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if x in (x0, x1) or z in (z0, z1):
                if z == z0 and GATE_X[0] <= x < GATE_X[1]:
                    continue
                for k in range(2):
                    w.set((x, GY + 1 + k, z), "cobble")
    # path from gate to door
    for z in range(z0 - 3, HOUSE[2]):
        for x in (10, 11):
            w.set((x, GY, z), "path")


def portal_frame(w):
    px, pz = PORTAL
    for (a, b) in [(1, 0), (2, 0), (0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (3, 3), (1, 4), (2, 4)]:
        w.set((px + a, GY + 1 + b, pz), "obsidian")


def portal_inner():
    px, pz = PORTAL
    return [(px + a, GY + 1 + b, pz) for a in (1, 2) for b in (1, 2, 3)]


def castle(w, breach=False):
    x0, x1, z0, z1 = CASTLE
    H = 6
    # moat (two wide) outside the walls
    for x in range(x0 - 3, x1 + 4):
        for z in range(z0 - 3, z1 + 4):
            ring = not (x0 - 1 <= x <= x1 + 1 and z0 - 1 <= z <= z1 + 1)
            if ring:
                w.set((x, GY, z), "water")
                w.set((x, GY - 1, z), "water")
                w.set((x, GY - 2, z), "dirt")
    # bridge over the south moat
    for z in range(z0 - 3, z0):
        for x in (11, 12, 13):
            w.set((x, GY, z), "planks")
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            if not (x in (x0, x1) or z in (z0, z1)):
                continue
            gate = z == z0 and 11 <= x <= 13
            for k in range(H):
                if gate and k < 3:
                    continue
                w.set((x, GY + 1 + k, z), "stone_bricks")
            if (x + z) % 2 == 0:
                w.set((x, GY + 1 + H, z), "stone_bricks")
    # walkway on top (inside lip)
    # towers
    for (cx, cz) in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
        for x in range(cx - 2, cx + 3):
            for z in range(cz - 2, cz + 3):
                ring = x in (cx - 2, cx + 2) or z in (cz - 2, cz + 2)
                for k in range(H + 5):
                    if ring or k in (H, ):
                        b = "stone_bricks"
                        if ring and k in (H + 2,) and (x == cx or z == cz):
                            b = "glass"
                        w.set((x, GY + 1 + k, z), b)
                if ring and (x + z) % 2 == 0:
                    w.set((x, GY + 1 + H + 5, z), "stone_bricks")
            w.set((cx, GY + 1 + H, cz), "stone_bricks")
    # keep
    kx0, kx1, kz0, kz1 = KEEP
    for x in range(kx0, kx1):
        for z in range(kz0, kz1):
            edge = x in (kx0, kx1 - 1) or z in (kz0, kz1 - 1)
            for k in range(12):
                if edge:
                    b = "stone_bricks"
                    if k in (2, 6, 10) and (x - kx0) % 3 == 1 and z in (kz0, kz1 - 1):
                        b = "glass"
                    if k in (2, 6, 10) and (z - kz0) % 3 == 1 and x in (kx0, kx1 - 1):
                        b = "glass"
                    w.set((x, GY + 1 + k, z), b)
                elif k in (3, 7, 11):
                    w.set((x, GY + 1 + k, z), "planks")
            if edge and (x + z) % 2 == 0:
                w.set((x, GY + 13, z), "stone_bricks")
            w.set((x, GY, z), "stone_bricks")
    w.set((12, GY + 1, kz0), None)
    w.set((12, GY + 2, kz0), None)
    # courtyard path
    for z in range(z0, kz0):
        for x in (11, 12, 13):
            w.set((x, GY, z), "path")
    if breach:
        c = np.array([x1 + 0.5, GY + 2.5, 15.5])
        for x in range(x1 - 4, x1 + 5):
            for y in range(GY - 1, GY + 9):
                for z in range(10, 22):
                    d = np.linalg.norm((np.array([x, y, z]) + 0.5 - c) * np.array([1, 1.1, 1]))
                    if d < 3.6 + 0.5 * np.sin(x * 2.1 + z * 1.3 + y):
                        if w.get((x, y, z)) in ("stone_bricks", "grass", "dirt", "glass"):
                            w.set((x, y, z), None)


def castle_farm(w):
    # farm moved inside the walls (east courtyard)
    for x in range(19, 25):
        for z in range(4, 26):
            w.set((x, GY, z), "water" if x == 22 else "farmland")


def castle_crops():
    return [(x, GY + 1, z) for x in range(19, 25) for z in range(4, 26) if x != 22]


def castle_pen_posts():
    posts = []
    for x in range(0, 6):
        for z in range(20, 27):
            if x in (0, 5) or z in (20, 26):
                if z == 20 and x in (2, 3):
                    continue
                posts.append((x, z))
    return posts


GARDEN_TREES = [(1, 4), (4, 7), (1, 10), (3, 15)]


def stage_world(name):
    """Pad voxel world for a stage."""
    w = VoxelWorld()
    pad_ground(w)
    if name == "P0":
        return w
    if name in ("P1", "P1b"):
        wood_house(w)
        if name == "P1b":
            farm(w)
        return w
    if name == "P2":
        wood_house(w)
        farm(w)
        return w
    if name == "P3":
        wood_house(w)
        farm(w)
        crater(w)
        return w
    if name == "P3c":                      # crater cleared, ready to rebuild
        farm(w)
        return w
    if name in ("P4", "P5"):
        stone_house(w)
        farm(w)
        stone_wall(w)
        if name == "P5":
            portal_frame(w)
            for p in portal_inner():
                w.set(p, "portal")
        return w
    if name in ("P6", "P7", "P8"):
        castle(w, breach=(name == "P7"))
        castle_farm(w)
        portal_frame(w)
        for p in portal_inner():
            w.set(p, "portal")
        if name == "P8":
            rng = np.random.default_rng(3)
            for (x, z) in GARDEN_TREES:
                tree(w, x, GY + 1, z, 5, False, rng)
        return w
    if name == "P6pre":                     # stone base cleared for the castle, moat trench dug
        farm(w)
        portal_frame(w)
        for p in portal_inner():
            w.set(p, "portal")
        x0, x1, z0, z1 = CASTLE
        for x in range(x0 - 3, x1 + 4):
            for z in range(z0 - 3, z1 + 4):
                if not (x0 - 1 <= x <= x1 + 1 and z0 - 1 <= z <= z1 + 1):
                    w.set((x, GY, z), None)
                    w.set((x, GY - 1, z), None)
        return w
    raise KeyError(name)


def stage_decor(name, crops="gold"):
    """Non-voxel decor for a stage: (opaque verts, torch bases)."""
    v, torches = [], []
    tile = {"young": "crop_young", "mid": "crop", "gold": "crop_gold"}[crops]
    if name not in ("P0", "P3c", "P6pre"):
        bx, bz = (7, 11) if name not in ("P6", "P7", "P8") else (8, 11)
        v.append(M.bed_vertices((bx, GY + 1, bz), 0.0))
    if name in ("P1", "P1b", "P2", "P4", "P5"):
        v.append(chest_vertices((13.5, GY + 1.5, 13.5), PI_))
    if name in ("P1", "P1b", "P2", "P3"):
        if name != "P3":
            v.append(M.door_vertices((DOOR[0], GY + 1, DOOR[1]), 0.0))
    if name in ("P2", "P3", "P3c", "P4", "P5", "P1b"):
        for c in farm_crops():
            v.append(M.crop_vertices(c, tile, 0.85 if crops == "gold" else 0.6))
    if name in ("P2", "P3", "P3c", "P4", "P5"):
        posts = set(pen_posts())
        for (x, z) in posts:
            v.append(M.fence_post((x, GY + 1, z), (x + 1, z) in posts, (x, z + 1) in posts))
    if name in ("P4", "P5"):
        v.append(M.door_vertices((DOOR[0], GY + 1, DOOR[1]), 0.0))
        x0, x1, z0, z1 = WALL
        for x in range(x0, x1 + 1, 4):
            torches += [(x + 0.5, GY + 3, z0 + 0.5), (x + 0.5, GY + 3, z1 + 0.5)]
        for z in range(z0 + 4, z1, 4):
            torches += [(x0 + 0.5, GY + 3, z + 0.5), (x1 + 0.5, GY + 3, z + 0.5)]
        hx0, hx1, hz0, hz1 = HOUSE
        torches += [(DOOR[0] - 0.5, GY + 1, DOOR[1] - 0.5), (DOOR[0] + 1.5, GY + 1, DOOR[1] - 0.5),
                    (FARM[0] - 0.5, GY + 1, FARM[2] - 0.5), (FARM[1] - 0.5, GY + 1, FARM[2] - 0.5),
                    (FARM[0] - 0.5, GY + 1, FARM[3] + 0.5), (FARM[1] - 0.5, GY + 1, FARM[3] + 0.5)]
    if name in ("P6", "P7", "P8"):
        for c in castle_crops():
            v.append(M.crop_vertices(c, tile, 0.85))
        posts = set(castle_pen_posts())
        for (x, z) in posts:
            v.append(M.fence_post((x, GY + 1, z), (x + 1, z) in posts, (x, z + 1) in posts))
        x0, x1, z0, z1 = CASTLE
        for (cx, cz) in ((x0, z0), (x1, z0), (x0, z1), (x1, z1)):
            torches.append((cx + 0.5, GY + 13, cz + 0.5))
        for x in range(x0 + 4, x1, 5):
            torches += [(x + 0.5, GY + 8, z0 + 0.5), (x + 0.5, GY + 8, z1 + 0.5)]
        for z in range(z0 + 5, z1, 5):
            torches += [(x0 + 0.5, GY + 8, z + 0.5)]
            if not (name == "P7" and 10 <= z <= 21):
                torches += [(x1 + 0.5, GY + 8, z + 0.5)]
        torches += [(11 - 0.5 + 0.0, GY + 1, z0 - 0.5), (14.5, GY + 1, z0 - 0.5), (11.5, GY + 1, KEEP[2] - 0.5),
                    (13.5, GY + 1, KEEP[2] - 0.5)]
        for x in (KEEP[0], KEEP[1] - 1):
            for z in (KEEP[2], KEEP[3] - 1):
                torches.append((x + 0.5, GY + 14, z + 0.5))
    cat = np.concatenate(v) if v else np.zeros((0, 12), np.float32)
    return cat, torches


def hut_world(open_=False):
    w = VoxelWorld()
    x0, x1, z0, z1 = HUT
    for x in range(x0, x1):
        for z in range(z0, z1):
            if x in (x0, x1 - 1) or z in (z0, z1 - 1):
                for k in range(3):
                    w.set((x, GY + 1 + k, z), "dirt")
            w.set((x, GY + 4, z), "dirt")
    # tiny window
    w.set((x0 + 2, GY + 2, z0), None)
    if open_:
        w.set((x1 - 1, GY + 1, z0 + 2), None)
        w.set((x1 - 1, GY + 2, z0 + 2), None)
    return w


def hut_blocks_ordered():
    w = hut_world()
    bl = sorted(w.blocks.items(), key=lambda kv: (kv[0][1], np.arctan2(kv[0][2] - SPAWN[2], kv[0][0] - SPAWN[0])))
    return bl


def lighthouse_world(terrain, glow=False):
    w = VoxelWorld()
    cx, cz = int(HILL_C[0]), int(HILL_C[1])
    y0 = max(terrain.top(cx + dx, cz + dz) for dx in range(-2, 3) for dz in range(-2, 3)) + 1
    Hh = 22
    for k in range(Hh):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                ring = abs(dx) == 2 or abs(dz) == 2
                if not ring or (abs(dx) == 2 and abs(dz) == 2):
                    if not ring:
                        continue
                band = (k // 4) % 2 == 0
                b = "stone_bricks" if band else "cobble"
                if k % 6 == 3 and (dx == 0 or dz == 0):
                    b = "glass"
                w.set((cx + dx, y0 + k, cz + dz), b)
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                if abs(dx) + abs(dz) == 2 and k < Hh - 1:
                    w.set((cx + dx, y0 + k, cz + dz), "stone_bricks")
    for dx in range(-3, 4):
        for dz in range(-3, 4):
            w.set((cx + dx, y0 + Hh, cz + dz), "stone_bricks")
            if (abs(dx) == 3 or abs(dz) == 3) and (dx + dz) % 2 == 0:
                w.set((cx + dx, y0 + Hh + 1, cz + dz), "stone_bricks")
    top = y0 + Hh + 1
    if glow:
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                for k in range(2):
                    w.set((cx + dx, top + k, cz + dz), "glowstone")
        w.set((cx, top + 2, cz), "glowstone")
    return w, (cx + 0.5, top, cz + 0.5), y0


def lava_pool_world(state="lava"):
    """state: 'lava' | 'obsidian' | 'mined'"""
    w = VoxelWorld()
    x0, x1, z0, z1 = LAVA_POOL
    for x in range(x0 - 1, x1 + 1):
        for z in range(z0 - 1, z1 + 1):
            rim = x in (x0 - 1, x1) or z in (z0 - 1, z1)
            if rim:
                w.set((x, GY, z), "stone")
            else:
                w.set((x, GY - 1, z), "stone")
                if state == "lava":
                    w.set((x, GY, z), "lava")
                elif state == "obsidian":
                    w.set((x, GY, z), "obsidian")
    return w


# ------------------------------------------------------------------ nether
def build_nether():
    """Local coordinates; caller adds NO. Lava ocean at y=0 (top surface 1); portal platform at (0, 16, 0)."""
    w = VoxelWorld()
    rng = np.random.default_rng(21)
    R = 70
    for x in range(-R, R):
        for z in range(-R, R):
            # cliff mass: a long ridge with the portal ledge, plus scattered pillars
            ridge = 16 + 3 * _noise2(x * 1.3, z * 1.3, 3) - 0.004 * (z - 0.25 * x) ** 2 * 2
            pill = 0
            for (px, pz, pr, ph) in ((-30, -25, 7, 30), (35, -30, 9, 34), (28, 30, 6, 26), (-35, 28, 8, 24), (5, -48, 10, 38)):
                d = np.hypot(x - px, z - pz)
                if d < pr:
                    pill = max(pill, ph * (1 - (d / pr) ** 3) + 2 * _noise2(x * 2, z * 2, 1))
            hgt = int(max(ridge, pill))
            if hgt >= 3:
                bottom = 1 if hgt < 24 else 1
                for y in range(max(1, hgt - 7), hgt + 1):
                    w.set((x, y, z), "netherrack")
                for y in range(1, max(1, hgt - 7)):
                    if (x + y + z) % 3 == 0 or abs(z - 0.25 * x) > 10:
                        w.set((x, y, z), "netherrack")
            else:
                w.set((x, 0, z), "lava")
    # glowstone clusters on the pillar flanks
    gl = []
    for (px, pz, pr, ph) in ((-30, -25, 7, 30), (35, -30, 9, 34), (28, 30, 6, 26), (5, -48, 10, 38)):
        for k in range(10):
            a = rng.uniform(0, 2 * np.pi)
            y = int(ph * 0.55 + rng.uniform(-3, 6))
            x = int(px + np.cos(a) * (pr * 0.75))
            z = int(pz + np.sin(a) * (pr * 0.75))
            for dx in (0, 1):
                for dy in (0, -1):
                    p = (x + dx, y + dy, z)
                    w.set(p, "glowstone")
                    gl.append(p)
    # portal frame on the ledge
    for (a, b) in [(1, 0), (2, 0), (0, 1), (0, 2), (0, 3), (3, 1), (3, 2), (3, 3), (1, 4), (2, 4)]:
        w.set((a - 2, 17 + b, 0), "obsidian")
    for a in range(-4, 4):
        for z in range(-2, 3):
            w.set((a, 16, z), "netherrack")
            for y in range(17, 23):
                if w.get((a, y, z)) == "netherrack":
                    w.set((a, y, z), None)
    return w, gl


def nether_portal_inner():
    return [(a - 2, 17 + b, 0) for a in (1, 2) for b in (1, 2, 3)]


# ------------------------------------------------------------------ cave (Day 2 set)
def day2_world_module():
    spec = importlib.util.spec_from_file_location("d2world", os.path.join(HERE, "..", "day2", "world.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ cache helpers
def cached_mesh(cache_dir, name, fn, offset=None, alpha_override=None):
    p = os.path.join(cache_dir, name + ".npz")
    if os.path.exists(p):
        d = np.load(p)
        return d["o"], d["t"]
    w = fn()
    o, t = w.mesh(alpha_override=alpha_override)
    if offset is not None:
        o = o.copy()
        t = t.copy()
        o[:, 0:3] += np.asarray(offset, np.float32)
        t[:, 0:3] += np.asarray(offset, np.float32)
    np.savez_compressed(p, o=o, t=t)
    return o, t


def nether_heights(cache_dir):
    """Walk-surface height map of the nether set (local coords, index [x+70, z+70])."""
    p = os.path.join(cache_dir, "nether_h.npy")
    if os.path.exists(p):
        return np.load(p)
    w, _ = build_nether()
    hm = np.full((140, 140), 1.0, np.float32)
    for (x, y, z), b in w.blocks.items():
        if b in ("netherrack", "obsidian") and -70 <= x < 70 and -70 <= z < 70:
            if y + 1 > hm[x + 70, z + 70] and y < 30:
                hm[x + 70, z + 70] = y + 1
    np.save(p, hm)
    return hm
