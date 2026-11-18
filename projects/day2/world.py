"""Sets for Day 2.

SURFACE: the finished base from 'First Night' (moat, tower, second wall, crater farm), rebuilt by
running the Day 1 film's own world code to its final state, plus Day 2 changes (iron ore in the hill).
UNDER: a new underground set at offset U: staircase, giant cavern with a lake, side tunnels,
lava chamber with the diamond ledge.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
from mcengine.geometry import VoxelWorld          # noqa: E402

FN_DIR = os.path.join(HERE, "..", "first_night")

U = np.array([400.0, 40.0, 0.0])          # underground origin (local -> world: + U)
HILL = (27, -18)
GY = 11

# underground layout (local coordinates)
N_STEPS = 32                 # step k: walk surface y=-k, x in [k, k+1), z in [0, 3), 4 tall
FLOOR = -32                  # cavern / chamber walk surface
CAV_C = np.array([60.0, -26.0, 4.0])
CAV_R = np.array([25.0, 10.0, 17.0])
LAKE_C, LAKE_R = (58.0, 11.5), (8.0, 4.0)
TUNNELS = [48, 59, 70]       # side tunnel x positions on the south wall (z<0)
LAVA_X = (92, 106)
LAVA_Z = (-4, 11)
BRIDGE_Z = (2, 5)
LEDGE_X = (106, 116)
ORES = [(116, -32, 3), (116, -32, 4), (116, -31, 3), (116, -31, 4)]
POURS = [(92, 95), (95, 98), (98, 101), (101, 104), (104, 106)]


def ug(x, y, z):
    return np.array([x, y, z], float) + U


def _load_day1():
    """Import first_night/story.py with its own script module."""
    keep = sys.modules.get("script")
    spec = importlib.util.spec_from_file_location("fn_script", os.path.join(FN_DIR, "script.py"))
    fs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fs)
    sys.modules["script"] = fs
    spec2 = importlib.util.spec_from_file_location("fn_story", os.path.join(FN_DIR, "story.py"))
    st = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(st)
    if keep is not None:
        sys.modules["script"] = keep
    else:
        del sys.modules["script"]
    return st


class Surface:
    def __init__(self, cache_dir):
        st = _load_day1()
        self.st = st
        f1 = st.Film(os.path.join(FN_DIR, "work", "cache"))
        self.f1 = f1
        g = f1._ground(4)
        props, _, _ = f1.props_world(f1.duration)
        w = VoxelWorld()
        for p, b in g.blocks.items():
            w.set(p, b)
        for p, b in props.blocks.items():
            w.set(p, b)
        self.base_world = w
        self.crops = [p for (_, p) in f1.crops]
        self.torches = [b for (_, b) in f1.torches]
        # iron ore on the hill's west face near the old tunnel
        hx, hz = HILL
        cands = []
        for (x, y, z), b in w.blocks.items():
            if hx - 9 <= x <= hx - 3 and hz - 6 <= z <= hz + 6 and 11 <= y <= 14 and b in ("stone", "dirt", "grass"):
                if w.get((x - 1, y, z)) is None and not (hz - 1 <= z <= hz + 2):
                    cands.append((x, y, z))
        cands.sort(key=lambda p: (p[0] - (hx - 7)) ** 2 + (p[1] - 12.5) ** 2 * 2 + (p[2] - (hz - 3)) ** 2)
        self.iron = cands[:6]
        self.iron.sort(key=lambda p: (p[2], -p[1]))
        for p in self.iron:
            w.set(p, None)            # ores live in the events world so they can be mined
        self.cache_dir = cache_dir

    def mesh(self):
        p = os.path.join(self.cache_dir, "surface.npz")
        if os.path.exists(p):
            d = np.load(p)
            return d["o"], d["t"]
        o, t = self.base_world.mesh()
        np.savez_compressed(p, o=o, t=t)
        return o, t

    def far(self):
        d = np.load(os.path.join(self.cache_dir, "far.npz"))
        return d["o"], d["t"]


def stair_air(x, y, z):
    """Air of the staircase and its top corridor (local coords)."""
    if not (0 <= z < 3):
        return False
    if -8 <= x < 0:
        if 0 <= y < 4:
            return True
        if -8 <= x < -5 and 0 <= y < 14:      # shaft up to daylight above the corridor
            return True
        return False
    if 0 <= x < N_STEPS + 1:
        k = x
        return -k <= y < -k + 4
    return False


def under_air(x, y, z):
    if stair_air(x, y, z):
        return True
    # porch joining the stairs to the cavern
    if 32 <= x < 40 and FLOOR <= y < FLOOR + 5 and -1 <= z < 5:
        return True
    # cavern
    if y >= FLOOR:
        e = ((x - CAV_C[0]) / CAV_R[0]) ** 2 + ((y - CAV_C[1]) / CAV_R[1]) ** 2 + ((z - CAV_C[2]) / CAV_R[2]) ** 2
        wob = 0.08 * np.sin(x * 0.7 + z * 0.5) + 0.06 * np.cos(y * 0.9 + x * 0.3)
        if e < 1.0 + wob:
            return True
    # side tunnels (south wall)
    for i, tx in enumerate(TUNNELS):
        if tx <= x < tx + 3 and -19 <= z < -6:
            drop = 0 if i != 1 else max(0, (-8 - z)) // 2
            if FLOOR - drop <= y < FLOOR + 4 - drop:
                return True
    # east tunnel to the lava chamber
    if 76 <= x < 89 and FLOOR <= y < FLOOR + 4 and 2 <= z < 5:
        return True
    # lava chamber
    if 88 <= x < 117 and LAVA_Z[0] <= z < LAVA_Z[1]:
        top = FLOOR + 7 + int(2 * np.sin(x * 0.4) * np.cos(z * 0.5))
        if FLOOR <= y < top:
            return True
    return False


def build_under(cache_dir):
    path = os.path.join(cache_dir, "under.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["o"], d["t"]
    w = VoxelWorld()
    air = set()
    for x in range(-10, 120):
        for z in range(-21, 23):
            for y in range(-46, 16):
                if under_air(x, y, z):
                    air.add((x, y, z))
    shell = set()
    for (x, y, z) in air:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    q = (x + dx, y + dy, z + dz)
                    if q not in air:
                        shell.add(q)
    for q in list(shell):                          # second layer for solidity
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            r = (q[0] + d[0], q[1] + d[1], q[2] + d[2])
            if r not in air:
                shell.add(r)
    for (x, y, z) in shell:
        if y >= 12 and -8 <= x < -5 and 0 <= z < 3:
            continue
        deep = y < -17
        rr = (x * 7 + z * 13 + y * 5) % 29
        b = "deepslate" if deep else "stone"
        if rr == 0:
            b = "cobble" if not deep else "stone"
        if rr == 7 and not deep and y > -12:
            b = "gravel"
        w.set((x, y, z), b)
    # lake
    for x in range(int(LAKE_C[0] - LAKE_R[0]) - 1, int(LAKE_C[0] + LAKE_R[0]) + 2):
        for z in range(int(LAKE_C[1] - LAKE_R[1]) - 1, int(LAKE_C[1] + LAKE_R[1]) + 2):
            if ((x - LAKE_C[0]) / LAKE_R[0]) ** 2 + ((z - LAKE_C[1]) / LAKE_R[1]) ** 2 < 1 and (x, FLOOR, z) in air:
                w.set((x, FLOOR - 1, z), "water")
                w.set((x, FLOOR - 2, z), "water")
                w.set((x, FLOOR - 3, z), "stone")
    # lava lake (bridge strip handled by events)
    for x in range(*LAVA_X):
        for z in range(*LAVA_Z):
            if BRIDGE_Z[0] <= z < BRIDGE_Z[1]:
                w.set((x, FLOOR - 1, z), None)
            else:
                w.set((x, FLOOR - 1, z), "lava")
            w.set((x, FLOOR - 2, z), "lava")
            w.set((x, FLOOR - 3, z), "deepslate")
    for p in ORES:
        w.set(p, None)
    o, t = w.mesh()
    # move to world space
    o = o.copy()
    t = t.copy()
    o[:, 0:3] += U.astype(np.float32)
    t[:, 0:3] += U.astype(np.float32)
    np.savez_compressed(path, o=o, t=t)
    return o, t
