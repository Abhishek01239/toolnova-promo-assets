"""Mesh building: textured boxes, voxel worlds with ambient occlusion.

Vertex layout (12 float32):
    pos.xyz  uv.xy  normal.xyz  ao  flash  alpha  emissive
"""
import numpy as np
from . import textures as tx

VERT = 12

# face order +x,-x,+y,-y,+z,-z ; corners TL,TR,BR,BL (seen from outside)
FACE_CORNERS = np.array([
    [(1, 1, 1), (1, 1, 0), (1, 0, 0), (1, 0, 1)],
    [(0, 1, 0), (0, 1, 1), (0, 0, 1), (0, 0, 0)],
    [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
    [(0, 0, 1), (1, 0, 1), (1, 0, 0), (0, 0, 0)],
    [(0, 1, 1), (1, 1, 1), (1, 0, 1), (0, 0, 1)],
    [(1, 1, 0), (0, 1, 0), (0, 0, 0), (1, 0, 0)],
], dtype=np.float32)
FACE_NORMALS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], np.float32)
TRI_ORDER = [0, 3, 2, 0, 2, 1]      # TL,BL,BR, TL,BR,TR


def _face_uv(tile):
    u0, v0, u1, v1 = tx.tile_uv(tile)
    return np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)], np.float32)


def box_vertices(size, tiles, matrix=None, offset=(0, 0, 0), flash=0.0, alpha=1.0,
                 emissive=0.0, centered=True, inflate=0.0):
    """Vertices for a box.

    size: (sx, sy, sz) in world units. tiles: list of 6 tile names (+x,-x,+y,-y,+z,-z)
    or a single name. offset: box min corner (or centre if centered) in local space.
    matrix: optional 4x4 transform applied to the local box.
    """
    if isinstance(tiles, str):
        tiles = [tiles] * 6
    s = np.array(size, np.float32) + 2 * inflate
    o = np.array(offset, np.float32)
    if centered:
        o = o - s / 2
    out = np.empty((36, VERT), np.float32)
    k = 0
    for f in range(6):
        corners = FACE_CORNERS[f] * s + o
        uv = _face_uv(tiles[f])
        for ci in TRI_ORDER:
            out[k, 0:3] = corners[ci]
            out[k, 3:5] = uv[ci]
            out[k, 5:8] = FACE_NORMALS[f]
            out[k, 8] = 1.0
            out[k, 9] = flash
            out[k, 10] = alpha
            out[k, 11] = emissive
            k += 1
    if matrix is not None:
        transform_vertices(out, matrix)
    return out


def transform_vertices(v, m):
    p = np.c_[v[:, 0:3], np.ones(len(v), np.float32)] @ m.T
    v[:, 0:3] = p[:, 0:3]
    n = v[:, 5:8] @ m[:3, :3].T
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    v[:, 5:8] = n / np.maximum(ln, 1e-6)
    return v


# ------------------------------------------------------------------ voxels
class VoxelWorld:
    """Sparse voxel world: {(x, y, z): block_name}."""

    def __init__(self):
        self.blocks = {}

    def set(self, p, name):
        p = tuple(int(c) for c in p)
        if name is None:
            self.blocks.pop(p, None)
        else:
            self.blocks[p] = name

    def get(self, p):
        return self.blocks.get(tuple(p))

    def copy(self):
        w = VoxelWorld()
        w.blocks = dict(self.blocks)
        return w

    def top_at(self, x, z, below=10_000):
        """Highest solid block y (top surface = y+1) in column below `below`."""
        best = None
        for (bx, by, bz), n in self.blocks.items():
            if bx == x and bz == z and by < below and n not in ("cloud",):
                if best is None or by > best:
                    best = by
        return best

    def mesh(self, alpha_override=None):
        """Return (opaque_vertices, translucent_vertices)."""
        solid = self.blocks
        opaque, trans = [], []
        ao_levels = np.array([0.5, 0.68, 0.84, 1.0], np.float32)
        for p, name in solid.items():
            faces = tx.block_faces(name)
            translucent = tx.is_translucent(name)
            emis = tx.BLOCKS[name][3].get("emissive", 0.0)
            x, y, z = p
            for f in range(6):
                n = FACE_NORMALS[f].astype(int)
                q = (x + n[0], y + n[1], z + n[2])
                nb = solid.get(q)
                if nb is not None:
                    if nb == name:
                        continue
                    if not tx.is_transparent(nb):
                        continue
                    if translucent and tx.is_translucent(nb):
                        continue
                corners = FACE_CORNERS[f]
                uv = _face_uv(faces[f])
                # ambient occlusion per corner
                ax = [i for i in range(3) if n[i] == 0]
                ao = np.empty(4, np.float32)
                for ci in range(4):
                    c = corners[ci]
                    d = [0, 0, 0]
                    s1p, s2p = list(q), list(q)
                    su = 1 if c[ax[0]] > 0.5 else -1
                    sw = 1 if c[ax[1]] > 0.5 else -1
                    s1p[ax[0]] += su
                    s2p[ax[1]] += sw
                    cp = list(q)
                    cp[ax[0]] += su
                    cp[ax[1]] += sw
                    s1 = _occ(solid.get(tuple(s1p)))
                    s2 = _occ(solid.get(tuple(s2p)))
                    cc = _occ(solid.get(tuple(cp)))
                    lvl = 0 if (s1 and s2) else 3 - (s1 + s2 + cc)
                    ao[ci] = ao_levels[lvl]
                quad = np.empty((6, VERT), np.float32)
                for k, ci in enumerate(TRI_ORDER):
                    quad[k, 0:3] = corners[ci] + (x, y, z)
                    quad[k, 3:5] = uv[ci]
                    quad[k, 5:8] = FACE_NORMALS[f]
                    quad[k, 8] = ao[ci]
                    quad[k, 9] = 0.0
                    quad[k, 10] = (alpha_override if alpha_override is not None else 1.0)
                    quad[k, 11] = emis
                (trans if translucent else opaque).append(quad)
        cat = lambda L: np.concatenate(L) if L else np.zeros((0, VERT), np.float32)
        return cat(opaque), cat(trans)


def _occ(name):
    return 1 if (name is not None and not tx.is_transparent(name)) else 0


# ------------------------------------------------------------ matrices
def translate(x, y, z):
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = (x, y, z)
    return m


def scale(sx, sy=None, sz=None):
    if sy is None:
        sy = sz = sx
    return np.diag([sx, sy, sz, 1]).astype(np.float32)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], np.float32)


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]], np.float32)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1
    return m


def look_at(eye, target, up=(0, 1, 0), roll=0.0):
    eye, target = np.array(eye, np.float32), np.array(target, np.float32)
    f = target - eye
    f /= max(np.linalg.norm(f), 1e-6)
    up = np.array(up, np.float32)
    if abs(np.dot(f, up)) > 0.999:
        up = np.array((0, 0, 1), np.float32)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    if roll:
        c, sn = np.cos(roll), np.sin(roll)
        s, u = s * c + u * sn, -s * sn + u * c
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m
