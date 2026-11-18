"""Models for '100 Days': cow, sheep, wolf (Biscuit), spider, ghast, boat, bed, door, fence, crops.
All face +z at yaw 0 and stand on the origin."""
import numpy as np
from .geometry import box_vertices, translate, rot_x, rot_y, rot_z, scale
PX = 1 / 16


def _cat(parts):
    return np.concatenate(parts) if parts else np.zeros((0, 12), np.float32)


def cow_vertices(pos, yaw=0.0, pose=None, flash=0.0, s=1.0):
    p = pose or {}
    root = translate(*pos) @ rot_y(yaw) @ scale(s)
    walk, ph = p.get("walk", 0.0), p.get("walk_phase", 0.0)
    kw = dict(flash=flash)
    out = []
    for i, (lx, lz) in enumerate(((3, 6), (-3, 6), (3, -7), (-3, -7))):
        a = np.sin(ph + (0 if i in (0, 3) else np.pi)) * 0.55 * walk
        m = root @ translate(lx * PX, 12 * PX, lz * PX) @ rot_x(a)
        out.append(box_vertices((4 * PX, 12 * PX, 4 * PX), "cow_leg", matrix=m, offset=(0, -6 * PX, 0), **kw))
    out.append(box_vertices((12 * PX, 10 * PX, 18 * PX), ["cow_side", "cow_side", "cow_top", "cow_top", "cow_end", "cow_end"],
                            matrix=root, offset=(0, 17 * PX, 0), **kw))
    hm = root @ translate(0, 20 * PX, 9 * PX) @ rot_x(p.get("head_pitch", 0.0) + 0.12 * np.sin(p.get("t", 0) * 1.3))
    out.append(box_vertices((8 * PX, 8 * PX, 6 * PX), ["cow_head", "cow_head", "cow_head", "cow_head", "cow_face", "cow_head"],
                            matrix=hm, offset=(0, 0, 3 * PX), **kw))
    for sx in (1, -1):
        out.append(box_vertices((1.5 * PX, 3 * PX, 1.5 * PX), "horn", matrix=hm, offset=(sx * 4.5 * PX, 4.5 * PX, 2 * PX), **kw))
    return _cat(out)


def sheep_vertices(pos, yaw=0.0, pose=None, flash=0.0, s=1.0):
    p = pose or {}
    root = translate(*pos) @ rot_y(yaw) @ scale(s)
    walk, ph = p.get("walk", 0.0), p.get("walk_phase", 0.0)
    out = []
    for i, (lx, lz) in enumerate(((3, 5), (-3, 5), (3, -6), (-3, -6))):
        a = np.sin(ph + (0 if i in (0, 3) else np.pi)) * 0.55 * walk
        m = root @ translate(lx * PX, 12 * PX, lz * PX) @ rot_x(a)
        out.append(box_vertices((4 * PX, 12 * PX, 4 * PX), "sheep_leg", matrix=m, offset=(0, -6 * PX, 0), flash=flash))
    out.append(box_vertices((12 * PX, 11 * PX, 17 * PX), "wool", matrix=root, offset=(0, 17 * PX, 0), flash=flash))
    hm = root @ translate(0, 19 * PX, 8 * PX) @ rot_x(p.get("head_pitch", 0.0) + 0.15 * np.sin(p.get("t", 0) * 1.1 + 1))
    out.append(box_vertices((6 * PX, 6 * PX, 8 * PX), ["sheep_head", "sheep_head", "wool", "sheep_head", "sheep_face", "sheep_head"],
                            matrix=hm, offset=(0, 1 * PX, 3 * PX), flash=flash))
    return _cat(out)


def wolf_vertices(pos, yaw=0.0, pose=None, flash=0.0, tame=False, s=1.0):
    """pose: walk, walk_phase, sit (0..1), tail (wag phase), head_yaw, angry"""
    p = pose or {}
    root = translate(*pos) @ rot_y(yaw) @ scale(s)
    walk, ph = p.get("walk", 0.0), p.get("walk_phase", 0.0)
    sit = p.get("sit", 0.0)
    out = []
    body = root @ translate(0, 10 * PX, 0) @ rot_x(-0.55 * sit) @ translate(0, -10 * PX, 0)
    for i, (lx, lz) in enumerate(((1.5, 4), (-1.5, 4), (1.5, -4), (-1.5, -4))):
        a = np.sin(ph + (0 if i in (0, 3) else np.pi)) * 0.6 * walk
        if i >= 2:
            a += 1.3 * sit
            m = root @ translate(lx * PX, (8 - 3 * sit) * PX, (lz + 1.5 * sit) * PX) @ rot_x(-a)
        else:
            m = root @ translate(lx * PX, 8 * PX, lz * PX) @ rot_x(a)
        out.append(box_vertices((2 * PX, 8 * PX, 2 * PX), "wolf_leg", matrix=m, offset=(0, -4 * PX, 0), flash=flash))
    out.append(box_vertices((6 * PX, 6 * PX, 9 * PX), "wolf_body", matrix=body, offset=(0, 10 * PX, -1 * PX), flash=flash))
    out.append(box_vertices((8 * PX, 7 * PX, 6 * PX), "wolf_mane", matrix=body, offset=(0, 10.5 * PX, 3.5 * PX), flash=flash))
    if tame:
        out.append(box_vertices((7 * PX, 2 * PX, 2 * PX), "collar", matrix=body, offset=(0, 11 * PX, 7 * PX), inflate=0.02))
    wag = np.sin(p.get("tail", 0.0)) * (0.6 if tame else 0.15)
    tm = body @ translate(0, 12 * PX, -5.5 * PX) @ rot_y(wag) @ rot_x(-0.9 + (0.5 if tame else 0.0))
    out.append(box_vertices((2 * PX, 2 * PX, 8 * PX), "wolf_body", matrix=tm, offset=(0, 0, -4 * PX), flash=flash))
    hm = body @ translate(0, 13 * PX, 7 * PX) @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(-p.get("head_pitch", 0.0))
    out.append(box_vertices((6 * PX, 6 * PX, 4 * PX), ["wolf_head", "wolf_head", "wolf_head", "wolf_head", "wolf_face", "wolf_head"],
                            matrix=hm, offset=(0, 0, 2 * PX), flash=flash))
    out.append(box_vertices((3 * PX, 3 * PX, 4 * PX), ["wolf_snout", "wolf_snout", "wolf_snout", "wolf_snout", "wolf_snout", "wolf_snout"],
                            matrix=hm, offset=(0, -1.5 * PX, 6 * PX), flash=flash))
    for sx in (1, -1):
        out.append(box_vertices((2 * PX, 2 * PX, 1 * PX), "wolf_head", matrix=hm, offset=(sx * 2 * PX, 4 * PX, 1 * PX), flash=flash))
    return _cat(out)


def spider_vertices(pos, yaw=0.0, pose=None, flash=0.0, s=1.0):
    """pose: walk, walk_phase, climb (0..1: body pitched up a wall)"""
    p = pose or {}
    root = translate(*pos) @ rot_y(yaw) @ rot_x(-p.get("climb", 0.0) * np.pi / 2) @ scale(s)
    walk, ph = p.get("walk", 0.0), p.get("walk_phase", 0.0)
    out = []
    out.append(box_vertices((10 * PX, 8 * PX, 12 * PX), "spider_body", matrix=root, offset=(0, 7 * PX, -5 * PX), flash=flash))
    out.append(box_vertices((6 * PX, 6 * PX, 6 * PX), "spider_head", matrix=root, offset=(0, 7 * PX, 2 * PX), flash=flash))
    out.append(box_vertices((8 * PX, 8 * PX, 8 * PX), ["spider_head", "spider_head", "spider_head", "spider_head", "spider_face",
                                                        "spider_head"], matrix=root, offset=(0, 7 * PX, 8 * PX), flash=flash))
    for side in (1, -1):
        for i in range(4):
            a = side * (0.35 + 0.28 * (i - 1.5)) + np.sin(ph + i * 1.7 + (side > 0) * np.pi) * 0.3 * walk
            m = root @ translate(side * 3 * PX, 7 * PX, (3 - i * 1.5) * PX) @ rot_y(side * np.pi / 2 - a * side) @ rot_z(side * 0.0) @ rot_x(0.35)
            out.append(box_vertices((2 * PX, 2 * PX, 16 * PX), "spider_leg", matrix=m, offset=(0, 0, 7 * PX), flash=flash))
    return _cat(out)


def ghast_vertices(pos, yaw=0.0, pose=None, flash=0.0, s=1.0):
    """pos = centre bottom of the head cube; 4 blocks wide. pose: open (bool), t"""
    p = pose or {}
    root = translate(*pos) @ rot_y(yaw) @ scale(s)
    face = "ghast_face_open" if p.get("open") else "ghast_face"
    out = [box_vertices((4, 4, 4), ["ghast_body", "ghast_body", "ghast_body", "ghast_body", face, "ghast_body"],
                        matrix=root, offset=(0, 2, 0), flash=flash, emissive=0.45)]
    tt = p.get("t", 0.0)
    for i, (tx_, tz) in enumerate(((-1.2, -1.2), (0, -1.2), (1.2, -1.2), (-1.2, 0), (0, 0), (1.2, 0), (-1.2, 1.2), (0, 1.2), (1.2, 1.2))):
        ln = 1.6 + 0.6 * ((i * 7) % 3) / 2
        m = root @ translate(tx_, 0, tz) @ rot_x(0.25 * np.sin(tt * 2.2 + i)) @ rot_z(0.2 * np.sin(tt * 1.7 + i * 2))
        out.append(box_vertices((0.25, ln, 0.25), "ghast_tentacle", matrix=m, offset=(0, -ln / 2, 0), flash=flash, emissive=0.4))
    return _cat(out)


def boat_vertices(pos, yaw=0.0, rock=0.0):
    root = translate(*pos) @ rot_y(yaw) @ rot_z(rock)
    out = [box_vertices((1.2, 0.12, 2.6), "planks", matrix=root, offset=(0, 0.06, 0))]
    for sx in (1, -1):
        out.append(box_vertices((0.12, 0.42, 2.6), "planks", matrix=root, offset=(sx * 0.6, 0.27, 0)))
    for sz in (1, -1):
        out.append(box_vertices((1.32, 0.42, 0.12), "planks", matrix=root, offset=(0, 0.27, sz * 1.3)))
    return _cat(out)


def bed_vertices(base, yaw=0.0):
    """Bed 1x2 blocks; base = min corner of the head block, foot extends to +z (rotated by yaw)."""
    m = translate(*base) @ translate(0.5, 0, 0.5) @ rot_y(yaw) @ translate(-0.5, 0, -0.5)
    return _cat([box_vertices((1, 0.56, 2), ["bed_side", "bed_side", "bed_top", "planks", "bed_side", "bed_side"], matrix=m,
                              offset=(0, 0, 0), centered=False),
                 box_vertices((1, 0.1, 0.62), "wool", matrix=m, offset=(0, 0.56, 0.06), centered=False)])


def door_vertices(base, yaw=0.0, open_=0.0):
    """2-tall door standing on base (min corner of block); hinge at x=0."""
    m = translate(*base) @ rot_y(yaw) @ rot_y(-open_ * np.pi / 2)
    return box_vertices((1, 2, 0.18), ["door", "door", "planks", "planks", "door", "door"], matrix=m, offset=(0, 0, 0.0), centered=False)


def fence_post(base, link_x=False, link_z=False):
    x, y, z = base
    out = [box_vertices((0.25, 1.0, 0.25), "planks", matrix=translate(x + 0.5, y + 0.5, z + 0.5))]
    for lx, lz, on in ((0.5, 0, link_x), (0, 0.5, link_z)):
        if on:
            for hy in (0.35, 0.8):
                sx, sz = (1.0, 0.12) if lx else (0.12, 1.0)
                out.append(box_vertices((sx, 0.14, sz), "planks", matrix=translate(x + 0.5 + lx, y + hy, z + 0.5 + lz)))
    return _cat(out)


def crop_vertices(base, tile="crop", h=0.8):
    x, y, z = base
    m = translate(x + 0.5, y, z + 0.5)
    return _cat([box_vertices((0.9, h, 0.02), tile, matrix=m @ rot_y(0.785), offset=(0, h / 2, 0)),
                 box_vertices((0.9, h, 0.02), tile, matrix=m @ rot_y(-0.785), offset=(0, h / 2, 0))])


def fireball_vertices(pos, s=1.0, t=0.0):
    m = translate(*pos) @ rot_y(t * 4) @ rot_x(t * 3)
    return box_vertices((0.8 * s, 0.8 * s, 0.8 * s), "fireball", matrix=m, emissive=1.0)
