"""Blocky character & mob models with procedural animation poses.

All models stand on their feet at the origin and face +z when yaw = 0.
1 pixel = 1/16 block, so a character is 32 px = 2 blocks tall.
"""
import numpy as np
from .geometry import box_vertices, translate, rot_x, rot_y, rot_z, scale
PX = 1 / 16


def _cat(parts):
    return np.concatenate(parts) if parts else np.zeros((0, 12), np.float32)


def pip_vertices(pos, yaw=0.0, pose=None, flash=0.0, alpha=1.0):
    """The hero, 'Pip'. pose keys (all optional):
       walk_phase, walk (0..1 amplitude), punch (0..1), head_yaw, head_pitch,
       arms_up (0..1), flail_phase, body_pitch, item ('water_bucket'|'bucket'|None),
       lean (forward tilt), roll (whole-body roll, radians)
    """
    p = pose or {}
    root = translate(*pos) @ rot_y(yaw)
    if p.get("body_pitch") or p.get("roll"):
        # pivot around body centre for tumbling while falling
        c = translate(0, 1.0, 0)
        root = root @ c @ rot_x(p.get("body_pitch", 0.0)) @ rot_z(p.get("roll", 0.0)) @ np.linalg.inv(c)
    walk = p.get("walk", 0.0)
    ph = p.get("walk_phase", 0.0)
    swing = np.sin(ph) * 0.8 * walk
    arms_up = p.get("arms_up", 0.0)
    fl = p.get("flail_phase", 0.0)
    kw = dict(flash=flash, alpha=alpha)
    out = []

    # legs (pivot at hip)
    for side, sgn in ((1, 1), (-1, -1)):
        a = swing * sgn + arms_up * 0.35 * np.sin(fl * 1.3 + side)
        m = root @ translate(side * 2 * PX, 12 * PX, 0) @ rot_x(a)
        out.append(box_vertices((4 * PX, 12 * PX, 4 * PX),
                                ["pip_leg", "pip_leg", "pip_shoe", "pip_shoe", "pip_leg", "pip_leg"],
                                matrix=m, offset=(0, -6 * PX, 0), **kw))
    # body
    lean = p.get("lean", 0.0)
    body_m = root @ translate(0, 12 * PX, 0) @ rot_x(lean) @ translate(0, -12 * PX, 0)
    out.append(box_vertices((8 * PX, 12 * PX, 4 * PX),
                            ["pip_body_side", "pip_body_side", "pip_body_top", "pip_body_top",
                             "pip_body_front", "pip_body_back"],
                            matrix=body_m, offset=(0, 18 * PX, 0), **kw))
    # arms (pivot at shoulder). side=+1 -> +x (character's left), -1 -> right
    punch = p.get("punch", 0.0)
    right_arm_m = None
    for side in (1, -1):
        a = -swing * side
        spread = 0.08 + arms_up * (1.2 + 0.35 * np.sin(fl * 2.0 + side * 1.7))
        if side == -1 and punch > 0:
            a = -1.45 * np.sin(np.pi * min(punch, 1.0))
        if side == -1 and p.get("item"):
            a = min(a, -0.5) if punch == 0 else a
        if side == -1 and p.get("wave"):
            w = p["wave"]
            a = a * (1 - w) + (-2.7 + 0.25 * np.sin(p.get("wave_phase", 0.0))) * w
            spread = spread + 0.35 * np.sin(p.get("wave_phase", 0.0)) * w
        if side == -1 and p.get("reach"):
            a = a * (1 - p["reach"]) + (-0.35) * p["reach"]   # arm thrust down/forward (bucket)
        if arms_up:
            a = a * (1 - arms_up) + (-2.6 + 0.4 * np.sin(fl * 2.3 + side)) * arms_up
        m = body_m @ translate(side * 6 * PX, 22 * PX, 0) @ rot_x(a) @ rot_z(side * spread * (1 - arms_up * 0.7))
        out.append(box_vertices((4 * PX, 12 * PX, 4 * PX),
                                ["pip_arm", "pip_arm", "pip_arm_top", "pip_hand", "pip_arm", "pip_arm"],
                                matrix=m, offset=(0, -4 * PX, 0), **kw))
        if side == -1:
            right_arm_m = m
    # held item
    item = p.get("item")
    if item and right_arm_m is not None:
        top = "bucket_top" if item == "water_bucket" else "bucket_side"
        m = right_arm_m @ translate(0, -11 * PX, 3 * PX)
        out.append(box_vertices((5 * PX, 5 * PX, 5 * PX),
                                ["bucket_side", "bucket_side", top, "bucket_side", "bucket_side", "bucket_side"],
                                matrix=m, **kw))
    # head (pivot at neck)
    hm = body_m @ translate(0, 24 * PX, 0) @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(-p.get("head_pitch", 0.0))
    out.append(box_vertices((8 * PX, 8 * PX, 8 * PX),
                            ["pip_head_side", "pip_head_side_m", "pip_hair_top", "pip_chin",
                             "pip_face", "pip_head_back"],
                            matrix=hm, offset=(0, 4 * PX, 0), **kw))
    return _cat(out)


def creeper_vertices(pos, yaw=0.0, pose=None, flash=0.0, alpha=1.0):
    """pose keys: walk_phase, walk, swell (0..1), head_yaw, head_pitch, rise (0..1 spawn-in scale)"""
    p = pose or {}
    swell = p.get("swell", 0.0)
    rise = p.get("rise", 1.0)
    sx = 1 + 0.18 * swell
    root = translate(*pos) @ rot_y(yaw) @ scale(sx * rise, (1 + 0.06 * swell) * rise, sx * rise)
    walk = p.get("walk", 0.0)
    ph = p.get("walk_phase", 0.0)
    kw = dict(flash=flash, alpha=alpha)
    out = []
    for i, (lx, lz) in enumerate(((2, 4), (-2, 4), (2, -4), (-2, -4))):
        a = np.sin(ph + (0 if i in (0, 3) else np.pi)) * 0.6 * walk
        m = root @ translate(lx * PX, 6 * PX, lz * PX) @ rot_x(a)
        out.append(box_vertices((4 * PX, 6 * PX, 4 * PX), "creeper_leg", matrix=m, offset=(0, -3 * PX, 0), **kw))
    out.append(box_vertices((8 * PX, 12 * PX, 4 * PX),
                            ["creeper_body_side", "creeper_body_side", "creeper_top", "creeper_top",
                             "creeper_body", "creeper_body"],
                            matrix=root, offset=(0, 12 * PX, 0), **kw))
    hm = root @ translate(0, 18 * PX, 0) @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(-p.get("head_pitch", 0.0))
    out.append(box_vertices((8 * PX, 8 * PX, 8 * PX),
                            ["creeper_head", "creeper_head", "creeper_head", "creeper_head",
                             "creeper_face", "creeper_head"],
                            matrix=hm, offset=(0, 4 * PX, 0), **kw))
    return _cat(out)


def block_entity_vertices(name, center, s=1.0, rot=(0, 0, 0), flash=0.0, alpha=1.0, crack=None):
    """A free-floating block (popping in, debris, dropped item)."""
    from .textures import block_faces, BLOCKS
    m = translate(*center) @ rot_y(rot[1]) @ rot_x(rot[0]) @ rot_z(rot[2]) @ scale(s)
    emis = BLOCKS[name][3].get("emissive", 0.0)
    v = [box_vertices((1, 1, 1), block_faces(name), matrix=m, flash=flash, alpha=alpha, emissive=emis)]
    if crack is not None:
        v.append(box_vertices((1, 1, 1), f"crack{crack}", matrix=m, inflate=0.004, alpha=alpha))
    return _cat(v)
