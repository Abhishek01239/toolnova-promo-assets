"""Blocky character & mob models with procedural animation poses.

All models stand on their feet at the origin and face +z when yaw = 0.
1 pixel = 1/16 block, so a character is 32 px = 2 blocks tall.
"""
import numpy as np
from .geometry import box_vertices, translate, rot_x, rot_y, rot_z, scale
PX = 1 / 16


def _cat(parts):
    return np.concatenate(parts) if parts else np.zeros((0, 12), np.float32)


def pip_vertices(pos, yaw=0.0, pose=None, flash=0.0, alpha=1.0, skin="pip"):
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
                                [f"{skin}_leg", f"{skin}_leg", f"{skin}_shoe", f"{skin}_shoe", f"{skin}_leg", f"{skin}_leg"],
                                matrix=m, offset=(0, -6 * PX, 0), **kw))
    # body
    lean = p.get("lean", 0.0)
    body_m = root @ translate(0, 12 * PX, 0) @ rot_x(lean) @ translate(0, -12 * PX, 0)
    out.append(box_vertices((8 * PX, 12 * PX, 4 * PX),
                            [f"{skin}_body_side", f"{skin}_body_side", f"{skin}_body_top", f"{skin}_body_top",
                             f"{skin}_body_front", f"{skin}_body_back"],
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
        if p.get("zombie"):
            a = -1.45 + 0.08 * np.sin(p.get("walk_phase", 0.0) * 0.5 + side)
            spread = 0.04
        if arms_up:
            a = a * (1 - arms_up) + (-2.6 + 0.4 * np.sin(fl * 2.3 + side)) * arms_up
        m = body_m @ translate(side * 6 * PX, 22 * PX, 0) @ rot_x(a) @ rot_z(side * spread * (1 - arms_up * 0.7))
        out.append(box_vertices((4 * PX, 12 * PX, 4 * PX),
                                [f"{skin}_arm", f"{skin}_arm", f"{skin}_arm_top", f"{skin}_hand", f"{skin}_arm", f"{skin}_arm"],
                                matrix=m, offset=(0, -4 * PX, 0), **kw))
        if side == -1:
            right_arm_m = m
    # held item
    item = p.get("item")
    if item and item.startswith("pickaxe") and right_arm_m is not None:
        # "pickaxe" (wood), "pickaxe:stone", "pickaxe:iron", "pickaxe:diamond"
        head = "pick_head" if ":" not in item else "pick_head_" + item.split(":")[1]
        m = right_arm_m @ translate(0, -11 * PX, 2 * PX) @ rot_x(1.2)
        out.append(box_vertices((1.5 * PX, 1.5 * PX, 11 * PX), "pick_handle", matrix=m, offset=(0, 0, 3 * PX), **kw))
        out.append(box_vertices((9 * PX, 1.8 * PX, 1.8 * PX), head, matrix=m, offset=(0, 0, 8.5 * PX),
                                emissive=0.3 if "diamond" in item else 0.0, flash=flash, alpha=alpha))
        item = None
    if item and item.startswith("sword") and right_arm_m is not None:
        # "sword" (diamond), "sword:iron", "sword:ench" (enchanted diamond, glowing purple)
        blade = {"sword": "blade_diamond", "sword:iron": "blade_iron", "sword:ench": "blade_ench"}.get(item, "blade_diamond")
        m = right_arm_m @ translate(0, -11 * PX, 2 * PX) @ rot_x(1.2)
        out.append(box_vertices((1.6 * PX, 1.6 * PX, 4 * PX), "pick_handle", matrix=m, offset=(0, 0, 1 * PX), **kw))
        out.append(box_vertices((5 * PX, 1.2 * PX, 1.2 * PX), "pick_handle", matrix=m, offset=(0, 0, 3.2 * PX), **kw))
        out.append(box_vertices((1.8 * PX, 1.0 * PX, 11 * PX), blade, matrix=m, offset=(0, 0, 9 * PX),
                                emissive=0.6 if item == "sword:ench" else (0.35 if blade == "blade_diamond" else 0.05),
                                flash=flash, alpha=alpha))
        item = None
    if item == "hoe" and right_arm_m is not None:
        m = right_arm_m @ translate(0, -11 * PX, 2 * PX) @ rot_x(1.2)
        out.append(box_vertices((1.5 * PX, 1.5 * PX, 11 * PX), "pick_handle", matrix=m, offset=(0, 0, 3 * PX), **kw))
        out.append(box_vertices((4 * PX, 1.8 * PX, 1.8 * PX), "hoe_head", matrix=m, offset=(-1.5 * PX, 0, 8.5 * PX), **kw))
        item = None
    if item in ("bone", "wheat", "emerald", "flint") and right_arm_m is not None:
        tile = {"bone": "bone", "wheat": "wheat_item", "emerald": "emerald", "flint": "flint_steel"}[item]
        m = right_arm_m @ translate(0, -12 * PX, 3 * PX) @ rot_x(1.2)
        out.append(box_vertices((7 * PX, 0.6 * PX, 7 * PX), tile, matrix=m, **kw))
        item = None
    if item == "glowstone" and right_arm_m is not None:
        m = right_arm_m @ translate(0, -12 * PX, 3 * PX)
        out.append(box_vertices((6 * PX, 6 * PX, 6 * PX), "glowstone", matrix=m, emissive=0.8, flash=flash, alpha=alpha))
        item = None
    if item in ("cobble", "dirt", "planks", "log", "stone_bricks", "obsidian", "wool") and right_arm_m is not None:
        from .textures import block_faces
        m = right_arm_m @ translate(0, -12 * PX, 3 * PX)
        out.append(box_vertices((6 * PX, 6 * PX, 6 * PX), block_faces(item), matrix=m, **kw))
        item = None
    if item == "torch" and right_arm_m is not None:
        m = right_arm_m @ translate(0, -11 * PX, 3 * PX)
        out.append(box_vertices((1.6 * PX, 8 * PX, 1.6 * PX), "torch_stick", matrix=m, offset=(0, 3 * PX, 0), **kw))
        out.append(box_vertices((2.2 * PX, 2.2 * PX, 2.2 * PX), "torch_flame", matrix=m, offset=(0, 7.5 * PX, 0),
                                emissive=1.0, flash=flash, alpha=alpha))
        item = None
    if item == "tnt" and right_arm_m is not None:
        m = right_arm_m @ translate(0, -12 * PX, 3 * PX)
        out.append(box_vertices((6 * PX, 6 * PX, 6 * PX), ["tnt_side", "tnt_side", "tnt_top", "tnt_top", "tnt_side", "tnt_side"],
                                matrix=m, **kw))
        item = None
    if item and right_arm_m is not None:
        top = "bucket_top" if item == "water_bucket" else "bucket_side"
        m = right_arm_m @ translate(0, -11 * PX, 3 * PX)
        out.append(box_vertices((5 * PX, 5 * PX, 5 * PX),
                                ["bucket_side", "bucket_side", top, "bucket_side", "bucket_side", "bucket_side"],
                                matrix=m, **kw))
    # head (pivot at neck)
    hm = body_m @ translate(0, 24 * PX, 0) @ rot_y(p.get("head_yaw", 0.0)) @ rot_x(-p.get("head_pitch", 0.0))
    out.append(box_vertices((8 * PX, 8 * PX, 8 * PX),
                            [f"{skin}_head_side", f"{skin}_head_side_m", f"{skin}_hair_top", f"{skin}_chin",
                             f"{skin}_face", f"{skin}_head_back"],
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


def zombie_vertices(pos, yaw=0.0, pose=None, flash=0.0, alpha=1.0):
    p = dict(pose or {})
    p.setdefault("zombie", True)
    return pip_vertices(pos, yaw, p, flash=flash, alpha=alpha, skin="zombie")


def torch_vertices(base, flicker=0.0):
    """A wall/ground torch standing on `base` (bottom centre)."""
    m = translate(*base)
    v = [box_vertices((2 * PX, 10 * PX, 2 * PX), "torch_stick", matrix=m, offset=(0, 5 * PX, 0)),
         box_vertices((3 * PX * (1 + flicker), 3 * PX * (1 + flicker), 3 * PX * (1 + flicker)), "torch_flame",
                      matrix=m, offset=(0, 11 * PX, 0), emissive=1.0)]
    return _cat(v)


def chest_vertices(center, yaw=0.0, lid=0.0):
    """Chest with an opening lid (lid 0..1)."""
    root = translate(*center) @ rot_y(yaw)
    v = [box_vertices((14 * PX, 10 * PX, 14 * PX), ["chest_side", "chest_side", "chest_top", "chest_top",
                                                     "chest_front", "chest_side"], matrix=root, offset=(0, -3 * PX, 0))]
    lm = root @ translate(0, 2 * PX, -7 * PX) @ rot_x(-1.6 * lid) @ translate(0, 0, 7 * PX)
    v.append(box_vertices((14 * PX, 4 * PX, 14 * PX), ["chest_side", "chest_side", "chest_top", "chest_top",
                                                        "chest_side", "chest_side"], matrix=lm, offset=(0, 2 * PX, 0)))
    return _cat(v)


def item_vertices(kind, center, spin=0.0, s=1.0):
    """Floating dropped item: 'diamond', 'log', 'tnt', 'coal'."""
    m = translate(*center) @ rot_y(spin) @ rot_x(0.5) @ scale(s)
    if kind == "diamond":
        return box_vertices((0.22, 0.22, 0.22), "diamond", matrix=m, emissive=0.6)
    if kind == "coal":
        return box_vertices((0.2, 0.2, 0.2), "black", matrix=m)
    if kind == "iron":
        return box_vertices((0.26, 0.1, 0.14), "ingot", matrix=m, emissive=0.15)
    if kind == "rawiron":
        return box_vertices((0.2, 0.2, 0.2), "iron_ore", matrix=m)
    if kind == "gravel":
        return block_entity_vertices("gravel", center, 0.28 * s, (0.5, spin, 0))
    name = {"log": "log", "tnt": "tnt", "planks": "planks", "cobble": "cobble"}[kind]
    return block_entity_vertices(name, center, 0.28 * s, (0.5, spin, 0))
