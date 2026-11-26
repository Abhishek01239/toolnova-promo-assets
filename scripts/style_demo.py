"""Before/after demo of the cinematic style on a hero shot (Pip on a cliff at sunset).

usage: python scripts/style_demo.py [out.png]
Renders the same 1080x1920 frame with the classic look (MC_STYLE=classic) and the cinematic look (default).
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mcengine import style                                   # noqa: E402
from mcengine.geometry import VoxelWorld                     # noqa: E402
from mcengine.lighting import environment                    # noqa: E402
from mcengine.models import pip_vertices                     # noqa: E402
from mcengine.renderer import Renderer                       # noqa: E402

W, H = 1080, 1920


def world():
    w = VoxelWorld()
    rng = np.random.default_rng(11)
    for x in range(-40, 41):
        for z in range(-50, 31):
            # cliff plateau near the camera, valley and hills behind
            h = 2 + 3 * np.sin(x * 0.12) * np.cos(z * 0.09) + 2 * np.sin((x + z) * 0.05)
            if z > -6:
                h = 8 + (1 if (x + z) % 7 == 0 else 0) * 0
            elif z > -14:
                h = -6
            else:
                h += 6 + 9 * max(0.0, np.sin(x * 0.06 + 1.2)) ** 2 + (z < -30) * 6
            h = int(round(h))
            for y in range(h - 6, h + 1):
                w.set((x, y, z), "grass" if y == h else ("dirt" if y >= h - 2 else "stone"))
    # cliff face detail
    for x in range(-40, 41):
        for y in range(-6, 9):
            if rng.random() < 0.18:
                w.set((x, y, -6), None)
    # trees
    for (x, z) in [(-9, 4), (-14, -2), (11, 6), (16, 0), (-20, -24), (-4, -26), (14, -28), (24, -34), (-28, -36)]:
        h = w.top_at(x, z)
        if h is None:
            continue
        for y in range(h + 1, h + 6):
            w.set((x, y, z), "log")
        top = h + 5
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                for dy in range(-1, 3):
                    if abs(dx) + abs(dz) + abs(dy) <= 3 and w.get((x + dx, top + dy, z + dz)) is None:
                        w.set((x + dx, top + dy, z + dz), "leaves")
    return w


def frame(r, static_keys):
    sky, light, fog = environment(1.0, 40, 260)       # golden hour / sunset
    sky = dict(sky); light = dict(light)
    sun = np.array([-0.55, 0.16, -0.82]); sun /= np.linalg.norm(sun)
    sky["sun_dir"] = tuple(sun); light["dir"] = tuple(sun)
    pip = pip_vertices((0.5, 9.0, -3.5), yaw=0.15, pose={"head_yaw": -0.25, "head_pitch": -0.05})
    cam = dict(eye=(2.6, 9.7, 1.6), target=(-0.4, 11.6, -4.5), fov=58)
    return dict(camera=cam, sky=sky, light=light, fog=fog, static=static_keys, dynamic=pip,
                post=dict(bloom=0.3, vignette=0.38, saturation=1.12))


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
    except OSError:
        f = ImageFont.load_default()
    d.rounded_rectangle((40, 40, 120 + d.textlength(text, font=f), 140), 24, fill=(0, 0, 0, 170))
    d.text((80, 52), text, font=f, fill=(255, 255, 255))
    return im


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "style_demo.png"
    w = world()
    o, t = w.mesh()
    shots = []
    for name, active in (("BEFORE", False), ("AFTER: cinematic", True)):
        style.ACTIVE = active
        r = Renderer(W, H)
        r.set_static("w", o)
        r.set_static("w_t", t)
        fs = frame(r, ["w"])
        fs["static_trans"] = ["w_t"]
        img = r.render(fs)
        Image.fromarray(img).save(out.replace(".png", f"_{'after' if active else 'before'}.png"))
        shots.append(label(img, name))
    sheet = Image.new("RGB", (W * 2 + 20, H), (20, 20, 20))
    sheet.paste(shots[0], (0, 0)); sheet.paste(shots[1], (W + 20, 0))
    sheet.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()
