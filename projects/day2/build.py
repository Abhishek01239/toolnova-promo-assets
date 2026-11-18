"""Build the long-form film 'Day 2: Until Sunset to Find Diamonds' (1920x1080, ~10 min).

usage: python build.py [stills t1,t2,...] [audio] [video] [thumb] [docs] [qc] [all]
       python build.py video --range START END   (render a slice to work/slice.mp4)
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mcengine import audio as A               # noqa: E402
from mcengine import overlay as ov            # noqa: E402
from mcengine import qc                       # noqa: E402
from mcengine.anim import seg                 # noqa: E402
from mcengine.render import render_video, mux, to_mp3   # noqa: E402
from mcengine.renderer import Renderer        # noqa: E402
import script as S                            # noqa: E402
import story as ST                            # noqa: E402

OUT = os.environ.get("MC_OUTPUT", "/home/user/output/day2")
WORK = os.path.join(HERE, "work")
W, H, FPS = ST.W, ST.H, 30
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def make_film(w=W, h=H):
    f = ST.Film(os.path.join(WORK, "cache"))
    r = Renderer(w, h)
    build_static(f, r)
    return f, r


def build_static(f, r):
    import world as WD
    from mcengine.geometry import VoxelWorld
    from scipy.ndimage import gaussian_filter
    f._r = r
    o, t = f.surface.far()
    r.set_static("far", o)
    r.set_static("far_t", t)
    o, t = f.surface.mesh()
    r.set_static("surface", o)
    r.set_static("surface_t", t)
    o, t = WD.build_under(os.path.join(WORK, "cache"))
    r.set_static("under", o)
    r.set_static("under_t", t)
    rng = np.random.default_rng(3)
    g = gaussian_filter(rng.normal(size=(30, 30)), 1.4)
    g /= g.std()
    cw = VoxelWorld()
    for gx in range(30):
        for gz in range(30):
            if g[gx, gz] > 0.7:
                cw.set((gx - 15, 0, gz - 15), "cloud")
    _, cv = cw.mesh(alpha_override=0.75)
    cv[:, 0:3] = cv[:, 0:3] * np.array([12, 3, 12], np.float32) + np.array([0, 70, 0], np.float32)
    r.set_static("clouds", cv)


def zoom_punch(img, z):
    if z <= 1.001:
        return img
    h, w = img.shape[:2]
    cw, ch = int(w / z), int(h / z)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return np.asarray(Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.BILINEAR))


def frame_fn_factory(f, r):
    cuts = f.cuts()

    def frame(t):
        img = r.render(f.frame_spec(t))
        z = 1.0
        for c in cuts:
            if c <= t < c + 0.18:
                z = 1 + 0.035 * (1 - seg(t, c, c + 0.18)) ** 2
        img = zoom_punch(img, z)
        layer = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
        f.draw_overlay(t, layer)
        return ov.composite(img, layer)
    return frame


def build_audio(f):
    dur = f.duration
    voice = A.load_audio(os.environ.get("NARRATION", os.path.join(HERE, "assets", "narration.wav")))
    mx = A.Mixer(dur)
    mx.add_voice(0.0, voice, target_db=-16.0)
    mx.add_music(A.compose(f.music_sections(), dur, bpm=112), gain_db=-15.0)
    for t, name, g, kw in f.sfx_events():
        mx.add_sfx(t, name, g, **kw)
    # ambient beds: forest wind by day, quiet wind in the cave, night wind
    mx.add_sfx(0.0, A.sfx_wind(dur) * 0.25, -24)
    mix = mx.render(duck_db=-9.0)
    A.write_wav(os.path.join(WORK, "mix.wav"), mix)
    vo = mx.voice / (np.abs(mx.voice).max() + 1e-9) * 0.89
    A.write_wav(os.path.join(WORK, "voice.wav"), vo)
    to_mp3(os.path.join(WORK, "voice.wav"), os.path.join(OUT, "voiceover.mp3"))
    print("audio done", mix.shape)


def build_video(f, r, start=0.0, end=None, out=None):
    fn = frame_fn_factory(f, r)
    vid = out or os.path.join(WORK, "video_only.mp4")
    t0 = time.time()
    render_video(vid, fn, f.duration, W, H, FPS, start=start, end=end, log_every=150)
    print(f"video rendered in {time.time() - t0:.0f}s")
    if out is None:
        mux(vid, os.path.join(WORK, "mix.wav"), os.path.join(OUT, "minecraft_video.mp4"))


def stills(f, r, times):
    fn = frame_fn_factory(f, r)
    ims = [Image.fromarray(fn(t)).resize((480, 270)) for t in times]
    cols = 4
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * 270))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * 480, (i // cols) * 270))
    p = os.path.join(WORK, "stills.png")
    sheet.save(p)
    print("saved", p)


def build_thumbnail():
    f, r = make_film(1280, 720)
    from world import ug
    t = f.T("five", 0.6)
    pip = f.pip_path.pos(t)
    cam = dict(eye=pip + np.array([2.3, 1.55, -2.1], np.float32), target=pip + np.array([0.0, 1.45, 0.2], np.float32),
               fov=52, roll=0.0)
    fs = f.frame_spec(t, cam)
    fs["post"].update(saturation=1.3, contrast=1.1, vignette=0.4, bloom=0.4)
    img = Image.fromarray(r.render(fs)).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov.draw_text(layer, "UNTIL SUNSET", (360, 100), 92, fill=(255, 225, 40), rot=4, stroke=12)
    ov.draw_text(layer, "TO FIND DIAMONDS", (370, 200), 70, fill=(255, 255, 255), rot=4, stroke=10)
    ov.draw_text(layer, "DAY 2", (1090, 620), 120, fill=(110, 235, 255), rot=-3, stroke=14)
    ov.paste_emoji(layer, "💎", (1130, 120), 140, rot=-10)
    ov.paste_emoji(layer, "⏰", (170, 600), 120, rot=12)
    img.alpha_composite(layer)
    img.convert("RGB").save(os.path.join(OUT, "thumbnail.png"))
    print("thumbnail saved")


def _ts(x):
    return f"{int(x // 60)}:{int(x % 60):02d}"


def docs(f):
    with open(os.path.join(OUT, "script.txt"), "w") as fh:
        fh.write(f"{S.TITLE}  —  long-form narration script (~{f.duration / 60:.1f} min)\n")
        fh.write("Narrator: Pip (first person). Voice: free local Kokoro 'puck_classic' (same as Hardcore Parts 2-5 "
                 "and First Night). Bracketed text = delivery direction (not spoken, not subtitled).\n")
        li = 0
        for num, title, lines in S.CHAPTERS:
            fh.write(f"\nCHAPTER {num}: {title}\n")
            for _, line in lines:
                fh.write(f"[{int(f.ls[li] // 60)}:{f.ls[li] % 60:04.1f}]  {line}\n")
                li += 1
    beats = [
        ("hook", "Sunrise on the Day 1 tower. Pip holds his one diamond; pull back over the base and the crater farm."),
        ("need3", "Diamond pickaxe recipe with 1 of 3 diamonds. Flash-forward: the lava chamber ('the bottom of the world')."),
        ("breakfast", "Daylight clock starts at 10:00. Harvesting the crater farm; 3 wheat = 1 bread; health refills."),
        ("plan", "THE PLAN checklist: IRON / GO DEEP / 2 DIAMONDS / HOME."),
        ("neediron", "Demo: a stone pickaxe breaks diamond ore into nothing."),
        ("hill", "CHAPTER 2. Back to the hill tunnel; stone pickaxe recipe; six iron ores mined (1/6 ... 6/6)."),
        ("furnace", "Furnace recipe, smelting, six ingots fly out; iron pickaxe and bucket recipes."),
        ("eight", "CHAPTER 3. The staircase: digging step by step, coal vein, daylight shrinking above, torches on the right."),
        ("gravel", "Gravel collapses into the stairs; Pip jumps back. Torch trick diagram. Deepslate. Break-through."),
        ("cavern", "CHAPTER 4. A giant cavern with an underground lake and torch reflections."),
        ("groan", "A zombie ambush: three pickaxe hits and it's gone. 'It's always night underground.'"),
        ("lava", "CHAPTER 5. The lava lake; diamonds sparkle on the far ledge."),
        ("pour1", "Water + lava = obsidian: the bridge is built pour by pour; Pip steps on lava and catches fire."),
        ("four", "CHAPTER 6. Four diamond ores mined with the iron pickaxe: 5 diamonds total."),
        ("ranback", "CHAPTER 7. Lost in the cavern: dead ends, then follow the torches on the left. A creeper appears."),
        ("run", "CHAPTER 8. The stair chase; the creeper blast collapses the tunnel behind Pip. Out into the sunset, dash home."),
        ("table", "CHAPTER 9. On the roof at night: diamond pickaxe and diamond sword recipes."),
        ("night2", "CHAPTER 10. Night two: three zombies, one sword hit after another, two knocked into the moat."),
        ("obsidian", "Flashback to the obsidian bridge; a vision of a 10-block Nether portal; end card asks for comments."),
    ]
    with open(os.path.join(OUT, "storyboard.txt"), "w") as fh:
        fh.write(f"STORYBOARD — {S.TITLE}  (1920x1080, 30 fps, {f.duration:.1f} s)\n\n")
        fh.write("Characters: Pip (original hero: orange hoodie, brown hair, teal pants), original zombies and creepers.\n"
                 "Sets: the base from 'First Night' (rebuilt from its own code) and a new underground world.\n\n")
        for key, txt in beats:
            fh.write(f"{_ts(f.T(key, 0))}  {txt}\n")
        fh.write("\nMusic (original, synthesized): adventure, build, cave suspense, panic stingers, chase, resolve.\n")
    chap = "\n".join(f"{_ts(0 if n == 1 else f.card(n)[0])} {f.chapters[n - 1]['title'].title()}" for n in range(1, 11))
    meta = f"""TITLE
{S.TITLE}

ALTERNATE TITLES
- I Had 10 Minutes of Daylight to Find Diamonds (Minecraft Survival Story, Day 2)
- I Built a Lava Bridge Out of Water to Reach 4 Diamonds (Minecraft)

DESCRIPTION
Yesterday I survived my first night with one single diamond. Today I need two more for a diamond pickaxe... and I have to be home before sunset.

In this video I mine iron, build a furnace, dig a staircase all the way down to deepslate, find a giant cave with an underground lake, fight a zombie with just a pickaxe, build an obsidian bridge across a lava lake using a water bucket, mine four diamonds, escape a creeper up a collapsing staircase, and craft my first diamond tools before night two.

Chapters
{chap}

Should I build a Nether portal on Day 3? Tell me in the comments!

#minecraft #minecraftsurvival #minecraftstory

HASHTAGS
#minecraft #minecraftsurvival #minecraftstory #diamonds #lava #obsidian #creeper #gaming

TAGS
minecraft, minecraft survival, minecraft day 2, minecraft story, minecraft animation, minecraft diamonds,
diamond pickaxe, obsidian bridge, water bucket lava, minecraft caves, deepslate, creeper, zombie, survival challenge,
minecraft tips, nether portal

THUMBNAIL CONCEPT
Pip on the diamond ledge in the lava chamber, five diamonds circling above him, lava glowing below.
Big tilted text "UNTIL SUNSET / TO FIND DIAMONDS" (yellow/white) top-left, "DAY 2" bottom-right, diamond + clock emoji.

CATEGORY / SETTINGS
Gaming · Not made for kids (general audience) · English · Synthetic narration voice (free local TTS).

NOTES
Every frame, the music and all sound effects are generated by code (procedural voxel engine, no game assets).
"Minecraft" is a trademark of Mojang/Microsoft; this is unofficial fan content.
"""
    open(os.path.join(OUT, "metadata.txt"), "w").write(meta)
    print("docs written")


def run_qc(f):
    path = os.path.join(OUT, "minecraft_video.mp4")
    rep = qc.report(path, W, H, 580, 630)
    times = [5 + i * (f.duration - 10) / 23 for i in range(24)]
    qc.contact_sheet(path, os.path.join(WORK, "qc_sheet.png"), times, thumb_w=320)
    json.dump(rep, open(os.path.join(WORK, "qc_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    need_r = any(a in args for a in ("video", "all", "stills"))
    f, r = make_film() if need_r else (ST.Film(os.path.join(WORK, "cache")), None)
    if "stills" in args:
        stills(f, r, [float(x) for x in args[args.index("stills") + 1].split(",")])
    if "audio" in args or "all" in args:
        build_audio(f)
    if "video" in args or "all" in args:
        if "--range" in args:
            i = args.index("--range")
            build_video(f, r, float(args[i + 1]), float(args[i + 2]), os.path.join(WORK, "slice.mp4"))
        else:
            build_video(f, r)
    if "thumb" in args or "all" in args:
        build_thumbnail()
    if "docs" in args or "all" in args:
        docs(f)
    if "qc" in args or "all" in args:
        run_qc(f)
