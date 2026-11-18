"""Build the long-form film 'I Had 10 Minutes to Survive My First Night' (1920x1080, ~5 min).

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

OUT = os.environ.get("MC_OUTPUT", "/home/user/output/first_night")
WORK = os.path.join(HERE, "work")
W, H, FPS = ST.W, ST.H, 30
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def make_film(w=W, h=H):
    f = ST.Film(os.path.join(WORK, "cache"))
    r = Renderer(w, h)
    f.build_static(r)
    return f, r


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
    t = f.t_boom + 0.18
    cam = dict(eye=np.array([-13.0, 16.0, 31.0], np.float32), target=np.array([1.0, 12.5, 12.0], np.float32),
               fov=58, roll=0.0)
    fs = f.frame_spec(t, cam)
    from mcengine.lighting import environment
    _, night_light, _ = environment(2.0)
    night_light = dict(night_light)
    night_light["ambient"] = tuple(np.array(night_light["ambient"]) + np.array([0.35, 0.18, 0.05]))
    fs["light"] = night_light                       # keep the scene readable, not blown out by the blast
    fs["post"].update(white_flash=0.0, saturation=1.3, contrast=1.1, vignette=0.45, bloom=0.45)
    img = Image.fromarray(r.render(fs)).convert("RGBA")
    # Pip close-up inset: render Pip at the wall looking at the blast
    cam2 = dict(eye=np.array([2.0, 12.4, 2.9], np.float32), target=np.array([2.0, 12.3, 5.6], np.float32), fov=40)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov.draw_text(layer, "10 MINUTES", (370, 115), 104, fill=(255, 225, 40), rot=4, stroke=13)
    ov.draw_text(layer, "TO SURVIVE", (360, 230), 86, fill=(255, 255, 255), rot=4, stroke=11)
    ov.draw_text(layer, "NIGHT 1", (1010, 610), 120, fill=(170, 190, 255), rot=-3, stroke=14)
    ov.paste_emoji(layer, "😱", (1150, 120), 150, rot=-10)
    ov.paste_emoji(layer, "🧨", (560, 600), 120, rot=12)
    img.alpha_composite(layer)
    img.convert("RGB").save(os.path.join(OUT, "thumbnail.png"))
    print("thumbnail saved")


def docs(f):
    T = f.T
    with open(os.path.join(OUT, "script.txt"), "w") as fh:
        fh.write(f"{S.TITLE}  —  long-form narration script (~{f.duration / 60:.1f} min)\n")
        fh.write("Narrator: Pip (first person). Voice: free local Kokoro 'puck_classic' (same as Hardcore Parts 2-5). "
                 "Bracketed text = delivery direction (not spoken, not subtitled).\n")
        li = 0
        for num, title, lines in S.CHAPTERS:
            fh.write(f"\nCHAPTER {num}: {title}\n")
            for line in lines:
                fh.write(f"[{int(f.ls[li] // 60)}:{f.ls[li] % 60:04.1f}]  {line}\n")
                li += 1
    beats = [
        (0, "Hook. Close-up on Pip, fast pull-back reveals a forest clearing; '10 MINUTES' + countdown clock 10:00."),
        (2, "Night flash-forward: zombies and a creeper in the dark field (tinted). ZOMBIES / SKELETONS / CREEPERS pops."),
        (3, "THE PLAN checklist (WOOD / STONE / BASE). Pip walks to a tree."),
        (4, "Punching the tree: logs break bottom-up and fly into the hotbar; FUN FACT; floating leaves joke."),
        (6, "Crafting montage UI: logs → planks → crafting table → sticks → FIRST PICKAXE. Pip mines stone."),
        (8, "CHAPTER 2 card. Approach to the hill cave; inside, the entrance glows; Pip mines into COAL."),
        (11, "Torches light the cavern; an old chest; lid creaks open: 6× TNT burst out; a glowing DIAMOND."),
        (16, "Pip grabs the diamond and sprints out into daylight."),
        (17, "CHAPTER 3 card. Time-lapse build: walls, floor, roof, '2 BLOCKS' marker, watchtower, moat fills with water, torches, BASE COMPLETE."),
        (24, "CHAPTER 4 card. Sunset → purple → night (NIGHT 1). Pip climbs the tower."),
        (27, "Zombies claw out of the ground, march to the moat and bump into the water. Sssss... a creeper at the east wall."),
        (33, "BOOM: the wall explodes, a hole opens, zombies pour in; hearts drop to 5."),
        (35, "CHAPTER 5 card. Pip remembers the TNT, jumps from the tower; chase across the bridge into the field; '½ HEART'."),
        (39, "Counting 1–6 he drops a TNT line, runs back and hides by the wall; the fuse sparks along the ground."),
        (41, "KABOOM chain explosion (+ creepers), slow debris, a giant crater; '6 TNT = 0 MONSTERS'."),
        (45, "CHAPTER 6 card. Sunrise; surviving zombies catch fire and vanish; CREEPERS DON'T BURN fact."),
        (48, "SURVIVED NIGHT 1; Pip watches the sunrise from the tower."),
        (50, "Repair time-lapse: patched hole, a second wall, the crater becomes a farm; full-base reveal."),
        (52, "The diamond close-up; end card: WHAT SHOULD I BUILD WITH IT? COMMENT BELOW • DAY 2 SOON."),
    ]
    with open(os.path.join(OUT, "storyboard.txt"), "w") as fh:
        fh.write(f"STORYBOARD — {S.TITLE}  (1920x1080, 30 fps, {f.duration:.1f} s)\n\n")
        fh.write("Characters: Pip (original hero: orange hoodie, star logo, brown hair, teal pants), original\n"
                 "zombies (grey-green skin, ragged brown shirt), creepers. One continuous story in 6 chapters.\n\n")
        for li, txt in beats:
            ts = f.ls[li]
            fh.write(f"{int(ts // 60)}:{ts % 60:05.2f}  {txt}\n")
        fh.write("\nMusic (original, synthesized): adventure → cave suspense → build drums → night suspense → "
                 "panic → epic chase → silence before the blast → resolve.\n")
    meta = f"""TITLE
{S.TITLE} (Minecraft)

ALTERNATE TITLES
- 10 Minutes, One Night, Zero Mistakes — Minecraft Survival Story
- I Lured an Entire Horde onto 6 TNT... (Minecraft First Night)

DESCRIPTION
I spawned in a brand-new world with exactly 10 minutes of daylight before my first night. No house, no tools, no weapons.

In this video I punch my first tree, craft a pickaxe, find a mysterious chest full of TNT (and a diamond!) in a cave, build a base with a watchtower and a moat... and then the night comes. Zombies, a creeper that blows a hole in my wall, and one very risky plan with six blocks of TNT.

Chapters
0:00 10 Minutes of Daylight
{int(f.card(2)[0] // 60)}:{int(f.card(2)[0] % 60):02d} The Cave
{int(f.card(3)[0] // 60)}:{int(f.card(3)[0] % 60):02d} The Base
{int(f.card(4)[0] // 60)}:{int(f.card(4)[0] % 60):02d} Nightfall
{int(f.card(5)[0] // 60)}:{int(f.card(5)[0] % 60):02d} The Trap
{int(f.card(6)[0] // 60)}:{int(f.card(6)[0] % 60):02d} Sunrise

What should I build with the diamond? Tell me in the comments — Day 2 is coming!

#minecraft #minecraftsurvival #minecraftstory

HASHTAGS
#minecraft #minecraftsurvival #minecraftstory #firstnight #tnt #creeper #zombie #gaming

TAGS
minecraft, minecraft survival, minecraft first night, minecraft story, minecraft animation, minecraft tnt,
tnt trap, creeper, zombie horde, minecraft base, minecraft moat, minecraft diamond, survival challenge,
minecraft challenge, 10 minutes

THUMBNAIL CONCEPT
Night field mid-explosion: a giant orange TNT fireball with zombies and blocks flying, the base and watchtower
silhouetted. Big tilted text "10 MINUTES / TO SURVIVE" (yellow/white) top-left, "NIGHT 1" bottom-right,
shocked emoji and a TNT emoji. High contrast, saturated.

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
    rep = qc.report(path, W, H, 280, 330)
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
