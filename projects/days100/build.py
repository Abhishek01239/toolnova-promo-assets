"""Build the long-form film '100 Days' (1920x1080, ~10 min).

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

OUT = os.environ.get("MC_OUTPUT", "/home/user/output/days100")
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
    f._r = r            # statics are registered lazily by Film.ensure()


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


PI_ = np.pi


def build_thumbnail():
    f, r = make_film(1280, 720)
    import world as WD
    Sx = WD.SPAWN
    # left: day 1 at the dirt hut (dawn, zombies burning)
    t1 = f.T("burn", 0.75)
    sh1 = f.shot_at(t1)
    p1 = next(a for a in sh1.actors if a.kind == "pip").pos(t1)
    cam1 = dict(eye=np.array(p1 + (3.6, 1.2, -4.6), np.float32), target=np.array(p1 + (-1.6, 1.5, 0.8), np.float32), fov=55, roll=0.0)
    fs = f.frame_spec(t1, cam1)
    fs["post"].update(saturation=1.25, contrast=1.1, vignette=0.45, bloom=0.4)
    left = Image.fromarray(r.render(fs))
    # right: day 100 at the castle (golden hour), Pip in diamond armor with Biscuit
    from mcengine.models import pip_vertices
    from mcengine import models100 as M
    t2 = f.T("reveal", 0.4)
    cam2 = dict(eye=np.array([12.6, 12.9, -7.4], np.float32), target=np.array([12.3, 14.6, 1.0], np.float32), fov=64, roll=0.0)
    fs = f.frame_spec(t2, cam2)
    fs["dynamic"] = np.concatenate([pip_vertices((12.1, 11.0, -2.8), PI_ + 0.1, {"item": "sword:ench", "wave": 1.0, "wave_phase": 1.0}, skin="pip_dia"),
                                    M.wolf_vertices((13.1, 11.0, -3.4), PI_ - 0.2, {"sit": 1.0, "tail": 2.0}, tame=True)])
    fs["post"].update(saturation=1.3, contrast=1.1, vignette=0.4, bloom=0.4)
    right = Image.fromarray(r.render(fs))
    img = Image.new("RGB", (1280, 720))
    img.paste(left.crop((320, 0, 960, 720)), (0, 0))
    img.paste(right.crop((320, 0, 960, 720)), (640, 0))
    img = img.convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    from PIL import ImageDraw
    d = ImageDraw.Draw(layer)
    d.polygon([(628, 0), (652, 0), (660, 720), (620, 720)], fill=(255, 255, 255, 255))
    ov.draw_text(layer, "100 DAYS", (640, 105), 150, fill=(255, 225, 40), stroke=16)
    ov.draw_text(layer, "DAY 1", (300, 640), 96, fill=(255, 255, 255), rot=4, stroke=12)
    ov.draw_text(layer, "DAY 100", (1010, 650), 104, fill=(110, 235, 255), rot=-4, stroke=14)
    ov.paste_emoji(layer, "➡️", (640, 420), 120)
    img.alpha_composite(layer)
    img.convert("RGB").save(os.path.join(OUT, "thumbnail.png"))
    print("thumbnail saved")


def _ts(x):
    return f"{int(x // 60)}:{int(x % 60):02d}"


STORY = {
    1: "Pip spawns in the middle of nowhere: forest, river and a big hill. First tree, crafting table, wooden and stone tools. The sun sets.",
    2: "A dirt hut built in seconds. Zombies groan outside, a spider climbs the roof. Sunrise: the zombies burn. Night 1 survived.",
    3: "A real wooden house by the river (time-lapse build), a door, sheep in the meadow, a bed that skips the night, a chest.",
    4: "Hoe, farmland, a water canal, golden wheat, bread. Cows follow the wheat home into a fence pen; a baby cow is born.",
    5: "A cave in the hill: coal, torches, iron, a furnace, an iron pickaxe and a full set of iron armor. A cave zombie, three hits.",
    6: "Bones, a wolf in the forest, two failed tries, then hearts and a red collar: Biscuit. He follows Pip home and saves him from a zombie.",
    7: "Peaceful days... until a creeper blows half the wooden house into the sky. Biscuit is okay. A promise: rebuild in stone.",
    8: "Cobblestone from the hill, a stone house, a wall around the base, torches everywhere. A creeper gets stuck at the wall.",
    9: "A staircase down to deepslate, a lava lake, a bridge of obsidian made with a water bucket, a fall into lava, six diamonds.",
    10: "A boat trip down the river with Biscuit to a village. Farmer, librarian, blacksmith, kids. Wheat traded for emeralds.",
    11: "Night at the village: ten zombies. Villagers hide; Pip and Biscuit fight in the square and save a villager behind a door.",
    12: "A lighthouse on the hill, seen from the village. Day 50 fireworks and the halfway stats. A glimpse of the Nether.",
    13: "Obsidian from a lava pool, a portal frame, flint and steel: the purple portal lights up. Biscuit stays home.",
    14: "The Nether: red fog and a lava ocean. Netherrack burns forever, water evaporates. A ghast's fireball leaves Pip on 3 hearts.",
    15: "Round two: Pip hits the ghast's fireball back at it. Glowstone mined and put on top of the lighthouse.",
    16: "Strip mining for 24 diamonds, full diamond armor, an enchanting table and a glowing Sharpness III sword.",
    17: "The castle: 30x30 stone brick walls, four towers, the keep, a moat and one bridge. Ten days, thousands of blocks.",
    18: "Day 85: a horde of zombies, spiders and creepers. Spiders climb the walls, a creeper blows a hole, a last stand in the gap. Sunrise.",
    19: "Repairs, a garden and baby cows. A visit to the village. Back to the dirt hut where it all started.",
    20: "The last night on the keep with Biscuit. Sunrise on Day 100, a look back at everything, and thanks to Biscuit.",
}


def docs(f):
    with open(os.path.join(OUT, "script.txt"), "w") as fh:
        fh.write(f"{S.TITLE}  —  long-form narration script (~{f.duration / 60:.1f} min)\n")
        fh.write("Narrator: Pip (first person). Voice: free local Kokoro 'puck_classic' (same as Hardcore Parts 2-5, First Night "
                 "and Day 2). Bracketed text = delivery direction (not spoken, not subtitled).\n")
        li = 0
        for num, label, title, lines in S.CHAPTERS:
            fh.write(f"\nCHAPTER {num}: {label} — {title}\n")
            for _, line in lines:
                fh.write(f"[{int(f.ls[li] // 60)}:{f.ls[li] % 60:04.1f}]  {line}\n")
                li += 1
    with open(os.path.join(OUT, "storyboard.txt"), "w") as fh:
        fh.write(f"STORYBOARD — {S.TITLE}  (1920x1080, 30 fps, {f.duration:.1f} s, {len(f.shots)} shots)\n\n")
        fh.write("Characters: Pip (original hero: orange hoodie, brown hair, teal pants; later iron and diamond armor),\n"
                 "Biscuit (his tamed wolf), original zombies, creepers, spiders, villagers and a ghast.\n"
                 "Sets: one procedural world (forest, river, hill, village), the evolving base (dirt hut -> wooden house -> stone base\n"
                 "-> castle), a cave/lava set and the Nether. A day counter is always on screen.\n\n")
        for num, label, title, lines in S.CHAPTERS:
            t0 = 0 if num == 1 else f.card(num)[0]
            fh.write(f"{_ts(t0)}  {label} — {title}: {STORY[num]}\n")
        fh.write("\nMusic (original, synthesized): adventure, build, suspense, panic, resolve.\n")
    chap = "\n".join(f"{_ts(0 if c['num'] == 1 else c['card_start'])} {c['label'].title()}: {c['title'].title()}" for c in f.chapters)
    meta = f"""TITLE
{S.TITLE}

ALTERNATE TITLES
- I Spawned in the Middle of Nowhere and Survived 100 Days in Minecraft
- 100 Days in Minecraft: From a Dirt Hut to a Castle

DESCRIPTION
I spawned in the middle of nowhere with nothing. No house, no tools, no food. My goal: survive 100 days.

In this video I build a dirt hut on night one, a wooden house, a farm and an animal pen, explore a cave for iron, tame a wolf named Biscuit, lose half my house to a creeper, rebuild in stone, cross a lava lake for diamonds, find a village and defend it from a zombie siege, build a lighthouse, light a Nether portal, get beaten by a ghast and come back to beat it, craft full diamond armor, build a castle and survive the biggest horde of the whole run.

Chapters
{chap}

Should I keep going to 200 days? Tell me in the comments!

#minecraft #100days #minecraftsurvival

HASHTAGS
#minecraft #100days #minecraftsurvival #minecraftstory #castle #nether #wolf #gaming

TAGS
minecraft, 100 days, 100 days minecraft, minecraft 100 days survival, minecraft survival, minecraft story, minecraft animation,
dirt hut, minecraft castle, minecraft village, nether portal, ghast, diamond armor, tame wolf, creeper, zombie horde, survival challenge

THUMBNAIL CONCEPT
Split screen. Left: "DAY 1" — Pip at the dirt hut at dawn with zombies burning. Right: "DAY 100" — Pip in diamond armor with Biscuit
inside his castle. Big yellow "100 DAYS" on top, arrow in the middle.

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
    rep = qc.report(path, W, H, 1100, 1300)
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
