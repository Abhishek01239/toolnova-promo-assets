"""Build pipeline for the 'One Block Hardcore' Short.

usage: python build.py [stills t1,t2,...] [audio] [video] [thumb] [docs] [qc] [all]
"""
import json
import os
import shutil
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from mcengine import audio as A                 # noqa: E402
from mcengine import voice as V                 # noqa: E402
from mcengine import overlay as ov              # noqa: E402
from mcengine import qc                         # noqa: E402
from mcengine.anim import seg                   # noqa: E402
from mcengine.render import render_video, mux, to_mp3  # noqa: E402
from mcengine.renderer import Renderer          # noqa: E402
import story as S                               # noqa: E402

OUT = os.environ.get("MC_OUTPUT", os.path.join(HERE, "..", "..", "output"))
ASSETS = os.path.join(HERE, "assets")
WORK = os.path.join(HERE, "work")
W, H, FPS = 1080, 1920, 30
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def load_words():
    asr = json.load(open(os.path.join(ASSETS, "words.json")))["words"]
    words = V.align(S.LINES, asr)
    json.dump(words, open(os.path.join(WORK, "aligned_words.json"), "w"), indent=1)
    return words


def make_story(words, w=W, h=H):
    st = S.OneBlockHardcore(words, os.path.join(WORK, "cache"))
    r = Renderer(w, h)
    st._r = r
    st.build_static(r)
    return st, r


def zoom_punch(img, z):
    if z <= 1.001:
        return img
    h, w = img.shape[:2]
    cw, ch = int(w / z), int(h / z)
    x0, y0 = (w - cw) // 2, (h - ch) // 2
    return np.asarray(Image.fromarray(img[y0:y0 + ch, x0:x0 + cw]).resize((w, h), Image.BILINEAR))


def frame_fn_factory(st, r):
    cuts = st.cuts()

    def frame(t):
        img = r.render(st.frame_spec(t))
        z = 1.0
        for c in cuts:
            if c <= t < c + 0.2:
                z = 1 + 0.07 * (1 - seg(t, c, c + 0.2)) ** 2
        img = zoom_punch(img, z)
        layer = Image.new("RGBA", (img.shape[1], img.shape[0]), (0, 0, 0, 0))
        st.draw_overlay(t, layer)
        return ov.composite(img, layer)
    return frame


def build_audio(st):
    dur = st.duration
    voice = A.load_audio(os.path.join(ASSETS, "take1.mp3"))
    mx = A.Mixer(dur)
    mx.add_voice(0.0, voice, target_db=-16.0)
    music = A.compose(st.music_sections(), dur)
    mx.add_music(music, gain_db=-13.0)
    for t, name, g, kw in st.sfx_events():
        mx.add_sfx(t, name, g, **kw)
    mx.add_sfx(st.T["jump"], st.wind(A), -9)
    mix = mx.render(duck_db=-8.0)
    A.write_wav(os.path.join(WORK, "mix.wav"), mix)
    # clean narration deliverable (levelled voice only)
    vo = mx.voice / (np.abs(mx.voice).max() + 1e-9) * 0.89
    A.write_wav(os.path.join(WORK, "voice.wav"), vo[: int((st.T["end"] + 0.4) * A.SR)])
    to_mp3(os.path.join(WORK, "voice.wav"), os.path.join(OUT, "voiceover.mp3"))
    print("audio done", mix.shape)


def build_video(st, r):
    fn = frame_fn_factory(st, r)
    vid = os.path.join(WORK, "video_only.mp4")
    t0 = time.time()
    render_video(vid, fn, st.duration, W, H, FPS)
    print(f"video rendered in {time.time() - t0:.0f}s")
    mux(vid, os.path.join(WORK, "mix.wav"), os.path.join(OUT, "minecraft_video.mp4"))


def stills(st, r, times):
    fn = frame_fn_factory(st, r)
    ims = []
    for t in times:
        ims.append(Image.fromarray(fn(t)).resize((360, 640)))
    cols = min(6, len(ims))
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 360, rows * 640))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * 360, (i // cols) * 640))
    p = os.path.join(WORK, "stills.png")
    sheet.save(p)
    print("saved", p)


def build_thumbnail(words):
    """1280x720 YouTube thumbnail rendered from the 3D scene + bold text."""
    st, r = make_story(words, 1280, 720)
    T = st.T
    t = T["exploded"] + 0.5           # after the red hurt-flash, fireball still blooming
    pip, _, _ = st.pip_state(t)
    cam = dict(eye=pip + np.array([3.0, -0.4, 1.6], np.float32),
               target=pip + np.array([-1.6, 2.3, -0.9], np.float32), fov=72, roll=-0.1)
    fs = st.frame_spec(t, cam)
    fs["post"].update(white_flash=0.0, saturation=1.35, contrast=1.12, vignette=0.45, bloom=0.5)
    img = Image.fromarray(r.render(fs)).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ov.draw_text(layer, "1 BLOCK", (330, 130), 140, fill=(255, 225, 40), rot=4, stroke=15)
    ov.draw_text(layer, "HARDCORE", (330, 275), 100, fill=(255, 60, 50), rot=4, stroke=13)
    ov.draw_text(layer, "I JUMPED?!", (900, 625), 108, fill=(255, 255, 255), rot=-3, stroke=14)
    ov.paste_emoji(layer, "😱", (1150, 130), 160, rot=-10)
    ov.draw_hearts(layer, 0.5, (330, 365), px=5, n=10)
    img.alpha_composite(layer)
    img.convert("RGB").save(os.path.join(OUT, "thumbnail.png"))
    print("thumbnail saved")


def docs(st, words):
    T = st.T
    with open(os.path.join(OUT, "script.txt"), "w") as f:
        f.write("ONE BLOCK. HARDCORE.  —  YouTube Short narration script\n")
        f.write("Narrator: Pip (first person). Voice: ElevenLabs 'Liam', model eleven_v4.\n")
        f.write("Bracketed text = delivery direction (not spoken, not subtitled).\n\n")
        lt = V.line_times(words, len(S.LINES))
        for line, (a, b) in zip(S.LINES, lt):
            f.write(f"[{a:05.2f}-{b:05.2f}]  {line}\n")
    sb = [
        ("0:00", T["two0"], "HOOK. Extreme close-up on Pip's face, fast pull-back reveals he stands on ONE grass block in an empty sky. Title 'ONE BLOCK'."),
        (T["two0"], T["hard"], "High angle looking down past the block: the ground is 200 blocks below, hazy through clouds. Title '200 BLOCKS UP'."),
        (T["hard"], T["life"], "Low hero angle. Hardcore hearts slam onto the HUD, red flash, impact hit. Title 'HARDCORE 💀'."),
        (T["life"], T["but"], "Orbit (Pip looks around) → over-the-edge shot of the void → wide shot, Pip waves → vertigo dolly-zoom straight down with roll."),
        (T["but"], T["wood"], "Low orbit around the block: golden shimmer particles. Pip punches it; it breaks and slot-machine cycles through block types."),
        (T["wood"], T["little"], "Three quick cuts: WOOD, STONE, WATER — each break regenerates a new block; a water bucket flies into Pip's hand and the hotbar appears."),
        (T["little"], T["then"], "Time-lapse: the island builds itself block by block (platform, tree, plank house, pool). Camera cranes out over the clouds, DAY 1→3, sunset."),
        (T["then"], T["creeper"], "Night falls, stars and square moon. 'NIGHT 3'. Over-shoulder: Pip breaks the block, it shakes and leaks green fumes. Snap zoom on 'NOT', heartbeats."),
        (T["creeper"], T["sword"], "A Creeper bursts out of the block. Low close-up push-in, impact, music cuts out. 😱"),
        (T["sword"], T["so"], "'No sword' (🗡️❌, Pip checks his empty hand) → top-down: Creeper advances, Pip backs to the edge → over-the-edge drop shot."),
        (T["so"], T["island"], "Hotbar selects the water bucket; Creeper starts hissing and flashing. Pip turns to the void… and jumps."),
        (T["island"], T["second"], "Falling: camera below Pip looking up as the island EXPLODES (flash, fireball, smoke, flying blocks, shake, hearts drop to ½). Then top-down fall with Y-meter, speed lines, falling through clouds, orbiting tumble shot."),
        (T["second"], T["survived"], "Slow-motion near the ground, desaturated. '1 SECOND LEFT ⏱️', heartbeat. On 'DOWN!' Pip places water — splash, flash. 'CLUTCH!'"),
        (T["survived"], T["but2"], "Low orbit around Pip standing in the water, celebrating. 'SURVIVED ✅'. Giant blinking half-heart: '½ HEART LEFT'."),
        (T["but2"], T["think"], "Pip looks up; telephoto tilt to the smoking wreck in the sky ('HOME: GONE 💔'). Dawn breaks; Pip starts pillaring back up."),
        (T["think"], st.duration, "End card: 'PART 2?  FOLLOW TO FIND OUT 👇' as the camera cranes up. Pip waves."),
    ]
    with open(os.path.join(OUT, "storyboard.txt"), "w") as f:
        f.write("STORYBOARD — ONE BLOCK. HARDCORE.  (1080x1920, 30 fps, %.1f s)\n\n" % st.duration)
        f.write("Characters: Pip (original hero: orange hoodie with star logo, brown hair, teal pants);\n"
                "            Creeper-style mob. Setting: floating sky island at Y=200 above procedural hills.\n\n")
        for a, b, d in sb:
            fa = a if isinstance(a, str) else f"{int(a // 60)}:{a % 60:05.2f}"
            fb = f"{int(b // 60)}:{b % 60:05.2f}"
            f.write(f"{fa} – {fb}\n    {d}\n\n")
        f.write("Music (original, synthesized): adventure → build-up drums → suspense → silence → panic → "
                "freefall → silence (slow-mo) → warm resolve.\n")
        f.write("SFX (synthesized): whooshes, block dig/place/pop, shimmer, heartbeat, creeper hiss, explosion, "
                "wind, splash, impacts, ding.\n")
    meta = """TITLE
I Survived ONE Block in the Sky... in HARDCORE

ALTERNATE TITLES
- One Block, One Life, 200 Blocks Up (Minecraft Hardcore)
- A Creeper Spawned on My One Block Island...

DESCRIPTION
I spawned on a single block, 200 blocks up in the sky — in HARDCORE mode. One life, no ground, and a block that gives me something new every time I break it. I built a home in the clouds... until night three, when the block gave me a Creeper. 😱

Can a water bucket clutch save a 200-block fall? Watch to the end.

Part 2 is coming — follow so you don't miss the climb back up!

#minecraft #minecraftshorts #oneblock #hardcore #shorts

HASHTAGS
#minecraft #minecraftshorts #oneblock #skyblock #hardcore #creeper #mlg #waterbucket #gaming #shorts

TAGS
minecraft, minecraft shorts, one block, minecraft one block, skyblock, minecraft hardcore, hardcore survival,
creeper, creeper explosion, water bucket clutch, mlg water bucket, minecraft survival, minecraft story,
minecraft animation, minecraft challenge, gaming shorts

THUMBNAIL CONCEPT
Night sky. Pip falls toward the camera with arms flailing while the sky island explodes in an orange fireball
behind him, blocks flying. Big tilted text "1 BLOCK" (yellow) / "HARDCORE" (red) top-left, "I JUMPED?!"
bottom-right, a shocked emoji and a row of hardcore hearts down to half a heart. High contrast, saturated.

CATEGORY / SETTINGS
Gaming · Shorts · Not made for kids (general audience) · Language: English
Contains synthetic (AI-generated) narration voice — disclose as altered/synthetic content if required.

NOTES
All visuals are rendered procedurally with the project's own voxel engine (Minecraft-style, no game assets).
Music and sound effects are synthesized from code (original). Narration: ElevenLabs TTS.
"Minecraft" is a trademark of Mojang/Microsoft; this is unofficial fan content.
"""
    open(os.path.join(OUT, "metadata.txt"), "w").write(meta)
    print("docs written")


def run_qc(st):
    path = os.path.join(OUT, "minecraft_video.mp4")
    rep = qc.report(path, W, H, 30, 60)
    times = [0.5, 3, 4.6, 7.5, 10, 12.5, 14.8, 16.8, 19.5, 22, 24.8, 26.2, 27.3, 28.2, 29.2, 30.8, 32, 33.4,
             35, 36.6, 38.2, 39.5, 41, 42.8, 44.5, 46.2, 48.5]
    times = [x for x in times if x < st.duration]
    qc.contact_sheet(path, os.path.join(WORK, "qc_sheet.png"), times, thumb_w=200)
    json.dump(rep, open(os.path.join(WORK, "qc_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    words = load_words()
    need_r = any(a in args for a in ("video", "all", "stills"))
    st, r = (make_story(words) if need_r else (S.OneBlockHardcore(words, os.path.join(WORK, "cache")), None))
    if "stills" in args:
        ts = [float(x) for x in args[args.index("stills") + 1].split(",")]
        stills(st, r, ts)
    if "audio" in args or "all" in args:
        build_audio(st)
    if "video" in args or "all" in args:
        build_video(st, r)
    if "thumb" in args or "all" in args:
        build_thumbnail(words)
    if "docs" in args or "all" in args:
        docs(st, words)
    if "qc" in args or "all" in args:
        run_qc(st)
