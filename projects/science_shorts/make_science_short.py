"""60-second science Short: 'A Planet Being Torn Apart' (WASP-12b).
Female free Kokoro voice (toolnova engine). Visuals drawn in skia. Score + SFX in numpy.
Facts checked against NASA exoplanet catalog text and arXiv papers (see script notes).
Output: /home/claude/wasp-explainer/out/wasp12b_short.mp4 (1080x1920, 30 fps)."""
import os, sys, math, subprocess
import numpy as np
import skia
import soundfile as sf
from scipy.signal import butter, lfilter

ENGINE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ENGINE)
os.environ.setdefault("KOKORO_DIR", os.path.join(ENGINE, "models"))
from mcengine import tts

OUT = os.environ.get("SCI_OUTPUT", os.path.join(os.getcwd(), "output"))
os.makedirs(OUT, exist_ok=True)
W, H, FPS = 1080, 1920, 30

LINES = [
    "[curious] This planet has a year that lasts just over one Earth day.",
    "[determined] It's called WASP-12b. It orbits so close to its star that NASA says it's being torn apart.",
    "[suspenseful] It circles its star in about 1.1 days. Earth takes 365 days to go around the sun.",
    "[determined] The planet is about 1.9 times the radius of Jupiter, puffed up by the intense heat.",
    "[curious] And its orbit is shrinking. Scientists have measured it getting closer to its star, year after year.",
    "[suspenseful] WASP-12b is the only hot Jupiter confirmed to have a decaying orbit.",
    "[whispering] One estimate says the planet may have as little as 3 million years left.",
    "[determined] Picture a whole planet, slowly pulled apart by its own sun. Slowly... and then all at once.",
    "[curious] Follow for one science fact every day!",
]
SCENE = ["orbit", "planet", "orbit", "planet", "orbit", "planet", "stretch", "stretch", "planet"]
HIGHLIGHT = {"year", "torn", "apart", "1.1", "1.9", "shrinking", "decaying", "3", "million", "sun", "planet", "orbit"}
COLOR_HI = {"torn": (255, 90, 150), "apart": (255, 90, 150), "shrinking": (255, 210, 80), "decaying": (255, 210, 80),
            "million": (255, 210, 80), "1.1": (90, 200, 255), "1.9": (90, 200, 255), "sun": (255, 170, 60)}

# ---------------- voice (female) ----------------
try:
    tts.PRESETS["girl"] = {"blend": {"af_bella": 0.6, "af_heart": 0.4}, "pitch": 0.0, "speed": 1.0}
    voice, SR, words = tts.synthesize(LINES, voice="girl")
except Exception as e:
    print("blend failed, using af_heart:", e)
    tts.PRESETS["girl"] = {"blend": {"af_heart": 1.0}, "pitch": 0.0, "speed": 1.0}
    voice, SR, words = tts.synthesize(LINES, voice="girl")
voice = np.asarray(voice, np.float32)
D = len(voice) / SR + 1.0
print("voice seconds:", round(len(voice) / SR, 2), "duration:", round(D, 2))

line_t0 = {}
for w in words:
    line_t0[w["line"]] = min(line_t0.get(w["line"], 1e9), w["start"])
scene_t = []
for li in range(len(LINES)):
    s = max(0, line_t0[li] - 0.2)
    e = (line_t0[li + 1] - 0.2) if li + 1 < len(LINES) else D
    scene_t.append((s, e, SCENE[li], li))

FONT_B = skia.Typeface.MakeFromName("Inter", skia.FontStyle.Bold())
FONT_S = skia.Typeface.MakeFromName("Inter", skia.FontStyle.Normal())
rng = np.random.default_rng(21)
STARS = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.5, 2.0)) for _ in range(160)]

def col(r, g, b, a=255):
    return skia.Color(r, g, b, a)

def glow(c, x, y, r, color, blur, alpha=255):
    p = skia.Paint(AntiAlias=True, Color=col(*color, alpha))
    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.BlurStyle.kNormal_BlurStyle, blur))
    c.drawCircle(x, y, r, p)

def space(c, t):
    g = skia.GradientShader.MakeLinear([skia.Point(0, 0), skia.Point(0, H)],
                                       [col(4, 6, 20), col(10, 8, 30), col(2, 2, 8)])
    c.drawRect(skia.Rect(0, 0, W, H), skia.Paint(Shader=g))
    for x, y, s in STARS:
        a = int(110 + 120 * math.sin(t * 1.5 + x))
        c.drawCircle(x, y, s, skia.Paint(Color=col(220, 225, 255, max(40, min(255, a)))))

def star(c, x, y, r):
    glow(c, x, y, r * 2.6, (255, 190, 90), r * 0.9, 150)
    g = skia.GradientShader.MakeRadial(skia.Point(x - r * 0.3, y - r * 0.3), r * 1.2,
                                       [col(255, 250, 220), col(255, 170, 60), col(200, 90, 20)],
                                       [0.0, 0.6, 1.0], skia.TileMode.kClamp)
    c.drawCircle(x, y, r, skia.Paint(AntiAlias=True, Shader=g))

def planet(c, x, y, r, stretch=0.0, t=0.0):
    """Shaded hot gas giant: radial shading, swirl bands, a glowing night-side rim."""
    if stretch > 0:  # tidal stretch: ellipse along the star direction
        c.save(); c.translate(x, y); c.rotate(-20); c.scale(1 + stretch, 1 - stretch * 0.5)
        c.translate(-x, -y)
    g = skia.GradientShader.MakeRadial(skia.Point(x - r * 0.4, y - r * 0.4), r * 1.6,
                                       [col(255, 210, 120), col(230, 110, 40), col(70, 20, 10)],
                                       [0.0, 0.55, 1.0], skia.TileMode.kClamp)
    c.drawCircle(x, y, r, skia.Paint(AntiAlias=True, Shader=g))
    for i in range(9):
        yy = y - r + (i + 0.5) * (2 * r / 9) + 6 * math.sin(t * 0.8 + i)
        half = math.sqrt(max(0, r * r - (yy - y) ** 2))
        c.drawRect(skia.Rect(x - half, yy - 6, x + half, yy + 6), skia.Paint(AntiAlias=True, Color=col(255, 160, 70, 60)))
    glow(c, x, y, r * 1.05, (255, 140, 60), r * 0.12, 60)
    if stretch > 0:
        c.restore()

def orbit_scene(c, t, prog):
    space(c, t)
    cx, cy = W / 2, H * 0.36
    star(c, cx - 210, cy - 150, 110)
    ex, ey = 380, 180
    c.drawOval(skia.Rect.MakeLTRB(cx - ex, cy - ey + 40, cx + ex, cy + ey + 40),
               skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=3, Color=col(120, 180, 255, 130),
                          PathEffect=skia.DashPathEffect.Make([18, 14], 0)))
    ang = t * 2.3
    px = cx + math.cos(ang) * ex * 0.6 - 0
    py = cy + 40 + math.sin(ang) * ey * 0.6
    planet(c, px, py, 46, 0.0, t)
    glow(c, px, py, 60, (255, 120, 60), 16, 90)
    # labels
    c.drawString("1 ORBIT ≈ 1.1 EARTH DAYS", W / 2 - 300, H * 0.53, skia.Font(FONT_B, 40),
                 skia.Paint(AntiAlias=True, Color=col(255, 210, 80)))

def planet_scene(c, t, prog):
    space(c, t)
    glow(c, W * 0.22, H * 0.36, 140, (255, 200, 110), 60, 150)
    star(c, W * 0.22, H * 0.36, 90)
    planet(c, W * 0.6, H * 0.36, 260 + 10 * math.sin(t), 0.0, t)
    c.drawString("WASP-12b", W * 0.6 - 130, H * 0.36 + 320, skia.Font(FONT_B, 50), skia.Paint(AntiAlias=True, Color=col(255, 255, 255)))
    c.drawString("A HOT GAS GIANT", W * 0.6 - 150, H * 0.36 + 365, skia.Font(FONT_S, 30), skia.Paint(AntiAlias=True, Color=col(180, 190, 220)))

def stretch_scene(c, t, prog):
    space(c, t)
    star(c, W * 0.14, H * 0.30, 160)
    s = min(1.0, prog * 1.5)
    planet(c, W * 0.6, H * 0.36, 250, stretch=0.10 * s, t=t)
    # tidal arrows pointing outward from the planet's pole
    for sign in (-1, 1):
        x0 = W * 0.6 + sign * 300
        c.drawLine(x0, H * 0.36, x0 + sign * 70, H * 0.36,
                   skia.Paint(AntiAlias=True, StrokeWidth=7, Color=col(90, 200, 255, 230)))
    c.drawString("TIDAL FORCES STRETCH IT", W / 2 - 330, H * 0.53, skia.Font(FONT_B, 40),
                 skia.Paint(AntiAlias=True, Color=col(90, 200, 255)))

def draw_disclaimer(c):
    c.drawString("SCIENTIFIC ILLUSTRATION · NOT REAL FOOTAGE", W / 2 - 330, H * 0.93, skia.Font(FONT_S, 26),
                 skia.Paint(AntiAlias=True, Color=col(150, 160, 190)))

def draw_captions(c, t, li):
    cur = [w for w in words if w["line"] == li and w["start"] <= t]
    if not cur:
        return
    recent = cur[-3:]
    raw = [w["word"] for w in recent]
    font = skia.Font(FONT_B, 96)
    total = sum(font.measureText(r.upper()) for r in raw) + 26 * (len(raw) - 1)
    size = int(96 * min(1.0, 900 / max(1, total)))
    font = skia.Font(FONT_B, size)
    widths = [font.measureText(r.upper()) for r in raw]
    total = sum(widths) + 26 * (len(raw) - 1)
    x = W / 2 - total / 2
    y = 1180
    for i, (r, wid) in enumerate(zip(raw, widths)):
        age = t - recent[i]["start"]
        s = 1.0 + 0.16 * max(0, 1 - age / 0.12)
        key = r.strip(".,!?'\"").lower()
        rgb = COLOR_HI.get(key)
        color = col(*rgb) if rgb else col(255, 255, 255)
        c.save()
        c.translate(x + wid / 2, y)
        c.scale(s, s)
        c.translate(-(x + wid / 2), -y)
        if rgb:
            glow(c, x + wid / 2, y - 30, 80, rgb, 22, 130)
        c.drawString(r.upper(), x, y, font, skia.Paint(AntiAlias=True, Color=color))
        c.restore()
        x += wid + 26

def render_frame(t):
    surf = skia.Surface(W, H)
    c = surf.getCanvas()
    s0, s1, sname, li = next((s for s in scene_t if s[0] <= t < s[1]), scene_t[-1])
    prog = (t - s0) / max(0.01, s1 - s0)
    if sname == "orbit":
        orbit_scene(c, t, prog)
    elif sname == "planet":
        planet_scene(c, t, prog)
    else:
        stretch_scene(c, t, prog)
    draw_disclaimer(c)
    draw_captions(c, t, li)
    return surf

def make_audio():
    N = int(D * SR)
    t = np.arange(N) / SR
    lp = lambda x, f: lfilter(*butter(2, f / (SR / 2)), x)
    drone = lp(np.sin(2*np.pi*55*t) + 0.5*np.sin(2*np.pi*82.4*t), 200) * (0.5 + 0.2*np.sin(2*np.pi*0.1*t))
    pad = sum(np.sin(2*np.pi*f*t + 0.4*np.sin(2*np.pi*3.3*t)) for f in (110.0, 164.8, 220.0)) / 3
    pad = lp(pad, 900) * np.clip(t / 2.0, 0, 1) * 0.16
    tick = np.zeros(N)
    for bt in np.arange(0.5, D, 0.5):
        i0 = int(bt * SR); n = int(0.03 * SR)
        if i0 + n < N:
            tt = np.arange(n) / SR
            tick[i0:i0+n] += np.sin(2*np.pi*1800*tt) * np.exp(-tt/0.008) * 0.04
    score = drone * 0.12 + pad + tick
    v = np.zeros(N); v[:min(N, len(voice))] = voice[:N]
    env = np.clip(lp(np.abs(v), 12) * 6, 0, 1)
    score = score * (1.0 - 0.5 * env)
    music = np.stack([score, np.roll(score, 180)], 1) * 0.6
    vo = np.stack([v, v], 1)
    mix = music + vo * 0.95
    mix /= max(1e-6, np.max(np.abs(mix))) * 1.12
    return mix.astype(np.float32)

if __name__ == "__main__":
    mix = make_audio()
    wav = os.path.join(OUT, "mix.wav")
    sf.write(wav, mix, SR, subtype="PCM_16")
    total = int(math.ceil(D * FPS))
    silent = os.path.join(OUT, "video_silent.mp4")
    ff = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                           "-an", silent], stdin=subprocess.PIPE)
    thumb_saved = False
    for f in range(total):
        t = f / FPS
        arr = np.asarray(render_frame(t).makeImageSnapshot().toarray(colorType=skia.ColorType.kRGBA_8888_ColorType), dtype=np.uint8)
        if not thumb_saved and t >= 3.0:
            from PIL import Image
            Image.fromarray(arr[:, :, :3]).resize((1280, 720)).save(os.path.join(OUT, "thumb.png"))
            thumb_saved = True
        ff.stdin.write(arr.tobytes())
    ff.stdin.close(); ff.wait()
    final = os.path.join(OUT, "wasp12b_short.mp4")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
                    "-shortest", "-movflags", "+faststart", final], check=True)
    print("done", final)
