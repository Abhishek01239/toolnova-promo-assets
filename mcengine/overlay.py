"""2D motion-graphics overlay drawn with Pillow: subtitles, titles, emoji, HUD."""
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from . import textures as tx
from .anim import ease_out_back, clamp01, ease_out

FONT_BLACK = "/usr/share/fonts/opentype/inter/InterDisplay-Black.otf"
FONT_BOLD = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FONT_EMOJI = "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"
_fonts = {}
_emoji_cache = {}
_icon_cache = {}


def font(size, path=FONT_BLACK):
    key = (path, int(size))
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, int(size))
    return _fonts[key]


def emoji_image(ch, size):
    key = (ch, int(size))
    if key not in _emoji_cache:
        f = ImageFont.truetype(FONT_EMOJI, 109)
        im = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((10, 10), ch, font=f, embedded_color=True)
        bb = im.getbbox() or (0, 0, 1, 1)
        im = im.crop(bb)
        s = size / max(im.size)
        _emoji_cache[key] = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    return _emoji_cache[key]


def paste_emoji(layer, ch, center, size, rot=0.0, alpha=1.0):
    im = emoji_image(ch, size)
    if rot:
        im = im.rotate(rot, expand=True, resample=Image.BICUBIC)
    if alpha < 1:
        a = np.asarray(im).copy()
        a[..., 3] = (a[..., 3] * alpha).astype(np.uint8)
        im = Image.fromarray(a)
    layer.alpha_composite(im, (int(center[0] - im.width / 2), int(center[1] - im.height / 2)))


def text_size(txt, f, stroke=0):
    bb = f.getbbox(txt, stroke_width=stroke)
    return bb[2] - bb[0], bb[3] - bb[1], bb


def draw_text(layer, txt, center, size, fill=(255, 255, 255), stroke=None, stroke_fill=(0, 0, 0),
              alpha=1.0, scale=1.0, shadow=True, path=FONT_BLACK, rot=0.0):
    """Draw centred outlined text. Renders on a temp image so it can be scaled/rotated."""
    sz = max(6, int(size * scale))
    f = font(sz, path)
    st = stroke if stroke is not None else max(3, sz // 9)
    w, h, bb = text_size(txt, f, st)
    pad = st + 12
    im = Image.new("RGBA", (w + pad * 2 + 10, h + pad * 2 + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    org = (pad - bb[0], pad - bb[1])
    if shadow:
        d.text((org[0] + sz * 0.05, org[1] + sz * 0.07), txt, font=f, fill=(0, 0, 0, 150),
               stroke_width=st, stroke_fill=(0, 0, 0, 150))
    d.text(org, txt, font=f, fill=tuple(fill) + (255,), stroke_width=st, stroke_fill=tuple(stroke_fill) + (255,))
    if rot:
        im = im.rotate(rot, expand=True, resample=Image.BICUBIC)
    if alpha < 1:
        a = np.asarray(im).copy()
        a[..., 3] = (a[..., 3] * clamp01(alpha)).astype(np.uint8)
        im = Image.fromarray(a)
    layer.alpha_composite(im, (int(center[0] - im.width / 2), int(center[1] - im.height / 2)))
    return w


# ------------------------------------------------------------------ subtitles
class Subtitles:
    """Shorts-style captions: 1-3 words at a time, active word highlighted & popped.

    words: list of {"word", "start", "end"}; keywords: {UPPERCASE_WORD: (r,g,b)}
    """

    def __init__(self, words, keywords=None, max_words=3, max_chars=16, y=1150, size=104,
                 highlight=(255, 221, 0), cx=540, max_w=960):
        self.cx, self.max_w = cx, max_w
        self.words = words
        self.kw = {k.upper(): v for k, v in (keywords or {}).items()}
        self.y, self.size, self.hl = y, size, highlight
        self.chunks = []
        cur = []
        for i, w in enumerate(words):
            cur.append(i)
            txt = " ".join(words[j]["word"] for j in cur)
            end_punct = re.search(r"[.!?,…]$", w["word"]) is not None
            nxt_len = len(words[i + 1]["word"]) if i + 1 < len(words) else 0
            if len(cur) >= max_words or end_punct or len(txt) + nxt_len + 1 > max_chars:
                self.chunks.append(cur)
                cur = []
        if cur:
            self.chunks.append(cur)

    @staticmethod
    def clean(w):
        return re.sub(r"[^\w'?!\-]", "", w).upper()

    def draw(self, layer, t):
        words = self.words
        for ci, ch in enumerate(self.chunks):
            t0 = words[ch[0]]["start"] - 0.05
            t1 = words[ch[-1]]["end"] + 0.25
            if ci + 1 < len(self.chunks):
                t1 = min(t1, words[self.chunks[ci + 1][0]]["start"] - 0.05)
            if not (t0 <= t < t1):
                continue
            appear = ease_out_back(clamp01((t - t0) / 0.12))
            items = []
            for j in ch:
                w = words[j]
                txt = self.clean(w["word"])
                if not txt:
                    continue
                active = w["start"] - 0.03 <= t < max(w["end"], w["start"] + 0.12)
                said = t >= w["start"]
                key = re.sub(r"[^A-Z]", "", txt)
                col = self.kw.get(key, None)
                if col is None:
                    col = self.hl if active else (255, 255, 255)
                pop = 1.0
                if active:
                    pop = 1.0 + 0.16 * (1 - clamp01((t - w["start"]) / 0.18))
                elif key in self.kw:
                    pop = 1.08
                alpha = 1.0 if said else 0.85
                items.append((txt, col, pop, alpha))
            if not items:
                return
            # layout on one line; shrink if too wide
            size = self.size * (0.7 + 0.3 * appear)
            gap = size * 0.28
            widths = [text_size(it[0], font(size * it[2]), int(size // 9))[0] for it in items]
            total = sum(widths) + gap * (len(items) - 1)
            if total > self.max_w:
                k = self.max_w / total
                size *= k
                gap *= k
                widths = [wd * k for wd in widths]
                total = self.max_w
            x = self.cx - total / 2
            for (txt, col, pop, alpha), wd in zip(items, widths):
                draw_text(layer, txt, (x + wd / 2, self.y), size, fill=col, scale=pop, alpha=alpha)
                x += wd + gap
            return


# ------------------------------------------------------------------ titles
def title_pop(layer, txt, t, t0, t1, center, size, fill=(255, 255, 255), rot=0.0, emoji=None,
              emoji_side="left", stroke_fill=(0, 0, 0)):
    """Text that pops in with overshoot, wobbles slightly, and fades out."""
    if not (t0 <= t < t1):
        return
    a = ease_out_back(clamp01((t - t0) / 0.22), 2.2)
    out = clamp01((t1 - t) / 0.18)
    wob = np.sin((t - t0) * 7) * 1.5
    w = draw_text(layer, txt, center, size, fill=fill, scale=max(0.05, a), alpha=out, rot=rot + wob,
                  stroke_fill=stroke_fill)
    if emoji:
        dx = (w * a / 2 + size * 0.75) * (-1 if emoji_side == "left" else 1)
        paste_emoji(layer, emoji, (center[0] + dx, center[1]), size * 1.1 * max(a, 0.05),
                    rot=-wob * 3, alpha=out)


# ------------------------------------------------------------------ HUD
HEART = [
    "..XX.XX..",
    ".XrrXrrX.",
    "XrwrrrrrX",
    "XrrrrrrrX",
    ".XrrrrrX.",
    "..XrrrX..",
    "...XrX...",
    "....X....",
]


def _heart_img(px, kind="full"):
    pal = {"X": (20, 20, 20, 255), "r": (220, 30, 40, 255), "w": (255, 200, 200, 255), ".": (0, 0, 0, 0)}
    if kind == "empty":
        pal = {"X": (20, 20, 20, 255), "r": (60, 30, 30, 200), "w": (60, 30, 30, 200), ".": (0, 0, 0, 0)}
    h, w = len(HEART), len(HEART[0])
    a = np.zeros((h, w, 4), np.uint8)
    for y, row in enumerate(HEART):
        for x, c in enumerate(row):
            a[y, x] = pal[c]
    if kind == "half":
        e = _heart_arr("empty")
        a[:, w // 2 + 1:] = e[:, w // 2 + 1:]
    return Image.fromarray(a).resize((w * px, h * px), Image.NEAREST)


def _heart_arr(kind):
    return np.asarray(_heart_img(1, kind))


def draw_hearts(layer, health, center, px=6, n=10, shake_amt=0.0, t=0.0):
    """health in hearts (0..10, halves allowed)."""
    w = 9 * px + px
    x0 = center[0] - n * w / 2
    for i in range(n):
        kind = "full" if health >= i + 1 else ("half" if health >= i + 0.5 else "empty")
        dy = 0
        if shake_amt:
            dy = np.sin(t * 40 + i * 1.3) * shake_amt * px
        layer.alpha_composite(_heart_img(px, kind), (int(x0 + i * w), int(center[1] + dy)))


def iso_block_icon(name, size):
    """Isometric block icon built from the procedural atlas textures."""
    key = (name, size)
    if key in _icon_cache:
        return _icon_cache[key]
    top_t, side_t, _, _ = tx.BLOCKS[name]

    def tile(n):
        return tx.tile_image(n, 64)

    S = 64
    out = Image.new("RGBA", (2 * S, 2 * S), (0, 0, 0, 0))
    top = tile(top_t).transform((2 * S, S), Image.AFFINE,
                                (0.5, 1.0, -S / 2, -0.5, 1.0, S / 2), Image.NEAREST)
    left = tile(side_t).transform((S, int(1.5 * S)), Image.AFFINE, (1, 0, 0, -0.5, 1, 0), Image.NEAREST)
    right = tile(side_t).transform((S, int(1.5 * S)), Image.AFFINE, (1, 0, 0, 0.5, 1, -S / 2), Image.NEAREST)
    shade = lambda im, k: Image.fromarray(np.dstack([(np.asarray(im)[..., :3] * k).astype(np.uint8),
                                                     np.asarray(im)[..., 3]]))
    out.alpha_composite(shade(left, 0.78), (0, S // 2))
    out.alpha_composite(shade(right, 0.6), (S, S // 2))
    out.alpha_composite(top, (0, 0))
    out = out.crop(out.getbbox()).resize((size, size), Image.NEAREST)
    _icon_cache[key] = out
    return out


def draw_hotbar(layer, items, selected, center, slot=96, alpha=1.0, emoji_items=None):
    """items: list of block names or emoji strings (prefix 'e:')."""
    n = 9
    x0 = int(center[0] - n * slot / 2)
    y0 = int(center[1] - slot / 2)
    d = ImageDraw.Draw(layer)
    d.rectangle([x0 - 6, y0 - 6, x0 + n * slot + 6, y0 + slot + 6], fill=(20, 20, 20, int(170 * alpha)))
    for i in range(n):
        x = x0 + i * slot
        d.rectangle([x + 3, y0 + 3, x + slot - 3, y0 + slot - 3], fill=(120, 120, 120, int(120 * alpha)),
                    outline=(60, 60, 60, int(255 * alpha)), width=3)
        if i < len(items) and items[i]:
            it = items[i]
            if it.startswith("e:"):
                paste_emoji(layer, it[2:], (x + slot / 2, y0 + slot / 2), slot * 0.62, alpha=alpha)
            elif it.startswith("t:"):
                ic = _item_icon(it, int(slot * 0.66))
                layer.alpha_composite(ic, (int(x + slot / 2 - ic.width / 2), int(y0 + slot / 2 - ic.height / 2)))
            else:
                ic = iso_block_icon(it, int(slot * 0.66))
                layer.alpha_composite(ic, (int(x + slot / 2 - ic.width / 2), int(y0 + slot / 2 - ic.height / 2)))
    xs = x0 + selected * slot
    d.rectangle([xs - 4, y0 - 4, xs + slot + 4, y0 + slot + 4], outline=(255, 255, 255, int(255 * alpha)), width=7)


def speed_lines(layer, t, strength, center=(540, 960), n=46, seed=3):
    if strength <= 0:
        return
    d = ImageDraw.Draw(layer)
    r = np.random.default_rng(seed + int(t * 30))
    for _ in range(int(n * strength)):
        a = r.uniform(0, 2 * np.pi)
        r0 = r.uniform(420, 700)
        r1 = r0 + r.uniform(250, 700)
        c, s = np.cos(a), np.sin(a)
        d.line([(center[0] + c * r0, center[1] + s * r0 * 1.6), (center[0] + c * r1, center[1] + s * r1 * 1.6)],
               fill=(255, 255, 255, int(r.uniform(60, 140) * strength)), width=int(r.uniform(3, 8)))


def depth_meter(layer, value, label, center, size=64, color=(255, 255, 255)):
    draw_text(layer, f"{label} {int(round(value))}", center, size, fill=color, path=FONT_BLACK)


def composite(frame, layer):
    """frame: HxWx3 uint8, layer: RGBA PIL image -> HxWx3 uint8"""
    a = np.asarray(layer)
    if a[..., 3].max() == 0:
        return frame
    al = a[..., 3:4].astype(np.float32) / 255.0
    out = frame.astype(np.float32) * (1 - al) + a[..., :3].astype(np.float32) * al
    return out.astype(np.uint8)


# ------------------------------------------------------------------ long-form graphics
def chapter_card(layer, t, t0, dur, number, title, W, H, color=(255, 196, 40)):
    """Pixel-block dissolve wipe that reveals a chapter title card, then dissolves out.
    Never fully black: the card is a dark-green panel with a subtle block pattern."""
    if not (t0 <= t < t0 + dur):
        return
    u = (t - t0) / dur
    cover = min(1.0, u / 0.18) if u < 0.5 else min(1.0, (1 - u) / 0.18)
    B = 80
    cols, rows = W // B + 1, H // B + 1
    rng = np.random.default_rng(7)
    order = rng.random((rows, cols))
    d = ImageDraw.Draw(layer)
    for r in range(rows):
        for c in range(cols):
            if order[r, c] < cover:
                shade = 34 + int(10 * ((r + c) % 2))
                d.rectangle([c * B, r * B, c * B + B - 1, r * B + B - 1], fill=(shade, 52 + shade // 3, 30, 255))
    if cover > 0.85:
        a = min(1.0, (cover - 0.85) / 0.15)
        pop = ease_out_back(clamp01((u - 0.12) / 0.25))
        draw_text(layer, f"CHAPTER {number}", (W / 2, H / 2 - 95), 64, fill=(230, 230, 230), alpha=a,
                  scale=max(pop, 0.05))
        draw_text(layer, title, (W / 2, H / 2 + 20), 120, fill=color, alpha=a, scale=max(pop, 0.05))


def countdown(layer, seconds, center, size=64, warn=60):
    m, sec = divmod(max(0, int(seconds)), 60)
    col = (255, 255, 255) if seconds > warn else (255, 90, 70)
    paste_emoji(layer, "☀️" if seconds > 0 else "🌙", (center[0] - size * 2.0, center[1]), size * 1.1)
    draw_text(layer, f"{m:02d}:{sec:02d}", center, size, fill=col)


def _item_icon(item, size):
    if item.startswith("e:"):
        return emoji_image(item[2:], int(size))
    if item.startswith("t:"):          # flat item from an atlas tile, e.g. "t:ingot"
        return tx.tile_image(item[2:], int(size * 0.8)).convert("RGBA")
    return iso_block_icon(item, int(size))


def crafting_grid(layer, center, grid, result, appear=1.0, cell=110):
    """3x3 crafting grid + arrow + result slot. grid: 9 item codes or None."""
    if appear <= 0.02:
        return
    s = cell * (0.6 + 0.4 * ease_out_back(appear))
    x0 = center[0] - s * 2.6
    y0 = center[1] - s * 1.5
    d = ImageDraw.Draw(layer)
    a = int(235 * min(1, appear * 1.5))
    d.rounded_rectangle([x0 - 24, y0 - 24, x0 + s * 5.3 + 24, y0 + s * 3 + 24], radius=18,
                        fill=(198, 198, 198, a), outline=(40, 40, 40, a), width=6)
    for i in range(9):
        cx, cy = x0 + (i % 3) * s, y0 + (i // 3) * s
        d.rectangle([cx + 4, cy + 4, cx + s - 4, cy + s - 4], fill=(139, 139, 139, a), outline=(60, 60, 60, a), width=3)
        if grid[i]:
            ic = _item_icon(grid[i], s * 0.7)
            layer.alpha_composite(ic, (int(cx + s / 2 - ic.width / 2), int(cy + s / 2 - ic.height / 2)))
    ax = x0 + s * 3.3
    d.polygon([(ax, y0 + s * 1.2), (ax + s * 0.6, y0 + s * 1.2), (ax + s * 0.6, y0 + s * 1.0),
               (ax + s * 1.0, y0 + s * 1.5), (ax + s * 0.6, y0 + s * 2.0), (ax + s * 0.6, y0 + s * 1.8),
               (ax, y0 + s * 1.8)], fill=(90, 90, 90, a))
    rx, ry = x0 + s * 4.3, y0 + s * 1.0
    d.rectangle([rx, ry, rx + s, ry + s], fill=(139, 139, 139, a), outline=(60, 60, 60, a), width=4)
    if result:
        ic = _item_icon(result, s * 0.75)
        layer.alpha_composite(ic, (int(rx + s / 2 - ic.width / 2), int(ry + s / 2 - ic.height / 2)))
