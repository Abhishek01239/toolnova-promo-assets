"""The "cinematic" look. ON by default; set MC_STYLE=classic for the old bright look.

When active:
- lighting.environment() uses richer sky presets (deep blue day, orange-to-purple
  sunset, violet night) - see lighting.CINE_PRESETS;
- apply() (called by the renderer on every frame) adds a big glowing sun with halo,
  blocky lit clouds, warm rim light on edges, purple-tinted shadows, floating glowing
  motes near the camera, wide bloom, god rays and an orange/purple split-tone grade.

Story-level effects (white flash, red hurt tint, desaturated sad scenes) still win
over the style defaults, so episodes keep their dramatic moments.
"""
import os

import numpy as np

ACTIVE = os.environ.get("MC_STYLE", "cinematic").strip().lower() != "classic"

POST = dict(bloom=0.36, bloom_threshold=0.76, bloom_wide=0.12, god_rays=0.32,
            saturation=1.24, contrast=1.10, vignette=0.40,
            shadow_tint=(0.90, 0.88, 1.12), highlight_tint=(1.05, 1.01, 0.94))


def _warmth(sky):
    """0 = cool/neutral sky, 1 = strong sunset horizon."""
    hz = np.array(sky["horizon"])
    return float(np.clip((hz[0] - hz[2]) * 1.8, 0, 1))


def _night(sky):
    return float(np.clip(1.0 - sky.get("sun_vis", 1.0), 0, 1))


def motes(t, eye, target, n=46, seed=3):
    """Glowing dust/firefly particles floating in a box in front of the camera (additive)."""
    rng = np.random.default_rng(seed)
    eye = np.asarray(eye, np.float32)
    fwd = np.asarray(target, np.float32) - eye
    fwd /= np.linalg.norm(fwd) + 1e-6
    right = np.cross(fwd, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right) + 1e-6
    up = np.cross(right, fwd)
    base = rng.uniform([-1, -1, 0], [1, 1, 1], (n, 3)).astype(np.float32)
    ph = rng.uniform(0, 6.283, n).astype(np.float32)
    drift = np.stack([np.sin(t * 0.35 + ph) * 0.12, (t * 0.03 + ph * 0.1) % 1.0, np.cos(t * 0.27 + ph) * 0.1], 1)
    p = base + drift
    p[:, 1] = (p[:, 1] + 1) % 2 - 1
    dist = 4.0 + p[:, 2] * 16.0
    spread = dist * 0.55
    pos = eye + fwd * dist[:, None] + right * (p[:, 0] * spread)[:, None] + up * (p[:, 1] * spread * 1.4)[:, None]
    tw = 0.5 + 0.5 * np.sin(t * 2.3 + ph * 3)
    warm = rng.uniform(0, 1, n) < 0.55
    col = np.where(warm[:, None], [1.0, 0.72, 0.35], [0.78, 0.5, 1.0]).astype(np.float32)
    alpha = (0.25 + 0.5 * tw) * np.clip(1.2 - p[:, 2], 0.3, 1)
    size = (0.018 + 0.03 * rng.uniform(0, 1, n)) * (dist / 6.0) ** 0.6
    return np.column_stack([pos, size, col, alpha]).astype(np.float32)


def apply(fs, t):
    fs = dict(fs)
    sky = dict(fs["sky"])
    light = dict(fs["light"])
    warm, night = _warmth(sky), _night(sky)
    # sky: big square sun with a halo, blocky clouds lit by the sun
    sky.setdefault("sun_size", 0.10 + 0.04 * warm)
    sky.setdefault("halo", 0.35 + 0.35 * warm)
    sky.setdefault("halo_col", tuple(np.array([1.0, 0.82, 0.55]) * (1 - warm) + np.array([1.0, 0.5, 0.18]) * warm))
    sky.setdefault("clouds", 0.85 * (1 - night) + 0.35 * night)
    sky.setdefault("cloud_lit", tuple(np.array([1.0, 0.97, 0.92]) * (1 - warm) + np.array([1.0, 0.62, 0.3]) * warm))
    sky.setdefault("cloud_dark", tuple(np.array([0.62, 0.66, 0.9]) * (1 - warm) + np.array([0.42, 0.24, 0.55]) * warm
                                       * (1 - night) + np.array([0.12, 0.08, 0.24]) * night))
    # light: stronger directional sun, warm rim on edges, purple-tinted shadows
    light.setdefault("sun_power", 0.72)
    light.setdefault("rim", 0.40 + 0.30 * warm)
    light.setdefault("rim_col", tuple(np.array([1.0, 0.86, 0.6]) * (1 - warm) + np.array([1.0, 0.55, 0.22]) * warm
                                      if night < 0.5 else (0.62, 0.5, 1.0)))
    light.setdefault("shadow_col", (0.92, 0.88, 1.08))
    fs["sky"], fs["light"] = sky, light
    # floating glowing motes in front of the camera
    cam = fs["camera"]
    m = motes(t, cam["eye"], cam["target"])
    pa = fs.get("particles_add")
    fs["particles_add"] = m if pa is None or len(pa) == 0 else np.concatenate([pa, m])
    # grade: style defaults, but keep story effects (flash, tints, sad scenes)
    story = dict(fs.get("post", {}))
    post = dict(POST)
    if night > 0.5:
        post["god_rays"] = 0.0
        post["highlight_tint"] = (1.0, 0.95, 1.08)
    for k in ("white_flash", "tint"):
        if k in story:
            post[k] = story[k]
    if story.get("saturation", 1.12) < 0.9:          # deliberately desaturated moment
        post["saturation"] = story["saturation"]
    post["vignette"] = max(post["vignette"], story.get("vignette", 0))
    fs["post"] = post
    return fs
