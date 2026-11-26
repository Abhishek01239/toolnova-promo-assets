"""Time-of-day presets (sky, sun/moon light, fog) with smooth blending.

phase: 0 = day, 1 = golden hour, 2 = night, 3 = dawn (fractions blend).
"""
import numpy as np

PRESETS = [
    dict(zenith=(0.30, 0.54, 0.96), horizon=(0.72, 0.85, 1.00), below=(0.62, 0.74, 0.90),
         sun_dir=(0.3, 0.75, -0.55), sun_vis=1.0, moon_vis=0.0, stars=0.0,
         sun=(1.0, 0.95, 0.86), ambient=(0.74, 0.78, 0.86), fog=(0.70, 0.82, 0.97), exposure=1.0),
    dict(zenith=(0.30, 0.40, 0.78), horizon=(1.00, 0.66, 0.40), below=(0.70, 0.52, 0.48),
         sun_dir=(0.85, 0.14, -0.5), sun_vis=1.0, moon_vis=0.0, stars=0.0,
         sun=(1.0, 0.70, 0.45), ambient=(0.62, 0.56, 0.62), fog=(0.92, 0.66, 0.50), exposure=1.05),
    dict(zenith=(0.015, 0.025, 0.08), horizon=(0.07, 0.10, 0.22), below=(0.05, 0.07, 0.15),
         sun_dir=(-0.4, -0.7, 0.5), sun_vis=0.0, moon_vis=1.0, stars=1.0,
         sun=(0.38, 0.44, 0.70), ambient=(0.36, 0.40, 0.58), fog=(0.06, 0.08, 0.17), exposure=1.3),
    dict(zenith=(0.24, 0.34, 0.70), horizon=(1.00, 0.62, 0.55), below=(0.55, 0.45, 0.55),
         sun_dir=(-0.85, 0.16, 0.45), sun_vis=1.0, moon_vis=0.2, stars=0.1,
         sun=(1.0, 0.76, 0.62), ambient=(0.58, 0.56, 0.66), fog=(0.86, 0.62, 0.62), exposure=1.05),
]


# Cinematic presets (MC_STYLE=cinematic): richer, more saturated, orange/purple palette.
CINE_PRESETS = [
    dict(zenith=(0.10, 0.26, 0.78), horizon=(0.98, 0.80, 0.62), below=(0.55, 0.50, 0.70),
         sun_dir=(0.45, 0.42, -0.6), sun_vis=1.0, moon_vis=0.0, stars=0.0,
         sun=(1.08, 0.92, 0.74), ambient=(0.60, 0.62, 0.82), fog=(0.62, 0.66, 0.88), exposure=0.95),
    dict(zenith=(0.16, 0.08, 0.38), horizon=(1.00, 0.46, 0.14), below=(0.42, 0.18, 0.34),
         sun_dir=(0.85, 0.10, -0.5), sun_vis=1.0, moon_vis=0.0, stars=0.25,
         sun=(0.98, 0.90, 0.70), ambient=(0.62, 0.62, 0.84), fog=(0.62, 0.34, 0.50), exposure=1.08),
    dict(zenith=(0.03, 0.01, 0.10), horizon=(0.20, 0.07, 0.34), below=(0.08, 0.03, 0.16),
         sun_dir=(-0.4, -0.7, 0.5), sun_vis=0.0, moon_vis=1.0, stars=1.0,
         sun=(0.50, 0.44, 0.95), ambient=(0.34, 0.28, 0.62), fog=(0.12, 0.05, 0.22), exposure=1.35),
    dict(zenith=(0.20, 0.14, 0.50), horizon=(1.00, 0.52, 0.48), below=(0.45, 0.28, 0.48),
         sun_dir=(-0.85, 0.14, 0.45), sun_vis=1.0, moon_vis=0.2, stars=0.15,
         sun=(1.15, 0.72, 0.58), ambient=(0.48, 0.40, 0.66), fog=(0.76, 0.48, 0.58), exposure=1.05),
]


def _mix(a, b, t):
    if isinstance(a, tuple):
        return tuple(float(x * (1 - t) + y * t) for x, y in zip(a, b))
    return float(a * (1 - t) + b * t)


def environment(phase, fog_start=60.0, fog_end=300.0):
    phase = float(np.clip(phase, 0, len(PRESETS) - 1))
    i = min(int(phase), len(PRESETS) - 2)
    t = phase - i
    t = t * t * (3 - 2 * t)
    from . import style
    table = CINE_PRESETS if style.ACTIVE else PRESETS
    a, b = table[i], table[i + 1]
    p = {k: _mix(a[k], b[k], t) for k in a}
    sun = np.array(p["sun_dir"]); sun /= np.linalg.norm(sun)
    light_dir = sun if sun[1] > 0.05 else -sun          # moonlight at night
    sky = dict(zenith=p["zenith"], horizon=p["horizon"], below=p["below"], sun_dir=tuple(sun),
               sun_vis=p["sun_vis"], moon_vis=p["moon_vis"], stars=p["stars"])
    light = dict(dir=tuple(light_dir), sun=p["sun"], ambient=p["ambient"], exposure=p["exposure"])
    fog = dict(color=p["fog"], start=fog_start, end=fog_end)
    return sky, light, fog
