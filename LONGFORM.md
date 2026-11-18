# Long-form videos (8–20 min "challenge" episodes)

Landscape 1920x1080 voxel films built by the same engine. Reference projects:

| Project | Video | Length |
|---|---|---|
| `projects/first_night` | I Had 10 Minutes to Survive My First Night | 5 min |
| `projects/day2` | I Had Until Sunset to Find Diamonds (Day 2) | 10 min |
| `projects/days100` | I Survived 100 Days in Minecraft | 20 min, chaptered shot engine |

`projects/days100` is the most reusable: `engine.py` (Shot/Actor/props/cameras),
`story.py` (Film: timing, statics, lighting, HUD overlay, chapter cards, SFX, music),
`world.py` (terrain + structures), `ch_*.py` (shots per chapter), `script.py` (narration lines),
`make_voice.py` (narration with the puck_classic voice via `mcengine.tts_classic`), `build.py`
(`stills <t,t,...>`, `audio`, `video`, `video --range a b`, `thumb`, `docs`, `qc`, `all`).

## Pipeline
1. Copy `projects/days100` to `projects/<slug>`; rewrite `script.py`, `world.py`, `ch_*.py`.
2. `python make_voice.py` → `assets/narration.wav` + `assets/words.json`.
3. `python build.py stills ...` and look at `work/stills.png` for every chapter; fix cameras in terrain.
4. `python build.py all` (writes output dir set in build.py `OUT`).
5. Re-encode audio: `loudnorm=I=-15:TP=-3:LRA=11,alimiter=limit=0.7` with `-c:v copy`.

## Render speed / container survival
- ~0.5 s per 1080p frame on 4 CPUs; run 2 processes in parallel on frame ranges
  (`render_video(..., start, end)`), then join with the ffmpeg concat demuxer (`-c copy`).
- A cloud container is reclaimed when the session is idle. Always keep a background
  command (an `until ...; do sleep 30; done` watcher) running while rendering, and re-arm it
  before its 2 h limit.
- If a render is killed, the partial mp4 has no moov atom: recover its H.264 by reading the
  length-prefixed NALs after `mdat` into Annex-B (prepend SPS/PPS from a tiny encode with the
  same x264 settings), cut at the last keyframe and render only the rest.
