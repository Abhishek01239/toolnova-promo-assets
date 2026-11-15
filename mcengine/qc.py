"""Quality control: probe streams, detect black frames / silence, loudness, contact sheets."""
import json
import re
import subprocess
from PIL import Image


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def blackdetect(path, min_dur=0.1):
    err = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vf", f"blackdetect=d={min_dur}:pix_th=0.06",
                          "-an", "-f", "null", "-"], capture_output=True, text=True).stderr
    return re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)


def silencedetect(path, noise_db=-45, min_dur=0.8):
    err = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af", f"silencedetect=n={noise_db}dB:d={min_dur}",
                          "-vn", "-f", "null", "-"], capture_output=True, text=True).stderr
    return re.findall(r"silence_start: ([\d.]+)", err)


def loudness(path):
    err = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-af", "ebur128=peak=true", "-vn", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.search(r"Integrated loudness:\s+I:\s+([-\d.]+) LUFS", err)
    p = re.search(r"True peak:\s+Peak:\s+([-\d.]+) dBFS", err)
    return (float(m.group(1)) if m else None, float(p.group(1)) if p else None)


def decode_check(path):
    """Full decode of every frame; returns ffmpeg error output (empty == clean)."""
    return subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "null", "-"],
                          capture_output=True, text=True).stderr.strip()


def contact_sheet(path, out, times, thumb_w=270):
    ims = []
    for t in times:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1",
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        import io
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        im = im.resize((thumb_w, int(im.height * thumb_w / im.width)))
        ims.append(im)
    cols = min(6, len(ims))
    rows = (len(ims) + cols - 1) // cols
    W, H = ims[0].size
    sheet = Image.new("RGB", (cols * W, rows * H), (0, 0, 0))
    for i, im in enumerate(ims):
        sheet.paste(im, ((i % cols) * W, (i // cols) * H))
    sheet.save(out)
    return out


def report(path, expect_w, expect_h, min_dur, max_dur):
    info = probe(path)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    dur = float(info["format"]["duration"])
    lufs, peak = loudness(path)
    r = {
        "resolution": f"{v['width']}x{v['height']}",
        "resolution_ok": v["width"] == expect_w and v["height"] == expect_h,
        "fps": v["r_frame_rate"],
        "duration": round(dur, 2),
        "duration_ok": min_dur <= dur <= max_dur,
        "video_codec": v["codec_name"], "pix_fmt": v["pix_fmt"],
        "audio": bool(a), "audio_codec": a["codec_name"] if a else None,
        "black_segments": blackdetect(path),
        "silences": silencedetect(path),
        "loudness_lufs": lufs, "true_peak_dbfs": peak,
        "decode_errors": decode_check(path),
    }
    r["pass"] = (r["resolution_ok"] and r["duration_ok"] and r["audio"] and not r["black_segments"]
                 and not r["decode_errors"])
    return r
