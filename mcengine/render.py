"""Frame loop -> ffmpeg encoder, plus muxing."""
import subprocess
import sys
import time
import numpy as np


class Encoder:
    def __init__(self, path, w, h, fps, crf=17, preset="medium"):
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
             "-pix_fmt", "yuv420p", "-profile:v", "high", path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg encode failed")


def render_video(path, frame_fn, duration, w, h, fps=30, start=0.0, end=None, log_every=60):
    """frame_fn(t) -> HxWx3 uint8. Renders [start, end) seconds."""
    end = duration if end is None else end
    n0, n1 = int(round(start * fps)), int(round(end * fps))
    enc = Encoder(path, w, h, fps)
    t_begin = time.time()
    for i in range(n0, n1):
        enc.write(frame_fn(i / fps))
        if (i - n0) % log_every == 0:
            el = time.time() - t_begin
            done = i - n0 + 1
            eta = el / done * (n1 - n0 - done)
            print(f"  frame {i}/{n1}  {el:6.1f}s elapsed  eta {eta:6.1f}s", file=sys.stderr, flush=True)
    enc.close()


def mux(video, audio, out, audio_bitrate="192k"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", audio_bitrate, "-shortest",
                    "-movflags", "+faststart", out], check=True)


def to_mp3(wav, mp3, bitrate="192k"):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav, "-c:a", "libmp3lame", "-b:a", bitrate, mp3],
                   check=True)
