"""视频读写 —— 用 ffmpeg 管道，避免落一堆临时帧文件。

读：ffmpeg 解码成 rawvideo 送到 stdout，逐帧读入
写：rawvideo 从 stdin 喂给 ffmpeg，同时把**原视频的音轨**带过去

音轨与帧率必须原样保留，否则交付出去的片子会变哑或变速。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass


class FFmpegMissing(RuntimeError):
    pass


def _need_ffmpeg():
    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise FFmpegMissing(
                f"找不到 {exe}。视频处理依赖 ffmpeg，请先安装："
                "sudo apt install ffmpeg  或到 ffmpeg.org 下载"
            )


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float
    has_audio: bool
    codec: str

    def describe(self) -> str:
        a = "有音轨" if self.has_audio else "无音轨"
        return (f"{self.width}×{self.height} @ {self.fps:.3g}fps · "
                f"{self.n_frames} 帧 · {self.duration:.2f}s · {self.codec} · {a}")


def probe(path: str) -> VideoInfo:
    """读视频元信息。"""
    _need_ffmpeg()
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    raw = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    meta = json.loads(raw)

    v = next((s for s in meta["streams"] if s["codec_type"] == "video"), None)
    if v is None:
        raise ValueError(f"文件里没有视频流：{path}")
    has_audio = any(s["codec_type"] == "audio" for s in meta["streams"])

    # 帧率可能是 "30000/1001" 这种分数
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "25/1"
    if "/" in rate:
        num, den = rate.split("/")
        fps = float(num) / float(den) if float(den) else 25.0
    else:
        fps = float(rate)

    dur = float(meta.get("format", {}).get("duration") or v.get("duration") or 0.0)
    n = int(v.get("nb_frames") or 0) or (int(dur * fps) if dur else 0)

    return VideoInfo(width=int(v["width"]), height=int(v["height"]), fps=fps,
                     n_frames=n, duration=dur, has_audio=has_audio,
                     codec=v.get("codec_name", "?"))


def iter_frames(path: str, info: VideoInfo | None = None, *, start: int = 0,
                count: int | None = None):
    """逐帧产出 BGR ndarray。count=None 表示到结尾。"""
    _need_ffmpeg()
    info = info or probe(path)
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", f"{start / info.fps:.6f}"]
    cmd += ["-i", path, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    w, h = info.width, info.height
    frame_bytes = w * h * 3
    got = 0
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            if count is not None and got >= count:
                break
            import numpy as np
            yield np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
            got += 1
    finally:
        proc.stdout.close()
        proc.wait()


def write_video(dst: str, frames_iter, info: VideoInfo, *, src_for_audio: str | None = None,
                crf: int = 18, preset: str = "medium", progress=None) -> dict:
    """把处理后的帧写成视频。

    src_for_audio 给定时，把该文件的音轨一并封装进去（保持原声）。
    """
    _need_ffmpeg()
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{info.width}x{info.height}", "-r", f"{info.fps}",
           "-i", "-"]
    if src_for_audio:
        cmd += ["-i", src_for_audio, "-map", "0:v:0", "-map", "1:a?",
                "-c:a", "copy", "-shortest"]
    cmd += ["-c:v", "libx264", "-crf", str(crf), "-preset", preset,
            "-pix_fmt", "yuv420p", dst]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    n = 0
    try:
        for f in frames_iter:
            proc.stdin.write(f.tobytes())
            n += 1
            if progress and n % 30 == 0:
                progress(n)
    except BrokenPipeError:
        pass
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        err = proc.stderr.read().decode("utf-8", "ignore")
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 写视频失败：\n{err[-800:]}")
    return {"frames_written": n, "output": os.path.abspath(dst),
            "size": os.path.getsize(dst) if os.path.exists(dst) else 0}
