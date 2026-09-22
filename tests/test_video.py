"""视频去水印的回归测试。

合成一段短测试视频作为夹具：运动背景 + 半透明白字水印，并保留干净版对照。
断言的是一条**视频特有的性质**：处理完之后，水印处的帧间抖动必须比
"逐帧独立修复"更小 —— 这正是时域稳定存在的理由。
"""
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

cv2 = pytest.importorskip("cv2")

from wmkit.video import io as V          # noqa: E402
from wmkit.video import process as PR    # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="需要 ffmpeg")

RECT = (415, 290, 600, 350)
W, H, FPS, DUR = 640, 360, 30, 2


def _make_pair(tmp_path):
    clean = str(tmp_path / "clean.mp4")
    wm = str(tmp_path / "wm.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi",
        "-i", f"color=c=0x3a6ea5:s={W}x{H}:d={DUR}:r={FPS}",
        "-vf", ("drawbox=x='mod(t*300\,900)-260':y=60:w=160:h=240:"
                "color=0x2a5080@1:t=fill,noise=alls=5:allf=t+u"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", clean,
    ], check=True, capture_output=True)
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", clean,
        "-vf", "drawtext=text='STUDIO':x=420:y=300:fontsize=40:"
               "fontcolor=white@0.4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", wm,
    ], check=True, capture_output=True)
    return clean, wm


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    d = tmp_path_factory.mktemp("video")
    return _make_pair(d)


def _read(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def test_probe_reports_metadata(pair):
    _, wm = pair
    info = V.probe(wm)
    assert info.width == W and info.height == H
    assert abs(info.fps - FPS) < 0.5
    assert info.n_frames >= DUR * FPS - 3


def test_build_mask_from_rect(pair):
    """矩形遮罩的像素数 —— 注意 OpenCV 的 rectangle 两端都是闭区间。"""
    mask = PR.build_mask((H, W), rects=[RECT])
    x0, y0, x1, y1 = RECT
    assert int((mask > 0).sum()) == (x1 - x0 + 1) * (y1 - y0 + 1)


def test_empty_mask_rejected(pair):
    _, wm = pair
    with pytest.raises(ValueError):
        PR.process(wm, "/tmp/_never.mp4", rects=[])


def test_temporal_stabilization_reduces_flicker(pair, tmp_path):
    """核心断言：开时域稳定后，水印处的帧间抖动必须更小。"""
    clean, wm = pair
    x0, y0, x1, y1 = RECT
    M = np.zeros((H, W), bool)
    M[y0:y1, x0:x1] = True

    outs = {}
    for tag, temporal in (("逐帧", 0), ("时域稳定", 1)):
        dst = str(tmp_path / f"{tag}.mp4")
        PR.process(wm, dst, rects=[RECT], temporal=temporal, crf=20)
        frames = _read(dst)
        jit, err = [], []
        prev = None
        for f, c in zip(frames, _read(clean)):
            err.append(np.abs(f.astype(int) - c.astype(int))[M].mean())
            if prev is not None:
                jit.append(np.abs(f.astype(int) - prev.astype(int))[M].mean())
            prev = f
        outs[tag] = (float(np.mean(jit)), float(np.mean(err)))

    assert outs["时域稳定"][0] < outs["逐帧"][0], (
        f"时域稳定没起作用：逐帧 {outs['逐帧'][0]:.3f} vs "
        f"稳定 {outs['时域稳定'][0]:.3f}")
    # 准确度也不能因为稳定而变差太多
    assert outs["时域稳定"][1] < 6.0, f"修复不准：{outs['时域稳定'][1]:.2f}"


def test_audio_stream_preserved(tmp_path):
    """带音轨的视频，处理后音轨必须还在。"""
    src = str(tmp_path / "a.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=s={W}x{H}:d=1:r={FPS}",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src,
    ], check=True, capture_output=True)
    assert V.probe(src).has_audio

    dst = str(tmp_path / "a-out.mp4")
    PR.process(src, dst, rects=[(10, 10, 60, 30)], crf=22)
    assert V.probe(dst).has_audio, "音轨丢了"
