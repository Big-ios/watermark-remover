"""视频去水印三方案对比实验。

视频与图片最本质的区别是**时序一致性**：逐帧独立修复会让水印处
忽明忽暗地"闪烁"，这是最影响观感的瑕疵。本脚本把这一点量化出来。

指标：
    水印区误差  —— 修得准不准
    帧间抖动    —— 相邻帧在修复区域的平均绝对变化，越小越不闪

做法：真实水印遮罩从"带水印版 与 干净版 的差分"反推得到，
避免手工标注位置不准导致结论失真（第一版就栽在这里）。
"""
from __future__ import annotations

import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/home/zcl/.pi/agent/skills/watermark-remover")
from wmkit.image import restore as R   # noqa: E402

CLEAN = "/tmp/clean.mp4"
WM = "/tmp/wm.mp4"


def read_all(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def main():
    clean = read_all(CLEAN)
    wm = read_all(WM)
    n = min(len(clean), len(wm))
    clean, wm = clean[:n], wm[:n]

    # ---- 用差分反推真实水印遮罩与强度 ----
    diff = np.mean([np.abs(w.astype(int) - c.astype(int)).sum(axis=2)
                    for w, c in zip(wm[:20], clean[:20])], axis=0)
    mask = (diff > 25).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    M = mask > 127
    # alpha：在遮罩内，wm ≈ clean*(1-a) + 255*a  →  a = (wm-clean)/(255-clean)
    num = np.mean([(w.astype(float) - c.astype(float)) for w, c in zip(wm[:20], clean[:20])], axis=0)
    den = np.mean([(255.0 - c.astype(float)) for c in clean[:20]], axis=0)
    alpha = float(np.clip((num[M] / np.maximum(den[M], 1)).mean(), 0.02, 0.95))
    print(f"共 {n} 帧 {wm[0].shape}")
    print(f"真实水印遮罩 {int(M.sum())} px（差分辨出）　推定 alpha = {alpha:.3f}")
    print()

    def metrics(name, frames, elapsed):
        errs, jit = [], []
        prev = None
        for o, c in zip(frames, clean):
            errs.append(np.abs(o.astype(int) - c.astype(int))[M].mean())
            if prev is not None:
                jit.append(np.abs(o.astype(int) - prev.astype(int))[M].mean())
            prev = o
        print(f"  {name:<20} 水印区误差 {np.mean(errs):6.2f} /255"
              f"   帧间抖动 {np.mean(jit):6.2f}"
              f"   {len(frames)/max(elapsed,1e-9):7.1f} fps")

    metrics("不处理（基线）", wm, 1e-9)

    t = time.time()
    out1 = [R.unblend(f, mask, alpha=alpha) for f in wm]
    metrics("① alpha 反算", out1, time.time() - t)

    t = time.time()
    base = [R.inpaint(f, mask, radius=3) for f in wm]
    metrics("② inpaint（逐帧）", base, time.time() - t)

    t = time.time()
    out3 = []
    for i in range(n):
        lo, hi = max(0, i - 1), min(n, i + 2)
        med = np.median(np.stack(base[lo:hi]).astype(np.int16), axis=0).astype(np.uint8)
        f = wm[i].copy()
        f[M] = med[M]
        out3.append(f)
    metrics("③ inpaint+时域中值", out3, time.time() - t)

    cv2.imwrite("/tmp/video-mask.png", mask)
    print()
    print("（水印遮罩已存 /tmp/video-mask.png，可查看识别出的区域）")


if __name__ == "__main__":
    main()
