"""视频水印定位。

通用思路（与内容无关）
----------------------
水印在视频里有一个图片没有的强特征：**它在时间维度上不变**。
画面内容在动，水印始终是同一块像素，所以「沿时间轴的标准差」在
水印处极小。这是视频去水印最可靠的通用线索。

但有两个坑必须处理：
  1. **静止背景也满足这个条件**（固定机位、纯色背景、黑边）
     → 用「与画面主体的反差」和「连通块形状」剔除，并交人工确认
  2. **噪声**：传感器噪声会让静止区域的 std 不为零
     → 阈值要按全片噪声水平自适应

因此本模块的产物是**候选区域 + 可视化**，最终由人（或 AI 看标注图）确认，
而不是直接开修。
"""
from __future__ import annotations

import numpy as np


def sample_frames(path: str, n: int = 40, *, info=None):
    """均匀抽取 n 帧用于分析。"""
    from . import io as V

    info = info or V.probe(path)
    total = info.n_frames or max(int(info.duration * info.fps), 1)
    step = max(total // max(n, 1), 1)
    out = []
    for i in range(0, total, step):
        for f in V.iter_frames(path, info, start=i, count=1):
            out.append(f)
            break
        if len(out) >= n:
            break
    return out, info


def detect_translucent(path: str, *, samples: int = 40, ratio: float = 0.72,
                       min_area: int = 120, info=None) -> dict:
    """找"时间波动被明显压低"的区域 —— 半透明水印的通用特征。

    为什么不能简单地"找完全静止的区域"（第一版就是这么写的，检出 0）：
    半透明水印不会让画面停止运动，它是**透过水印看内容**，
    水印处的像素仍在变化，只是被按 (1-a) 压缩了幅度。
    所以要看的是**相对周边的波动比值**，不是绝对值。

        obs = orig×(1-a) + C×a
        std(obs) ≈ std(orig) × (1-a)      ← 波动被按比例压低

    ratio  波动比低于该值视为疑似水印（0.72 对应 a≳0.28）
    """
    import cv2

    frames, info = sample_frames(path, samples, info=info)
    if len(frames) < 3:
        return {"mask": None, "boxes": [], "info": info,
                "stats": {"error": "采样帧太少"}}

    stack = np.stack([f.astype(np.float32) for f in frames])
    tstd = stack.std(axis=0).mean(axis=2)          # 每个像素沿时间的波动

    # 局部参考：用大核中值得到"这一带本该有多大的波动"
    tstd_u8 = np.clip(tstd * 8, 0, 255).astype(np.uint8)   # 放大以便中值处理
    ref = cv2.medianBlur(tstd_u8, 61).astype(np.float32) / 8.0
    ref = np.maximum(ref, 1e-3)
    rel = tstd / ref                                # 相对波动

    # 噪声底：相对波动的最小 20% 分位，用它估计"完全没被压缩"的水平
    noise = float(np.percentile(rel, 20))
    static = ((rel < ratio) & (ref > 0.5)).astype(np.uint8) * 255
    if static.sum() == 0:
        # 退一步：找相对波动最低的那一小撮
        thr = np.percentile(rel, 3.0)
        static = (rel <= thr).astype(np.uint8) * 255

    # 去噪：开运算 + 小连通块过滤
    static = cv2.morphologyEx(static, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(static, 8)
    boxes = []
    keep = np.zeros_like(static)
    h, w = static.shape
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        # 排除"铺满画面"的整块背景（水印通常只占一小块）
        if bw * bh > 0.55 * w * h:
            continue
        keep[lab == i] = 255
        boxes.append({"x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                      "area": int(area),
                      "cx": round(float(cent[i][0]), 1), "cy": round(float(cent[i][1]), 1)})

    boxes.sort(key=lambda b: -b["area"])
    return {
        "mask": keep,
        "boxes": boxes,
        "info": info,
        "stats": {
            "samples": len(frames),
            "noise_floor": round(noise, 3),
            "ratio_threshold": ratio,
            "static_px": int((keep > 0).sum()),
            "ratio": round(float((keep > 0).sum()) / keep.size, 4),
            "size": (w, h),
        },
    }


def render_overlay(frame, mask, boxes=(), *, alpha: float = 0.45):
    """把识别结果画在首帧上，供人工确认。"""
    import cv2

    out = frame.copy()
    if mask is not None:
        m = mask > 0
        # 半透明红覆盖
        red = np.zeros_like(out)
        red[:, :, 2] = 255
        out[m] = (out[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    for b in boxes:
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]),
                      (0, 255, 0), 2)
    return out
