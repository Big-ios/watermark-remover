"""修复：把水印遮罩内的像素还原。

两种手段
    unblend  反算 —— observed = original*(1-a) + C*a，解出 original。
             已知 alpha 与颜色时近乎无损（实测误差 1/255），**首选**。
    inpaint  重绘 —— 用周边内容填补。适用于不透明水印（logo、贴图），
             或反算后仍有残留的部分。误差较大（实测 31/255），是兜底手段。
"""
from __future__ import annotations

import numpy as np

WHITE = (255, 255, 255)


def unblend(img: np.ndarray, mask: np.ndarray, *, alpha: float,
            color=WHITE, clip: bool = True) -> np.ndarray:
    """反算：original = (observed - C*a) / (1 - a)。"""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha 必须在 (0,1) 之间，收到 {alpha}")
    m = mask > 127
    if not m.any():
        return img.copy()

    out = img.astype(np.float32).copy()
    c = np.array(color if not isinstance(color, str) else WHITE, dtype=np.float32)
    if isinstance(color, str):
        raise ValueError("color 请传 (R,G,B) 三元组")
    for ch in range(3):
        v = (out[:, :, ch] - c[ch] * alpha) / (1.0 - alpha)
        out[:, :, ch][m] = v[m]
    if clip:
        out = out.clip(0, 255)
    return out.astype(np.uint8)


def inpaint(img: np.ndarray, mask: np.ndarray, *, radius: int = 3,
            method: str = "telea", engine: str = "auto") -> np.ndarray:
    """重绘：用周边内容填补遮罩区域。

    engine  "lama"   用 LaMA 模型（质量最好，CPU 约 4 秒/次）
            "opencv" 用 OpenCV 邻域插值（很快，视频首选）
            "auto"   有模型就用 LaMA，否则退回 OpenCV

    为什么视频默认不用 LaMA：模型输入被固定为 512×512，无论原图多大，
    单次推理都是 4 秒上下（实测 ROI 裁剪也省不掉 —— 瓶颈在模型本身）。
    1080p 视频按帧算就是小时级，所以视频侧把它作为可选项。
    """
    import cv2

    if engine == "auto":
        from . import lama as _lama
        engine = "lama" if _lama.available() else "opencv"

    if engine == "lama":
        from . import lama as _lama
        return _lama.inpaint(img, mask)

    flag = cv2.INPAINT_TELEA if method.lower().startswith("t") else cv2.INPAINT_NS
    return cv2.inpaint(img, mask, radius, flag)


def unblend_then_inpaint(img: np.ndarray, mask: np.ndarray, *, alpha: float,
                         color=WHITE, residual_thresh: int = 18,
                         radius: int = 3, method: str = "telea") -> np.ndarray:
    """先反算，再把"反算失败（越界被夹取）"的像素交给重绘。

    反算时若 (observed - C*a) 落在 [0,255] 之外，说明该处不符合单纯叠加模型
    （可能是不透明贴图、或多层水印），这些像素反算会失真，改由重绘处理。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha 必须在 (0,1) 之间，收到 {alpha}")
    m = mask > 127
    if not m.any():
        return img.copy()

    c = np.array(color, dtype=np.float32)
    f = img.astype(np.float32)
    bad = np.zeros(img.shape[:2], bool)
    out = f.copy()
    for ch in range(3):
        v = (f[:, :, ch] - c[ch] * alpha) / (1.0 - alpha)
        bad |= (v < -residual_thresh) | (v > 255 + residual_thresh)
        out[:, :, ch][m] = v[m].clip(0, 255)
    res = out.astype(np.uint8)

    bad &= m
    if bad.sum() > max(10, 0.002 * m.sum()):
        fix = inpaint(res, (bad.astype(np.uint8) * 255), radius=radius, method=method)
        res[bad] = fix[bad]
    return res


def estimate_alpha_candidates(img: np.ndarray, mask: np.ndarray, *, color=WHITE,
                              n: int = 5) -> list:
    """给出几个待试的 alpha 值，供人看效果挑选（不自动定夺）。

    依据：反算后水印区出现明显越界（过曝/欠曝）说明 alpha 估偏大；
    若水印区与周边的局部对比度差异仍很大，说明估偏小。
    """
    import cv2

    m = mask > 127
    if not m.any():
        return []
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(g, (5, 5))
    sq = cv2.blur(g * g, (5, 5))
    lv = (sq - mean * mean).clip(0)
    ring = (cv2.dilate((m * 255).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & ~m
    target = lv[ring].mean() if ring.any() else lv.mean()

    scored = []
    for a in np.arange(0.05, 0.85, 0.05):
        rec = unblend(img, mask, alpha=float(a), color=color, clip=False)
        over = int((rec < 0).sum() + (rec > 255).sum())
        r = rec.clip(0, 255).astype(np.uint8)
        g2 = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY).astype(np.float32)
        m2 = cv2.blur(g2, (5, 5))
        s2 = cv2.blur(g2 * g2, (5, 5))
        lv2 = (s2 - m2 * m2).clip(0)
        contrast_gap = abs(lv2[m].mean() - target)
        scored.append((float(a), over, contrast_gap))
    # 越界少、对比度接近周边的排在前面
    scored.sort(key=lambda t: (t[1] / max(m.sum(), 1) * 4 + t[2] / max(target, 1e-6)))
    return [s[0] for s in scored[:n]]
