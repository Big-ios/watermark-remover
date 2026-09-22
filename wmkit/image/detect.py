"""遮罩定位：找出水印覆盖了哪些像素。

三条线索，按可靠性排序：
  1. 模板匹配 —— 用户给一张水印样本，在图上找出所有出现位置（最准）
  2. 颜色/亮度筛选 —— 半透明水印会显著抬高/压低局部亮度
  3. 手动矩形 —— 兜底，用户直接框选
"""
from __future__ import annotations

import numpy as np


def by_template(img: np.ndarray, template: np.ndarray, *, threshold: float = 0.75,
                mask_alpha: int = 40) -> tuple:
    """模板匹配定位。返回 (mask, hits)，hits 是匹配到的矩形列表。"""
    import cv2

    g_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template
    res = cv2.matchTemplate(g_img, g_tpl, cv2.TM_CCOEFF_NORMED)
    th, tw = g_tpl.shape[:2]

    mask = np.zeros(g_img.shape, np.uint8)
    hits = []
    ys, xs = np.where(res >= threshold)
    for y, x in zip(ys, xs):
        # 非极大值抑制
        if any(abs(y - hy) < th * 0.5 and abs(x - hx) < tw * 0.5 for hy, hx in hits):
            continue
        hits.append((int(y), int(x)))
        mask[y:y + th, x:x + tw] = 255

    if mask_alpha != 255:
        # 需要更精细的像素级遮罩时，用模板的"有内容处"作为形状
        tm = (g_tpl > 30).astype(np.uint8) * 255
        fine = np.zeros_like(mask)
        for y, x in hits:
            sub = fine[y:y + th, x:x + tw]
            fine[y:y + th, x:x + tw] = np.maximum(sub, tm)
        if fine.sum():
            mask = fine
    return mask, hits


def by_brightness(img: np.ndarray, *, mode: str = "bright", percent: float = 96.0,
                  max_saturation: float = 60.0) -> np.ndarray:
    """半透明水印常表现为"局部过亮/过暗且色彩饱和度低"。

    mode='bright' 找偏白的（最常见），'dark' 找偏黑的。
    """
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float)
    s = hsv[:, :, 1].astype(float)

    thr = np.percentile(v, percent)
    if mode == "bright":
        m = (v >= thr) & (s <= max_saturation)
    else:
        thr = np.percentile(v, 100 - percent)
        m = (v <= thr) & (s <= max_saturation)
    return (m.astype(np.uint8) * 255)


def by_rect(img: np.ndarray, rects) -> np.ndarray:
    """手动矩形 → 遮罩。rects 为 [(x0, y0, x1, y1), ...]。"""
    mask = np.zeros(img.shape[:2], np.uint8)
    for x0, y0, x1, y1 in rects:
        mask[int(y0):int(y1), int(x0):int(x1)] = 255
    return mask


# ⚠️ 关于 auto() 的实测结论（别对它期望过高）
# ---------------------------------------------------------------
# 在自带的样例照片（渐变天空 + 半透明白字，alpha=0.35）上实测：
# 只能找到约 1/3 的水印像素，且三处水印只命中一处。
# 原因是半透明叠加的对比度本来就低（实测差值仅 ~20/255），
# 而"背景估计"又要跨过水印尺度（水印块约 200×60），两者互相牵制：
#   核小了 → 背景把水印算进去，差值被抹平
#   核大了 → 渐变背景本身也被抹平，同样测不到
# 试过 31px / 61px 核、调整亮度与饱和度阈值，都没能稳定解决。
#
# 所以定位的可靠性排序仍然是：
#   框选 --rect（最可靠）> 模板 --template > auto()（仅作粗筛）
# 若要把这一步真正做到通用，正确路径是训练/接入分割模型，
# 而不是继续堆启发式规则。
# ---------------------------------------------------------------
def auto(img: np.ndarray, *, sensitivity: float = 1.0) -> tuple:
    """自动定位水印像素 —— 实验性，仅作粗筛（见上方实测结论）。

    思路与 PDF 侧一致：**找物理特征，不找内容**。图片水印通常满足：
      1. 叠了一层半透明膜 → 该处比周围"发白/发灰"，且**饱和度低**
      2. 边缘清晰且规则 → 局部边缘密度高于自然纹理
      3. 同一图案反复出现 → 自相关峰（平铺水印）

    做法：把"亮度异常 + 低饱和度"的像素作为种子，再用形态学去掉零散噪点，
    最后返回 (mask, 诊断信息)。**务必配合 try 出的对比图人工确认后再执行。**
    """
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # 局部背景：核必须**大于水印笔画的尺度**（否则背景估计会把水印自己算进去，
    # 差值被抹平 —— 实测 31px 核只找到 5% 的水印像素，61px 才够）。
    bg_px = 61
    val_bg = cv2.medianBlur(val.astype(np.uint8), bg_px).astype(np.float32)
    sat_bg = cv2.medianBlur(sat.astype(np.uint8), bg_px).astype(np.float32)

    # 比周围亮、且比周围"发灰"（饱和度低）→ 白色半透明水印的特征
    bright = (val - val_bg) > (4.0 * sensitivity)
    flat = (sat_bg - sat) > (5.0 * sensitivity)
    seed = (bright & flat)
    # 水印是"成片"的，用闭运算把笔画连起来，再滤掉零散噪点
    seed = cv2.morphologyEx(seed.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0

    mask = (seed.astype(np.uint8) * 255)
    # 连通域去噪：太小的块多半是自然高光
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    clean = np.zeros_like(mask)
    kept = 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            clean[lab == i] = 255
            kept += 1

    info = {
        "seed_px": int(seed.sum()),
        "kept_components": kept,
        "final_px": int((clean > 0).sum()),
        "ratio": round(float((clean > 0).sum()) / clean.size, 4),
    }
    return clean, info


def refine(img: np.ndarray, coarse: np.ndarray, *, mode: str = "bright",
           min_delta: float = 6.0, open_px: int = 15) -> np.ndarray:
    """把"粗略框选"精化成"只盖住水印笔画"的遮罩。

    为什么需要这一步：用户框选的范围总比水印本身大，框内的正常像素也被
    反算就会被改变（实测整图误差几乎没改善就是这个原因）。

    原理：用大核形态学开运算估计「这一带的背景应该是什么样」，再拿观测值
    与它比较 —— 明显比背景亮（白水印）或暗（黑水印）的像素才是水印。
    """
    import cv2

    if not (coarse > 127).any():
        return coarse
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1, open_px * 2 + 1))
    if mode == "bright":
        bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)     # 闭运算抹掉亮笔画
        delta = g.astype(np.int16) - bg.astype(np.int16)
        keep = delta > min_delta
    else:
        bg = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)      # 开运算抹掉暗笔画
        delta = bg.astype(np.int16) - g.astype(np.int16)
        keep = delta > min_delta

    out = np.zeros_like(coarse)
    out[(coarse > 127) & keep] = 255
    # 精化后若几乎空了，说明该区域本来就没水印，退回原样避免误伤
    if int((out > 0).sum()) < 0.02 * int((coarse > 127).sum()):
        return coarse
    return out


def dilate(mask: np.ndarray, px: int = 2) -> np.ndarray:
    """把遮罩略微膨胀，盖住水印边缘的抗锯齿残留。"""
    import cv2

    if px <= 0:
        return mask
    k = np.ones((px * 2 + 1, px * 2 + 1), np.uint8)
    return cv2.dilate(mask, k)
