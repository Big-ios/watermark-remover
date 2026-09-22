"""字形几何分析 —— 识破 ToUnicode 造假。

为什么需要这一层
----------------
PDF 生成方可以任意设置字形的 ToUnicode 映射：让一个「格子辅助线」的字形
在文字层里读作「速创字帖」。只信文字层的自动去水印工具会因此误删正文图案
—— 本工具的开发过程中就真的这样误删过 120 个田字格辅助线。

判据（实测得到，100% 分离）
--------------------------
统计该字形**自身颜色**的像素（排除底下的格子线干扰），然后看投影：

    高覆盖率贯穿（≥90% bbox 宽/高）的行数与列数

    · 格子辅助线（十字/米字/虚线）：横 ≥4 行、竖 ≥5 列同时成立
    · 任何汉字（郑/清/婉/速/创/字/帖……）：横 0、竖 0

注意：**墨迹密度不能用作判据** —— 实测「创」字密度 3.3%，
比辅助线的 5.4% 还低，用密度判断必然出错。

所以判定规则是「横竖都有高覆盖率贯穿」这一条，简单且可靠。
"""
from __future__ import annotations

import numpy as np

SPAN_RATIO = 0.90      # 投影覆盖 bbox 该方向的 90% 以上，才算「贯穿」


def _rgb255(color) -> list:
    return [int(round(float(c) * 255)) for c in color]


def profile(page, bbox, color, *, pad: float = 4.0, dpi: int = 200,
            tol: float = 60.0) -> dict:
    """对页面上某处、某颜色的内容做几何画像。

    color 请传该字形自身的颜色（0~1 的 RGB），这样统计时只会计入
    这个字形，不会被底下的格子线或其他元素干扰。
    """
    import pymupdf

    r = pymupdf.Rect(bbox).__add__((-pad, -pad, pad, pad))
    if r.width < 3 or r.height < 3:
        return {"ok": False, "reason": "区域太小", "ink": 0, "density": 0.0}

    px = page.get_pixmap(clip=r, dpi=dpi)
    a = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width, px.n)[:, :, :3].astype(int)
    target = np.array(_rgb255(color))
    m = np.abs(a - target).sum(axis=2) < tol * 3

    ink = int(m.sum())
    total = int(m.size)
    if ink < 20:
        return {"ok": False, "reason": "该颜色几乎没有墨迹", "ink": ink,
                "density": ink / max(total, 1)}

    rows = m.sum(axis=1)
    cols = m.sum(axis=0)
    w, h = m.shape[1], m.shape[0]

    def longest_run(flags) -> int:
        best = cur = 0
        for f in flags:
            cur = cur + 1 if f else 0
            best = max(best, cur)
        return best

    return {
        "ok": True,
        "ink": ink,
        "total": total,
        "density": round(ink / total, 4),
        "h_span": int((rows > w * SPAN_RATIO).sum()),
        "v_span": int((cols > h * SPAN_RATIO).sum()),
        "h_run": longest_run(rows > w * SPAN_RATIO),
        "v_run": longest_run(cols > h * SPAN_RATIO),
        "size": (w, h),
    }


def classify(prof: dict) -> str:
    """'lines'（线条结构）/ 'text'（文字）/ 'unknown'。"""
    if not prof.get("ok"):
        return "unknown"
    if prof["h_span"] >= 1 and prof["v_span"] >= 1:
        return "lines"
    return "text"          # 没有高覆盖率贯穿 → 按文字处理


def explain(prof: dict) -> str:
    """给人看的一句话解释。"""
    if not prof.get("ok"):
        return f"无法判定（{prof.get('reason', '未知')}）"
    return (f"墨迹密度 {prof['density']*100:.1f}% · "
            f"高覆盖贯穿：横 {prof['h_span']} 行 / 竖 {prof['v_span']} 列"
            f"（最长连续 {prof['h_run']}/{prof['v_run']}）")


def analyze(page, span) -> dict:
    """对 texttrace 里的一个 span 做完整判定，返回可直接进报告的结论。"""
    prof = profile(page, span["bbox"], span["color"])
    verdict = classify(prof)
    return {
        "verdict": verdict,
        "profile": {k: v for k, v in prof.items() if k != "m"},
        "explain": explain(prof),
    }
