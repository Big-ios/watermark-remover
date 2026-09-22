"""水印评分 —— 只看「物理特征」，不看它写了什么字。

为什么要重写成评分制
--------------------
最初版本靠"推广关键词"命中来判水印，那只对公众号/平台推广有效；
换个水印（"内部资料""某某公司"、纯图形 logo）就完全失灵。
**通用的判据必须是内容无关的物理特征。**

PDF 里最强的通用信号是**半透明度** —— 实测同一份文件里：
    真水印「速创字帖」  opacity = 0.020 / 0.047 / 0.098
    正文（所有文字）    opacity = 1.0
    格子辅助线          opacity = 1.0
水印之所以"看起来淡"，就是因为它是半透明绘制的；这一点与文字内容无关。

其余信号（旋转、重复、页边、浅色、独立字体）都是加分项，单独任何一条
都不足以定罪，所以做成打分而不是硬规则。
"""
from __future__ import annotations

# 半透明阈值：低于此值基本可断定是叠加式水印
OPAQUE = 0.95
TRANSLUCENT = 0.60
VERY_FAINT = 0.25

# 推广关键词（加分项，不再是唯一依据）
PROMO_KEYWORDS = (
    "公众号", "微信", "水印", "会员", "扫码", "二维码", "关注", "免费", "下载",
    "网址", "www.", "http", "©", "®", "版权", "侵权", "定制", "咨询", "联系",
    "客服", "淘宝", "抖音", "小红书", "店铺", "搜", "加我", "仅供", "内部",
)


def score_span(span, *, count: int = 1, at_edge: bool = False,
               body_fonts=(), body_colors=()) -> dict:
    """给一个文字对象打分。返回 {score, breakdown, opacity}。

    body_fonts / body_colors 用于判断"是否与正文风格不同"。
    """
    text = "".join(chr(c[0]) for c in span["chars"])
    opacity = float(span.get("opacity", 1.0) or 1.0)
    color = tuple(round(float(v) * 255) for v in span["color"])
    font = span.get("font", "")
    size = float(span.get("size", 0))

    pts = {}
    # ---- 半透明：最强信号 ----
    if opacity < VERY_FAINT:
        pts["半透明（很淡）"] = 55
    elif opacity < TRANSLUCENT:
        pts["半透明"] = 45
    elif opacity < OPAQUE:
        pts["轻微透明"] = 20

    # ---- 重复 ----
    if count >= 10:
        pts["大量重复"] = 18
    elif count >= 3:
        pts["重复出现"] = 12
    elif count == 2:
        pts["出现两次"] = 5

    # ---- 颜色浅 ----
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    if lum > 200:
        pts["颜色很浅"] = 10
    elif lum > 150:
        pts["颜色偏浅"] = 6

    # ---- 页边 ----
    if at_edge:
        pts["位于页边"] = 8

    # ---- 关键词（保留为加分） ----
    kw = next((k for k in PROMO_KEYWORDS if k in text), None)
    if kw:
        pts[f"含推广词「{kw}」"] = 20

    # ---- 与正文风格不同 ----
    if body_fonts and font and font not in body_fonts:
        pts["字体与正文不同"] = 10
    if body_colors and color not in body_colors:
        pts["颜色与正文不同"] = 5

    # ---- 反向信号：与正文完全一致、且不透明 ----
    if opacity >= OPAQUE and body_fonts and font in body_fonts:
        pts["字体颜色与正文一致"] = -15

    total = sum(pts.values())
    return {"score": total, "breakdown": pts, "opacity": round(opacity, 4),
            "color": color, "font": font, "size": round(size, 1), "text": text}


def grade(score: int) -> str:
    """分数 → 建议档位。"""
    if score >= 50:
        return "safe_delete"
    if score >= 25:
        return "likely_delete"
    return "review"


def explain(sig: dict) -> str:
    items = "、".join(f"{k}(+{v})" if v > 0 else f"{k}({v})"
                      for k, v in sorted(sig["breakdown"].items(),
                                         key=lambda kv: -abs(kv[1])))
    return f"得分 {sig['score']}｜不透明度 {sig['opacity']}｜{items}" if items else \
           f"得分 {sig['score']}｜不透明度 {sig['opacity']}｜无显著特征"
