"""水印候选识别 —— 输出一份"发现什么、建议删什么、风险多大"的报告。

设计原则
--------
**只报告，不改文件。** 每个候选都带证据链与风险评级，由人确认后才执行。

候选来源（三条互补的线索）
  1. 文字层：按 (字体, 字号, 颜色, 文本) 分组，重复出现或压在别的内容之上
  2. 几何层：用 glyph 判据验证"这到底是不是文字"——识破 ToUnicode 造假
  3. 位置层：贴近页面边缘（页眉/页脚/页边）的内容多为推广信息

风险评级
  keep          几何判定为线条结构 → 几乎肯定是图案/辅助线，**建议保留**
  safe_delete   几何确认是文字 + 重复出现 → 典型平铺水印
  likely_delete 几何确认是文字 + 只在页边出现一次 → 典型页脚推广
  review        其余情况 → 交给人判断
"""
from __future__ import annotations

import collections
import os

from . import glyph, score as SC

# 距页面边界多近算"页边"（按页面尺寸的比例）
EDGE_RATIO = 0.06

# 判定原则（两次修正后定稿）：
#   1. 最初靠「推广关键词」—— 只对公众号类水印有效，换个水印就失灵
#   2. 后来靠「重复次数」—— 练字帖的范字本来就重复，会误伤
#   3. 现在靠**物理特征评分**（半透明度为主，关键词只作加分）
#      —— 水印之所以看起来淡，是因为它半透明绘制，这与内容无关
#   评分 >= 50 自动判删，25~49 提示，其余交人工。


def _page_bounds(page):
    r = page.rect
    return r.width, r.height


def _near_edge(bbox, page_w, page_h):
    x0, y0, x1, y1 = bbox
    mx, my = page_w * EDGE_RATIO, page_h * EDGE_RATIO
    return x0 < mx or y0 < my or x1 > page_w - mx or y1 > page_h - my


def _infer_body_style(doc, groups) -> tuple:
    """推定正文风格：出现最多、且不透明的那些字体与颜色。

    用来判断某个对象"是不是外来户"（与正文风格不符）。
    """
    import collections as _c

    tally = _c.Counter()
    for key, items in groups.items():
        font, size, color, opacity, text = key
        if opacity >= 0.95:
            tally[(font, color)] += len(items)
    top = [k for k, _ in tally.most_common(3)]
    return {f for f, _ in top}, {c for _, c in top}


def analyze(path: str, *, max_pages: int | None = None) -> dict:
    """扫描 PDF，返回水印报告（不修改文件）。"""
    import pymupdf

    doc = pymupdf.open(path)
    try:
        groups: dict = collections.defaultdict(list)
        for pno, page in enumerate(doc):
            if max_pages is not None and pno >= max_pages:
                break
            pw, ph = _page_bounds(page)
            for si, span in enumerate(page.get_texttrace()):
                text = "".join(chr(c[0]) for c in span["chars"])
                if not text.strip():
                    continue
                # 键里带上不透明度：同名文字若透明度不同，是两回事
                key = (span["font"], round(span["size"], 1),
                       tuple(round(float(v), 3) for v in span["color"]),
                       round(float(span.get("opacity", 1) or 1), 3), text)
                groups[key].append({
                    "page": pno,
                    "span_index": si,
                    "bbox": tuple(round(float(v), 1) for v in span["bbox"]),
                    "edge": _near_edge(span["bbox"], pw, ph),
                })

        # 先推定「正文风格」：出现最多且不透明的那几种字体/颜色
        body_fonts, body_colors = _infer_body_style(doc, groups)

        candidates = []
        for idx, (key, items) in enumerate(
                sorted(groups.items(),
                       key=lambda kv: -len(kv[1])), start=1):
            font, size, color, opacity, text = key
            count = len(items)
            at_edge = all(i["edge"] for i in items)

            # ---- 几何验证：不信文字层（防 ToUnicode 造假）----
            page0 = doc[items[0]["page"]]
            prof = glyph.profile(page0, items[0]["bbox"], color)
            verdict = glyph.classify(prof)

            # ---- 物理特征评分：看它"长什么样"，不看它写什么 ----
            span0 = doc[items[0]["page"]].get_texttrace()[items[0]["span_index"]]
            sig = SC.score_span(span0, count=count, at_edge=at_edge,
                                body_fonts=body_fonts, body_colors=body_colors)

            if verdict == "lines":
                risk, why = "keep", (
                    "字形几何呈线条结构（高覆盖贯穿线），文字层可能被伪造 —— "
                    "这是图案/辅助线，不应删除")
            else:
                risk = SC.grade(sig["score"])
                why = SC.explain(sig)

            candidates.append({
                "id": idx,
                "text": text if len(text) <= 40 else text[:37] + "…",
                "font": font,
                "size": size,
                "color": [round(float(c), 3) for c in color],
                "count": count,
                "at_edge": at_edge,
                "geometry": {
                    "verdict": verdict,
                    "explain": glyph.explain(prof),
                    "profile": {k: v for k, v in prof.items() if k != "m"},
                },
                "risk": risk,
                "reason": why,
                "positions": [i["bbox"] for i in items[:5]],
                "pages": sorted({i["page"] for i in items}),
            })

        summary = collections.Counter(c["risk"] for c in candidates)
        return {
            "file": os.path.abspath(path),
            "pages": doc.page_count,
            "size_bytes": os.path.getsize(path),
            "candidates": candidates,
            "summary": dict(summary),
        }
    finally:
        doc.close()


def render_report(rep: dict, *, verbose: bool = False) -> str:
    """把报告渲染成给人看的文本。"""
    risk_label = {
        "keep": "保留  ",
        "safe_delete": "删★   ",
        "likely_delete": "删?   ",
        "review": "待定  ",
    }
    _ = EDGE_RATIO
    lines = []
    s = rep["summary"]
    lines.append(f"文件：{rep['file']}")
    lines.append(f"页数：{rep['pages']}　候选：{len(rep['candidates'])} 组"
                 f"　→ 建议删除 {s.get('safe_delete', 0) + s.get('likely_delete', 0)} 组，"
                 f"保留 {s.get('keep', 0)} 组，待定 {s.get('review', 0)} 组")
    lines.append("")
    for c in rep["candidates"]:
        tag = risk_label.get(c["risk"], c["risk"])
        if c["risk"] == "review" and not verbose:
            continue
        lines.append(f"[{tag}] #{c['id']} {c['text']!r}")
        lines.append(f"         {c['font']} {c['size']}pt "
                     f"色 rgb{tuple(round(v*255) for v in c['color'])} × {c['count']} 处"
                     + ("（页边）" if c["at_edge"] else ""))
        lines.append(f"         几何：{c['geometry']['explain']}")
        lines.append(f"         结论：{c['reason']}")
        lines.append("")
    return "\n".join(lines)
