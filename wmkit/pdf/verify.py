"""量化验证 —— 删除前后对比，确保只清了水印、没伤正文。

为什么必须有这一层
------------------
开发过程中曾发生：按文字层删除，把 120 个田字格辅助线一起删了，
而"文字层看起来完全正常"。肉眼看图不一定发现，必须用数字说话。

指标分两类：
  · 消失类（期望下降）：目标水印文本的出现次数
  · 保护类（期望不变）：矢量对象数、正文文字、页面尺寸
"""
from __future__ import annotations

import collections
import os


def snapshot(path: str) -> dict:
    """给文件拍一张"可量化的指纹"。"""
    import numpy as np
    import pymupdf

    doc = pymupdf.open(path)
    try:
        texts = collections.Counter()
        colors = collections.Counter()
        drawings = 0
        images = 0
        for page in doc:
            drawings += len(page.get_drawings())
            images += len(page.get_images(full=True))
            for span in page.get_texttrace():
                t = "".join(chr(c[0]) for c in span["chars"]).strip()
                if t:
                    texts[t] += 1
                colors[tuple(round(float(v), 2) for v in span["color"])] += len(span["chars"])
            px = page.get_pixmap(dpi=72)
            a = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width, px.n)[:, :, :3]
            for rgb, name in (((110, 123, 139), "grayblue"),
                              ((84, 139, 84), "green"),
                              ((0, 0, 0), "black")):
                m = (np.abs(a.astype(int) - np.array(rgb)).sum(axis=2) < 60).sum()
                colors[(name,)] = colors.get((name,), 0) + int(m)
        return {
            "path": os.path.abspath(path),
            "size": os.path.getsize(path),
            "pages": doc.page_count,
            "drawings": drawings,
            "images": images,
            "texts": dict(texts),
            "colors": {str(k): v for k, v in colors.items()},
            "bytes": sum(len(p.read_contents()) for p in doc),
        }
    finally:
        doc.close()


def compare(before: dict, after: dict, expect_gone=()) -> dict:
    """对比两份指纹，给出结论。"""
    issues = []
    notes = []

    # 保护类：结构性指标必须不变
    for key, label in (("pages", "页数"), ("drawings", "矢量对象数"),
                       ("images", "内嵌图片数")):
        if before[key] != after[key]:
            issues.append(f"{label}变化：{before[key]} → {after[key]}（本不该变）")

    # 消失类：目标文本应归零
    for t in expect_gone:
        n = after["texts"].get(t, 0)
        if n:
            issues.append(f"目标水印 {t!r} 仍残留 {n} 处")
        else:
            notes.append(f"已清除 {t!r}（原 {before['texts'].get(t, 0)} 处）")

    # 颜色层面的变化量（提示性）
    for name in ("grayblue", "green"):
        k = f"('{name}',)"
        b, a = before["colors"].get(k, 0), after["colors"].get(k, 0)
        if b:
            pct = 100 * (a - b) / b
            notes.append(f"{name} 像素 {b} → {a}（{pct:+.2f}%）")
            if name == "green" and abs(pct) > 5:
                issues.append(f"绿色像素变化 {pct:.2f}% —— 可能误删了格子线类元素")

    return {
        "ok": not issues,
        "issues": issues,
        "notes": notes,
        "before": {k: before[k] for k in ("size", "pages", "drawings", "images")},
        "after": {k: after[k] for k in ("size", "pages", "drawings", "images")},
    }


def render(res: dict) -> str:
    lines = []
    head = "✅ 验证通过" if res["ok"] else "❌ 验证发现问题"
    lines.append(head)
    for n in res["notes"]:
        lines.append(f"   · {n}")
    for i in res["issues"]:
        lines.append(f"   ⚠ {i}")
    return "\n".join(lines)
