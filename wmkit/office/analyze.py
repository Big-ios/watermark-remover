"""扫描 OOXML 包，列出候选水印。

与 PDF 模块共用同一套评级思路：
  · safe_delete  文本含推广关键词 —— 最强信号
  · review       其余一律交人工（位置、重复都不足以定罪）
  · 内嵌图片     单独列出，由人确认是不是 logo/二维码

不做几何造假检测 —— OOXML 里文本就是文本，没有 PDF 那种 ToUnicode 陷阱。
"""
from __future__ import annotations

import collections
import os
import xml.etree.ElementTree as ET

from . import common as C


def _iter_texts(data: bytes):
    """产出 (元素标签, 文本)。容忍命名空间差异。"""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return
    for el in root.iter():
        if el.text and el.text.strip():
            tag = el.tag.rsplit("}", 1)[-1]
            if tag in ("t", "instrText"):     # w:t / a:t / x:t
                yield tag, el.text


def analyze(path: str) -> dict:
    """扫描 Office 文件，返回水印候选报告（只读）。"""
    groups = collections.defaultdict(list)
    media = []
    with C.open_package(path) as z:
        parts = z.namelist()
        for name in parts:
            if C.is_media_part(name):
                info = z.getinfo(name)
                media.append({"part": name, "size": info.file_size,
                              "ext": os.path.splitext(name)[1].lower()})
                continue
            if not C.is_text_part(name):
                continue
            for _tag, txt in _iter_texts(z.read(name)):
                groups[txt.strip()].append(name)

    candidates = []
    for idx, (text, where) in enumerate(
            sorted(groups.items(), key=lambda kv: -len(kv[1])), start=1):
        kinds = collections.Counter(C.kind_of(w) for w in where)
        promo = next((k for k in C.PROMO_KEYWORDS if k in text), None)
        if promo:
            risk, why = "safe_delete", f"文本含推广关键词「{promo}」"
        else:
            risk, why = "review", (
                f"出现在 {'/'.join(kinds)} 部件里 {len(where)} 处 —— 线索不足，需人工确认")
        candidates.append({
            "id": idx,
            "text": text if len(text) <= 60 else text[:57] + "…",
            "count": len(where),
            "parts": sorted(set(where)),
            "kinds": dict(kinds),
            "risk": risk,
            "reason": why,
        })

    return {
        "file": os.path.abspath(path),
        "size_bytes": os.path.getsize(path),
        "parts": len(parts),
        "media": media,
        "candidates": candidates,
        "summary": dict(collections.Counter(c["risk"] for c in candidates)),
    }


def render_report(rep: dict, *, verbose: bool = False) -> str:
    label = {"keep": "保留  ", "safe_delete": "删★   ",
             "likely_delete": "删?   ", "review": "待定  "}
    s = rep["summary"]
    out = [f"文件：{rep['file']}",
           f"部件：{rep['parts']} 个　内嵌图片：{len(rep['media'])} 个",
           f"文字候选：{len(rep['candidates'])} 组　→ "
           f"建议删除 {s.get('safe_delete', 0)} 组，待定 {s.get('review', 0)} 组",
           ""]
    for c in rep["candidates"]:
        if c["risk"] == "review" and not verbose:
            continue
        out.append(f"[{label.get(c['risk'], c['risk'])}] #{c['id']} {c['text']!r}")
        out.append(f"         出现 {c['count']} 处，部件类型 {c['kinds']}")
        out.append(f"         结论：{c['reason']}")
        out.append("")
    if rep["media"]:
        out.append("内嵌图片（可能是 logo / 二维码，需人确认）：")
        for m in rep["media"]:
            out.append(f"   {m['part']}  {m['size']} 字节")
    return "\n".join(out)
