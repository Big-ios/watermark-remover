"""按报告删除 OOXML 包里的水印。

做法：把目标文本所在的**整个段落**删掉，或者只删文字本身（保留段落）。
默认删段落 —— 页眉页脚的推广文字通常自己独占一段，删段落更干净。

**包结构必须原样保留**：条目顺序、压缩方式、[Content_Types].xml 的位置
都不能动，否则 Office 会报文件损坏。
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from . import common as C


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _drop_paragraphs(data: bytes, targets: set, *, whole_paragraph: bool) -> tuple:
    """删除含目标文本的节点。返回 (新字节, 删了几个)。"""
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return data, 0

    removed = 0
    if whole_paragraph:
        # 自底向上找"段落"级容器，其全部文字命中目标就整段删
        containers = ("p", "tr", "tbl")
        for parent in list(root.iter()):
            for child in list(parent):
                if _local(child.tag) not in containers:
                    continue
                text = "".join(t.text or "" for t in child.iter()
                               if _local(t.tag) == "t")
                if text.strip() and any(tg in text for tg in targets):
                    try:
                        parent.remove(child)
                        removed += 1
                    except ValueError:
                        pass
    else:
        for el in list(root.iter()):
            if _local(el.tag) != "t" or not el.text:
                continue
            if any(tg in el.text for tg in targets):
                el.text = ""
                removed += 1

    if not removed:
        return data, 0

    # 写回时保留 XML 声明，并声明常用命名空间前缀
    ET.register_namespace("w", C.NS["w"])
    ET.register_namespace("a", C.NS["a"])
    ET.register_namespace("p", C.NS["p"])
    ET.register_namespace("r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships")
    body = ET.tostring(root, encoding="UTF-8", xml_declaration=True)
    return body, removed


def apply(src: str, dst: str, *, delete_texts, delete_media=(),
          whole_paragraph: bool = True) -> dict:
    """执行删除。

    delete_texts   要删的文本片段（候选里的 text）
    delete_media   要删的 media 部件名（如 'word/media/image1.png'）
    whole_paragraph True=整段删；False=只清空文字保留段落
    """
    targets = {t.rstrip("…") for t in delete_texts if t}
    drop_media = set(delete_media)
    counters = {"text_nodes": 0, "parts_touched": 0, "media_removed": 0}

    def transform(name: str, data: bytes):
        if name in drop_media:
            counters["media_removed"] += 1
            return None
        if not C.is_text_part(name):
            return data
        new, n = _drop_paragraphs(data, targets, whole_paragraph=whole_paragraph)
        if n:
            counters["text_nodes"] += n
            counters["parts_touched"] += 1
        return new

    st = C.rewrite_package(src, dst, transform)
    st.update(counters)
    st["output"] = os.path.abspath(dst)
    st["size"] = os.path.getsize(dst)
    return st
