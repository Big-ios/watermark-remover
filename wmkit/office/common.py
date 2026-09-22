"""OOXML 包结构的公共工具。"""
from __future__ import annotations

import os
import zipfile

# 各命名空间
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",   # Word
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",          # Drawing（PPT 文本）
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",     # PPT
    "x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",      # Excel
}

# 哪些部件里可能出现水印文字
TEXT_PART_HINTS = (
    "word/document.xml", "word/header", "word/footer", "word/footnotes",
    "word/endnotes", "word/comments",
    "ppt/slides/slide", "ppt/slideLayouts/", "ppt/slideMasters/",
    "ppt/notesSlides/", "xl/sharedStrings.xml", "xl/worksheets/",
    "xl/drawings/",
)

# 内嵌图片
MEDIA_HINT = "media/"

# 推广类关键词（与 PDF 模块同一套，保持行为一致）
PROMO_KEYWORDS = (
    "公众号", "微信", "水印", "会员", "扫码", "二维码", "关注", "免费", "下载",
    "网址", "www.", "http", "©", "®", "版权", "侵权", "定制", "咨询", "联系",
    "客服", "淘宝", "抖音", "小红书", "店铺", "搜", "加我",
)


def is_text_part(name: str) -> bool:
    return any(h in name for h in TEXT_PART_HINTS) and name.endswith(".xml")


def is_media_part(name: str) -> bool:
    return MEDIA_HINT in name and not name.endswith("/")


def kind_of(name: str) -> str:
    """给部件分类，供报告展示。"""
    if "word/document.xml" == name:
        return "正文"
    if "header" in name:
        return "页眉"
    if "footer" in name:
        return "页脚"
    if name.startswith("ppt/slides/slide"):
        return "幻灯片"
    if "slideMaster" in name or "slideLayout" in name:
        return "版式"
    if name.startswith("xl/"):
        return "工作表"
    if is_media_part(name):
        return "内嵌图片"
    return "其他"


def open_package(path: str) -> zipfile.ZipFile:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return zipfile.ZipFile(path, "r")


def rewrite_package(src: str, dst: str, transform) -> dict:
    """把 src 的每个部件交给 transform(name, data) -> bytes|None，写进 dst。

    transform 返回 None 表示"删除这个部件"。
    保留原有的条目顺序与压缩方式 —— Office 对顺序敏感。
    """
    stats = {"parts": 0, "changed": 0, "removed": 0, "bytes_in": 0, "bytes_out": 0}
    with zipfile.ZipFile(src, "r") as zin:
        infos = zin.infolist()
        with zipfile.ZipFile(dst, "w") as zout:
            for info in infos:
                data = zin.read(info.filename)
                stats["parts"] += 1
                stats["bytes_in"] += len(data)
                new = transform(info.filename, data)
                if new is None:
                    stats["removed"] += 1
                    continue
                if new != data:
                    stats["changed"] += 1
                # 用原条目的压缩方式与元数据写回
                zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                zi.compress_type = info.compress_type
                zi.external_attr = info.external_attr
                zout.writestr(zi, new)
                stats["bytes_out"] += len(new)
    return stats
