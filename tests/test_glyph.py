"""字形几何判据的回归测试（用真实案例）。

这两个断言就是本工具存在的理由：
  · 格子辅助线的 ToUnicode 谎称是「速创字帖」，几何判据必须识破
  · 真水印「速创字帖」不能被误判成线条
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wmkit.pdf import glyph  # noqa: E402

SAMPLE = os.path.expanduser(
    "~/workspace/chat/郑清婉姓名贴.pdf")


@pytest.fixture(scope="module")
def page():
    pymupdf = pytest.importorskip("pymupdf")
    if not os.path.isfile(SAMPLE):
        pytest.skip("样例文件不在: " + SAMPLE)
    doc = pymupdf.open(SAMPLE)
    yield doc[0]
    doc.close()


def _spans(page):
    return list(page.get_texttrace())


def test_aux_line_glyph_detected_as_lines(page):
    """格子辅助线：ToUnicode 说是「速创字帖」，几何上是线条 → 必须判 lines。"""
    hits = []
    for s in _spans(page):
        if s["font"].startswith("gezi"):
            r = glyph.analyze(page, s)
            hits.append(r["verdict"])
    assert hits, "样例里应该有 gezi 字形的辅助线"
    assert all(v == "lines" for v in hits), f"辅助线应全部判为 lines，实际 {set(hits)}"


def test_real_watermark_is_not_lines(page):
    """真水印「速创字帖」（changguikaiti 字体）不能判成线条。"""
    checked = 0
    for s in _spans(page):
        t = "".join(chr(c[0]) for c in s["chars"])
        if s["font"].startswith("gezi") or not t.strip():
            continue
        if t.strip() in ("速创字帖", "速", "创", "字", "帖"):
            r = glyph.analyze(page, s)
            if r["verdict"] == "unknown":
                continue
            checked += 1
            assert r["verdict"] == "text", f"{t!r} 被误判为 {r['verdict']}"
    assert checked > 0, "样例里应该有可判定的水印文字"


def test_chinese_names_are_text(page):
    """正文汉字（郑清婉）必须是 text。"""
    checked = 0
    for s in _spans(page):
        t = "".join(chr(c[0]) for c in s["chars"])
        if s["font"].startswith("gezi"):
            continue
        if t.strip() in ("郑", "清", "婉"):
            r = glyph.analyze(page, s)
            if r["verdict"] == "unknown":
                continue
            checked += 1
            assert r["verdict"] == "text", f"{t!r} 被误判为 {r['verdict']}"
    assert checked > 0
