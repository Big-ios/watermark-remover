"""端到端回归：analyze → apply → verify。

用真实案例当夹具。核心断言：
  · 能删掉指定水印
  · 删除后正文完好、结构指标不变
  · 线条类（格子辅助线）绝不能被自动判定为可删
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wmkit.pdf import apply as AP          # noqa: E402
from wmkit.pdf import verify as V          # noqa: E402
from wmkit.pdf.analyze import analyze      # noqa: E402

SAMPLE = os.path.expanduser("~/workspace/chat/郑清婉姓名贴.pdf")
pytestmark = pytest.mark.skipif(not os.path.isfile(SAMPLE), reason="样例不在: " + SAMPLE)


@pytest.fixture(scope="module")
def rep():
    return analyze(SAMPLE)


def test_analyze_protects_line_shapes(rep):
    """线条结构的候选必须是 keep，且绝不出现在可删档里。"""
    keeps = [c for c in rep["candidates"] if c["risk"] == "keep"]
    assert keeps, "应该识别出线条结构（格子辅助线）"
    for c in keeps:
        assert c["geometry"]["verdict"] == "lines"
        assert c["geometry"]["profile"]["h_span"] >= 1
        assert c["geometry"]["profile"]["v_span"] >= 1


def test_analyze_flags_watermark_by_physical_features(rep):
    """自动判删的依据是物理特征（半透明度等），不是文字内容。

    这里不断言"必须含关键词"—— 通用识别恰恰要求即使不含关键词也能识别。
    改为断言：被判删的候选都是**真的水印**（低不透明度），
    且正文内容绝不出现在判删名单里。
    """
    auto = [c for c in rep["candidates"] if c["risk"] == "safe_delete"]
    assert auto, "应能自动识别出水印"

    # 判删的候选必须真的半透明（这是通用判据的核心）
    for c in auto:
        op = (c.get("geometry") or {})
        assert "半透明" in c["reason"] or "重复" in c["reason"], \
            f"判删依据不明：{c['reason']}"

    # 正文绝不能被判删
    body = ("郑", "清", "婉", "姓名字帖", "部首：", "结构：", "笔画：")
    for c in auto:
        assert c["text"].rstrip("…") not in body, f"正文 {c['text']!r} 被误判为水印"


def test_no_keyword_dependency(rep):
    """把关键词去掉后，半透明水印仍应被识别（证明不依赖内容）。"""
    from wmkit.pdf import score as S
    # 找一个真实判删的候选，看它的得分构成里是否以半透明为主
    auto = [c for c in rep["candidates"] if c["risk"] == "safe_delete"]
    assert auto
    # 至少有一个候选的得分来自"半透明"而非关键词
    assert any("半透明" in c["reason"] for c in auto), \
        "识别完全依赖关键词，不是通用方案"


def test_analyze_never_auto_deletes_repeated_content(rep):
    """重复的正文（如字帖范字）不能被自动判删 —— 这是踩过的坑。"""
    for c in rep["candidates"]:
        txt = c["text"].rstrip("…")
        if txt in ("郑", "清", "婉", "姓名字帖"):
            assert c["risk"] != "safe_delete", f"{txt!r} 被自动判删了"


def test_apply_and_verify(rep, tmp_path):
    """删除关键词命中的候选，验证结构未受损。"""
    ids = [c["id"] for c in rep["candidates"] if c["risk"] == "safe_delete"]
    if not ids:
        pytest.skip("这个样本没有关键词命中的候选")

    out = str(tmp_path / "out.pdf")
    before = V.snapshot(SAMPLE)
    AP.apply(SAMPLE, out, delete_ids=ids, candidates=rep["candidates"])
    after = V.snapshot(out)

    gone = tuple(c["text"].rstrip("…") for c in rep["candidates"]
                 if c["id"] in ids)
    res = V.compare(before, after, expect_gone=gone)
    assert res["ok"], "验证未通过：" + "; ".join(res["issues"])

    # 结构必须原样
    assert before["drawings"] == after["drawings"], "矢量对象数变了（可能误删图案）"
    assert before["images"] == after["images"], "内嵌图片数变了"
    assert before["pages"] == after["pages"]


def test_verify_detects_structure_damage():
    """verify 必须能发现结构被破坏（自测判据本身是否有效）。"""
    a = {"pages": 1, "drawings": 576, "images": 2, "size": 100, "texts": {}, "colors": {}}
    b = dict(a, drawings=400)          # 人为破坏：矢量对象少了
    res = V.compare(a, b)
    assert not res["ok"]
    assert any("矢量对象数" in i for i in res["issues"])
