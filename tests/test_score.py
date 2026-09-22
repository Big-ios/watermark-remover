"""通用评分判据的回归测试。

这个模块存在的理由：早期版本靠"推广关键词"判水印，只对公众号类有效。
现在主判据是**半透明度**等物理特征，与文字内容无关 —— 本文件就是把这个
性质钉死：换个水印文字，识别结果不许变。
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wmkit.pdf import score as S  # noqa: E402


def _span(text="测试水印", *, opacity=1.0, color=(0.0, 0.0, 0.0),
          font="somefont", size=48.0):
    return {
        "chars": [(ord(c), 0, (0, 0), (0, 0, 0, 0)) for c in text],
        "color": color, "font": font, "size": size,
        "opacity": opacity,
        "get": lambda k, d=None: {"opacity": opacity}.get(k, d),
    }


def test_translucent_is_strongest_signal():
    """半透明 —— 与水印文字无关的最强判据。"""
    faint = S.score_span(_span("随便什么字", opacity=0.05))
    solid = S.score_span(_span("随便什么字", opacity=1.0))
    assert faint["score"] - solid["score"] >= 40
    assert S.grade(faint["score"]) == "safe_delete"


def test_content_independent():
    """核心性质：换文字不影响判定。

    刻意取都不含推广关键词的样本，才能验证"与内容无关"这一点。
    """
    samples = ["AAA", "第 3 季度财务报表", "xyz-2024", "折线图示例"]
    scores = [S.score_span(_span(t, opacity=0.08))["score"] for t in samples]
    assert max(scores) - min(scores) <= 5, f"文字内容影响了得分：{list(zip(samples, scores))}"
    for t, sc in zip(samples, scores):
        assert S.grade(sc) == "safe_delete", f"{t!r} 应因半透明被判删，实际 {sc} 分"


def test_very_faint_outweighs_plain():
    assert S.score_span(_span(opacity=0.02))["score"] > \
           S.score_span(_span(opacity=0.4))["score"]


def test_opaque_body_text_not_flagged():
    """与正文同字体同颜色的不透明文字，必须拿不到高分。"""
    sig = S.score_span(_span("正文内容", opacity=1.0, font="body", color=(0, 0, 0)),
                       count=20, at_edge=False,
                       body_fonts={"body"}, body_colors={(0, 0, 0)})
    assert S.grade(sig["score"]) != "safe_delete", f"正文被误判：{sig}"


def test_repetition_alone_is_not_enough():
    """重复本身不能定罪 —— 字帖范字、表格表头都会重复。"""
    sig = S.score_span(_span("范字", opacity=1.0, font="body", color=(0, 0, 0)),
                       count=50, at_edge=True,
                       body_fonts={"body"}, body_colors={(0, 0, 0)})
    assert S.grade(sig["score"]) == "review", f"仅凭重复就判删了：{sig}"


def test_edge_alone_is_not_enough():
    """页边也不能定罪 —— 标题、页码也在页边。"""
    sig = S.score_span(_span("标题", opacity=1.0, font="body", color=(0, 0, 0)),
                       count=1, at_edge=True,
                       body_fonts={"body"}, body_colors={(0, 0, 0)})
    assert S.grade(sig["score"]) == "review"


def test_promo_keyword_is_bonus_not_requirement():
    """关键词降级为加分项：有它更容易判删，没它也能靠半透明判删。"""
    with_kw = S.score_span(_span("公众号水印", opacity=0.8))
    without = S.score_span(_span("某某公司", opacity=0.8))
    assert with_kw["score"] > without["score"]
    # 但没有关键词时，半透明仍足以判删
    assert S.grade(S.score_span(_span("某某公司", opacity=0.05))["score"]) == "safe_delete"


def test_breakdown_is_explainable():
    sig = S.score_span(_span("x", opacity=0.05), count=5, at_edge=True)
    assert sig["breakdown"], "应给出可解释的加分明细"
    assert "半透明" in S.explain(sig)
