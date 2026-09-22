"""PDF 引擎：识别、规划、执行、验证。

用法
    from wmkit.pdf import analyze as A
    rep = A.analyze("in.pdf")          # 只读，产出报告
    from wmkit.pdf import apply as AP
    AP.apply("in.pdf", rep, "out.pdf")  # 按报告执行
"""
from . import analyze, apply, glyph, verify  # noqa: F401

__all__ = ["analyze", "apply", "glyph", "verify"]
