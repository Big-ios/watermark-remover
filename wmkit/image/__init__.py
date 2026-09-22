"""位图水印清理：定位遮罩 → 修复。

与 PDF 的根本差别
----------------
PDF 里水印是**独立对象**，删掉即可，正文分毫不动。
位图里水印**融进了像素**：observed = original*(1-a) + C*a，
只能"反算"（若能估出 alpha 与颜色）或"重绘"（inpainting）来近似还原。

实测结论（见 tests）
    · 已知 alpha 时，反算误差 ≈ 1/255 —— 近乎无损，远胜 inpainting（≈31/255）
    · alpha 未知时，自动估计**不可靠**（笔画细、背景有纹理时都会失准）
因此定位：工具提供可靠的反算/重绘能力与多种遮罩来源，
**alpha 这类参数交由人（或 AI 看对比图）来定**，不硬猜。
"""
from . import detect, restore  # noqa: F401

__all__ = ["detect", "restore"]
