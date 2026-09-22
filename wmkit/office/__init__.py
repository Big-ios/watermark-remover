"""Office 水印清理。

原理
----
DOCX / PPTX / XLSX 本质上都是 ZIP 包，内部是一堆 XML（OOXML 规范）。
水印通常出现在三处：
    1. 页眉 / 页脚（header*.xml、footer*.xml）—— 最常见，推广信息、落款
    2. 正文各页（document.xml、slide*.xml）—— 平铺水印、背景文字
    3. 内嵌图片（media/*）—— logo、二维码、印章

处理方式：解压 → 定位并删除对应节点/文件 → 按原结构重新打包。

与 PDF 的差别：OOXML 里文本就是文本，没有 PDF 那种 ToUnicode 造假陷阱；
但**包结构不能乱**（条目顺序、[Content_Types].xml 的位置都有讲究），
所以本模块只改该改的部件，其余原样写回。
"""
__all__ = ["analyze", "apply", "cli", "common"]
