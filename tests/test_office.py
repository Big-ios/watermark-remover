"""Office 水印清理的回归测试。

样本是现场构造的最小 DOCX：正文 + 页眉水印 + 页脚推广。
关键断言：删水印之后**包结构必须与原件一致**（否则 Office 会报文件损坏）。
"""
import os
import sys
import zipfile

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from wmkit.office import apply as AP            # noqa: E402
from wmkit.office.analyze import analyze        # noqa: E402

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _make_docx(path, *, body="正文内容", header="速创字帖 免费版",
               footer="公众号：速创字帖　超级会员可去除水印"):
    doc = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document xmlns:w="{W}"><w:body><w:p><w:r><w:t>{body}</w:t></w:r></w:p></w:body></w:document>'
    hdr = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:hdr xmlns:w="{W}"><w:p><w:r><w:t>{header}</w:t></w:r></w:p></w:hdr>'
    ftr = f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:ftr xmlns:w="{W}"><w:p><w:r><w:t>{footer}</w:t></w:r></w:p></w:ftr>'
    ct = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
          '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="xml" ContentType="application/xml"/></Types>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>')
        z.writestr("word/document.xml", doc)
        z.writestr("word/header1.xml", hdr)
        z.writestr("word/footer1.xml", ftr)


@pytest.fixture()
def docx(tmp_path):
    p = tmp_path / "in.docx"
    _make_docx(str(p))
    return str(p)


def test_analyze_finds_header_footer_watermark(docx):
    rep = analyze(docx)
    texts = {c["text"]: c for c in rep["candidates"]}
    assert any("速创字帖" in t for t in texts), "应识别出页眉水印"
    promo = [c for c in rep["candidates"] if c["risk"] == "safe_delete"]
    assert len(promo) >= 2, "页眉与页脚都应命中关键词"
    # 正文不能被自动判删
    body = [c for c in rep["candidates"] if "正文内容" in c["text"]]
    assert body and body[0]["risk"] == "review"


def test_remove_preserves_package_structure(docx, tmp_path):
    out = str(tmp_path / "out.docx")
    rep = analyze(docx)
    texts = [c["text"] for c in rep["candidates"] if c["risk"] == "safe_delete"]
    AP.apply(docx, out, delete_texts=texts)

    zi, zo = zipfile.ZipFile(docx), zipfile.ZipFile(out)
    assert zi.namelist() == zo.namelist(), "包内条目与顺序必须一致"


def test_remove_clears_watermark_keeps_body(docx, tmp_path):
    out = str(tmp_path / "out.docx")
    rep = analyze(docx)
    texts = [c["text"] for c in rep["candidates"] if c["risk"] == "safe_delete"]
    AP.apply(docx, out, delete_texts=texts)

    rep2 = analyze(out)
    left = [c["text"] for c in rep2["candidates"]]
    assert not any("速创字帖" in t for t in left), f"水印残留: {left}"
    assert not any("公众号" in t for t in left), f"推广残留: {left}"
    assert any("正文内容" in t for t in left), "正文被误删了"


def test_partial_removal_keeps_untargeted_text(docx, tmp_path):
    """只删页脚时，页眉应原样保留 —— 证明删除是精确到节点的。"""
    out = str(tmp_path / "out2.docx")
    rep = analyze(docx)
    footer = [c["text"] for c in rep["candidates"] if "公众号" in c["text"]]
    assert footer
    AP.apply(docx, out, delete_texts=footer)

    rep2 = analyze(out)
    left = [c["text"] for c in rep2["candidates"]]
    assert any("速创字帖 免费版" in t for t in left), "页眉不该被删"
    assert not any("公众号" in t for t in left)


def test_drop_media_removes_part(docx, tmp_path):
    """删除内嵌图片部件（logo/二维码）。"""
    src = str(tmp_path / "withimg.docx")
    with zipfile.ZipFile(docx) as zin, zipfile.ZipFile(src, "w") as zout:
        for n in zin.namelist():
            zout.writestr(n, zin.read(n))
        zout.writestr("word/media/image1.png", b"\x89PNG\r\n\x1a\n fake")
    out = str(tmp_path / "noimg.docx")
    st = AP.apply(src, out, delete_texts=[], delete_media=["word/media/image1.png"])
    assert st["media_removed"] == 1
    assert "word/media/image1.png" not in zipfile.ZipFile(out).namelist()
