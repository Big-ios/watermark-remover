"""按计划执行删除 —— 内容流块级手术。

铁律（都是踩坑换来的）
  1. **只删文字绘制块**（BT…ET）与图片绘制调用（Do），
     绝不用"红删/涂白"—— 那会连带擦掉底下的格线、表格线。
  2. **保留 q/Q 配对**：只摘 BT…ET 之间的内容，图形状态不会失衡。
  3. **不用 garbage>=2**：它会重排对象编号，让手写的资源引用失效
     （开发中曾因此丢掉内嵌图片）。
  4. **原文件不动**，永远输出到新文件。
"""
from __future__ import annotations

import os
import re

BLOCK = re.compile(rb"BT\n(.*?)ET\n", re.S)
Tj = re.compile(rb"<([0-9A-Fa-f]{4,})>\s*Tj")


def _cids_of_block(body: bytes) -> str:
    return b"".join(Tj.findall(body)).decode("ascii").upper()


def collect_targets(doc, wanted_ids, candidates) -> dict:
    """把"要删的候选 id"翻译成「内容流里该删哪些块」的判据。

    返回 {"cids": set(...) 精确匹配的 CID 串, "texts": set(...) 文本}
    """
    by_id = {c["id"]: c for c in candidates}
    texts = set()
    for i in wanted_ids:
        c = by_id.get(i)
        if c:
            texts.add(c["text"].rstrip("…"))
    return texts


def apply(src: str, dst: str, *, delete_ids, candidates,
          keep_gezi: bool = True, verbose: bool = False) -> dict:
    """删除指定候选对应的文字块。

    delete_ids   要删的候选 id 列表
    candidates   analyze() 产出的候选列表
    keep_gezi    跳过 gezi 字体的块（格子辅助线的常见载体），默认 True
    """
    import pymupdf

    texts = collect_targets(None, delete_ids, candidates)
    if verbose:
        print(f"目标文本：{sorted(texts)}")

    doc = pymupdf.open(src)
    stats = {"blocks_removed": 0, "bytes_before": 0, "bytes_after": 0, "pages": 0}
    try:
        for page in doc:
            raw = page.read_contents()
            stats["bytes_before"] += len(raw)
            cuts = []
            for m in BLOCK.finditer(raw):
                body = m.group(1)
                if keep_gezi:
                    # gezi 字体的块一律保留（辅助线的 ToUnicode 会被伪装成水印文本，
                    # 文本匹配在这里不可信；几何判据已在 analyze 阶段判过）
                    head = raw[max(0, m.start() - 400): m.start()]
                    if b"/F16 " in head or b"gezi" in head:
                        continue
                cids = _cids_of_block(body)
                if not cids:
                    continue
                # 用 CID 反查这个块对应的文本，与目标比对
                t = _text_of_cids(page, cids)
                if t is not None and t in texts:
                    cuts.append((m.start(), m.end()))
            if cuts:
                new = raw
                for s, e in sorted(set(cuts), reverse=True):
                    new = new[:s] + new[e:]
                doc.update_stream(page.get_contents()[0], new)
                stats["blocks_removed"] += len(cuts)
                stats["bytes_after"] += len(new)
            else:
                stats["bytes_after"] += len(raw)
            stats["pages"] += 1

        os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
        doc.save(dst, garbage=1, deflate=True, clean=True)
    finally:
        doc.close()
    stats["output"] = os.path.abspath(dst)
    stats["size"] = os.path.getsize(dst) if os.path.exists(dst) else 0
    return stats


def _text_of_cids(page, cids_hex: str):
    """把一段 CID 串反查回文本（借助页面的 texttrace 建立映射）。"""
    if not hasattr(page, "_wm_cid_map"):
        mapping = {}
        for span in page.get_texttrace():
            h = "".join(f"{c[1]:04X}" for c in span["chars"])
            t = "".join(chr(c[0]) for c in span["chars"])
            mapping[h] = t
        page._wm_cid_map = mapping
    return page._wm_cid_map.get(cids_hex)
