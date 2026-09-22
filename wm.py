#!/usr/bin/env python3
"""wm —— 通用水印清理工具（CLI 主体）。

    wm analyze <文件>               只读扫描，打印水印报告
    wm remove  <文件> --auto        删除自动判定的部分（关键词命中）
    wm remove  <文件> --ids 8,15    按候选编号删除
    wm remove  <文件> --ids all --yes   删全部候选
    wm batch   <目录> --yes         批量

默认只预演、不写文件；加 --yes 才执行。原文件永不被改动。
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _default_out(src: str) -> str:
    root, ext = os.path.splitext(src)
    return f"{root}-cleaned{ext or '.pdf'}"


def cmd_analyze(args) -> int:
    from wmkit.pdf.analyze import analyze, render_report

    rep = analyze(args.file)
    print(render_report(rep, verbose=args.verbose))
    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"（报告已写入 {args.json}）")
    return 0


def cmd_remove(args) -> int:
    from wmkit.pdf import apply as AP
    from wmkit.pdf import verify as V
    from wmkit.pdf.analyze import analyze

    rep = analyze(args.file)
    cands = rep["candidates"]

    if args.ids == "all":
        ids = [c["id"] for c in cands]
    elif args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = [c["id"] for c in cands if c["risk"] == "safe_delete"]

    by_id = {c["id"]: c for c in cands}
    ids = [i for i in ids if i in by_id]
    if not ids:
        print("没有要删除的候选。用 --ids 指定编号（先跑 wm analyze 查看）。")
        return 0

    print("将要删除：")
    for i in ids:
        c = by_id[i]
        print(f"   #{i} {c['text']!r} × {c['count']} 处  [{c['risk']}]")

    out = args.out or _default_out(args.file)
    if not args.yes:
        print()
        print(f"这是预演，未写文件。确认后加 --yes 执行（输出到 {out}）。")
        return 0

    before = V.snapshot(args.file)
    st = AP.apply(args.file, out, delete_ids=ids, candidates=cands, verbose=args.verbose)
    after = V.snapshot(out)

    gone = tuple(by_id[i]["text"].rstrip("…") for i in ids)
    res = V.compare(before, after, expect_gone=gone)
    print()
    print(f"已写出：{out}　（删除 {st['blocks_removed']} 个文字块）")
    print(V.render(res))
    return 0 if res["ok"] else 1


def cmd_batch(args) -> int:
    from wmkit.pdf.analyze import analyze

    files = []
    for root, _dirs, names in os.walk(args.dir):
        for n in names:
            if n.lower().endswith(".pdf") and not n.endswith("-cleaned.pdf"):
                files.append(os.path.join(root, n))
    if not files:
        print("目录里没有 PDF")
        return 0
    print(f"找到 {len(files)} 个 PDF")
    rc = 0
    for f in files:
        rep = analyze(f)
        ids = [c["id"] for c in rep["candidates"] if c["risk"] == "safe_delete"]
        print(f"  {os.path.basename(f):<44} 自动判定可删 {len(ids)} 组")
        if ids and args.yes:
            ns = argparse.Namespace(file=f, ids=",".join(map(str, ids)), out=None,
                                    yes=True, verbose=False)
            rc |= cmd_remove(ns)
    if not args.yes:
        print()
        print("（预演模式，未写文件；加 --yes 执行）")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="wm", description="通用水印清理工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("analyze", help="扫描文件，打印水印报告（不改文件）")
    p.add_argument("file")
    p.add_argument("--json", default=None, help="把报告另存为 JSON")
    p.add_argument("-v", "--verbose", action="store_true", help="连待定项一起列出")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("remove", help="按报告删除水印")
    p.add_argument("file")
    p.add_argument("--ids", default=None, help="要删的候选编号，如 8,15,16 或 all")
    p.add_argument("--auto", action="store_true", help="只删自动判定的（关键词命中）")
    p.add_argument("--out", default=None, help="输出路径")
    p.add_argument("--yes", action="store_true", help="确认执行（否则只预演）")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_remove)

    # 位图与 Office 相关子命令
    from wmkit.image.cli import add_image_parser
    from wmkit.office.cli import add_office_parser
    from wmkit.video.cli import add_video_parser
    add_image_parser(sub)
    add_office_parser(sub)
    add_video_parser(sub)

    p = sub.add_parser("batch", help="批量处理目录下的 PDF")
    p.add_argument("dir")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_batch)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
