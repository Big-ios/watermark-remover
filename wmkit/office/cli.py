"""Office 水印清理的命令行实现。

    wm office analyze <文件.docx>          只读扫描，打印候选报告
    wm office remove  <文件.docx> --auto   删除关键词命中的（预演）
    wm office remove  <文件.docx> --ids 2,3 --yes
    wm office remove  <文件.docx> --drop-media word/media/image1.png --yes
    wm office batch   <目录> --auto --yes
"""
from __future__ import annotations

import os


def _default_out(src: str) -> str:
    root, ext = os.path.splitext(src)
    return f"{root}-cleaned{ext}"


def cmd_office_analyze(args) -> int:
    from wmkit.office.analyze import analyze, render_report

    rep = analyze(args.file)
    print(render_report(rep, verbose=args.verbose))
    if args.json:
        import json
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"（报告已写入 {args.json}）")
    return 0


def cmd_office_remove(args) -> int:
    from wmkit.office import apply as AP
    from wmkit.office.analyze import analyze, render_report

    rep = analyze(args.file)
    cands = rep["candidates"]
    by_id = {c["id"]: c for c in cands}

    if args.ids == "all":
        ids = [c["id"] for c in cands]
    elif args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        ids = [c["id"] for c in cands if c["risk"] == "safe_delete"]
    ids = [i for i in ids if i in by_id]

    drop = args.drop_media or []

    if not ids and not drop:
        print("没有要删除的内容。先跑 wm office analyze 看看有哪些候选。")
        return 0

    print("将要删除：")
    for i in ids:
        print(f"   #{i} {by_id[i]['text']!r} × {by_id[i]['count']} 处  [{by_id[i]['risk']}]")
    for m in drop:
        print(f"   内嵌图片 {m}")

    out = args.out or _default_out(args.file)
    if not args.yes:
        print()
        print(f"这是预演，未写文件。确认后加 --yes（输出到 {out}）。")
        return 0

    texts = [by_id[i]["text"] for i in ids]
    st = AP.apply(args.file, out, delete_texts=texts, delete_media=drop,
                  whole_paragraph=not args.keep_paragraph)
    print()
    print(f"已写出：{out}")
    print(f"   改动部件 {st['parts_touched']} 个，删除文本节点 {st['text_nodes']} 处"
          + (f"，删除内嵌图片 {st['media_removed']} 个" if st["media_removed"] else ""))

    # 验证：重扫，确认目标文本没了、其余还在
    rep2 = analyze(out)
    left = [c["text"] for c in rep2["candidates"]]
    gone = [t.rstrip("…") for t in texts]
    ok = True
    for t in gone:
        if any(t in x or x in t for x in left):
            print(f"   ⚠ 仍有残留：{t!r}")
            ok = False
    if ok:
        print("   ✅ 验证通过：目标文字已清除")
    print(f"   处理后剩 {len(left)} 组文字")
    import zipfile
    same = zipfile.ZipFile(args.file).namelist() == zipfile.ZipFile(out).namelist()
    print(f"   {'✅' if same else '⚠'} 包结构与原文件{'一致' if same else '不一致'}")
    return 0 if ok and same else 1


def cmd_office_batch(args) -> int:
    from wmkit.office.analyze import analyze

    exts = (".docx", ".pptx", ".xlsx", ".docm", ".pptm")
    files = []
    for root, _d, names in os.walk(args.dir):
        for n in names:
            if n.lower().endswith(exts) and "-cleaned" not in n:
                files.append(os.path.join(root, n))
    if not files:
        print("目录里没有 Office 文件")
        return 0
    print(f"找到 {len(files)} 个文件")
    rc = 0
    for f in files:
        rep = analyze(f)
        ids = [c["id"] for c in rep["candidates"] if c["risk"] == "safe_delete"]
        print(f"  {os.path.basename(f):<44} 自动判定可删 {len(ids)} 组")
        if ids and args.yes:
            ns = type(args)(file=f, ids=",".join(map(str, ids)), out=None, yes=True,
                            verbose=False, drop_media=None, keep_paragraph=False)
            rc |= cmd_office_remove(ns)
    if not args.yes:
        print()
        print("（预演模式，未写文件；加 --yes 执行）")
    return rc


def add_office_parser(sub) -> None:
    p = sub.add_parser("office", help="Office 文档水印清理（DOCX/PPTX/XLSX）")
    sp = p.add_subparsers(dest="offcmd", required=True)

    q = sp.add_parser("analyze", help="扫描文件，打印候选报告（不改文件）")
    q.add_argument("file")
    q.add_argument("--json", default=None)
    q.add_argument("-v", "--verbose", action="store_true")
    q.set_defaults(func=cmd_office_analyze)

    q = sp.add_parser("remove", help="按报告删除水印")
    q.add_argument("file")
    q.add_argument("--ids", default=None, help="候选编号，如 2,3 或 all")
    q.add_argument("--auto", action="store_true", help="只删关键词命中的")
    q.add_argument("--drop-media", action="append", default=None, dest="drop_media",
                   help="删除指定内嵌图片部件（如 word/media/image1.png）")
    q.add_argument("--keep-paragraph", action="store_true", dest="keep_paragraph",
                   help="保留空段落（默认整段删除）")
    q.add_argument("--out", default=None)
    q.add_argument("--yes", action="store_true")
    q.add_argument("-v", "--verbose", action="store_true")
    q.set_defaults(func=cmd_office_remove)

    q = sp.add_parser("batch", help="批量处理目录")
    q.add_argument("dir")
    q.add_argument("--yes", action="store_true")
    q.set_defaults(func=cmd_office_batch)
