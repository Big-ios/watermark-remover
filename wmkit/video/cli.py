"""视频水印清理命令行。

    wm video info   <视频>                      看规格
    wm video probe  <视频> [--out 图.png]        定位水印，输出标注图供确认
    wm video remove <视频> --rect x0,y0,x1,y1 --yes
    wm video batch  <目录> --rect ... --yes

定位方式：**框选 --rect 最可靠**（水印位置通常固定，框一次即可）。
--auto 是实验性的粗筛，会把半透明水印的笔画检成碎片，仅作参考。
"""
from __future__ import annotations

import os


def _out(src: str) -> str:
    root, ext = os.path.splitext(src)
    return f"{root}-clean{ext or '.mp4'}"


def cmd_video_info(args) -> int:
    from wmkit.video import io as V

    info = V.probe(args.file)
    print(f"文件：{args.file}")
    print(f"规格：{info.describe()}")
    return 0


def cmd_video_probe(args) -> int:
    import cv2

    from wmkit.video import probe as P

    res = P.detect_translucent(args.file, samples=args.samples)
    st = res["stats"]
    if "error" in st:
        print(f"分析失败：{st['error']}")
        return 1
    print(f"采样 {st['samples']} 帧 · 阈值比 {st['ratio_threshold']}")
    print(f"检出「波动被压缩」的区域：{st['static_px']} px（占 {st['ratio']*100:.2f}%）")
    print()
    if res["boxes"]:
        print("候选区域（面积从大到小）:")
        for b in res["boxes"][:10]:
            print(f"   ({b['x']},{b['y']}) {b['w']}x{b['h']}　面积 {b['area']}")
    else:
        print("未检出候选区域。")

    # 出标注图供人看
    from wmkit.video import io as V
    info = res["info"]
    frame = next(V.iter_frames(args.file, info, count=1))
    ov = P.render_overlay(frame, res["mask"], res["boxes"])
    out = args.out or (os.path.splitext(args.file)[0] + "-探测.png")
    cv2.imwrite(out, ov)
    print()
    print(f"标注图已写出：{out}")
    print("提示：自动定位对半透明水印只能给粗线索（笔画细、波动被压缩的量级接近噪声）。")
    print("      实际处理建议用 --rect 框选，水印位置固定的话框一次就够。")
    return 0


def _parse_rects(items):
    rects = []
    for s in items or []:
        p = [int(float(x)) for x in s.split(",")]
        if len(p) != 4:
            raise SystemExit("--rect 需要 x0,y0,x1,y1 四个数")
        rects.append(tuple(p))
    return rects


def cmd_video_remove(args) -> int:
    from wmkit.video import io as V
    from wmkit.video import process as PR

    info = V.probe(args.file)
    rects = _parse_rects(args.rect)
    if not rects and not args.mask:
        raise SystemExit("请用 --rect x0,y0,x1,y1 指定水印位置（可多次），或 --mask 给遮罩图")

    print(f"输入：{info.describe()}")
    print(f"遮罩：{'框选 ' + str(rects) if rects else ''}"
          f"{' 遮罩图 ' + args.mask if args.mask else ''}")
    print(f"修复：{'alpha 反算 ' + str(args.alpha) if args.alpha else 'inpaint'}"
          f"　时域稳定：{'开（窗口半径 %d）' % args.temporal if args.temporal else '关'}")

    out = args.out or _out(args.file)
    if not args.yes:
        print()
        print(f"这是预演，未写文件。确认后加 --yes（输出到 {out}）。")
        return 0

    def prog(n):
        print(f"\r   已处理 {n} 帧…", end="", flush=True)

    st = PR.process(args.file, out, rects=rects, mask_image=args.mask,
                    alpha=args.alpha, temporal=args.temporal,
                    inpaint_radius=args.inpaint_radius, dilate=args.dilate,
                    crf=args.crf, engine=args.engine, progress=prog)
    print()
    print(f"已写出：{out}")
    print(f"   {st['frames_written']} 帧 · {st['size']/1e6:.1f} MB · "
          f"遮罩 {st['mask_px']} px（{st['mask_ratio']*100:.2f}%）· 模式 {st['mode']}")
    return 0


def cmd_video_batch(args) -> int:
    exts = (".mp4", ".mov", ".mkv", ".avi", ".webm")
    files = []
    for root, _d, names in os.walk(args.dir):
        for n in names:
            if n.lower().endswith(exts) and "-clean" not in n:
                files.append(os.path.join(root, n))
    if not files:
        print("目录里没有视频文件")
        return 0
    rects = _parse_rects(args.rect)
    if not rects and not args.mask:
        raise SystemExit("批量模式必须给 --rect 或 --mask（水印位置需一致）")
    print(f"找到 {len(files)} 个视频")
    rc = 0
    for f in files:
        ns = type(args)(file=f, rect=args.rect, mask=args.mask, alpha=args.alpha,
                        temporal=args.temporal, inpaint_radius=args.inpaint_radius,
                        dilate=args.dilate, crf=args.crf, engine=args.engine,
                        out=None, yes=args.yes)
        rc |= cmd_video_remove(ns)
    return rc


def add_video_parser(sub) -> None:
    p = sub.add_parser("video", help="视频水印清理（MP4/MOV/MKV）")
    sp = p.add_subparsers(dest="vidcmd", required=True)

    q = sp.add_parser("info", help="看视频规格")
    q.add_argument("file")
    q.set_defaults(func=cmd_video_info)

    q = sp.add_parser("probe", help="定位水印（实验性），输出标注图")
    q.add_argument("file")
    q.add_argument("--samples", type=int, default=40)
    q.add_argument("--out", default=None)
    q.set_defaults(func=cmd_video_probe)

    def common(q_):
        q_.add_argument("file")
        q_.add_argument("--rect", action="append", default=None,
                        help="水印位置 x0,y0,x1,y1（可多次；最可靠）")
        q_.add_argument("--mask", default=None, help="遮罩图（白=要修的地方）")
        q_.add_argument("--alpha", type=float, default=None,
                        help="水印强度：给了就走反算，不给走 inpaint")
        q_.add_argument("--temporal", type=int, default=1,
                        help="时域稳定窗口半径，默认 1；0=关闭（会抖）")
        q_.add_argument("--inpaint-radius", type=int, default=3, dest="inpaint_radius")
        q_.add_argument("--dilate", type=int, default=2, help="遮罩外扩像素，默认 2")
        q_.add_argument("--crf", type=int, default=18, help="输出画质，默认 18")
        q_.add_argument("--engine", choices=["opencv", "lama", "auto"], default="opencv",
                        help="修复引擎：opencv 快（默认）；lama 质量更好但约 4 秒/帧，"
                             "整段视频会慢到小时级")

    q = sp.add_parser("remove", help="执行处理")
    common(q)
    q.add_argument("--out", default=None)
    q.add_argument("--yes", action="store_true")
    q.set_defaults(func=cmd_video_remove)

    q = sp.add_parser("batch", help="批量处理目录")
    common(q)
    q.add_argument("--dir", default=None)
    q.set_defaults(func=cmd_video_batch)
