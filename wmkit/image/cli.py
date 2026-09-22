"""图片水印清理的命令行实现。

    wm image info   <图>                        看尺寸、可用的遮罩线索
    wm image try    <图> [遮罩参数]              生成多个 alpha 的对比图，供挑选
    wm image remove <图> --alpha 0.28 [遮罩参数] 执行修复
    wm image batch  <目录> --alpha 0.28 ...      批量

遮罩参数（三选一）
    --rect x0,y0,x1,y1        手动框选（可重复）
    --bright 97               按亮度筛选（半透明白水印）
    --dark  97                按亮度筛选（半透明黑水印）
    --template wm.png         用样本图做模板匹配

为什么 alpha 要人来定：实测自动估计不可靠（笔画细、背景有纹理时都会失准），
而 alpha 猜错会留下残影或引入过曝。工具负责生成候选对比图，人看图挑选。
"""
from __future__ import annotations

import os
import sys

import numpy as np


def _lama_ok() -> bool:
    """LaMA 模型是否可用（装了 onnxruntime 且权重在位）。"""
    try:
        from wmkit.image import lama as _L
        return _L.available()
    except Exception:
        return False


def _load(path: str):
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"读不了图片：{path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _build_mask(img, args):
    from wmkit.image import detect as D

    if args.rect:
        rects = []
        for r in args.rect:
            parts = [float(x) for x in r.split(",")]
            if len(parts) != 4:
                raise SystemExit("--rect 需要 x0,y0,x1,y1 四个数")
            rects.append(parts)
        mask = D.by_rect(img, rects)
        how = f"手动框选 {len(rects)} 处"
    elif args.bright is not None:
        mask = D.by_brightness(img, mode="bright", percent=args.bright)
        how = f"亮度筛选（亮，{args.bright} 百分位）"
    elif args.dark is not None:
        mask = D.by_brightness(img, mode="dark", percent=args.dark)
        how = f"亮度筛选（暗，{args.dark} 百分位）"
    elif args.template:
        import cv2
        tpl = cv2.imread(args.template, cv2.IMREAD_UNCHANGED)
        if tpl is None:
            raise SystemExit(f"读不了模板：{args.template}")
        mask, hits = D.by_template(img, tpl, threshold=args.match_threshold)
        how = f"模板匹配命中 {len(hits)} 处"
    elif getattr(args, "auto", False):
        mask, info = D.auto(img, sensitivity=args.sensitivity)
        how = (f"自动定位（亮度异常+低饱和度）种子 {info['seed_px']}px"
               f" → 保留 {info['kept_components']} 块 {info['final_px']}px"
               f"，占画面 {info['ratio']*100:.1f}%")
    else:
        raise SystemExit(
            "请指定遮罩来源：--auto（自动）/ --rect（框选）/ --template（模板）/ --bright")
    if args.refine:
        before = int((mask > 127).sum())
        mask = D.refine(img, mask, mode=("bright" if args.color != "black" else "dark"),
                        min_delta=args.refine_delta)
        after = int((mask > 127).sum())
        if before:
            how += f" + 精化 {before}→{after} px（只留水印笔画）"
    if args.grow:
        mask = D.dilate(mask, args.grow)
        how += f" + 膨胀 {args.grow}px"
    return mask, how


def _color(args):
    if args.color == "white":
        return (255, 255, 255)
    if args.color == "black":
        return (0, 0, 0)
    parts = [int(x) for x in args.color.split(",")]
    if len(parts) != 3:
        raise SystemExit("--color 需要 white / black / R,G,B")
    return tuple(parts)


def cmd_image_info(args) -> int:
    img = _load(args.file)
    h, w = img.shape[:2]
    print(f"图片：{args.file}")
    print(f"尺寸：{w} × {h}")
    gray = img.mean(axis=2)
    print(f"亮度：min {gray.min():.0f} / mean {gray.mean():.0f} / max {gray.max():.0f}")
    print()
    print("可用的遮罩线索（挑一个试）：")
    print("  --rect x0,y0,x1,y1     手动框选（最可靠）")
    print("  --bright 97            找最亮的 3% 像素（半透明白水印）")
    print("  --dark 97              找最暗的 3% 像素")
    print("  --template 样本.png    有重复水印样本时最准")
    return 0


def cmd_image_try(args) -> int:
    """生成多个 alpha 的对比图供挑选。"""
    import cv2
    from wmkit.image import restore as R

    img = _load(args.file)
    mask, how = _build_mask(img, args)
    color = _color(args)
    n_px = int((mask > 127).sum())
    print(f"遮罩：{how}　覆盖 {n_px} 像素（占 {100*n_px/(img.shape[0]*img.shape[1]):.1f}%）")
    if n_px == 0:
        print("遮罩是空的 —— 换个参数再试")
        return 1

    # 网格铺开多个 alpha 的结果
    tiles = []
    alphas = [0.10, 0.20, 0.30, 0.40, 0.55, 0.70]
    for a in alphas:
        out = R.unblend_then_inpaint(img, mask, alpha=a, color=color)
        cv2.putText(out, f"a={a:.2f}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        tiles.append(out)
    h, w = img.shape[:2]
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = t
    # 缩小以便查看
    scale = min(1.0, 1600.0 / grid.shape[1])
    if scale < 1.0:
        grid = cv2.resize(grid, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out_path = args.out or (os.path.splitext(args.file)[0] + "-alpha对比.png")
    cv2.imwrite(out_path, grid)
    print(f"已生成对比图：{out_path}")
    print("  挑一个残留最少、又没有出现白斑/过曝的 alpha，再用 remove 执行")
    return 0


def cmd_image_remove(args) -> int:
    import cv2
    from wmkit.image import restore as R

    if args.alpha is None:
        raise SystemExit("请用 --alpha 指定强度（先跑 wm image try 看对比图挑一个）")
    if not args.yes:
        print("这是预演。确认后加 --yes 写出文件。")
        return 0

    img = _load(args.file)
    mask, how = _build_mask(img, args)
    color = _color(args)
    n_px = int((mask > 127).sum())
    print(f"遮罩：{how}　覆盖 {n_px} 像素")
    if n_px == 0:
        return 1

    if args.engine == "lama" or (args.engine == "auto" and _lama_ok()):
        from wmkit.image import lama as _L
        base = R.unblend(img, mask, alpha=args.alpha, color=color) if args.alpha else img
        out = _L.inpaint(base, mask)
    else:
        out = R.unblend_then_inpaint(img, mask, alpha=args.alpha, color=color,
                                     radius=args.inpaint_radius)
    dst = args.out or (os.path.splitext(args.file)[0] + "-cleaned.png")
    cv2.imwrite(dst, out)

    # 量化：遮罩内外的连续性
    import numpy as _np
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(float)
    m = mask > 127
    ring = (cv2.dilate(mask, _np.ones((25, 25), _np.uint8)) > 0) & ~m
    if ring.any():
        print(f"遮罩内均值 {g[m].mean():6.2f}　周边均值 {g[ring].mean():6.2f}"
              f"　差 {abs(g[m].mean()-g[ring].mean()):5.2f}")
    print(f"已写出：{dst}")
    return 0


def cmd_image_batch(args) -> int:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
    files = []
    for root, _d, names in os.walk(args.dir):
        for n in names:
            if n.lower().endswith(exts) and "-cleaned" not in n:
                files.append(os.path.join(root, n))
    if not files:
        print("目录里没有图片")
        return 0
    print(f"找到 {len(files)} 张图片　alpha={args.alpha}")
    rc = 0
    for f in files:
        ns = type(args)(file=f, out=None, alpha=args.alpha, yes=True, grow=args.grow,
                        color=args.color, rect=args.rect, bright=args.bright,
                        dark=args.dark, template=args.template,
                        match_threshold=args.match_threshold,
                        refine=args.refine, refine_delta=args.refine_delta,
                        auto=args.auto, sensitivity=args.sensitivity,
                        inpaint_radius=args.inpaint_radius)
        rc |= cmd_image_remove(ns)
    return rc


def add_image_parser(sub) -> None:
    p = sub.add_parser("image", help="位图水印清理")
    sp = p.add_subparsers(dest="imgcmd", required=True)

    def mask_args(sp_, *, positional="file"):
        """所有涉及遮罩的子命令共用这一组参数（避免各处定义漂移）。"""
        if positional == "file":
            sp_.add_argument("file")
        else:
            sp_.add_argument("dir")
        sp_.add_argument("--auto", action="store_true",
                         help="自动定位水印（亮度异常+低饱和度），不再需要框选")
        sp_.add_argument("--sensitivity", type=float, default=1.0,
                         help="自动定位灵敏度，越大圈得越多（默认 1.0）")
        sp_.add_argument("--rect", action="append", default=None,
                         help="手动框选 x0,y0,x1,y1（可重复；最可靠）")
        sp_.add_argument("--bright", type=float, default=None,
                         help="按亮度筛选：最亮的 N%% 像素（如 97）")
        sp_.add_argument("--dark", type=float, default=None,
                         help="按亮度筛选：最暗的 N%% 像素")
        sp_.add_argument("--template", default=None, help="水印样本图，做模板匹配")
        sp_.add_argument("--match-threshold", type=float, default=0.75,
                         dest="match_threshold")
        sp_.add_argument("--color", default="white", help="水印颜色 white/black/R,G,B")
        sp_.add_argument("--refine", action="store_true",
                         help="把粗遮罩精化成「只盖水印笔画」（框选时强烈建议开启，"
                              "否则框内的正常像素也会被改动）")
        sp_.add_argument("--refine-delta", type=float, default=6.0, dest="refine_delta",
                         help="精化阈值：比背景亮/暗多少才算水印，默认 6")
        sp_.add_argument("--grow", type=int, default=2,
                         help="遮罩膨胀像素，默认 2（盖住水印边缘的抗锯齿）")
        sp_.add_argument("--engine", choices=["auto", "lama", "opencv"], default="auto",
                         help="修复引擎：auto 有模型就用 LaMA（约 4 秒/张，质量更好），"
                              "否则退回 OpenCV 插值（毫秒级）")

    q = sp.add_parser("info", help="看图片与可用的遮罩线索")
    q.add_argument("file")
    q.set_defaults(func=cmd_image_info)

    q = sp.add_parser("try", help="生成多个 alpha 的对比图，供挑选")
    mask_args(q)
    q.add_argument("--out", default=None)
    q.set_defaults(func=cmd_image_try)

    q = sp.add_parser("remove", help="执行修复")
    mask_args(q)
    q.add_argument("--alpha", type=float, default=None, help="水印强度 0~1（先 try 挑）")
    q.add_argument("--out", default=None)
    q.add_argument("--yes", action="store_true")
    q.add_argument("--inpaint-radius", type=int, default=3, dest="inpaint_radius")
    q.set_defaults(func=cmd_image_remove)

    q = sp.add_parser("batch", help="批量处理目录")
    mask_args(q, positional="dir")
    q.add_argument("--alpha", type=float, required=True)
    q.add_argument("--yes", action="store_true")
    q.add_argument("--inpaint-radius", type=int, default=3, dest="inpaint_radius")
    q.set_defaults(func=cmd_image_batch)
