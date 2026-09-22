"""图片修复的回归测试。

核心断言：反算必须远优于 inpainting（这是选择反算作为首选手段的依据）。
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

cv2 = pytest.importorskip("cv2")
from wmkit.image import detect as D       # noqa: E402
from wmkit.image import restore as R      # noqa: E402


@pytest.fixture()
def scene():
    """带纹理的背景 + 半透明白色水印。"""
    h, w = 300, 400
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = (50 + xx * 0.3).clip(0, 255)
    img[:, :, 1] = (110 + yy * 0.2).clip(0, 255)
    img[:, :, 2] = 170
    rng = np.random.default_rng(0)
    img = (img + rng.normal(0, 8, img.shape)).clip(0, 255).astype(np.uint8)

    mask = np.zeros((h, w), np.uint8)
    cv2.putText(mask, "TEST", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 6)
    alpha = 0.30
    wm = (img.astype(float) * (1 - alpha) + 255 * alpha).astype(np.uint8)
    return img, mask, wm, alpha


def test_unblend_recovers_almost_perfectly(scene):
    img, mask, wm, alpha = scene
    out = R.unblend(wm, mask, alpha=alpha)
    m = mask > 127
    err = np.abs(out.astype(int) - img.astype(int))[m].mean()
    assert err < 3.0, f"alpha 正确时反算误差应 <3/255，实测 {err:.2f}"


def test_unblend_beats_inpainting(scene):
    """这是把反算设为默认手段的依据，必须成立。"""
    img, mask, wm, alpha = scene
    m = mask > 127
    e_un = np.abs(R.unblend(wm, mask, alpha=alpha).astype(int) - img.astype(int))[m].mean()
    e_in = np.abs(R.inpaint(wm, D.dilate(mask, 2)).astype(int) - img.astype(int))[m].mean()
    assert e_un * 5 < e_in, f"反算({e_un:.1f})应远优于重绘({e_in:.1f})"


def test_unblend_rejects_bad_alpha(scene):
    img, mask, wm, alpha = scene
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            R.unblend(wm, mask, alpha=bad)


def test_out_of_range_pixels_fall_back_to_inpaint():
    """反算"越界"的像素交给重绘，而不是留下过曝/欠曝的光斑。

    什么情况会越界：按「白色水印 + 某个 alpha」反算时，若观测值远低于
    C*a（例如该处其实是深色内容或完全不透明的贴图），解出的值会跑到
    0 以下。这些像素不符合叠加模型，必须换手段。
    """
    h, w = 120, 160
    img = np.full((h, w, 3), 120, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (40, 40), (80, 80), 255, -1)
    wm = img.copy()
    wm[40:80, 40:80] = 30          # 很暗的一块，却按"白色水印 a=0.7"去反算
    out = R.unblend_then_inpaint(wm, mask, alpha=0.7)
    m = mask > 127
    # 反算会得到负数；兜底后应接近周边背景（120 上下），而不是 0 或 255
    assert 60 < out[m].mean() < 200, f"兜底失败，输出均值 {out[m].mean():.0f}"


def test_opaque_watermark_needs_inpaint_mode():
    """完全不透明的水印（观测值≈水印色）反算不出来，这是已知限制。

    记录成测试是为了把限制写清楚：这类水印要走 inpainting，
    或者让用户知道"反算对它无效"。
    """
    h, w = 120, 160
    img = np.full((h, w, 3), 120, np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (40, 40), (80, 80), 255, -1)
    wm = img.copy()
    wm[40:80, 40:80] = 250          # 不透明白块

    m = mask > 127
    un = R.unblend(wm, mask, alpha=0.3)
    e_un = np.abs(un.astype(int) - img.astype(int))[m].mean()
    inp = R.inpaint(wm, D.dilate(mask, 2))
    e_inp = np.abs(inp.astype(int) - img.astype(int))[m].mean()

    # 不透明水印：反算无力，重绘才是对的手段 —— 二者关系与半透明时正好相反
    assert e_inp < e_un, (
        f"不透明水印应当重绘优于反算，实测 反算{e_un:.1f} vs 重绘{e_inp:.1f}")
    assert e_inp < 60, f"重绘应能大致恢复背景，实测 {e_inp:.1f}"


def test_refine_keeps_watermark_pixels(scene):
    """精化应当保留水印笔画（在纯色/渐变背景上）。"""
    img, mask, wm, alpha = scene
    coarse = D.by_rect(wm, [(30, 100, 260, 210)])      # 比水印大的框
    fine = D.refine(wm, coarse)
    m = mask > 127
    kept = (fine > 0) & m
    # 水印像素应大部分被保留
    assert kept.sum() > 0.4 * m.sum(), f"精化把水印也删了：{kept.sum()}/{m.sum()}"
    # 且应当比粗遮罩小（剔掉了一些非水印像素）
    assert (fine > 0).sum() <= (coarse > 0).sum()


def test_refine_falls_back_when_nothing_found():
    """框内本来就没什么水印时，精化应退回原样，避免把区域判成空。"""
    h, w = 80, 100
    img = np.full((h, w, 3), 128, np.uint8)
    coarse = D.by_rect(img, [(10, 10, 80, 60)])
    fine = D.refine(img, coarse)
    assert (fine > 0).sum() == (coarse > 0).sum()


def test_mask_sources_produce_nonempty(scene):
    img, mask, wm, alpha = scene
    assert (D.by_brightness(wm, percent=99) > 0).sum() > 100
    assert (D.by_rect(wm, [(10, 10, 50, 50)]) > 0).sum() == 40 * 40
    tpl = wm[110:190, 40:230]
    mk, hits = D.by_template(wm, tpl, threshold=0.5)
    assert len(hits) >= 1, "模板匹配应至少命中一次"
