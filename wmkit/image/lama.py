"""LaMA 图像修复（ONNX 运行时，CPU 可跑）。

为什么用它
----------
实测经典方法在真实条件下**分不开水印与内容**：13 个内容无关的物理特征里，
最好的可分性只有 0.40σ（要 3σ 以上才能可靠分离）—— 根因是半透明水印
带来的差异比画面本身起伏还小，JPG 压缩又把细节磨掉。

所以定位交给模型/多线索，**修复也交给模型**：LaMA 是基于学习的 inpainting，
能"理解"周围内容再补，而不是像 OpenCV inpaint 那样只做邻域插值。

模型文件
    models/lama_fp32.onnx（约 208MB）
    来自 hf-mirror.com/Carve/LaMa-ONNX —— 原始地址在 GitHub，国内不通

约束
    输入尺寸必须是 8 的倍数，内部会自动 pad 再裁回
    单次推理约数秒（CPU，1080p）
"""
from __future__ import annotations

import os
import threading

import numpy as np

MODEL_CANDIDATES = (
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "models", "lama_fp32.onnx"),   # 包内 models/
    os.path.expanduser("~/.cache/watermark-remover/lama_fp32.onnx"),
)

_session = None
_lock = threading.Lock()


def model_path() -> str | None:
    for p in MODEL_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return model_path() is not None


def _pad8(x, mode="edge"):
    """把尺寸补到 8 的倍数，返回 (补齐后, 原尺寸)。LaMA 的 stride 要求。"""
    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        pad = [(0, 0)] * (x.ndim - 2) + [(0, ph), (0, pw)]
        x = np.pad(x, pad, mode=mode)
    return x, (h, w)


def get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort

            p = model_path()
            if p is None:
                raise FileNotFoundError(
                    "找不到 LaMA 模型。请下载 lama_fp32.onnx 放到 "
                    f"{MODEL_CANDIDATES[0]}（见 SKILL.md 的说明）")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _session = ort.InferenceSession(
                p, so, providers=["CPUExecutionProvider"])
        return _session


# 该导出版本把输入尺寸固定为 512×512（见模型签名），所以内部做缩放。
# 这是常见做法：LaMA 原版是全卷积、可任意尺寸，ONNX 导出时被固定了。
FIXED = 512
# 裁剪送推理时的外扩边距：给模型足够的上下文来"理解"该补什么
CROP_PAD = 96


def _roi(mask: np.ndarray, shape, pad: int = CROP_PAD):
    """算出遮罩的外接矩形（带外扩），用于只把需要的部分送进模型。

    这一步很关键：1080p 整图直接推理每帧要 10 秒量级，
    而水印通常只占几个百分点 —— 按 ROI 裁剪能把单帧耗时压到毫秒级，
    视频才跑得动。
    """
    import cv2

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        return None, None
    h, w = shape[:2]
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    # 至少给一点尺寸，避免退化
    if y1 - y0 < 32:
        y0 = max(0, y0 - 16); y1 = min(h, y1 + 16)
    if x1 - x0 < 32:
        x0 = max(0, x0 - 16); x1 = min(w, x1 + 16)
    return (x0, y0, x1, y1), cv2


def inpaint(img_bgr: np.ndarray, mask: np.ndarray, *, size: int = FIXED,
            crop: bool = True) -> np.ndarray:
    """用 LaMA 修复遮罩区域。输入输出都是 BGR uint8。

    crop=True（默认）只把遮罩所在的外接矩形送去推理，速度提升显著。
    """
    import cv2

    sess = get_session()
    h, w = img_bgr.shape[:2]
    if (mask > 127).sum() == 0:
        return img_bgr.copy()

    if crop:
        box, _ = _roi(mask, img_bgr.shape)
        x0, y0, x1, y1 = box
        sub_img = img_bgr[y0:y1, x0:x1]
        sub_mask = mask[y0:y1, x0:x1]
        fixed = inpaint(sub_img, sub_mask, size=size, crop=False)
        res = img_bgr.copy()
        sel = sub_mask > 127
        region = res[y0:y1, x0:x1]
        region[sel] = fixed[sel]
        res[y0:y1, x0:x1] = region
        return res

    # 缩放到模型要求的尺寸（遮罩用最近邻，保持硬边）
    img_s = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

    img = img_s[:, :, ::-1].astype(np.float32) / 255.0           # BGR→RGB
    m = (mask_s > 127).astype(np.float32)
    img_t = np.transpose(img, (2, 0, 1))[None]                   # 1,3,S,S
    m_t = m[None, None]                                          # 1,1,S,S

    names = [i.name for i in sess.get_inputs()]
    feed = {}
    for name in names:
        feed[name] = m_t.astype(np.float32) if "mask" in name.lower() \
            else img_t.astype(np.float32)
    if len(names) == 1:      # 个别导出把 image+mask 拼成一个输入
        feed[names[0]] = np.concatenate([img_t, m_t], axis=1).astype(np.float32)

    out = sess.run(None, feed)[0]
    out = np.transpose(np.squeeze(out, 0), (1, 2, 0))
    # 实测该导出的**输出已经是 0~255 值域**（输出均值约 169，与图像一致），
    # 再乘 255 会把结果彻底冲爆（误差从 1 涨到 144）。这里只做裁剪。
    if out.max() <= 1.5:
        out = out * 255.0
    out = np.clip(out, 0, 255).astype(np.uint8)                    # RGB, S×S
    out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]  # →BGR

    # 只替换遮罩内的像素，其余保持原样（模型可能有轻微全局改动）
    res = img_bgr.copy()
    sel = mask > 127
    res[sel] = out[sel]
    return res


def describe() -> str:
    p = model_path()
    if p is None:
        return "LaMA：未安装模型"
    mb = os.path.getsize(p) / 1e6
    return f"LaMA ONNX：{mb:.0f} MB @ {p}"
