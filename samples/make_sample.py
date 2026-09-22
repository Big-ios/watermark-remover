"""生成一张贴近真实的"带水印照片"，用于演示与回归。

模拟的是最常见的场景：一张有内容的照片（渐变天空 + 山形 + 草地 + 噪点），
被平台叠加了半透明白色水印文字，另存为 JPG。
"""
import os

import cv2
import numpy as np

H, W = 600, 900
rng = np.random.default_rng(7)

# 天空渐变
img = np.zeros((H, W, 3), np.uint8)
for y in range(H):
    t = y / H
    img[y, :] = (int(200 - 60 * t), int(170 - 40 * t), int(120 - 20 * t))

# 远山
pts = np.array([[0, 380], [150, 250], [320, 360], [480, 220], [650, 340], [W, 260], [W, H], [0, H]])
cv2.fillPoly(img, [pts], (90, 110, 95))

# 草地
cv2.rectangle(img, (0, 470), (W, H), (60, 120, 70), -1)

# 太阳
cv2.circle(img, (700, 150), 60, (240, 240, 235), -1)

# 加噪声与轻度模糊，像真实照片
img = (img + rng.normal(0, 6, img.shape)).clip(0, 255).astype(np.uint8)
img = cv2.GaussianBlur(img, (3, 3), 0)
truth = img.copy()

# 平台水印：半透明白字，斜着铺三处
mask = np.zeros((H, W), np.uint8)
for (x, y) in ((120, 180), (380, 380), (560, 560)):
    cv2.putText(mask, "STUDIO", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 255, 4)
alpha = 0.35
wm = (truth.astype(float) * (1 - alpha) + 255 * alpha)
wm = (wm + rng.normal(0, 3, wm.shape)).clip(0, 255).astype(np.uint8)

here = os.path.dirname(os.path.abspath(__file__))
JPEG = [cv2.IMWRITE_JPEG_QUALITY, 92]
# 两份都必须走同样的 JPG 压缩，否则"有损 vs 无损"的差异会污染全图，
# 让真值遮罩失效（第一版就栽在这里：真值遮罩检出了 98% 的画面）。
cv2.imwrite(os.path.join(here, "photo-watermarked.jpg"), wm, JPEG)
cv2.imwrite(os.path.join(here, "photo-truth.jpg"), truth, JPEG)
np.save(os.path.join(here, "photo-mask.npy"), (mask > 127))
print("样本已生成：")
print("   photo-watermarked.jpg   带水印（JPG q92）")
print("   photo-truth.jpg         无水印（同样 JPG q92，可做逐像素对比）")
print("   photo-mask.npy          水印真实遮罩")
print(f"水印真实覆盖 {int((mask>127).sum())} px（占 {100*(mask>127).sum()/mask.size:.2f}%），alpha={alpha}")
