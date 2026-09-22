# watermark-remover

通用水印清理工具。支持四种载体：**PDF 文档、位图图片、Office 文档、视频**。

做成 pi 的全局 skill，既能在会话里直接用，也能独立当命令行工具跑。

## 核心设计：先报告，后执行

**永远不一上来就删。** 水印和正文常常长得一样，必须先列候选、判清楚、再动手。

```bash
WM=/home/zcl/.pi/agent/skills/watermark-remover/wm

"$WM" analyze 文档.pdf                 # ① 只读扫描，输出候选报告（绝不动文件）
"$WM" remove  文档.pdf --ids 8,15 --yes  # ② 按判断执行
```

`remove` 默认只预演（打印「将要删除什么」但不写盘），加 `--yes` 才真正执行。
输出写到 `xxx-cleaned.pdf`，**原文件永不改动**。

## 四种载体

| 载体 | 识别方式 | 修复方式 |
|---|---|---|
| **PDF** | 物理特征评分（**半透明度**为主，与文字内容无关）| 内容流块级手术 |
| **图片** | 遮罩来源（框选 / 模板 / 亮度）| **LaMA 模型** 或 OpenCV 插值 |
| **Office** | OOXML 部件结构 + 评分 | 解压改 XML |
| **视频** | 同图片（水印位置通常固定）| 逐帧修复 + **时域稳定** |

## 三个别人做不到的点

### 1. 不信文字层（识破 ToUnicode 造假）

PDF 生成方可以把**任意字形**的 ToUnicode 映射成**任意文本**。开发中遇到的真例：
120 个「田字格辅助线」的字形，文字层读出来是「速创字帖」—— 按文字匹配删除，
就把格子线全删了，而文字层看起来完全正常。

**解法**：字形几何分析 —— 统计该字形自身颜色的像素做投影，
同时存在高覆盖率贯穿的横线与竖线就判定为线条结构。

    辅助线：横贯 4~9 行/列    → 线条，保留
    所有汉字：横贯 0、竖贯 0  → 文字，可删

（顺带发现：**墨迹密度不能当判据** —— 「创」字 3.3%，比辅助线 5.4% 还低。）

### 2. 通用识别靠物理特征，不靠内容

初版靠「推广关键词」判水印（公众号/水印/会员…）—— 只对平台推广有效，
换个水印就失灵。现在主判据是**半透明度**：

    真水印「速创字帖」  opacity = 0.020 / 0.047 / 0.098
    正文（全部）        opacity = 1.0

水印「看起来淡」是因为它半透明绘制，**这与它写什么字无关**。
实测把水印文字换成「内部资料」「CONFIDENTIAL」「某某公司」「AAA」，
识别结果一致。

### 3. 视频必须做时域稳定

逐帧独立修复会让原本静止的水印处「忽明忽暗」地闪：

    方法                水印区误差   帧间抖动
    不处理                7.38       0.148   ← 水印本身是静止的
    逐帧 inpaint           1.48       0.42    ← 修准了，但抖了
    + 时域中值             1.10       0.076   ← 又准又稳

时域中值把相邻帧的修复结果取中值，压掉随机起伏。
**这一层与修复手段解耦**，换成任何模型进去都能用。

## 安装

```bash
cd /home/zcl/.pi/agent/skills/watermark-remover
python3 setup.py --check    # 先体检，看缺什么
python3 setup.py            # 缺什么装什么（幂等）
```

**ffmpeg 需系统安装**（视频功能依赖）：`sudo apt install ffmpeg`。
装不上时 PDF/图片/Office 仍可用，只有视频不可用。

**模型权重**（208MB）国内直连不通，走镜像：

```bash
curl -L -o models/lama_fp32.onnx \
  https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx
```

**不需要 torch** —— LaMA 走 onnxruntime（15MB）推理就够，
torch 会白占 1.2GB。`requirements.txt` 里刻意没有它。

## 用法

```bash
# PDF
./wm analyze 文档.pdf
./wm remove  文档.pdf --auto --yes
./wm remove  文档.pdf --ids 8,15,16 --yes
./wm batch   ./目录 --yes

# 图片
./wm image info   图.png
./wm image try    图.png --rect 100,120,320,200      # 生成 alpha 候选对比图
./wm image remove 图.png --rect 100,120,320,200 --alpha 0.35 --yes

# Office
./wm office analyze 文件.docx -v
./wm office remove  文件.docx --auto --yes
./wm office remove  文件.docx --drop-media word/media/image2.png --yes

# 视频
./wm video info   片.mp4
./wm video remove 片.mp4 --rect 415,290,600,350 --yes
./wm video remove 片.mp4 --rect ... --engine lama --yes   # 质量优先（慢）
```

## 已知边界（诚实记录）

- **图片/视频的全自动定位尚未解决**。经典方法已用数据证明到顶：
  13 个内容无关特征里最好的可分性只有 **0.40σ**（要 3σ 才可靠）——
  半透明水印的差异比画面自身起伏还小，JPG 压缩又把细节磨掉。
  这两个载体目前需要**框选一次**（视频框一次管全片）。
  要真正全自动，得接专门的水印检测模型。

- **LaMA 固定 512×512 输入**，单次推理约 4 秒（CPU），
  且 **ROI 裁剪省不掉时间**（瓶颈在模型本身）。所以视频默认用 OpenCV inpaint，
  模型作为 `--engine lama` 可选。

- **不透明水印**（实心 logo）反算无解 —— 观测值就约等于水印色，
  解不出原始内容。这类要走重绘。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q      # 35 项
```

测试用真实案例做夹具（含「辅助线不得被判为水印」这类反向断言），
并且会现场用 ffmpeg 合成视频验证时域稳定确实起作用。

## 结构

```
watermark-remover/
├── wm / wm.py           命令行入口
├── setup.py             环境配置器（--check 只体检）
├── wmkit/
│   ├── pdf/             内容流手术 + 字形几何判据（glyph.py 是核心）
│   ├── image/           遮罩定位 + 反算/重绘 + LaMA(ONNX)
│   ├── office/          OOXML 解压改 XML
│   ├── video/           ffmpeg 管道 + 时域稳定
│   └── ...
├── tests/               35 项回归测试
└── samples/             可复现的测试样本与实验脚本
```

配套的**环境配置 skill**：`~/.pi/agent/skills/watermark-setup/`
