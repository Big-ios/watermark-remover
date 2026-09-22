#!/usr/bin/env python3
"""watermark-remover 的环境配置器。

    python3 setup.py --check           只体检，报告缺什么（不动任何东西）
    python3 setup.py                   缺什么装什么（幂等，可反复跑）
    python3 setup.py --skip-model      不装 LaMA 权重（只装代码依赖）
    python3 setup.py --mirror URL      指定 pip 镜像

设计原则
--------
1. **幂等**：已经装好的不重复装，反复跑结果一致
2. **可预检**：--check 先看清楚缺什么，再决定装不装
3. **认得出网络现实**：国内直连 GitHub / HuggingFace 不通，
   模型权重默认走 hf-mirror.com；pip 可用镜像加速
4. **不用 torch**：LaMA 走 onnxruntime（15MB）就够了，
   torch 会带来 1.2GB 体积（早期误装过，已移除）
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
PY = os.path.join(VENV, "bin", "python")
MODEL_DIR = os.path.join(HERE, "models")
MODEL = os.path.join(MODEL_DIR, "lama_fp32.onnx")

MODEL_URLS = (
    "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
    "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
)
MODEL_MIN_BYTES = 150 * 1024 * 1024      # 正常约 208MB，小于此视为不完整

PIP_MIRRORS = (
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple",
)

# 需要 import 的模块 → pip 包名
PACKAGES = {
    "pymupdf": "pymupdf",
    "numpy": "numpy",
    "scipy": "scipy",
    "PIL": "pillow",
    "cv2": "opencv-python-headless",
    "onnxruntime": "onnxruntime",
    "pytest": "pytest",
}


class R:
    OK = "\033[32m✓\033[0m"
    NO = "\033[31m✗\033[0m"
    WARN = "\033[33m!\033[0m"


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def check_python() -> tuple:
    v = sys.version_info
    ok = v >= (3, 10)
    return ok, f"Python {v.major}.{v.minor}.{v.micro}"


def check_venv() -> tuple:
    if os.path.isfile(PY):
        out = _run([PY, "-V"]).stdout.strip()
        return True, out or "已存在"
    return False, f"未创建（{VENV}）"


def check_ffmpeg() -> tuple:
    missing = [e for e in ("ffmpeg", "ffprobe") if shutil.which(e) is None]
    if missing:
        return False, "缺少 " + ", ".join(missing)
    out = _run(["ffmpeg", "-version"]).stdout.split("\n")[0][:40]
    return True, out


def python_bin() -> str:
    return PY if os.path.isfile(PY) else sys.executable


def check_packages() -> tuple:
    py = python_bin()
    missing = []
    for mod, pkg in PACKAGES.items():
        r = _run([py, "-c", f"import {mod}"])
        if r.returncode != 0:
            missing.append(pkg)
    if missing:
        return False, "缺少 " + ", ".join(missing)
    return True, f"{len(PACKAGES)} 个模块齐全"


def check_model() -> tuple:
    if not os.path.isfile(MODEL):
        return False, "未下载"
    size = os.path.getsize(MODEL)
    if size < MODEL_MIN_BYTES:
        return False, f"不完整（{size/1e6:.0f}MB）"
    return True, f"{size/1e6:.0f}MB"


def check_all() -> list:
    return [
        ("Python 版本", *check_python()),
        ("虚拟环境", *check_venv()),
        ("ffmpeg", *check_ffmpeg()),
        ("Python 依赖", *check_packages()),
        ("LaMA 模型权重", *check_model()),
    ]


def report(rows, *, title="环境体检"):
    print(f"\n{title}")
    print("-" * 52)
    for name, ok, detail in rows:
        print(f"  {R.OK if ok else R.NO} {name:<16} {detail}")
    bad = [n for n, ok, _ in rows if not ok]
    print("-" * 52)
    if bad:
        print(f"  待处理：{'、'.join(bad)}")
    else:
        print("  全部就绪 ✓")
    return not bad


# ---------------------------------------------------------------- 安装动作

def ensure_venv() -> bool:
    if os.path.isfile(PY):
        print(f"  {R.OK} 虚拟环境已存在，跳过")
        return True
    print(f"  → 创建虚拟环境 {VENV}")
    r = _run([sys.executable, "-m", "venv", VENV])
    if r.returncode != 0:
        print(f"  {R.NO} 创建失败：{r.stderr[-300:]}")
        return False
    print(f"  {R.OK} 已创建")
    return True


def install_packages(extra_index=None) -> bool:
    py = python_bin()
    req = os.path.join(HERE, "requirements.txt")
    print(f"  → 安装 Python 依赖（源：{extra_index or '默认'}）")
    cmd = [py, "-m", "pip", "install", "-q", "-r", req]
    if extra_index:
        cmd += ["-i", extra_index]
    r = _run(cmd)
    if r.returncode != 0:
        # 镜像失败就回退官方源
        print(f"  {R.WARN} 镜像失败，回退默认源重试")
        cmd = [py, "-m", "pip", "install", "-q", "-r", req]
        r = _run(cmd)
    if r.returncode != 0:
        print(f"  {R.NO} 安装失败：{r.stderr[-400:]}")
        return False
    print(f"  {R.OK} 依赖就绪")
    return True


def download_model() -> bool:
    if check_model()[0]:
        print(f"  {R.OK} 模型已存在，跳过")
        return True
    os.makedirs(MODEL_DIR, exist_ok=True)
    tmp = MODEL + ".part"
    for url in MODEL_URLS:
        if shutil.which("curl") is None:
            break
        print(f"  → 下载模型（约 208MB）\n     {url}")
        cmd = ["curl", "-L", "--retry", "3", "-C", "-", "--progress-bar",
               "-o", tmp, url]
        r = subprocess.run(cmd)
        if r.returncode == 0 and os.path.isfile(tmp) and \
                os.path.getsize(tmp) >= MODEL_MIN_BYTES:
            os.replace(tmp, MODEL)
            print(f"  {R.OK} 模型就绪（{os.path.getsize(MODEL)/1e6:.0f}MB）")
            return True
        print(f"  {R.WARN} 该源失败，换下一个")
    print(f"  {R.NO} 模型下载失败。可手动下载后放到：{MODEL}")
    print("     hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx")
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="setup.py", description="配置去水印工具所需环境")
    ap.add_argument("--check", action="store_true", help="只体检，不安装")
    ap.add_argument("--skip-model", action="store_true", help="不下载 LaMA 权重")
    ap.add_argument("--mirror", default=None, help="pip 镜像地址")
    ap.add_argument("--no-test", action="store_true", help="装完不跑测试")
    a = ap.parse_args(argv)

    print("watermark-remover 环境配置")
    print("=" * 52)

    if a.check:
        ok = report(check_all())
        if not ok:
            print("\n执行 `python3 setup.py` 可自动补齐。")
        return 0 if ok else 1

    rows = check_all()
    need_venv = not rows[1][1]
    need_pkgs = not rows[3][1]
    need_model = not rows[4][1]

    if not (need_venv or need_pkgs or need_model):
        report(rows, title="环境已就绪")
        return 0

    print("\n开始安装")
    print("-" * 52)
    ok = True
    if need_venv and not ensure_venv():
        ok = False
    if ok and need_pkgs:
        mirror = a.mirror or (PIP_MIRRORS[0] if not shutil.which("pip") else None)
        ok = install_packages(a.mirror or PIP_MIRRORS[0])
    if ok and need_model and not a.skip_model:
        ok = download_model()

    print("\n最终状态")
    report(check_all())

    # ffmpeg 是系统包，装不了，只提示
    if not check_ffmpeg()[0]:
        print(f"\n{R.WARN} ffmpeg 需要系统安装（视频功能依赖）：")
        print("     Debian/Ubuntu:  sudo apt install ffmpeg")
        print("     macOS:          brew install ffmpeg")

    if ok and not a.no_test and os.path.isfile(PY):
        print("\n跑一遍回归测试")
        print("-" * 52)
        r = _run([PY, "-m", "pytest", "tests/", "-q"], cwd=HERE)
        tail = (r.stdout or "").strip().split("\n")[-1]
        print("  " + (tail or "(无输出)"))
        ok = ok and r.returncode == 0

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
