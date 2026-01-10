#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing as mp

# ========== 只需改这两个路径 ==========
src_dir = r"D:\yz\1_yz\dataset\cocodataset\val2017_label"  # 原始 PNG 标签目录
dst_dir = r"D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\val"  # 整理后保存目录
# =====================================

# 像素值映射表
PX_MAP_ARR = np.empty(256, dtype=np.uint8)
PX_MAP_ARR[:] = np.arange(256)          # 默认 1:1
PX_MAP_ARR[0:8] = [7, 0, 1, 2, 3, 4, 5, 6]  # 只改 0-7

Path(dst_dir).mkdir(parents=True, exist_ok=True)

def remap_one(png_path: Path) -> None:
    """单张图的重映射 + 保存"""
    # 直接 palette 模式读取，省内存
    img = Image.open(png_path)
    if img.mode != "P":
        img = img.convert("P")
    idx = np.array(img, dtype=np.uint8)          # 0-255 索引
    idx = PX_MAP_ARR[idx]                        # 向量化映射
    out = Image.fromarray(idx, mode="P")
    # 保留原调色板（颜色不变，仅索引变）
    out.putpalette(img.getpalette())
    out.save(Path(dst_dir) / png_path.name)

if __name__ == "__main__":
    png_list = list(Path(src_dir).glob("*.png"))
    n_cpu = mp.cpu_count()          # 用满 CPU
    Parallel(n_jobs=n_cpu, backend="loky")(
        delayed(remap_one)(p) for p in tqdm(png_list, desc="Remap")
    )
    print("✅ 像素映射完成！已保存至 →", dst_dir)