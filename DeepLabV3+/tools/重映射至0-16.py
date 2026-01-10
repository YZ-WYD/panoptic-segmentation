#!/usr/bin/env python3
"""
批量映射 .tif 像素值
0  -> 16
11 -> 0
12 -> 1
21 -> 2
22 -> 3
23 -> 4
24 -> 5
31 -> 6
41 -> 7
42 -> 8
43 -> 9
52 -> 10
71 -> 11
81 -> 12
82 -> 13
90 -> 14
95 -> 15
其余 -> 255 (nodata)
"""
import os
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm

# ========== 1. 只改这里 ==========
TIF_DIR = r"I:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\SegmentationClass_TIF"          # 原始影像文件夹
OUT_DIR = r"I:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\SegmentationClass_TIF_new"   # 结果文件夹（可与 TIF_DIR 相同，实现原地覆盖）
# =================================

# 2. 构造 0-255 查找表，默认 255
LUT = np.full(256, 255, dtype=np.uint8)

# 填充映射
LUT[0]  = 16
for new, old in enumerate([1, 2, 3, 4, 5, 6], start=0):
    LUT[old] = new

def remap_tile(src_path, dst_path):
    with rasterio.open(src_path) as src:
        prof = src.profile.copy()
        prof.update(dtype=np.uint8, nodata=255)
        with rasterio.open(dst_path, 'w', **prof) as dst:
            for _, win in src.block_windows(1):
                raw = src.read(1, window=win)  # 二维
                out = LUT[raw]                 # 矢量查找
                dst.write(out, 1, window=win)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tifs = list(Path(TIF_DIR).glob("*.tif"))
    for p in tqdm(tifs, desc="remap"):
        remap_tile(p, Path(OUT_DIR) / p.name)
    print("全部完成 →", OUT_DIR)

if __name__ == "__main__":
    main()