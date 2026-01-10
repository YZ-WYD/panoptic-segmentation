#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
count_folder_tif_classes.py
一次性统计指定文件夹内所有单通道.tif的像素类别总数、像素数及占比
路径直接写在代码里，无需命令行参数
"""

import os
import numpy as np
from osgeo import gdal

# ========== 1. 这里改成你的文件夹 ==========================================
FOLDER = r"I:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\new_label"   # <--- 直接改这里
# ===========================================================================

def accumulate_pixels(folder, counter):
    """累加文件夹内所有单通道tif的像素统计"""
    for root, _, files in os.walk(folder):          # 含子目录
        for name in files:
            if name.lower().endswith('.tif'):
                path = os.path.join(root, name)
                ds = gdal.Open(path)
                if ds is None or ds.RasterCount != 1:
                    continue
                arr = ds.GetRasterBand(1).ReadAsArray()
                uniq, cnt = np.unique(arr, return_counts=True)
                for v, c in zip(uniq, cnt):
                    counter[v] = counter.get(v, 0) + int(c)
                ds = None

def main():
    counter = {}
    accumulate_pixels(FOLDER, counter)

    if not counter:
        print("未找到单通道tif文件！")
        return

    total = sum(counter.values())
    print("文件夹累计像素统计")
    print("类别\t像素数\t占比")
    for val in sorted(counter):
        n = counter[val]
        print(f"{val}\t{n}\t{n/total*100:.2f}%")
    print(f"总像素：{total:,}，共 {len(counter)} 个类别")

if __name__ == "__main__":
    main()