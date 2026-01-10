#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 安全多进程 + tqdm
纯随机 15000 张（只剔除全图=7的未标注图）
"""
import os
import random
import numpy as np
from PIL import Image
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

LABEL_DIR = r'D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\SegmentationClass'
SAVE_TXT  = r'D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\ImageSets\Segmentation\train_random15k.txt'
NEED      = 15000
SEED      = 42
random.seed(SEED)
NPROC     = min(16, os.cpu_count())

def keep_if_valid(pth):
    """返回basename；只要0-6类任意像素>0就保留"""
    cnt = Counter(np.array(Image.open(pth), dtype=np.uint8).flatten())
    return os.path.basename(pth)[:-4] if any(cnt[i] > 0 for i in range(7)) else None

def main():
    png_paths = [os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR)
                 if f.lower().endswith('.png')]
    print(f'共发现 {len(png_paths)} 张标签图')

    # 多进程过滤
    valid = []
    with ProcessPoolExecutor(max_workers=NPROC) as exe:
        futures = [exe.submit(keep_if_valid, p) for p in png_paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc='过滤'):
            name = f.result()
            if name:
                valid.append(name)

    print(f'剔除全7未标注图后剩余 {len(valid)} 张')
    if len(valid) < NEED:
        raise RuntimeError(f'库存不足 {NEED} 张')
    chosen = random.sample(valid, NEED)

    # 写txt
    os.makedirs(os.path.dirname(SAVE_TXT) or '.', exist_ok=True)
    with open(SAVE_TXT, 'w', encoding='utf-8') as f:
        for name in tqdm(chosen, desc='写txt', leave=False):
            f.write(f'{name}\n')
    print('完成！→', SAVE_TXT)

if __name__ == '__main__':
    main()