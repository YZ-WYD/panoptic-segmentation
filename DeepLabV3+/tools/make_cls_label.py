#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""COCO 2017 7类单通道label（PNG）· 机械盘优化 · 映射正确 · 未标注=7"""
import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ================= 用户配置 =================
COCO_ROOT   = r"D:\yz\1_yz\dataset\cocodataset"          # COCO根目录
OUT_LBL_DIR = r"D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\val"  # label输出
DONE_TXT    = r"D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\done_val.txt"          # 断点续跑
WORKERS     = 2                 # 机械盘建议4进程
BATCH       = 5000              # 5k张刷盘
CHUNK       = 512               # 机械盘128大块
# ==========================================

INST_JSON  = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
STUFF_JSON = os.path.join(COCO_ROOT, "annotations", "stuff_val2017.json")
IMG_DIR    = os.path.join(COCO_ROOT, "val2017")

os.makedirs(OUT_LBL_DIR, exist_ok=True)

# 21行最终映射：0~6 对应 7类，其余默认 5/6
CATEGORY_MAP = {
    # 1. 生物 → 0  (10 thing)
    1: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0,

    # 2. 不透水面 → 1  (5 thing + 19 stuff)
    10: 1, 11: 1, 13: 1, 14: 1, 15: 1,
    95: 1, 96: 1, 107: 1, 112: 1, 113: 1, 114: 1, 115: 1, 116: 1, 117: 1, 118: 1,
    128: 1, 140: 1, 144: 1, 145: 1, 146: 1, 147: 1, 149: 1, 151: 1, 158: 1, 161: 1,
    171: 1, 172: 1, 173: 1, 174: 1, 175: 1, 176: 1, 177: 1, 180: 1, 181: 1,

    # 3. 草地/低矮植被 → 2  (9 stuff)
    97: 2, 119: 2, 122: 2, 124: 2, 134: 2, 142: 2, 153: 2, 163: 2, 170: 2,

    # 4. 树木 → 3  (3 stuff)
    94: 3, 129: 3, 169: 3,

    # 5. 天空 → 4  (3 stuff)
    106: 4, 120: 4, 157: 4,
}

def remap(cat_id, is_thing):
    return CATEGORY_MAP.get(cat_id, 5 if is_thing else 6)

# 断点续跑
done = set()
if os.path.exists(DONE_TXT):
    with open(DONE_TXT, 'r') as f:
        done = {line.strip() for line in f}

def work(img_id, coco_t, coco_s):
    info = coco_t.loadImgs(img_id)[0]
    name = info['file_name']
    if name in done:
        return None, None
    h, w = info['height'], info['width']
    lbl = np.full((h, w), 7, dtype=np.uint8)          # 未标注区先填 7

    # stuff 先画
    for ann in coco_s.loadAnns(coco_s.getAnnIds(imgIds=img_id)):
        lbl[coco_s.annToMask(ann) == 1] = remap(ann['category_id'], False)
    # things 后画
    for ann in coco_t.loadAnns(coco_t.getAnnIds(imgIds=img_id)):
        lbl[coco_t.annToMask(ann) == 1] = remap(ann['category_id'], True)
    return (name, lbl), None

def write_batch(batch, done_fp):
    for name, lbl in batch:
        cv2.imwrite(os.path.join(OUT_LBL_DIR, name.replace('.jpg', '.png')),
                    lbl, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        done_fp.write(name + '\n')
    done_fp.flush()

if __name__ == '__main__':
    coco_t = COCO(INST_JSON)
    coco_s = COCO(STUFF_JSON)
    ids = list(set(coco_t.getImgIds()) & set(coco_s.getImgIds()))
    ids = [i for i in ids if coco_t.loadImgs(i)[0]['file_name'] not in done]
    print(f'待处理 {len(ids)} 张')

    batch, ok, fail = [], 0, 0
    with open(DONE_TXT, 'a') as done_fp, mp.Pool(WORKERS) as pool:
        for ret, err in tqdm(
            pool.imap(partial(work, coco_t=coco_t, coco_s=coco_s),
                      ids, chunksize=CHUNK),
            total=len(ids), desc='label_only'
        ):
            if ret:
                batch.append(ret); ok += 1
            else:
                fail += 1
                if err: print(err)
            if len(batch) >= BATCH:
                write_batch(batch, done_fp); batch.clear()
        if batch:
            write_batch(batch, done_fp)
    print(f'完成！成功 {ok} 张 / 失败 {fail} 张')