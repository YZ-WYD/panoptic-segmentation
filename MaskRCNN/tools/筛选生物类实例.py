#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
COCO 2017 → 仅保留 11 类（人 + 10 种动物）标注
类别 ID：1, 16-25
输出新的 instances_train2017.json / instances_val2017.json
"""
import os
import json
from pycocotools.coco import COCO
from tqdm import tqdm

# ========== 用户路径 ==========
COCO_ROOT = r'D:\yz\1_yz\dataset\cocodataset'          # 原始 COCO 根目录
NEW_ROOT  = r'D:\yz\1_yz\mask-rcnn-pytorch-master\datasets\coco\Jsons'  # 新输出根目录
# ================================

KEEP_CAT_IDS = {1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}

def filter_and_save(split: str):
    in_ann  = os.path.join(COCO_ROOT, 'annotations', f'instances_{split}2017.json')
    out_ann = os.path.join(NEW_ROOT,  'annotations', f'instances_{split}2017.json')
    os.makedirs(os.path.dirname(out_ann), exist_ok=True)

    coco = COCO(in_ann)

    # 1. 收集相关图片 ID
    img_ids = set()
    for cat_id in tqdm(KEEP_CAT_IDS, desc=f'{split} 收集图片'):
        img_ids.update(coco.getImgIds(catIds=[cat_id]))

    # 2. 收集对应标注
    ann_ids = []
    for img_id in tqdm(img_ids, desc=f'{split} 收集标注'):
        ann_ids.extend(coco.getAnnIds(imgIds=[img_id], catIds=list(KEEP_CAT_IDS), iscrowd=None))
    anns = coco.loadAnns(ann_ids)

    # 3. 构建新 JSON
    new_coco = {
        "info":       coco.dataset["info"],
        "licenses":   coco.dataset["licenses"],
        "images":     coco.loadImgs(list(img_ids)),
        "annotations": anns,
        "categories": [c for c in coco.loadCats(coco.getCatIds()) if c["id"] in KEEP_CAT_IDS]
    }

    # 4. 写入
    with open(out_ann, 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)

    print(f'{split} 完成 → 共 {len(img_ids)} 张图，{len(anns)} 条实例')

def main():
    filter_and_save('train')
    filter_and_save('val')

if __name__ == '__main__':
    main()