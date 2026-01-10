#!/usr/bin/env python
import os, cv2, numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# ====== 仅机械盘 =======
COCO_ROOT = r"D:\yz\1_yz\dataset\cocodataset"
WORKERS   = 2          # 机械盘 2 进程足够
BATCH     = 5000       # 5 k 张刷一次
CHUNK     = 512        # imap 大块
# =======================

INST_JSON  = os.path.join(COCO_ROOT, "annotations", "instances_val2017.json")
STUFF_JSON = os.path.join(COCO_ROOT, "annotations", "stuff_val2017.json")
IMG_DIR    = os.path.join(COCO_ROOT, "val2017")
OUT_IMG    = os.path.join(COCO_ROOT, "val2017_img")
OUT_LBL    = os.path.join(COCO_ROOT, "val2017_label")

os.makedirs(OUT_IMG, exist_ok=True); os.makedirs(OUT_LBL, exist_ok=True)

T_MAP = {1:1,17:1,18:1,19:1,20:1,21:1,22:1,23:1,24:1,25:1,26:1,
         2:6,3:6,4:6,5:6,6:6,7:6,8:6,9:6,10:6,11:6}
S_MAP = {7:2,8:2,12:2,13:2,62:2, 26:3,58:3, 27:4, 28:5}

def work(img_id, coco_t, coco_s):
    info = coco_t.loadImgs(img_id)[0]
    name = info['file_name']
    h, w = info['height'], info['width']
    img = cv2.imread(os.path.join(IMG_DIR, name))
    if img is None: return None, f"bad {name}"
    lbl = np.zeros((h, w), dtype=np.uint8)
    # stuff 先画
    for ann in coco_s.loadAnns(coco_s.getAnnIds(imgIds=img_id)):
        lbl[coco_s.annToMask(ann) == 1] = S_MAP.get(ann['category_id'], 7)
    # things 后画
    for ann in coco_t.loadAnns(coco_t.getAnnIds(imgIds=img_id)):
        lbl[coco_t.annToMask(ann) == 1] = T_MAP.get(ann['category_id'], 6)
    return (name, img, lbl), None

def write_batch(batch):
    for name, img, lbl in batch:
        cv2.imwrite(os.path.join(OUT_IMG, name), img)
        cv2.imwrite(os.path.join(OUT_LBL, name.replace('.jpg', '.png')), lbl)

if __name__ == '__main__':
    coco_t = COCO(INST_JSON); coco_s = COCO(STUFF_JSON)
    ids = list(set(coco_t.getImgIds()) & set(coco_s.getImgIds()))
    print(f'机械盘模式 | 共 {len(ids)} 张')

    batch = []
    with mp.Pool(WORKERS) as pool:
        for ret, err in tqdm(
            pool.imap(partial(work, coco_t=coco_t, coco_s=coco_s),
                      ids, chunksize=CHUNK),
            total=len(ids), desc='HDD'
        ):
            if ret: batch.append(ret)
            if len(batch) >= BATCH:
                write_batch(batch); batch.clear()
        if batch: write_batch(batch)
    print('机械盘完成！')