import os
import shutil
from tqdm import tqdm   # 进度条库

# --------- 只需改下面 3 个路径 ---------
TXT_PATH      = r"D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\ImageSets\Segmentation\trainval.txt"          # 15500 行无后缀清单
SRC_IMG_DIR   = r'D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\JPEGImages'             # 原始 jpg 所在文件夹
DST_IMG_DIR   = r'D:\yz\1_yz\mask-rcnn-pytorch-master\datasets\coco\JPEGImages'    # 要把挑出来的图拷到哪里
# --------------------------------------

os.makedirs(DST_IMG_DIR, exist_ok=True)

# 1. 读清单
with open(TXT_PATH, 'r', encoding='utf-8') as f:
    names = [line.strip() for line in f if line.strip()]

# 2. 拷贝 + 进度条
missed = []
for name in tqdm(names, desc='拷贝进度', unit='张'):
    src = os.path.join(SRC_IMG_DIR, f'{name}.jpg')
    if os.path.isfile(src):
        shutil.copy2(src, DST_IMG_DIR)   # copy2 保留原文件元数据
    else:
        missed.append(name)

# 3. 简单报告
print(f'\n全部处理完成！成功拷贝 {len(names) - len(missed)} 张，缺失 {len(missed)} 张。')
if missed:
    print('缺失示例（前 10 个）:', missed[:10])