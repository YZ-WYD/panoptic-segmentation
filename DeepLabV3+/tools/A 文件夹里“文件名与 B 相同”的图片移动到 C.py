#!/usr/bin/env python3
"""
直接写死路径版本——把 A 文件夹里“文件名与 B 相同”的图片移动到 C。
改完路径后双击/命令行直接跑即可。
"""
import os
import shutil
from pathlib import Path

# ========== 只用改这里 ==========
FOLDER_A = Path(r"D:\yz\1_yz\mask-rcnn-pytorch-master\datasets\coco\JPEGImages").expanduser()   # A 文件夹路径
FOLDER_B = Path(r"D:\yz\1_yz\dataset\cocodataset\val2017").expanduser()   # B 文件夹路径
FOLDER_C = Path(r"D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\c").expanduser()   # 目标文件夹路径
# ================================

def move_common_images():
    if not FOLDER_A.is_dir() or not FOLDER_B.is_dir():
        exit("❌ A 或 B 文件夹不存在，请检查路径！")

    # 1. 收集 B 所有文件名（忽略大小写）
    b_names = {p.name.lower() for p in FOLDER_B.iterdir() if p.is_file()}
    if not b_names:
        exit("⚠️  B 文件夹里没有文件，无事可做。")

    # 2. 创建 C
    FOLDER_C.mkdir(parents=True, exist_ok=True)

    # 3. 遍历 A，命中就移动
    moved = 0
    for img in FOLDER_A.iterdir():
        if img.is_file() and img.name.lower() in b_names:
            shutil.move(str(img), str(FOLDER_C / img.name))
            moved += 1

    print(f"✅ 已完成：{moved} 张图片已移动到 {FOLDER_C}")

if __name__ == "__main__":
    move_common_images()