#!/usr/bin/env python3
"""
将 img 文件夹中的影像复制到新文件夹，并改名为与 label 完全一致
不删除、不覆盖原始文件
"""
from pathlib import Path
import shutil

# ========== 1. 只改这里 ==========
LABEL_DIR  = r"I:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\new_label"          # 原始 label 文件夹
IMG_DIR    = r"I:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\JPEGImages"          # 原始 img 文件夹
OUT_DIR    = r"I:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\new_img"  # 新文件夹（自动创建）
# =====================================

def main():
    label_dir = Path(LABEL_DIR)
    img_dir   = Path(IMG_DIR)
    out_dir   = Path(OUT_DIR)

    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = list(label_dir.glob("*_nlcd.tif"))
    if not label_files:
        print("未找到任何 *_nlcd.tif，请检查 LABEL_DIR")
        return

    for lbl in label_files:
        prefix   = lbl.stem.replace("_nlcd", "")        # m_3807503_ne_18_1
        src_img  = img_dir / f"{prefix}_naip-new.tif"   # 原始 img
        dst_img  = out_dir / lbl.name                   # 新名字 = xxx_nlcd.tif

        if not src_img.exists():
            print(f"⚠️  未找到对应影像：{src_img}")
            continue

        shutil.copy2(src_img, dst_img)  # 复制+保留元数据
        print(f"✅ 已复制：{src_img.name}  ->  {dst_img}")

    print("全部复制完成！→", out_dir.resolve())

if __name__ == "__main__":
    main()