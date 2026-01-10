import os
import numpy as np
from osgeo import gdal

def remap_label(src_path, dst_path=None):
    """
    重映射单波段tif标签：
    1→0, 2→1, 3→2, 4→3, 5→4, 6→5, 其他→255
    """
    dataset = gdal.Open(src_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"无法打开文件: {src_path}")

    band = dataset.GetRasterBand(1)
    label = band.ReadAsArray()

    # 创建映射表
    remap = np.arange(256, dtype=np.uint8)
    remap[1:7] = np.arange(6, dtype=np.uint8)  # 1→0, ..., 6→5
    remap[~(np.isin(remap, np.arange(6)))] = 5  # 其他→255

    label_remapped = remap[label]

    # 保存结果
    driver = gdal.GetDriverByName('GTiff')
    if dst_path is None:
        dst_path = src_path  # 覆盖原文件

    out_dataset = driver.CreateCopy(dst_path, dataset, strict=0)
    out_band = out_dataset.GetRasterBand(1)
    out_band.WriteArray(label_remapped)

    # 清理
    out_band.FlushCache()
    dataset = out_dataset = None

def batch_remap_label_folder(input_folder, output_folder=None, overwrite=False):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.tif'):
            src_path = os.path.join(input_folder, filename)
            if overwrite or output_folder is None:
                dst_path = src_path
            else:
                dst_path = os.path.join(output_folder, filename)
            print(f"处理: {src_path} → {dst_path}")
            remap_label(src_path, dst_path)

# ==================== 使用示例 ====================
if __name__ == "__main__":
    input_dir = r"H:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\SegmentationClass_TIF（1-6）"  # ←←← 修改这里
    output_dir = r"H:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\SegmentationClass_TIF"     # ←←← 修改这里（可为None表示覆盖）

    batch_remap_label_folder(input_dir, output_dir, overwrite=False)