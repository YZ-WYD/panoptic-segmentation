import time
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from osgeo import gdal

from deeplab import DeeplabV3

# -------------------------------------------------------------------------#
#   配置区域
# -------------------------------------------------------------------------#
# 输入输出文件夹
dir_origin_path = "img/"
dir_save_path = "img_out/"

# 关键参数
in_channels = 4
num_classes = 6

# 滑窗参数
stride = 256
window_size = 512

if __name__ == "__main__":
    deeplab = DeeplabV3(in_channels=in_channels, num_classes=num_classes)

    if not os.path.exists(dir_save_path):
        os.makedirs(dir_save_path)

    img_names = os.listdir(dir_origin_path)

    for img_name in tqdm(img_names):
        if not img_name.lower().endswith(('.tif', '.tiff')):
            continue

        image_path = os.path.join(dir_origin_path, img_name)

        # 1. 使用 GDAL 读取数据
        ds = gdal.Open(image_path)
        if ds is None: continue

        # 校验波段
        if ds.RasterCount < in_channels:
            print(f"Skipping {img_name}: Not enough bands.")
            continue

        # 读取指定波段 (保留原始 16位/8位 类型)
        bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(in_channels)]
        image_data = np.stack(bands, axis=-1)  # (H, W, C)

        # 2. 滑窗预测
        h, w = image_data.shape[:2]
        pred_result = np.zeros((h, w), dtype=np.uint8)

        # Padding
        pad_h = (window_size - h % window_size) % window_size
        pad_w = (window_size - w % window_size) % window_size

        image_pad = np.pad(image_data, ((0, pad_h), (0, pad_w), (0, 0)), 'reflect')

        for y in range(0, h + pad_h, stride):
            for x in range(0, w + pad_w, stride):
                # 裁剪 Patch
                patch = image_pad[y: y + window_size, x: x + window_size, :]

                # 如果 Patch 超出边界 (最后一步)，裁剪掉多余的
                if patch.shape[0] != window_size or patch.shape[1] != window_size:
                    # 这种情况通常被 pad 解决了，但为了保险
                    temp = np.zeros((window_size, window_size, in_channels), dtype=image_data.dtype)
                    temp[:patch.shape[0], :patch.shape[1]] = patch
                    patch = temp

                # 推理 (直接传入 numpy array, get_miou_png 会处理)
                # get_miou_png 返回的是 probability map (H, W, Classes)
                pr = deeplab.get_miou_png(patch)

                # Argmax 得到 label
                pred_patch = np.argmax(pr, axis=-1).astype(np.uint8)

                # 填充回大图 (注意边界)
                h_end = min(y + window_size, h)
                w_end = min(x + window_size, w)

                # 这里的切片逻辑需要细致，因为 stride < window_size 时会有重叠
                # 简单策略：中心区域保留，边缘丢弃 (这里为了简单直接覆盖)
                # 更好的策略：重叠区域取中心

                # 简单覆盖逻辑:
                valid_h = h_end - y
                valid_w = w_end - x
                pred_result[y:h_end, x:w_end] = pred_patch[:valid_h, :valid_w]

        # 3. 保存结果 (带地理坐标)
        save_file = os.path.join(dir_save_path, img_name)
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(save_file, w, h, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
        out_ds.SetGeoTransform(ds.GetGeoTransform())
        out_ds.SetProjection(ds.GetProjection())
        out_ds.GetRasterBand(1).WriteArray(pred_result)
        out_ds = None
        ds = None