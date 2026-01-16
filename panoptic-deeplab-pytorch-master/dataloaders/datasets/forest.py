import os
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from dataloaders import custom_transforms as tr
from dataloaders.datasets.make_gaussian import make_gaussian
import warnings

# 【修改1】屏蔽 Python 警告
warnings.filterwarnings("ignore")

# 尝试导入 GDAL，如果没有安装会报错提示
try:
    from osgeo import gdal

    # 【修改2】屏蔽 GDAL 烦人的 TIFFReadDirectory 警告
    gdal.PushErrorHandler('CPLQuietErrorHandler')
except ImportError:
    print("错误: 找不到 gdal 库。请安装: pip install gdal 或 conda install gdal")
    raise


class ForestPanoptic(data.Dataset):
    NUM_CLASSES = 7

    def __init__(self, args, split="train"):
        self.args = args
        self.split = split

        self.root = args.base_dir
        self.images_dir = os.path.join(self.root, split)
        self.targets_dir = os.path.join(self.root, f"panoptic_{split}")

        self.files = [f for f in os.listdir(self.images_dir) if f.endswith('.tif')]
        if len(self.files) == 0:
            raise Exception(f"在 {self.images_dir} 没找到 .tif 文件，请检查路径！")

        print(f"ForestPanoptic [{split}]: Found {len(self.files)} images.")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def rgb2id(color):
        """
        【修改3】将 RGB 编码的 PNG 转换回 ID 数组
        公式：ID = R + G*256 + B*256*256
        """
        if isinstance(color, np.ndarray) and len(color.shape) == 3:
            quant = color.astype(np.int32)
            return quant[:, :, 0] + quant[:, :, 1] * 256 + quant[:, :, 2] * 256 * 256
        return color

    def __getitem__(self, index):
        img_filename = self.files[index]
        img_path = os.path.join(self.images_dir, img_filename)

        lbl_filename = img_filename.replace('.tif', '.png')
        lbl_path = os.path.join(self.targets_dir, lbl_filename)

        # 1. 使用 GDAL 读取 4 波段影像
        ds = gdal.Open(img_path)
        if ds is None:
            raise IOError(f"GDAL 无法打开文件: {img_path}")

        image_data = ds.ReadAsArray()
        if len(image_data.shape) == 3:
            image_data = image_data.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        image_data = image_data.astype(np.uint8)

        # 转为 PIL RGBA 模式
        _img = Image.fromarray(image_data, mode='RGBA')

        # 2. 读取 Panoptic ID Map (RGB PNG)
        _panoptic_rgb = Image.open(lbl_path).convert("RGB")
        panoptic_arr_rgb = np.array(_panoptic_rgb, dtype=np.int32)

        # 【关键修复】将 RGB (H,W,3) 解码为 ID (H,W)
        panoptic_arr = self.rgb2id(panoptic_arr_rgb)

        # 3. 解析 Semantic Mask
        semantic_arr = panoptic_arr.copy()
        mask_thing = panoptic_arr >= 1000
        semantic_arr[mask_thing] = panoptic_arr[mask_thing] // 1000

        # 这里的 _target 现在是单通道的索引图了，可以安全转为 tensor
        _target = Image.fromarray(semantic_arr.astype(np.uint8))

        # 4. 生成热图
        _center, _x_reg, _y_reg = self.generate_target(panoptic_arr, _img.size)

        _center = Image.fromarray((_center * 255).astype(np.uint8))
        _x_reg = Image.fromarray(_x_reg.astype(np.int32), "I")
        _y_reg = Image.fromarray(_y_reg.astype(np.int32), "I")

        sample = {
            "image": _img,
            "label": _target,
            "center": _center,
            "x_reg": _x_reg,
            "y_reg": _y_reg,
        }

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == "val":
            return self.transform_val(sample)

        return sample

    def generate_target(self, panoptic, size):
        width, height = size
        centers_image = np.zeros((height, width), dtype=np.float32)
        x_reg = np.zeros((height, width), dtype=np.float32)
        y_reg = np.zeros((height, width), dtype=np.float32)

        unique_ids = np.unique(panoptic)
        for uid in unique_ids:
            if uid < 1000: continue
            mask = (panoptic == uid).astype(np.uint8)
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0: continue

            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            h_box = y_max - y_min
            w_box = x_max - x_min
            c_x = x_min + w_box // 2
            c_y = y_min + h_box // 2

            gaussian_patch = make_gaussian([w_box, h_box], center=[w_box // 2, h_box // 2])

            x0 = max(0, c_x - w_box // 2)
            y0 = max(0, c_y - h_box // 2)
            x1 = min(width, x0 + gaussian_patch.shape[1])
            y1 = min(height, y0 + gaussian_patch.shape[0])

            patch_x0 = 0
            patch_y0 = 0
            patch_x1 = x1 - x0
            patch_y1 = y1 - y0

            # 只有当 patch 尺寸有效时才进行赋值
            if patch_x1 > patch_x0 and patch_y1 > patch_y0:
                centers_image[y0:y1, x0:x1] = np.maximum(
                    centers_image[y0:y1, x0:x1],
                    gaussian_patch[patch_y0:patch_y1, patch_x0:patch_x1]
                )

                mask_roi = mask[y0:y1, x0:x1]
                if np.sum(mask_roi) > 0:
                    y_grid, x_grid = np.mgrid[y0:y1, x0:x1]
                    off_x = c_x - x_grid
                    off_y = c_y - y_grid
                    current_x_reg = x_reg[y0:y1, x0:x1]
                    current_y_reg = y_reg[y0:y1, x0:x1]
                    current_x_reg[mask_roi == 1] = off_x[mask_roi == 1]
                    current_y_reg[mask_roi == 1] = off_y[mask_roi == 1]
                    x_reg[y0:y1, x0:x1] = current_x_reg
                    y_reg[y0:y1, x0:x1] = current_y_reg

        return centers_image, x_reg, y_reg

    def transform_tr(self, sample):
        composed_transforms = tr.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(
                mean=(0.089016, 0.146244, 0.112765, 0.588012),
                std=(0.065457, 0.096030, 0.119629, 0.292853)
            ),
            tr.ToTensor()
        ])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = tr.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(
                mean=(0.089016, 0.146244, 0.112765, 0.588012),
                std=(0.065457, 0.096030, 0.119629, 0.292853)
            ),
            tr.ToTensor()
        ])
        return composed_transforms(sample)