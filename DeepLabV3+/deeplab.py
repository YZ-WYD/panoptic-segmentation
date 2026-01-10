import colorsys
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from osgeo import gdal
import cv2

from nets.deeplabv3_plus import DeepLab
from utils.utils import cvtColor, preprocess_input, resize_image


class DeeplabV3(object):
    _defaults = {
        "model_path": 'logs_deeplab/best_epoch_weights.pth',
        "num_classes": 6,
        "backbone": "mobilenet",
        "input_shape": [512, 512],
        "downsample_factor": 16,
        "mix_type": 0,
        "cuda": True,
        # 【修改】默认通道数
        "in_channels": 4
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 画框设置 (可视化用)
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128),
                           (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    def generate(self):
        # 传递 in_channels
        self.net = DeepLab(num_classes=self.num_classes, backbone=self.backbone,
                           downsample_factor=self.downsample_factor, pretrained=False,
                           in_channels=self.in_channels)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = self.net.cuda()

    def read_tif(self, tif_path):
        ds = gdal.Open(tif_path)
        if ds is None: return None
        # 读取所有波段
        bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)]
        return np.stack(bands, axis=-1)

    def get_miou_png(self, image):
        """
        推理核心逻辑 (支持 PIL Image 或 Numpy array)
        """
        # 如果是 Numpy，不需要转 PIL，直接 resize 和 preprocess
        if isinstance(image, np.ndarray):
            # 1. Resize (使用 CV2)
            # image shape: (H, W, C)
            new_shape = (self.input_shape[1], self.input_shape[0])
            image_data = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

            # 2. Preprocess (自动判断 /10000 or /255)
            # 形状转为 (C, H, W)
            image_data = np.transpose(preprocess_input(image_data), (2, 0, 1))
            image_data = np.expand_dims(image_data, 0)

            # 记录原始尺寸用于后续 resize 回去 (如果需要的话)
            # 这里是 get_miou_png，通常返回 crop 后的结果
        else:
            # 回退兼容 PIL
            image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data = np.transpose(preprocess_input(image_data), (2, 0, 1))
            image_data = np.expand_dims(image_data, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()

            # 如果输入时 resize 了，这里应该不需要特殊的 crop，除非使用了 padding
            # 为了简单起见，这里直接 resize 回输入大小 (input_shape)
            pr = cv2.resize(pr, (self.input_shape[1], self.input_shape[0]), interpolation=cv2.INTER_LINEAR)

            return pr

    def detect_image(self, image, count=False, name_classes=None):
        # 这里的 image 参数可能是图片路径，也可能是 PIL 对象
        # 如果是路径且是 TIF，用 GDAL 读取
        if isinstance(image, str) and image.lower().endswith('.tif'):
            image_data = self.read_tif(image)
        elif isinstance(image, Image.Image):
            image_data = np.array(image)
        else:
            image_data = np.array(image)

        # 备份原始用于显示
        # 如果是 16位数据，显示时压缩到 8位
        if image_data.max() > 255:
            display_img = (image_data / 10000.0 * 255).astype(np.uint8)
        else:
            display_img = image_data.astype(np.uint8)

        # 只取前3波段显示
        if len(display_img.shape) == 3 and display_img.shape[2] > 3:
            display_img = display_img[:, :, :3]
        old_img = Image.fromarray(display_img)

        # 推理
        pr = self.get_miou_png(image_data)

        # Resize 回原图大小
        orininal_h, orininal_w = image_data.shape[:2]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)

        seg_img = np.argmax(pr, axis=-1).astype(np.uint8)

        # 混合显示
        image = Image.fromarray(seg_img)
        # 给 mask 上色
        if self.mix_type == 0:
            # seg_img -> color
            seg_img = Image.fromarray(seg_img).convert("P")
            seg_img.putpalette(np.array(self.colors).astype(np.uint8))
            image = Image.blend(old_img.convert("RGB"), seg_img.convert("RGB"), 0.5)

        elif self.mix_type == 1:
            image = Image.fromarray(seg_img).convert("P")
            seg_img.putpalette(np.array(self.colors).astype(np.uint8))
            image = seg_img

        return image