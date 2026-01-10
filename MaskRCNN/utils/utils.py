# -------------------------------------------------------------------------
# utils/utils.py
# -------------------------------------------------------------------------
import numpy as np
from PIL import Image
import cv2

# 获取类别
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = [c.strip() for c in f.readlines()]
    return class_names, len(class_names)

# 图像转 RGB (仅针对普通图片，多波段数据不走这里)
def cvtColor(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return image

# 图像 Resize (支持多波段)
def resize_image(image, size):
    w, h = size
    return image.resize((w, h), Image.BILINEAR)

# 预处理 (防止旧代码报错，实际逻辑由 config.py 控制)
def preprocess_input(image):
    return image / 255.0