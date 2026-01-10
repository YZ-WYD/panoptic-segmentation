#!/usr/bin/env python3
"""
detect_single.py  →  权重是否有货，10 秒见分晓
"""
import os
import numpy as np
import torch
from PIL import Image
from mask_rcnn import MASK_RCNN   # 你的封装类

WEIGHT_PATH = r'D:\yz\1_yz\mask-rcnn-pytorch-master\logs_pytorch\best_epoch_weights.pth'  # 改成你的
IMAGE_PATH  = None                # None → 用随机图；或填真实路径

def main():
    print('1. 初始化模型并强制加载权重...')
    net = MASK_RCNN(confidence=0.05, nms_iou=0.5)   # 用最低阈值
    print('   权重文件:', WEIGHT_PATH)

    print('\n2. 生成测试图...')
    if IMAGE_PATH and os.path.isfile(IMAGE_PATH):
        img = Image.open(IMAGE_PATH).convert('RGB')
        print('   使用真实图片:', IMAGE_PATH)
    else:
        img = Image.fromarray(np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8))
        print('   使用随机噪声图 600×800')

    print('\n3. 推理...')
    boxes, scores, classes, masks, _ = net.get_map_out(img)

    print('\n4. 结果统计')
    if boxes is None:
        print('   输出框数: 0  →  权重或预处理有问题！')
    else:
        print('   输出框数:', len(boxes))
        print('   前 3 个 score:', scores[:3].tolist())
        if len(boxes) > 10 and scores[0] > 0.1:
            print('   ✅ 权重正常，能检出目标')
        else:
            print('   ⚠️  框数或分数偏低，权重可能未训练好')

if __name__ == '__main__':
    main()