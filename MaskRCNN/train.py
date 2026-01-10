#!/usr/bin/env python3
# -------------------------------------------------------------------------
# train.py (已修复：官方模型 + 多波段支持 + 移除nets依赖)
# -------------------------------------------------------------------------
import os
import datetime
import warnings
import torch
import torch.nn as nn
import torch.utils.data as tdata
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# 官方模型组件
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np

# 引入我们的新工具
from utils.config import (
    CLASSES_PATH, PRETRAIN_PATH, SAVE_DIR,
    NUM_BANDS, IMAGE_DIVISOR, IMAGE_MAX_DIM,
    BATCH_SIZE, INIT_EPOCH, MAX_EPOCH,
    INIT_LR, MIN_LR, LR_DECAY_TYPE, OPTIMIZER_TYPE,
    RPN_ANCHOR_SCALES
)
from utils.utils import get_classes
from utils.dataloader import MultiBandCocoDetection, collate_fn

warnings.filterwarnings("ignore")

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print(f'Config: Bands={NUM_BANDS}, Divisor={IMAGE_DIVISOR}')

# 获取类别
class_names, num_classes_raw = get_classes(CLASSES_PATH)
NUM_CLASSES = num_classes_raw + 1  # +1 背景
print(f'Num Classes (incl. background): {NUM_CLASSES}')


# --------------------------------------------------#
#  构建模型 (核心修改：支持多波段)
# --------------------------------------------------#
def get_model(num_classes):
    print("Building Mask R-CNN Model...")
    # 1. 加载官方预训练模型
    # 注意：这里会下载权重，如果网络不通，请手动下载 maskrcnn_resnet50_fpn_coco.pth
    model = maskrcnn_resnet50_fpn(pretrained=True,
                                  min_size=IMAGE_MAX_DIM,
                                  max_size=IMAGE_MAX_DIM)

    # 2. 修改第一层输入 (适配多波段)
    if NUM_BANDS != 3:
        print(f"!!! Modifying First Layer for {NUM_BANDS} Bands !!!")
        original_conv1 = model.backbone.body.conv1

        # 创建新卷积层 (输入通道 = NUM_BANDS)
        new_conv1 = nn.Conv2d(NUM_BANDS, original_conv1.out_channels,
                              kernel_size=original_conv1.kernel_size,
                              stride=original_conv1.stride,
                              padding=original_conv1.padding,
                              bias=False)

        # 初始化策略：复制 RGB 权重，额外通道复用或随机
        with torch.no_grad():
            if NUM_BANDS > 3:
                # 复制前3通道
                new_conv1.weight[:, :3, :, :] = original_conv1.weight
                # 简单粗暴：把第1个通道的权重复制给剩余通道 (或者用均值)
                for i in range(3, NUM_BANDS):
                    new_conv1.weight[:, i, :, :] = original_conv1.weight[:, 0, :, :]
            else:
                # 如果是单波段灰度，取 RGB 均值
                new_conv1.weight[:, 0, :, :] = torch.mean(original_conv1.weight, dim=1)

        model.backbone.body.conv1 = new_conv1

    # 3. 修改分类头 (Box Head)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 4. 修改分割头 (Mask Head)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


# --------------------------------------------------#
#  主训练循环
# --------------------------------------------------#
def main():
    # 1. 数据集
    TRAIN_IMG = 'datasets/coco/JPEGImages'
    TRAIN_JSON = 'datasets/coco/Jsons/train_annotations.json'  # 注意对应 coco_annotation 生成的文件名
    VAL_IMG = 'datasets/coco/JPEGImages'
    VAL_JSON = 'datasets/coco/Jsons/val_annotations.json'

    train_ds = MultiBandCocoDetection(TRAIN_IMG, TRAIN_JSON, train=True)
    val_ds = MultiBandCocoDetection(VAL_IMG, VAL_JSON, train=False)

    train_loader = tdata.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                    num_workers=0, collate_fn=collate_fn)
    val_loader = tdata.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=0, collate_fn=collate_fn)

    # 2. 模型
    model = get_model(NUM_CLASSES)
    model.to(device)

    # 3. 优化器
    params = [p for p in model.parameters() if p.requires_grad]
    if OPTIMIZER_TYPE == 'adam':
        optimizer = Adam(params, lr=INIT_LR, weight_decay=1e-4)
    else:
        optimizer = SGD(params, lr=INIT_LR, momentum=0.9, weight_decay=1e-4)

    if LR_DECAY_TYPE == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=MAX_EPOCH, eta_min=MIN_LR)
    else:
        scheduler = StepLR(optimizer, step_size=MAX_EPOCH // 3, gamma=0.1)

    # 4. 日志
    writer = SummaryWriter(os.path.join(SAVE_DIR, f'logs_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'))
    os.makedirs(SAVE_DIR, exist_ok=True)

    best_loss = float('inf')

    for epoch in range(INIT_EPOCH, MAX_EPOCH):
        # --- Train ---
        model.train()
        train_loss = 0.
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{MAX_EPOCH} Train')

        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            train_loss += losses.item()
            pbar.set_postfix({'loss': losses.item()})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # --- Val ---
        # 注意：Torchvision MaskRCNN 只有在 train 模式下才返回 loss
        # 如果想看 val loss，必须保持 train 模式但可以冻结 BN (这里简化处理，直接 train 模式)
        val_loss = 0.
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc='Val'):
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss for loss in loss_dict.values())

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)

        # 保存权重
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best_epoch_weights.pth'))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f'epoch_{epoch + 1}_weights.pth'))

    writer.close()
    print("Done.")


if __name__ == '__main__':
    main()