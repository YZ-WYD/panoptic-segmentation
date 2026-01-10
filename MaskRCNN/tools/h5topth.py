#!/usr/bin/env python3
"""
离线转换：Keras Mask R-CNN .h5 → PyTorch .pth
输入输出路径直接在脚本里指定
"""
import h5py
import torch
import numpy as np
from torchvision.models.detection import maskrcnn_resnet50_fpn
from collections import OrderedDict
import re

# ---------------------------------------------------------#
#  1. 直接在这里写死路径
# ---------------------------------------------------------#
H5_PATH  = r'D:\yz\1_yz\mask-rcnn-pytorch-master\model_data\mask_rcnn_coco.h5'      # 输入
PTH_PATH = r'D:\yz\1_yz\mask-rcnn-pytorch-master\model_data\mask_rcnn_coco.pth'     # 输出

# ---------------------------------------------------------#
#  2. 核心层映射（可继续扩充）
# ---------------------------------------------------------#
KEY_MAP = {
    r"res2a_branch2a/kernel:0":        "backbone.body.conv1.weight",
    r"res2a_branch2a/bn.*gamma:0":     "backbone.body.bn1.weight",
    r"res2a_branch2a/bn.*beta:0":      "backbone.body.bn1.bias",
    r"res2a_branch2a/bn.*moving_mean:0": "backbone.body.bn1.running_mean",
    r"res2a_branch2a/bn.*moving_variance:0": "backbone.body.bn1.running_var",

    r"rpn_conv_1/kernel:0":     "rpn.head.conv.weight",
    r"rpn_conv_1/bias:0":       "rpn.head.conv.bias",
    r"rpn_class_raw/kernel:0":  "rpn.head.cls_logits.weight",
    r"rpn_class_raw/bias:0":    "rpn.head.cls_logits.bias",
    r"rpn_bbox_pred/kernel:0":  "rpn.head.bbox_pred.weight",
    r"rpn_bbox_pred/bias:0":    "rpn.head.bbox_pred.bias",

    r"mrcnn_class_fc1/kernel:0":     "roi_heads.box_head.fc1.weight",
    r"mrcnn_class_fc1/bias:0":       "roi_heads.box_head.fc1.bias",
    r"mrcnn_class_logits/kernel:0":  "roi_heads.box_predictor.cls_score.weight",
    r"mrcnn_class_logits/bias:0":    "roi_heads.box_predictor.cls_score.bias",
    r"mrcnn_bbox_fc/kernel:0":       "roi_heads.box_predictor.bbox_pred.weight",
    r"mrcnn_bbox_fc/bias:0":         "roi_heads.box_predictor.bbox_pred.bias",
    r"mrcnn_mask_conv1/kernel:0":    "roi_heads.mask_head.mask_fcn1.weight",
    r"mrcnn_mask_conv1/bias:0":      "roi_heads.mask_head.mask_fcn1.bias",
}

# ---------------------------------------------------------#
#  3. 工具函数
# ---------------------------------------------------------#
def load_h5(h5_path):
    w = {}
    with h5py.File(h5_path, 'r') as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                w[name] = np.array(obj)
        f.visititems(visitor)
    return w

def convert():
    pytorch_model = maskrcnn_resnet50_fpn(pretrained=False)
    tgt_state = pytorch_model.state_dict()
    h5_weights = load_h5(H5_PATH)
    new_state = OrderedDict()

    for pth_key in tgt_state.keys():
        for pat, tgt_pat in KEY_MAP.items():
            if re.search(tgt_pat, pth_key):
                np_w = h5_weights.get(pat, None)
                if np_w is None:
                    continue
                if len(np_w.shape) == 4:          # conv
                    np_w = np_w.transpose(3, 2, 0, 1)
                elif len(np_w.shape) == 2:        # fc
                    np_w = np_w.T
                new_state[pth_key] = torch.from_numpy(np_w)
                break
        else:
            new_state[pth_key] = tgt_state[pth_key]

    pytorch_model.load_state_dict(new_state, strict=False)
    torch.save(pytorch_model.state_dict(), PTH_PATH)
    print(f"转换完成 → {PTH_PATH}")

# ---------------------------------------------------------#
#  4. 一键运行
# ---------------------------------------------------------#
if __name__ == '__main__':
    convert()