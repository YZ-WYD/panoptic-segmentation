# -------------------------------------------------------------------------
# utils/utils_map.py
# -------------------------------------------------------------------------
import json
import os
import os.path as osp
import torch
import numpy as np
import pycocotools.mask as mask_util


class Make_json:
    def __init__(self, map_out_path: str, coco_label_map: dict, model2coco: dict = None):
        self.map_out_path = map_out_path
        self.bbox_data = []
        self.mask_data = []
        # model2coco: 0-based index -> COCO category_id
        if model2coco is None:
            model2coco = {v - 1: k for k, v in coco_label_map.items()}
        self.model2coco = model2coco

    def add_bbox(self, image_id: int, category_id: int, bbox: list, score: float, map_id: bool = True):
        # map_id=True: 输入是 0-based 索引，需要映射回 COCO ID
        if map_id:
            category_id = self.model2coco[int(category_id)]
        bbox = [round(float(x), 2) for x in bbox]
        self.bbox_data.append({
            'image_id': int(image_id),
            'category_id': int(category_id),
            'bbox': bbox,
            'score': float(score)
        })

    def add_mask(self, image_id: int, category_id: int, segmentation: np.ndarray, score: float, map_id: bool = True):
        if map_id:
            category_id = self.model2coco[int(category_id)]
        rle = mask_util.encode(np.asfortranarray(segmentation.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        self.mask_data.append({
            'image_id': int(image_id),
            'category_id': int(category_id),
            'segmentation': rle,
            'score': float(score)
        })

    def dump(self):
        dump_arguments = [
            (self.bbox_data, osp.join(self.map_out_path, "bbox_detections.json")),
            (self.mask_data, osp.join(self.map_out_path, "mask_detections.json"))
        ]
        for data, path in dump_arguments:
            os.makedirs(osp.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)


def prep_metrics(pred_boxes, pred_confs, pred_classes, pred_masks, image_id, make_json):
    # 统一转 numpy
    if isinstance(pred_boxes, torch.Tensor): pred_boxes = pred_boxes.cpu().numpy()
    if isinstance(pred_confs, torch.Tensor): pred_confs = pred_confs.cpu().numpy()
    if isinstance(pred_classes, torch.Tensor): pred_classes = pred_classes.cpu().numpy()
    if isinstance(pred_masks, torch.Tensor): pred_masks = pred_masks.cpu().numpy()

    pred_classes = pred_classes.astype(np.int32)

    for i in range(pred_boxes.shape[0]):
        y1, x1, y2, x2 = pred_boxes[i]
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w * h > 0:
            coco_box = np.array([x1, y1, w, h], dtype=np.float32)

            # 【核心修改】：移除硬编码列表！
            # 模型输出 label 是 1,2,3... (背景是0)
            # 我们减 1 变成 0-based 索引，交给 Make_json 去查表映射
            cls_idx_0_based = pred_classes[i] - 1

            make_json.add_bbox(image_id, cls_idx_0_based, coco_box, pred_confs[i], map_id=True)
            # 注意：masks 的维度是 [N, H, W] 还是 [H, W, N] 取决于 mask_rcnn.py 的输出
            # 新版 mask_rcnn.py 输出的是 [N, H, W]，所以这里取 [i, :, :]
            # 但如果你原来的代码习惯是 [:, :, i]，需要确认一下。
            # 通常 torch 输出是 [N, H, W]，所以应该是 pred_masks[i, :, :]
            make_json.add_mask(image_id, cls_idx_0_based, pred_masks[i, :, :], pred_confs[i], map_id=True)