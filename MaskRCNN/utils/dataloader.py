# -------------------------------------------------------------------------
# utils/dataloader.py
# 极简版 Dataset，只负责读取数据，不做复杂变换
# -------------------------------------------------------------------------
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from pycocotools.coco import COCO
from pycocotools import mask as mask_util
from utils.config import NUM_BANDS, IMAGE_DIVISOR


class MultiBandCocoDetection(Dataset):
    def __init__(self, root, annFile, train=True):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.train = train

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        full_path = os.path.join(self.root, path)

        # 1. 打开图片
        image = Image.open(full_path)

        # 2. 通道处理
        if NUM_BANDS == 3:
            if image.mode != 'RGB':
                image = image.convert('RGB')
        # 如果是多波段 (TIFF)，PIL 打开后通常能直接读取，这里不做强制转换

        return image

    def __getitem__(self, index):
        # 1. 读取 ID 和 图片
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_target = coco.loadAnns(ann_ids)

        img = self._load_image(img_id)
        w, h = img.size

        # 2. 处理标注 (Boxes & Labels & Masks)
        boxes = []
        labels = []
        masks = []

        for obj in coco_target:
            # 过滤掉 crowd
            if 'iscrowd' in obj and obj['iscrowd']:
                continue

            # Bbox [x, y, w, h] -> [x1, y1, x2, y2]
            xmin = obj['bbox'][0]
            ymin = obj['bbox'][1]
            xmax = xmin + obj['bbox'][2]
            ymax = ymin + obj['bbox'][3]

            # 简单裁剪防越界
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])  # 注意：这里假设 category_id 与我们训练的 id 一致

            # Mask
            seg = obj['segmentation']
            if isinstance(seg, list):
                # Polygon -> Mask
                rles = mask_util.frPyObjects(seg, h, w)
                rle = mask_util.merge(rles)
            elif isinstance(seg['counts'], list):
                rle = mask_util.frPyObjects(seg, h, w)
            else:
                rle = seg
            mask = mask_util.decode(rle)
            masks.append(mask)

        # 3. 转 Tensor
        # 图片转 Tensor 并缩放
        img_tensor = F.pil_to_tensor(img).float()
        img_tensor = img_tensor / IMAGE_DIVISOR

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (
                        target["boxes"][:, 2] - target["boxes"][:, 0])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # 负样本 (无目标)
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            target["image_id"] = torch.tensor([img_id])
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target

    def __len__(self):
        return len(self.ids)


# 简单的 collate_fn
def collate_fn(batch):
    return tuple(zip(*batch))