# -------------------------------------------------------------------------
# mask_rcnn.py (已修复：官方模型 + 多波段推理 + 评估接口)
# -------------------------------------------------------------------------
import os
import cv2
import numpy as np
import torch
import torchvision
import time
import colorsys
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import functional as F

# 引入配置
from utils.config import (
    CLASSES_PATH, NUM_BANDS, IMAGE_DIVISOR, IMAGE_MAX_DIM, RPN_ANCHOR_SCALES
)
from utils.utils import get_classes


class MASK_RCNN(object):
    _defaults = {
        # 指向训练好的权重文件 (默认指向 best_epoch_weights.pth)
        "model_path": 'logs_pytorch/best_epoch_weights.pth',
        "classes_path": CLASSES_PATH,
        # 只有得分大于置信度的预测框会被保留
        "confidence": 0.5,
        # 非极大抑制所用到的nms_iou大小
        "nms_iou": 0.3,
    }

    @classmethod
    def get_defaults(cls, n):
        return cls._defaults[n] if n in cls._defaults else "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 1. 动态获取类别
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.num_classes += 1  # 必须 +1 (背景)

        # 2. 生成颜色表
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        # 3. 加载模型
        self.generate()

    def generate(self):
        print(f"Loading model... Bands={NUM_BANDS}, Classes={self.num_classes}, Divisor={IMAGE_DIVISOR}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ------------------------------------------------------#
        #   构建网络结构 (与 train.py 保持一致)
        # ------------------------------------------------------#
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=False, pretrained_backbone=False,
            num_classes=self.num_classes,
            min_size=IMAGE_MAX_DIM, max_size=IMAGE_MAX_DIM,
            rpn_anchor_generator=torchvision.models.detection.rpn.AnchorGenerator(
                tuple((s,) for s in RPN_ANCHOR_SCALES), (0.5, 1.0, 2.0) * 5)
        )

        # ------------------------------------------------------#
        #   修改输入层 (适配多波段)
        # ------------------------------------------------------#
        if NUM_BANDS != 3:
            print(f"Modifying input layer to accept {NUM_BANDS} bands...")
            original_conv1 = self.model.backbone.body.conv1
            new_conv1 = torch.nn.Conv2d(NUM_BANDS, original_conv1.out_channels,
                                        kernel_size=original_conv1.kernel_size,
                                        stride=original_conv1.stride,
                                        padding=original_conv1.padding, bias=False)
            self.model.backbone.body.conv1 = new_conv1

        # ------------------------------------------------------#
        #   加载权重
        # ------------------------------------------------------#
        model_path = os.path.expanduser(self.model_path)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, map_location=device)
            # 兼容保存的是 state_dict 还是整个 checkpoint
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt

            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError as e:
                print(f"[Warning] Loading weights mismatch: {e}")
                print("Trying with strict=False (e.g. input channels changed)...")
                self.model.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Error] Model path {model_path} not found!")

        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully.")

    # ---------------------------------------------------
    #  核心推理函数 (单张图片)
    # ---------------------------------------------------
    def detect_image(self, image):
        """
        image: PIL Image 对象 (支持 RGB 或 多波段 Tiff)
        """
        # 1. 预处理 (与训练完全一致: 转 Tensor -> 除以 Divisor)
        # F.pil_to_tensor 会保留原始数据类型和数值
        image_tensor = F.pil_to_tensor(image).float()
        image_tensor = image_tensor / IMAGE_DIVISOR
        image_tensor = image_tensor.unsqueeze(0)  # 加 Batch 维

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # 2. 推理
        with torch.no_grad():
            outputs = self.model(image_tensor)

        pred = outputs[0]

        # 3. 后处理与绘图
        return self.draw_result(image, pred)

    # ---------------------------------------------------
    #  绘图逻辑
    # ---------------------------------------------------
    def draw_result(self, image, pred):
        # 将 PIL 转为 numpy 方便 cv2 绘图
        image_np = np.array(image)

        # 处理显示通道：多波段只取前3个，单通道转RGB
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] > 3:
            image_np = image_np[:, :, :3]  # 只取 RGB

        # 数值范围适配：如果是 16bit 数据，压缩到 0-255 用于显示
        # 这里假设 IMAGE_DIVISOR 是最大值（如 10000），简单线性拉伸
        if IMAGE_DIVISOR > 255:
            # 简单的可视化策略：归一化后乘 255
            image_np = (image_np / IMAGE_DIVISOR * 255).clip(0, 255).astype(np.uint8)
        else:
            image_np = image_np.astype(np.uint8)

        # 颜色转换 RGB -> BGR (OpenCV)
        image_vis = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 解析预测结果
        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        masks = pred['masks'].cpu().numpy().squeeze(1)  # [N, H, W]

        # 阈值过滤
        keep = scores >= self.confidence
        boxes, labels, scores, masks = boxes[keep], labels[keep], scores[keep], masks[keep]

        # print(f"Detected {len(boxes)} objects.")

        # 绘图循环
        for i in range(len(boxes)):
            box = boxes[i].astype(int)
            label_idx = labels[i]
            score = scores[i]
            mask = masks[i]

            # 颜色
            color = self.colors[label_idx % len(self.colors)]

            # 1. 画 Mask (阈值 0.5)
            mask_bool = mask > 0.5
            if mask_bool.any():
                roi = image_vis[mask_bool]
                blended = (roi * 0.5 + np.array(color) * 0.5).astype(np.uint8)
                image_vis[mask_bool] = blended

            # 2. 画框
            cv2.rectangle(image_vis, (box[0], box[1]), (box[2], box[3]), color, 2)

            # 3. 标签
            # 此时 label_idx 是 1-based，对应 class_names[idx-1]
            name_idx = label_idx - 1
            label_name = self.class_names[name_idx] if 0 <= name_idx < len(self.class_names) else f'Cls_{label_idx}'
            caption = f"{label_name} {score:.2f}"
            cv2.putText(image_vis, caption, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return Image.fromarray(cv2.cvtColor(image_vis, cv2.COLOR_BGR2RGB))

    # ---------------------------------------------------
    #  eval.py 专用接口 (获取原始数据)
    # ---------------------------------------------------
    def get_map_out(self, image):
        """
        eval.py 调用此函数计算 mAP
        返回: boxes, scores, labels, masks_bool(bool类型的mask)
        """
        image_tensor = F.pil_to_tensor(image).float()
        image_tensor = image_tensor / IMAGE_DIVISOR
        image_tensor = image_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        with torch.no_grad():
            outputs = self.model(image_tensor)

        pred = outputs[0]

        boxes = pred['boxes'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        masks = pred['masks'].cpu().numpy().squeeze(1)

        keep = scores >= self.confidence
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks = masks[keep]

        if len(boxes) == 0:
            return None, None, None, None, None

        masks_bool = masks > 0.5
        return boxes, scores, labels, masks_bool, None

    # ---------------------------------------------------
    #  FPS 测试接口
    # ---------------------------------------------------
    def get_FPS(self, image, test_interval):
        image_tensor = F.pil_to_tensor(image).float()
        image_tensor = image_tensor / IMAGE_DIVISOR
        image_tensor = image_tensor.unsqueeze(0)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # 预热
        with torch.no_grad():
            self.model(image_tensor)

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                self.model(image_tensor)
        t2 = time.time()

        return (t2 - t1) / test_interval