# -------------------------------------------------------------------------
# summary.py (已适配：PyTorch 版模型概览)
# -------------------------------------------------------------------------
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utils.config import NUM_CLASSES, NUM_BANDS, IMAGE_MAX_DIM


def get_model():
    # 1. 构建基础模型
    model = maskrcnn_resnet50_fpn(pretrained=False,
                                  num_classes=NUM_CLASSES,
                                  min_size=IMAGE_MAX_DIM,
                                  max_size=IMAGE_MAX_DIM)

    # 2. 如果是多波段，修改第一层
    if NUM_BANDS != 3:
        original_conv1 = model.backbone.body.conv1
        new_conv1 = torch.nn.Conv2d(NUM_BANDS, original_conv1.out_channels,
                                    kernel_size=original_conv1.kernel_size,
                                    stride=original_conv1.stride,
                                    padding=original_conv1.padding,
                                    bias=False)
        model.backbone.body.conv1 = new_conv1

    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model()
    model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Device: {device}")
    print(f"Model: Mask R-CNN (ResNet50-FPN)")
    print(f"Input Bands: {NUM_BANDS}")
    print(f"Num Classes: {NUM_CLASSES}")
    print("-" * 30)
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable Parameters: {trainable_params / 1e6:.2f} M")
    print("-" * 30)

    # 尝试打印结构（部分）
    # PyTorch 模型不像 Keras 那样有非常直观的 summary() 表格
    # 但我们可以打印 backbone 或 head 的结构
    print("\n[Backbone Structure]")
    print(model.backbone)

    print("\n[ROI Heads Structure]")
    print(model.roi_heads)