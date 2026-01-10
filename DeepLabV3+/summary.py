# --------------------------------------------#
#   该文件用于查看网络结构，不计算FLOPs
# --------------------------------------------#
import torch
from torchsummary import summary

from nets.deeplabv3_plus import DeepLab

if __name__ == "__main__":
    # ------------------------------------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # ------------------------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 6
    backbone = 'mobilenet'
    input_shape = [512, 512]

    # 【修改】设置输入波段数
    in_channels = 4

    # 【修改】初始化模型时传入 in_channels
    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=16,
                    pretrained=False, in_channels=in_channels).to(device)

    # 【修改】summary 统计时，输入 shape 的第一个维度必须对应波段数
    summary(model, (in_channels, input_shape[0], input_shape[1]))

    # dummy testing
    dummy_input = torch.randn(1, in_channels, input_shape[0], input_shape[1]).to(device)
    output = model(dummy_input)
    print("Output shape:", output.shape)