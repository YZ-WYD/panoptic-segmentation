from modeling.backbone import resnet, xception, drn, mobilenet


# 【修改点1】增加 in_channels 参数，默认值为 3
def build_backbone(backbone, output_stride, BatchNorm, in_channels=3):
    if backbone == "resnet":
        # 【修改点2】把 in_channels 传给 ResNet101
        return resnet.ResNet101(output_stride, BatchNorm, in_channels=in_channels)

    elif backbone == "xception":
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == "drn":
        return drn.drn_d_54(BatchNorm)
    elif backbone == "mobilenet":
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    elif backbone == "mobilenet_3stage":
        return mobilenet.MobileNetV2_3Stage(output_stride, BatchNorm)
    elif backbone == "xception_3stage":
        return xception.AlignedXception3Stage(output_stride, BatchNorm)

    if backbone == "resnet_3stage":
        # 【修改点3】如果是 3Stage 版本的 ResNet，也要传参数
        return resnet.ResNet101_3Stage(output_stride, BatchNorm, in_channels=in_channels)
    else:
        raise NotImplementedError