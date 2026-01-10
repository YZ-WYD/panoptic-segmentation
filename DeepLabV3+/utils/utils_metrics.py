import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from osgeo import gdal


def read_png_label(path):
    try:
        ds = gdal.Open(path)
        if ds is not None:
            return ds.GetRasterBand(1).ReadAsArray()
    except:
        pass
    return np.array(Image.open(path))


def f_score(gt_dir, pred_dir, image_ids, num_classes, name_classes=None):
    hist = np.zeros((num_classes, num_classes))
    for image_id in image_ids:
        gt_path = os.path.join(gt_dir, image_id + ".tif")
        pred_path = os.path.join(pred_dir, image_id + ".png")

        label = read_png_label(gt_path)
        prediction = read_png_label(pred_path)
        if label is None or prediction is None: continue

        if label.shape != prediction.shape:
            import cv2
            prediction = cv2.resize(prediction, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 【修改】忽略标签 0 (以及超范围的标签)
        # 生成掩码：标签 > 0 且 < num_classes 的像素才参与计算
        mask = (label > 0) & (label < num_classes)

        # 只取掩码区域的像素进行混淆矩阵计算
        label = label[mask]
        prediction = prediction[mask]

        # 注意：由于 label 已经过滤掉了 0，剩下的都是 1..N
        # 为了对应 hist 矩阵 (0..N-1)，这里的 label 需要 -1 ?
        # 答：通常 num_classes 是指有效类别数。如果 label 是 1..6，num_classes=6.
        # hist 索引是 0..5。所以这里需要 label - 1。
        # 如果您的训练设置是 label 0..5 (0是第一类)，那就不减。
        # 但既然您说 0 是未标注，那有效 label 应该是 1..N。
        # 这里假设您的模型输出也是对应 1..N。如果模型输出 0 对应 label 1，则需要调整。
        # **为了通用性，这里保持原样，但依赖于您的 label 设置。**
        # 如果您的 label 是 1, 2, 3... 对应模型输出 channel 0, 1, 2...
        # 那么这里 label - 1 比较合适。
        # **稳妥起见，我们假设您的 label 和预测值是直接对应的（模型输出 channel 1 预测 label 1）**
        # 那么 hist 大小应该是 num_classes + 1 (包含 0) 还是怎样？
        # Bubbling 代码通常假设 label 是 0, 1, 2...

        # 鉴于您之前的训练逻辑，这里直接 flatten 计算即可，mask 已经排除了 0。
        # 如果 prediction 预测了 0，但在 mask 区域 label 肯定不为 0，这算错分，没问题。
        hist += fast_hist(label.flatten(), prediction.flatten(), num_classes)

    IoUs = per_class_iu(hist)
    return hist, IoUs, 0, 0


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def CE_Loss(inputs, target, cls_weights, num_classes=21):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1, ct)[:, :-1]

    temp_inputs = temp_inputs[temp_target.sum(-1) == 1]
    temp_target = temp_target[temp_target.sum(-1) == 1]
    temp_target = temp_target.argmax(dim=-1)

    # 这里的 ignore_index=num_classes 配合 Dataloader 的映射，已经实现了忽略 0 (被映射为 num_classes)
    loss = F.cross_entropy(temp_inputs, temp_target, weight=cls_weights, ignore_index=num_classes)
    return loss


def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1, ct)[:, :-1]

    temp_inputs = temp_inputs[temp_target.sum(-1) == 1]
    temp_target = temp_target[temp_target.sum(-1) == 1]
    temp_target = temp_target.argmax(dim=-1)

    logpt = -F.cross_entropy(temp_inputs, temp_target, weight=cls_weights, ignore_index=num_classes, reduction='none')
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # 【核心修改】利用 target 的最后一个通道 (ignore通道) 制作掩码
    # target shape: [n, pixel_count, num_classes + 1]
    # 最后一个通道为 1 表示该像素被忽略 (比如 label=0)
    # 掩码：valid_mask = 1 - ignore_channel
    ignore_mask = temp_target[..., -1:]  # shape: [n, pixel_count, 1]
    valid_mask = 1.0 - ignore_mask  # 1 for valid, 0 for ignore

    # 将 mask 广播应用到 predictions (inputs)
    # 如果像素被忽略，其预测值被置为 0，从而不产生 TP, FP
    temp_inputs = temp_inputs * valid_mask

    # --------------------------------------------#
    #   计算dice loss (只计算前 num_classes 个通道)
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss