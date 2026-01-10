import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from osgeo import gdal

from deeplab import DeeplabV3
from utils.utils_metrics import f_score, fast_hist, per_class_iu, per_class_PA_Recall

if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ---------------------------------------------------------------------------#
    #   num_classes     分类个数+1、如2+1
    # ---------------------------------------------------------------------------#
    num_classes = 6
    # ---------------------------------------------------------------------------#
    #   VOCdevkit_path  数据集路径
    # ---------------------------------------------------------------------------#
    VOCdevkit_path = 'G:/newdata/GF1-GF6（全）/配对/pms1/test'
    in_channels = 4

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass_TIF/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    # 1. 预测阶段
    if miou_mode == 0 or miou_mode == 1:
        print("Load model.")
        deeplab = DeeplabV3(in_channels=in_channels, num_classes=num_classes)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # 使用 GDAL 读取 4波段 TIF
            image_path = os.path.join(VOCdevkit_path, "VOC2007/TIFFImages/" + image_id + ".tif")
            ds = gdal.Open(image_path)
            if ds is None: continue

            bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(in_channels)]
            image = np.stack(bands, axis=-1)

            # 推理
            # deeplab.detect_image 内部已经处理了 numpy -> preprocess -> predict -> argmax
            # 但 detect_image 返回的是为了显示的 PIL (RGB)，我们需要原始 label
            # 所以这里直接调 get_miou_png
            pr = deeplab.get_miou_png(image)
            image_label = np.argmax(pr, axis=-1).astype(np.uint8)

            # 保存预测结果 (PNG)
            image = Image.fromarray(image_label)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    # 2. 计算 mIoU 阶段
    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        # f_score 函数我们在 utils/utils_metrics.py 里已经改过了
        # 它会自动忽略标签 0
        hist, IoUs, PA_Recall, Precision = f_score(gt_dir, pred_dir, image_ids, num_classes, None)

        print("Get miou done.")
        # 结果输出 (略微调整打印逻辑，跳过第0类)
        # 注意 IoUs 是数组，如果需要忽略第0类显示：
        valid_ious = IoUs[1:] if len(IoUs) > 1 else IoUs
        print(f"mIoU (ignoring 0): {np.nanmean(valid_ious) * 100:.2f}%")