import os
import random

# --------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用于指定该文件运行时计算的内容
#   annotation_mode为0代表整个标签处理过程
#   annotation_mode为1代表获得VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt
#   annotation_mode为2代表获得VOCdevkit/VOC2007/ImageSets/Segmentation/val.txt
# --------------------------------------------------------------------------------------------------------------------------------#
annotation_mode = 0

# -------------------------------------------------------------------#
#   数据集路径，指向包含 VOC2007 的上一级目录
# -------------------------------------------------------------------#
VOCdevkit_path = 'G:/newdata/GF1-GF6（全）/配对/pms1/test'

# -------------------------------------------------------------------#
#   train_val_percent   训练验证集与测试集的比例（通常用于划分测试集，这里设为1表示全用于训练验证）
#   train_percent       训练集与验证集的比例
# -------------------------------------------------------------------#
train_val_percent = 1.0
train_percent = 0.9

# -------------------------------------------------------#
#   指向存放标签 TIF 的文件夹
# -------------------------------------------------------#
seg_filepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass_TIF')
saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

if not os.path.exists(saveBasePath):
    os.makedirs(saveBasePath)

if __name__ == "__main__":
    print("Generate txt in ImageSets.")

    # 1. 扫描标签文件夹下的所有文件
    temp_seg = os.listdir(seg_filepath)
    total_seg = []

    for seg in temp_seg:
        # 只要是 .tif 或 .png 结尾的都算有效标签
        if seg.endswith(".tif") or seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    list = range(num)
    tv = int(num * train_val_percent)
    tr = int(tv * train_percent)

    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub size", tr)

    # 生成 train.txt 和 val.txt
    # 注意：文件里只保存文件名（不带后缀），Dataloader 会自动添加 .tif
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_seg[i][:-4] + '\n'  # 去掉 .tif 后缀
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")