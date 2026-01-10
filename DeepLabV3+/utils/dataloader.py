import os
import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from utils.utils import preprocess_input
from osgeo import gdal


class DeeplabDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, in_channels=3):
        super(DeeplabDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.in_channels = in_channels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # 1. 读取数据 (使用 GDAL)
        jpg_path = os.path.join(self.dataset_path, "VOC2007/TIFFImages", name + ".tif")
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass_TIF", name + ".tif")

        image, _ = self.read_tif(jpg_path)
        label, _ = self.read_tif(png_path)

        # 2. 数据增强 (OpenCV)
        image, label = self.get_random_data(image, label, self.input_shape, random=self.train)

        # 3. 预处理 (自动 /10000 或 /255)
        image = np.transpose(preprocess_input(image), [2, 0, 1])

        # 4. 标签处理
        if len(label.shape) == 3:
            label = label[:, :, 0]
        label = np.array(label)

        # 【核心修改】将 0 (未标注) 映射为 num_classes (忽略索引)
        label[label == 0] = self.num_classes
        # 将其他超范围的标签也映射为 num_classes
        label[label >= self.num_classes] = self.num_classes

        # 5. One-hot 编码
        # 生成 num_classes + 1 个通道。最后一个通道 (index=num_classes) 是忽略通道
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        return image, label, seg_labels

    def read_tif(self, path):
        ds = gdal.Open(path)
        if ds is None:
            raise ValueError(f"无法打开: {path}")

        c = ds.RasterCount
        bands = []
        for i in range(c):
            bands.append(ds.GetRasterBand(i + 1).ReadAsArray())

        img = np.stack(bands, axis=-1)
        if c == 1 and "SegmentationClass" in path:
            img = img.squeeze(-1)
        return img, ds.GetGeoTransform()

    def get_random_data(self, image, label, input_shape, random=True):
        shape = (input_shape[1], input_shape[0])
        image = cv2.resize(image, shape, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, shape, interpolation=cv2.INTER_NEAREST)

        if random and np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
        return image, label


def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs = torch.from_numpy(np.array(pngs)).long()
    seg_labels = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels