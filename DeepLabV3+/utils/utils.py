import numpy as np
from PIL import Image
import cv2


# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    # 如果是 numpy 数组 (从 GDAL/OpenCV 读取)
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=-1)
        elif image.shape[2] in [3, 4]:  # 3或4波段直接返回
            return image
        else:
            return image  # 其他情况暂不处理，原样返回

    # 如果是 PIL Image
    if len(np.shape(image)) == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize
# ---------------------------------------------------#
def resize_image(image, size):
    w, h = size

    # ---------------------------------------------------#
    #   兼容 PIL Image 和 Numpy Array
    # ---------------------------------------------------#
    if isinstance(image, Image.Image):
        iw, ih = image.size
        # PIL resize
        nw = int(iw * min(w / iw, h / ih))
        nh = int(ih * min(w / iw, h / ih))
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh

    elif isinstance(image, np.ndarray):
        ih, iw = image.shape[:2]
        # OpenCV resize
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)

        # 生成背景 (根据通道数动态决定)
        if len(image.shape) == 3:
            num_channels = image.shape[2]
            new_image = np.full((h, w, num_channels), 128, dtype=image.dtype)
        else:
            new_image = np.full((h, w), 128, dtype=image.dtype)

        # 粘贴
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        new_image[dy:dy + nh, dx:dx + nw] = image
        return new_image, nw, nh
    else:
        raise ValueError("Unknown image type")


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
#   设置种子
# ---------------------------------------------------#
def seed_everything(seed=11):
    import random
    import os
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------#
#   设置worker种子
# ---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    import random
    worker_seed = rank + seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    import torch
    torch.manual_seed(worker_seed)


# ---------------------------------------------------#
#   【核心修改】预处理
#   根据通道数自动判断归一化系数
# ---------------------------------------------------#
def preprocess_input(image):
    # 如果是 PIL Image，转 numpy
    if isinstance(image, Image.Image):
        image = np.array(image)

    image = image.astype(np.float32)

    # 获取通道数
    if len(image.shape) == 3:
        channels = image.shape[2]
    else:
        channels = 1

    # 判断逻辑：如果是 4 波段，通常是 16位遥感数据，除以 10000
    if channels == 4:
        image /= 10000.0
    else:
        # 否则默认视为 8位数据，除以 255
        image /= 255.0

    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    download_urls = {
        'mobilenet': 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'xception': 'https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth',
    }
    url = download_urls[backbone]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)