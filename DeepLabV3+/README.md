# PyTorch DeepLabV3+ (支持多波段遥感影像)

这是一个基于 `bubbliiiing` 版 DeepLabV3+ 深度改造的语义分割项目，专门针对**多波段遥感影像**（如 4 波段 RGB+NIR、16位 Tiff 数据）进行了全流程适配。

相较于原版，本项目做了以下核心重构：
1.  **多波段支持**：支持 3 通道、4 通道甚至更多波段的输入（自动调整网络首层）。
2.  **遥感预处理**：
    * 移除 ImageNet 均值归一化，改为动态线性缩放。
    * **3波段**：自动除以 255。
    * **4波段**：自动除以 10000（适配 16位 影像）。
3.  **忽略无效值**：**训练时自动忽略标签为 0 的区域**（背景/未标注区域），只对有效类别计算 Loss 和精度。
4.  **GDAL内核**：底层读取逻辑完全改用 `GDAL`，完美保留 TIFF 的地理坐标信息和原始像素值。

---

## 目录
1. [⚡ 快速开始 Quick Start](#-快速开始-quick-start)
2. [环境配置 Requirements](#环境配置)
3. [核心配置 Config](#核心配置-重要)
4. [数据准备 Data](#数据准备)
5. [训练步骤 Training](#训练步骤)
6. [预测步骤 Inference](#预测步骤)
7. [评估步骤 Evaluation](#评估步骤)
8. [常见问题 FAQ](#常见问题)

---

## ⚡ 快速开始 (Quick Start)

拿到本项目后，请严格按照以下顺序操作，即可跑通全流程：

### 1. 核心配置 (Config)
**这是最关键的一步！** 打开 `train.py`，根据你的数据情况修改参数：
* **`in_channels`**: 输入波段数（如 4）。
* **`num_classes`**: 有效类别数（例如你有 5 类地物，这里填 6，包含背景）。
* **`VOCdevkit_path`**: 指向你的数据根目录。

### 2. 生成索引 (Index)
确保图片在 `TIFFImages`，标签在 `SegmentationClass_TIF`，然后运行：

    python voc_annotation.py

> 作用：自动扫描数据集，在 `ImageSets/Segmentation/` 下生成 `train.txt` 和 `val.txt`。

### 3. 开始训练 (Train)
确保配置无误后，直接运行：

    python train.py

> 作用：自动下载官方预训练权重（MobileNet/Xception），开始训练。权重文件会保存在 `logs_deeplab/` 文件夹下。

### 4. 预测推理 (Predict)
训练完成后，修改 `deeplab.py` 中的 `model_path` 指向你最新的权重文件，然后运行：

    python predict.py

> 作用：对指定文件夹下的 TIFF 影像进行滑窗预测，并保存带有地理坐标的结果图。

### 5. 精度评估 (Eval)
如果需要计算 mIoU 指标（自动忽略标签 0）：

    python get_miou.py

---

## 环境配置

请确保安装了 PyTorch 和 GDAL 环境。推荐使用 Anaconda。

    # 1. 安装 GDAL (必须优先安装，推荐 conda)
    conda install gdal -c conda-forge

    # 2. 安装 PyTorch (根据你的 CUDA 版本调整 URL)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # 3. 安装其他依赖
    pip install opencv-python pillow tqdm tensorboardX scipy matplotlib

---

## 环境配置

请确保安装了 PyTorch 和 GDAL 环境。推荐使用 Anaconda。

    # 1. 安装 GDAL (必须优先安装，推荐 conda)
    conda install gdal -c conda-forge

    # 2. 安装 PyTorch (根据你的 CUDA 版本调整 URL)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # 3. 安装其他依赖
    pip install opencv-python pillow tqdm tensorboardX scipy matplotlib

---

## 核心配置 (重要!)

本项目的所有核心参数分散在 `train.py` 和 `deeplab.py` 中。**在开始任何操作前，请务必检查！**

### 1. 训练配置 (`train.py`)

    # 输入波段数 (3=普通RGB, 4=RGB+NIR)
    in_channels = 4  

    # 分类个数 (有效类别 + 1个背景)
    # 注意：虽然我们在计算 Loss 时忽略了背景(0)，但网络输出通道数依然需要包含它
    num_classes = 6 

    # 数据集根目录
    VOCdevkit_path = 'G:/newdata/GF1-GF6（全）/配对/pms1/test'


### 2. 推理配置 (`deeplab.py`)

打开 `deeplab.py`，确保 `_defaults` 字典里的参数与训练时一致：

    _defaults = {
        # 指向训练好的权重文件
        "model_path"        : 'logs_deeplab/best_epoch_weights.pth',
        
        # 必须与训练时保持一致
        "num_classes"       : 6,
        "in_channels"       : 4,
        
        # 骨干网络 (需与训练一致)
        "backbone"          : "mobilenet",
        
        # 输入尺寸 (预测时会自动resize到这个尺寸进行处理)
        "input_shape"       : [512, 512],
        
        # 下采样倍率 (8或16)
        "downsample_factor" : 16,
        
        # 是否使用 GPU
        "cuda"              : True,
    }
---

## 数据准备

本项目采用类似 VOC 的目录结构，但支持 TIFF 格式。

1.  **准备数据**：
    请按照以下结构组织您的数据：
    
        VOCdevkit/
        └── VOC2007/
            ├── TIFFImages/             <-- 存放原始多波段 TIFF 影像 (.tif)
            └── SegmentationClass_TIF/  <-- 存放单通道标签图 (.tif/.png)
    
    *注意：标签图必须是单通道的（8-bit），背景像素值为 0，其他类别为 1, 2, 3...*

2.  **生成索引**：
    运行以下命令：

        python voc_annotation.py

    *运行后，会在 `VOCdevkit/VOC2007/ImageSets/Segmentation/` 下生成 `train.txt` 和 `val.txt`。*

---

## 训练步骤

1.  **检查配置**：再次确认 `train.py` 中的 `in_channels` 和 `num_classes`。
2.  **开始训练**：

        python train.py

3.  **查看进度**：
    * 终端会显示每个 Epoch 的 Loss 和 F-score（已剔除背景类）。
    * 使用 Tensorboard 查看可视化曲线：
        
        tensorboard --logdir=logs_deeplab
        
    * 权重会保存在 `logs_deeplab/` 目录下（如 `best_epoch_weights.pth`）。


---

## 预测步骤

用于大图滑窗预测，支持保留地理坐标。

1.  **修改配置**：
    打开 `predict.py`，修改以下参数：
    * `dir_origin_path`: 待预测图片文件夹
    * `dir_save_path`: 结果保存文件夹
    * `in_channels`: 4

2.  **运行预测**：

        python predict.py

    * 程序会自动读取 TIFF 的地理信息，进行滑窗预测，并将结果拼接回带有地理坐标的 TIFF 文件。

---

## 评估步骤

计算 mIoU (Mean Intersection over Union) 指标。

1.  **配置**：打开 `get_miou.py`，确认 `num_classes` 和 `VOCdevkit_path`。
2.  **运行评估**：

        python get_miou.py

    * 结果将显示每个类别的 IoU，以及 mIoU。
    * **注意**：脚本已修改为自动忽略标签 0，只计算有效类别的精度。

---

## 常见问题

**Q: 为什么训练时 Loss 不下降？**
A: 请检查 `utils/utils.py` 中的 `preprocess_input`。
* 如果数据是 16位 (0-10000)，必须除以 10000。
* 如果数据是 8位 (0-255)，必须除以 255。
* 本项目已实现自动判断：通道=4时除以10000，否则除以255。如果您的数据不符合此规律（例如 3波段的16位数据），请手动修改该函数。

**Q: 报错 `RuntimeError: Given groups=1, weight of size ...`**
A: 这通常是第一层通道数不匹配。请确保 `train.py` 里的 `in_channels` 设置正确，且没有加载错误的预训练权重（代码中已加入自动跳过第一层权重的逻辑，理论上不会报错）。

**Q: 标签中的 0 是什么？**
A: 在本项目中，**0 代表背景或无效区域**。我们在 Loss 计算和精度评估中都显式忽略了它。您的有效类别应当从 1 开始编号。