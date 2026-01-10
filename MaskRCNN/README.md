# PyTorch Mask R-CNN (支持多波段遥感影像)

这是一个基于 `torchvision` 实现的 Mask R-CNN 改进版本，专门针对**多波段遥感影像**（如 4 波段 RGB+NIR、16位 Tiff 数据）进行了适配。

相较于原版（bubbliiiing Keras版），本项目做了以下核心重构：
1.  **内核迁移**：完全移除 Keras/TensorFlow 依赖，拥抱 PyTorch 官方生态。
2.  **多波段支持**：支持 3 通道、4 通道甚至更多波段的输入（自动调整网络首层）。
3.  **遥感预处理**：移除了不适合遥感数据的 ImageNet 均值归一化，改为更灵活的线性缩放（如 `/10000`）。
4.  **极简架构**：删除了复杂的 `nets` 手写网络，直接调用经过工业界验证的 `torchvision` 官方模型。

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
**这是最关键的一步！** 打开 `utils/config.py`，根据你的数据情况修改参数：
* **`NUM_BANDS`**: 数据波段数（如 3 或 4）。
* **`IMAGE_DIVISOR`**: 归一化除数（如 255.0 或 10000.0）。
* **`CLASSES_PATH`**: 指向你的类别文件（如 `model_data/shengwulei_classes.txt`）。

### 2. 生成数据 (Data)
将 Labelme 格式的图片和 JSON 放入 `datasets/before` 文件夹，然后运行：

    python coco_annotation.py

> 作用：自动转换格式、划分训练验证集，并生成 COCO 格式标签到 `datasets/coco/` 目录下。

### 3. 开始训练 (Train)
确保配置无误后，直接运行：

    python train.py

> 作用：自动下载官方预训练权重，开始训练。权重文件会保存在 `logs_pytorch/` 文件夹下。

### 4. 预测推理 (Predict)
训练完成后，修改 `mask_rcnn.py` 中的 `model_path` 指向你最新的权重文件（如 `logs_pytorch/best_epoch_weights.pth`），然后运行：

    python predict.py

> 默认模式为单张图片预测，输入图片路径即可看到结果。

### 5. 精度评估 (Eval)
如果需要计算 mAP 指标：

    python eval.py

---

## 环境配置

请确保安装了 PyTorch 环境。推荐使用 Anaconda。

    # 安装 PyTorch (根据你的 CUDA 版本调整 URL)
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

    # 安装其他依赖
    pip install pycocotools opencv-python pillow tqdm tensorboardX

---

## 核心配置 (重要!)

本项目的所有核心参数（波段数、归一化方式、类别路径）均由 `utils/config.py` 统一管理。**在开始任何操作前，请务必检查此文件！**

打开 `utils/config.py`，重点修改以下参数：

    # =========================
    #  数据流配置 (根据你的数据修改!)
    # =========================
    # 输入波段数 (3=普通RGB, 4=RGB+NIR)
    NUM_BANDS = 4  

    # 归一化除数 (Scale Factor)
    # - 如果是普通 8-bit 图片 (0-255)，设为 255.0
    # - 如果是 16-bit 遥感影像 (0-10000)，设为 10000.0
    IMAGE_DIVISOR = 10000.0

    # 类别文件路径
    CLASSES_PATH = 'model_data/shengwulei_classes.txt'

---

## 数据准备

本项目支持 **Labelme** 格式的标注数据（JSON + 图片）。

1.  **准备数据**：
    将你的图片（支持 `.tif`, `.jpg`, `.png`）和对应的 `.json` 标签文件放入 `datasets/before` 文件夹中。
    
    *注意：如果是多波段 Tiff，请确保 JSON 里的 `imagePath` 即使写的是 `.jpg` 也没关系，脚本会自动优先寻找同名的 `.tif` 文件。*

2.  **转换格式**：
    运行以下命令，将 Labelme 格式转换为 COCO 格式，并自动划分训练/验证集：

        python coco_annotation.py

    *运行后，会在 `datasets/coco/` 下生成 `JPEGImages`（图片副本）和 `Jsons`（COCO标签）。*

---

## 训练步骤

1.  **检查配置**：再次确认 `utils/config.py` 中的 `NUM_BANDS` 和 `IMAGE_DIVISOR` 与你的数据一致。
2.  **开始训练**：

        python train.py

3.  **查看进度**：
    * 终端会显示每个 Epoch 的 Loss。
    * 使用 Tensorboard 查看可视化曲线：
        
        tensorboard --logdir=logs_pytorch
        
    * 训练好的权重会保存在 `logs_pytorch/` 目录下（如 `best_epoch_weights.pth`）。

---

## 预测步骤

用于单张图片预测、批量预测或视频检测。

1.  **修改模型路径**：
    打开 `mask_rcnn.py`，修改 `model_path` 指向你训练好的权重文件。

        "model_path": 'logs_pytorch/best_epoch_weights.pth',

2.  **运行预测**：

        python predict.py

    * 默认模式为 `predict`：输入图片路径，弹窗显示结果。
    * 如需批量预测文件夹，请在 `predict.py` 中修改 `mode = "dir_predict"`。

---

## 评估步骤

计算 mAP (Mean Average Precision) 指标。

1.  **准备验证集**：确保你已经运行过 `coco_annotation.py`，它会生成 `val_annotations.json`。
2.  **运行评估**：

        python eval.py

    结果将显示 BBox mAP 和 Mask mAP。

---

## 常见问题

**Q: 为什么预测结果全黑或没有框？**
A: 请检查 `utils/config.py` 中的 `IMAGE_DIVISOR`。如果你的训练数据是 16位 (0-10000) 但预测时除以了 255，或者反之，会导致模型输入数值范围错误。

**Q: 报错 `RuntimeError: Given groups=1, weight of size ...`**
A: 这通常是因为模型权重的第一层通道数与当前配置的 `NUM_BANDS` 不匹配。请确保加载的权重是对应波段数训练出来的。如果只是想加载预训练权重并忽略第一层，代码会自动处理（strict=False），但请确保日志里有相关提示。

**Q: 如何修改类别？**
A: 修改 `model_data/shengwulei_classes.txt`，一行一个类别，不需要包含背景。修改后记得重新运行 `coco_annotation.py`。