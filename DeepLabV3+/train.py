import os
import datetime

import torch
from torch.utils.data import DataLoader

from nets.deeplabv3_plus import DeepLab
from nets.deeplabv3_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit import fit_one_epoch

if __name__ == "__main__":
    # ------------------------------------------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ------------------------------------------------------------------#
    Cuda = True
    # ------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认关闭distributed，Windows下调用DDP需要详解DDP原理
    # ------------------------------------------------------------------#
    distributed = False
    # ------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式开启
    # ------------------------------------------------------------------#
    sync_bn = False
    # ------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ------------------------------------------------------------------#
    fp16 = False
    # ------------------------------------------------------------------#
    #   num_classes     训练自己的数据集必须要修改的
    #                   自己需要的分类个数+1，如2+1
    # ------------------------------------------------------------------#
    num_classes = 6
    # ------------------------------------------------------------------#
    #   backbone        用于选择所使用的骨干网络
    #                   mobilenet、xception
    # ------------------------------------------------------------------#
    backbone = "mobilenet"
    # ------------------------------------------------------------------#
    #   pretrained      是否使用主干网络的预训练权重
    # ------------------------------------------------------------------#
    pretrained = True
    # ------------------------------------------------------------------#
    #   model_path      用于加载预训练权重/上次训练的权重
    #                   如果不使用预训练权重，请设置为空，即 model_path = ''
    # ------------------------------------------------------------------#
    model_path = ""
    # ------------------------------------------------------------------#
    #   downsample_factor   下采样倍率 8或者16
    #                       8下采样倍率下模型性能更好，但显存要求高
    #                       16下采样倍率下模型性能较差，但显存要求低
    # ------------------------------------------------------------------#
    downsample_factor = 16
    # ------------------------------------------------------------------#
    #   input_shape     输入图片的大小，必须为32的倍数
    # ------------------------------------------------------------------#
    input_shape = [512, 512]

    # ------------------------------------------------------------------#
    #   【关键修改】输入图片的通道数
    #   3 = RGB, 4 = RGB+NIR
    # ------------------------------------------------------------------#
    in_channels = 4

    # ------------------------------------------------------------------#
    #   Init_Epoch      模型开始的epoch
    #   Freeze_Epoch    模型冻结训练的epoch
    #   Freeze_batch_size   模型冻结训练的batch_size
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 50
    Freeze_batch_size = 8
    # ------------------------------------------------------------------#
    #   Freeze_lr       模型冻结训练的学习率
    # ------------------------------------------------------------------#
    Freeze_lr = 5e-4
    # ------------------------------------------------------------------#
    #   UnFreeze_Epoch  模型总共训练的epoch
    #   Unfreeze_batch_size 模型解冻训练的batch_size
    # ------------------------------------------------------------------#
    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 4
    # ------------------------------------------------------------------#
    #   Unfreeze_lr     模型解冻训练的学习率
    # ------------------------------------------------------------------#
    Unfreeze_lr = 5e-5
    # ------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    Freeze_Train = True

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率下降方式、优化器等
    # ------------------------------------------------------------------#
    lr_decay_type = 'cos'
    save_period = 5
    save_dir = 'logs_deeplab'
    eval_flag = True
    eval_period = 1

    # ------------------------------------------------------------------#
    #   VOC数据集路径
    # ------------------------------------------------------------------#
    VOCdevkit_path = 'G:/newdata/GF1-GF6（全）/配对/pms1/test'
    # ------------------------------------------------------------------#
    #   Dice_Loss   是否使用Dice Loss
    #   Focal_Loss  是否使用Focal Loss
    #   cls_weights 是否使用类别平衡权重
    # ------------------------------------------------------------------#
    dice_loss = True
    focal_loss = False
    cls_weights = np.ones([num_classes], np.float32)

    # ------------------------------------------------------#
    #   设置显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        raise NotImplementedError(
            "Windows/Simple version does not support DistributedDataParallel. Set distributed=False.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    if pretrained and model_path == '':
        download_weights(backbone)

    # ------------------------------------------------------#
    #   创建模型，传入 in_channels
    # ------------------------------------------------------#
    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=downsample_factor,
                    pretrained=pretrained, in_channels=in_channels)

    if not pretrained:
        weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key Num:", len(no_load_key))

    # ------------------------------------------------------#
    #   记录Loss
    # ------------------------------------------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape, in_channels=in_channels)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        model_train = model_train.cuda()

    # ---------------------------#
    #   读取数据集列表
    # ---------------------------#
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch, \
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Freeze_lr, Min_lr=Unfreeze_lr * 0.01, optimizer_type="adam", num_train=num_train, num_val=num_val, \
        in_channels=in_channels
    )

    # ---------------------------------------------------------#
    #   总训练循环
    # ---------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        # 冻结阶段
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

            batch_size = Freeze_batch_size
            lr = Freeze_lr
            start_epoch = Init_Epoch
            end_epoch = Freeze_Epoch

            optimizer = torch.optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, lr, Unfreeze_lr * 0.01, UnFreeze_Epoch)

            train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path, in_channels)
            val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path, in_channels)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate)

            # EvalCallback 也要传入 in_channels
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period, in_channels=in_channels)

            for epoch in range(start_epoch, end_epoch):
                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                              len(train_lines) // batch_size, len(val_lines) // batch_size, gen, gen_val,
                              end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                              save_period, save_dir, local_rank)
            UnFreeze_flag = True

        # 解冻阶段
        if UnFreeze_flag:
            for param in model.backbone.parameters():
                param.requires_grad = True

            batch_size = Unfreeze_batch_size
            lr = Unfreeze_lr
            start_epoch = Freeze_Epoch
            end_epoch = UnFreeze_Epoch

            optimizer = torch.optim.Adam(model_train.parameters(), lr, weight_decay=5e-4)
            lr_scheduler_func = get_lr_scheduler(lr_decay_type, lr, Unfreeze_lr * 0.01, UnFreeze_Epoch)

            train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path, in_channels)
            val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path, in_channels)
            gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=deeplab_dataset_collate)
            gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate)

            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                         eval_flag=eval_flag, period=eval_period, in_channels=in_channels)

            for epoch in range(start_epoch, end_epoch):
                set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
                fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                              len(train_lines) // batch_size, len(val_lines) // batch_size, gen, gen_val,
                              end_epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler,
                              save_period, save_dir, local_rank)