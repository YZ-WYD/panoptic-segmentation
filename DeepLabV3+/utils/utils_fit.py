import os
import torch
from nets.deeplabv3_training import weights_init
from tqdm import tqdm


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0

    val_loss = 0
    val_f_score = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(imgs)
            # ----------------------#
            #   计算损失
            # ----------------------#
            if focal_loss:
                from utils.utils_metrics import Focal_Loss
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                from utils.utils_metrics import CE_Loss
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                from utils.utils_metrics import Dice_loss
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice

            with torch.no_grad():
                # ----------------------#
                #   计算f_score
                # ----------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(imgs)
                if focal_loss:
                    from utils.utils_metrics import Focal_Loss
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    from utils.utils_metrics import CE_Loss
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    from utils.utils_metrics import Dice_loss
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            outputs = model_train(imgs)
            if focal_loss:
                from utils.utils_metrics import Focal_Loss
                loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
            else:
                from utils.utils_metrics import CE_Loss
                loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

            if dice_loss:
                from utils.utils_metrics import Dice_loss
                main_dice = Dice_loss(outputs, labels)
                loss = loss + main_dice
            # ----------------------#
            #   计算f_score
            # ----------------------#
            _f_score = f_score(outputs, labels)

            val_loss += loss.item()
            val_f_score += _f_score.item()

            if local_rank == 0:
                pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                    'f_score': val_f_score / (iteration + 1),
                                    'lr': get_lr(optimizer)})
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        if eval_callback is not None:
            eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (
            epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_losses) <= 1 or val_loss / epoch_step_val <= min(loss_history.val_losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def f_score(inputs, target, beta=1, smooth=1e-5, threshold=0.5):
    """
    计算 F-score，针对 '忽略标签0' 进行了特殊适配
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = torch.nn.functional.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # ----------------------------------------------------------- #
    # 【核心修改】应用掩膜机制 + 排除第0类
    # ----------------------------------------------------------- #
    # 1. 获取掩膜: target的最后一维是 ignore_channel (1表示忽略)
    ignore_mask = temp_target[..., -1:]  # shape: (n, pixels, 1)
    valid_mask = 1.0 - ignore_mask  # 1表示有效，0表示忽略

    # 2. 将忽略区域的预测置为0，防止产生FP
    temp_inputs = temp_inputs * valid_mask

    # 3. 计算 TP, FP, FN
    # 注意：这里 temp_target[..., :-1] 包含所有类别通道 (0, 1, ..., num_classes-1)
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    # 4. 计算每个类别的 F-score
    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # 5. 【关键】计算平均分时，跳过第0类 (背景/未标注)
    # 假设 score 的 shape 是 [num_classes]
    # 如果您确定第0类是未标注/背景且不需要参与评估：
    if score.shape[0] > 1:
        score = score[1:]  # 抛弃索引0，只计算 1~N 类的平均分

    score = torch.mean(score)
    return score