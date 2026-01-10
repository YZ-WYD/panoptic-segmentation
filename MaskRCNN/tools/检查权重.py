#!/usr/bin/env python3
# ==================== 一键运行版 ====================
ckpt_path = r'D:\yz\1_yz\mask-rcnn-pytorch-master\logs_pytorch\best_epoch_weights.pth'   # <— 改成你的权重路径
import torch, os, sys
sys.argv = [sys.argv[0], ckpt_path]   # 假装命令行已给路径
# ===================================================

def main(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))

    print(f'\n==========  {ckpt_path}  ==========')
    print('TYPE :', type(state))
    if isinstance(state, dict):
        print('KEYS :', list(state.keys())[:10], '...' if len(state) > 10 else '')
        print('\n---- 与类别数直接相关的参数 ----')
        for k, v in state.items():
            if 'cls_score' in k or 'bbox_pred' in k or 'mask_fcn_logits' in k:
                print(f'{k:<45}  {tuple(v.shape)}')
        cls_weight = state.get('roi_heads.box_predictor.cls_score.weight')
        if cls_weight is not None:
            num_cls = cls_weight.shape[0]
            print(f'\n推断类别数（含背景）: {num_cls}')
        else:
            print('\n未找到 cls_score，无法推断类别数')
    else:
        print('警告：权重不是 dict 格式，无法检查！')
    print('=' * 60 + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python check_ckpt.py <xxx.pth>')
        sys.exit(1)
    main(sys.argv[1])