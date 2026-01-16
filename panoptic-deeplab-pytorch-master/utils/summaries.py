import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step, centers=None, reg_x=None,
                        reg_y=None):
        # image: (B, C, H, W)
        # target: (B, H, W)
        # output: (B, Num_Classes, H, W) -> 需要 argmax

        # 1. Input Image
        disp_image = image[:3, :3, :, :].clone()  # 只取前3张图，前3个通道
        grid_image = make_grid(disp_image, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)

        # 2. Ground Truth (Label)
        # target 已经是 (B, H, W)，不需要 torch.max
        # 注意：decode_seg_map_sequence 需要 numpy 输入
        target_display = target[:3].detach().cpu().numpy()
        grid_image = make_grid(decode_seg_map_sequence(target_display, dataset=dataset), 3, normalize=False)
        writer.add_image('Groundtruth label', grid_image, global_step)

        # 3. Prediction
        # output 是 logits (B, C, H, W)，需要 argmax 变成 (B, H, W)
        output_display = torch.argmax(output[:3], dim=1).detach().cpu().numpy()
        grid_image = make_grid(decode_seg_map_sequence(output_display, dataset=dataset), 3, normalize=False)
        writer.add_image('Predicted label', grid_image, global_step)

        # 4. Centers
        if centers is not None:
            grid_image = make_grid(centers[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Center Heatmap', grid_image, global_step)

        # 5. Regressions
        if reg_x is not None:
            grid_image = make_grid(reg_x[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Regression X', grid_image, global_step)

        if reg_y is not None:
            grid_image = make_grid(reg_y[:3].clone().cpu().data, 3, normalize=True)
            writer.add_image('Regression Y', grid_image, global_step)