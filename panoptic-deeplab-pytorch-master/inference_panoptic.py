import argparse
import os
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict

# 导入你的项目关联模块
from dataloaders import make_data_loader
from modeling.panoptic_deeplab import PanopticDeepLab
from modeling.sync_batchnorm.replicate import patch_replication_callback
from dataloaders.datasets.forest import ForestPanoptic
from torch.utils.data import DataLoader


class Tester(object):
    def __init__(self, args):
        self.args = args

        # 1. 核心改进：针对推理阶段优化 Data Loader
        # 如果是 forest 数据集，我们手动初始化，避免 make_data_loader 去寻找不存在的 train 文件夹
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        if args.dataset == 'forest':
            print(f"=> 正在加载林业测试集: {args.base_dir}")
            test_set = ForestPanoptic(args, split='test')  # 这里只加载 test 文件夹
            self.test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)
            self.nclass = 7
        else:
            # 兼容其他数据集（Cityscapes 等）的原有逻辑
            _, _, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # 2. 定义网络 (保持完整参数传递)
        self.model = PanopticDeepLab(
            num_classes=self.nclass,
            backbone=args.backbone,
            output_stride=args.out_stride,
            sync_bn=args.sync_bn,
            freeze_bn=args.freeze_bn,
            in_channels=4,
        )

        # 3. 使用 CUDA 及分布式配置
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()

        # 4. 加载权重 (带清洗逻辑)
        if not os.path.isfile(args.resume):
            raise RuntimeError(f"=> 没找到权重文件: '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location='cuda' if args.cuda else 'cpu')
        state_dict = checkpoint["state_dict"]

        if args.cuda:
            self.model.module.load_state_dict(state_dict)
        else:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)

        print(f"=> 加载成功: {args.resume} (Epoch {checkpoint.get('epoch', 'unknown')})")

        # 5. 推理逻辑分支参数
        if args.dataset == 'forest':
            self.things_category = [1, 2, 3, 4, 5, 6]
            self.reg_factor = 32.0
            self.center_threshold = 0.1
        else:
            self.things_category = [24, 25, 26, 27, 28, 31, 32, 33]
            self.reg_factor = 1.0  # Cityscapes 通常在 post-processing 里有单独逻辑
            self.center_threshold = 0.5

    def test_and_save(self):
        self.model.eval()
        tbar = tqdm(self.test_loader, desc="Inference")

        # 结果保存路径
        save_dir = os.path.join(os.path.dirname(self.args.resume), "test_results")
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        for i, sample in enumerate(tbar):
            image = sample["image"]
            if self.args.cuda: image = image.cuda()

            with torch.no_grad():
                output = self.model(image)

            sem_pred, center_pred, x_off_pred, y_off_pred = output

            # 1. 语义预测
            sem_labels = torch.argmax(sem_pred, dim=1)[0]

            # 2. 偏移量还原
            if self.args.dataset == 'forest':
                x_off = x_off_pred[0] * self.reg_factor
                y_off = y_off_pred[0] * self.reg_factor
            else:
                x_off = x_off_pred[0] / 4.0
                y_off = y_off_pred[0] / 2.0

            # 3. 实例聚合逻辑
            instances = self.get_instances(sem_labels, center_pred[0], x_off, y_off)

            # 4. 全景图生成 (Semantic * 1000 + Instance)
            final_panoptic = sem_labels.float()
            mask = torch.zeros_like(final_panoptic, dtype=torch.bool)
            for cat in self.things_category:
                mask |= (final_panoptic == cat)
            final_panoptic = torch.where(mask.cuda(), final_panoptic * 1000 + instances, final_panoptic)

            # 5. 保存 ID 图像 (用于指标计算)
            res_np = final_panoptic.cpu().numpy().astype(np.int32)
            Image.fromarray(res_np, mode='I').save(os.path.join(save_dir, f"test_{i}_id.png"))

            # 6. 保存彩色图 (用于论文展示)
            self.save_colored_result(res_np, os.path.join(save_dir, f"test_{i}_vis.png"))

    def get_instances(self, semantic_labels, center, x_offset, y_offset):
        center_heatmap = torch.sigmoid(center[0])
        # NMS 找中心点
        centers_max = F.max_pool2d(center_heatmap.unsqueeze(0).unsqueeze(0), 7, stride=1, padding=3).squeeze()
        centers_select = (center_heatmap == centers_max) & (center_heatmap > self.center_threshold)

        center_points = centers_select.nonzero()
        if center_points.shape[0] < 1:
            return torch.zeros_like(semantic_labels).float().cuda()

        h, w = semantic_labels.shape
        grid_y, grid_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        grid_x, grid_y = grid_x.cuda().float(), grid_y.cuda().float()

        target_x = grid_x + x_offset.squeeze()
        target_y = grid_y + y_offset.squeeze()

        dist_min = torch.full((h, w), 1e6).cuda()
        instances = torch.zeros((h, w)).cuda()

        for idx, cp in enumerate(center_points):
            cy, cx = cp[0].float(), cp[1].float()
            dist = torch.sqrt((target_x - cx) ** 2 + (target_y - cy) ** 2)
            # 只有前景像素参与分配
            closer_mask = (dist < dist_min) & (semantic_labels > 0).cuda()
            instances[closer_mask] = idx + 1
            dist_min[closer_mask] = dist[closer_mask]
        return instances

    def save_colored_result(self, panoptic_np, save_path):
        h, w = panoptic_np.shape
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        for uid in np.unique(panoptic_np):
            if uid == 0: continue
            vis[panoptic_np == uid] = [np.random.randint(50, 255) for _ in range(3)]
        Image.fromarray(vis).save(save_path)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Panoptic Deeplab Inference")

    # ==========================================
    # 完整参数列表 - 不再省略任何项
    # ==========================================
    parser.add_argument("--backbone", type=str, default="resnet_3stage",
                        choices=["xception_3stage", "mobilenet_3stage", "resnet_3stage"])
    parser.add_argument("--out-stride", type=int, default=16)
    parser.add_argument("--dataset", type=str, default="forest", choices=["pascal", "coco", "cityscapes", "forest"])
    parser.add_argument("--task", type=str, default="panoptic", choices=["segmentation", "panoptic"])
    parser.add_argument("--use-sbd", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-size", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--sync-bn", type=bool, default=False)
    parser.add_argument("--freeze-bn", type=bool, default=False)
    parser.add_argument("--loss-type", type=str, default="ce", choices=["ce", "focal"])

    # 训练/优化相关参数 (推理时作为兼容性保留)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--use-balanced-weights", action="store_true", default=False)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-scheduler", type=str, default="poly")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--nesterov", action="store_true", default=False)

    # 环境与路径参数
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--resume", type=str, default="run/forest/panoptic-deeplab-resnet_3stage/model_best.pth.tar")
    parser.add_argument("--checkname", type=str, default="exp_10")
    parser.add_argument("--ft", action="store_true", default=False)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--no-val", action="store_true", default=False)

    # 数据集路径
    parser.add_argument("--base-dir", type=str, default="H:/gaofeng2020_yzq/dataset/test/img_tif_RGBN/")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        args.gpu_ids = [int(s) for s in args.gpu_ids.split(",")]

    print(f"Inference Configuration:\nBackbone: {args.backbone}\nDataset: {args.dataset}\nResume: {args.resume}")

    torch.manual_seed(args.seed)
    tester = Tester(args)
    tester.test_and_save()


if __name__ == "__main__":
    main()