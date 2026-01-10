import os
import torch
import scipy.signal
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.utils import preprocess_input, resize_image
from utils.utils_metrics import f_score
from osgeo import gdal
from PIL import Image
import cv2


class LossHistory():
    def __init__(self, log_dir, model, input_shape, in_channels=3):
        self.log_dir = log_dir
        self.losses = []
        self.val_losses = []

        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2, in_channels, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_losses.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'coral', linewidth=2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_losses, num, 3), '#8B4513', linestyle='--', linewidth=2,
                     label='smooth val loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, eval_flag=True, period=1,
                 in_channels=3):
        super(EvalCallback, self).__init__()
        self.net = net
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.image_ids = image_ids
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.cuda = cuda
        self.eval_flag = eval_flag
        self.period = period
        self.in_channels = in_channels
        self.image_ids = [image_id.split()[0] for image_id in image_ids]
        self.mious = [0]
        self.epoches = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def read_tif(self, path):
        ds = gdal.Open(path)
        if ds is None: return None
        bands = [ds.GetRasterBand(i + 1).ReadAsArray() for i in range(ds.RasterCount)]
        return np.stack(bands, axis=-1)

    def get_miou_png(self, image):
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = np.transpose(preprocess_input(image_data), (2, 0, 1))
        image_data = np.expand_dims(image_data, 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda: images = images.cuda()
            pr = self.net(images)[0]
            pr = torch.nn.functional.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            return pr

    def on_epoch_end(self, epoch, model_eval):
        if epoch % self.period == 0 and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "VOC2007/SegmentationClass_TIF")
            pred_dir = os.path.join(self.log_dir, "miou_pr_dir")
            if not os.path.exists(pred_dir): os.makedirs(pred_dir)

            print("Get miou...")
            for image_id in tqdm(self.image_ids):
                image_path = os.path.join(self.dataset_path, "VOC2007/TIFFImages/" + image_id + ".tif")
                image = self.read_tif(image_path)
                if image is None: continue
                org_h, org_w = image.shape[:2]
                pr = self.get_miou_png(image)
                pr = cv2.resize(pr, (org_w, org_h), interpolation=cv2.INTER_LINEAR)
                image = pr.argmax(axis=-1)

                # 保存为 png 供 f_score 计算 (注意：f_score 会通过 read_png_label 读取)
                image = Image.fromarray(np.uint8(image))
                image.save(os.path.join(pred_dir, image_id + ".png"))

            print("Calculate miou...")
            # f_score 已经在 utils_metrics.py 中修改，会忽略 GT=0 的区域
            _, IoUs, _, _ = f_score(gt_dir, pred_dir, self.image_ids, self.num_classes, name_classes=None)

            _miou = np.nanmean(IoUs) * 100
            self.mious.append(_miou)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(_miou))
                f.write("\n")

            print("Get miou done.")
            import shutil
            shutil.rmtree(pred_dir)