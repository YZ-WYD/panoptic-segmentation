from osgeo import gdal
import os
import random
from tqdm import tqdm   # ← 新增

def make_same_random_crops(img_ds, label_ds, out_img_dir, out_label_dir,
                           base_name, crop_size=224, num_crops=50):
    cols = img_ds.RasterXSize
    rows = img_ds.RasterYSize

    if cols != label_ds.RasterXSize or rows != label_ds.RasterYSize:
        print(f"❌ img/label 尺寸不一致：{base_name}")
        return
    if cols < crop_size or rows < crop_size:
        print(f"⚠️ 图像太小，跳过：{base_name}")
        return

    # 生成 50 个随机窗口
    windows = [(random.randint(0, cols - crop_size),
                random.randint(0, rows - crop_size)) for _ in range(num_crops)]

    # 同一行内动态刷新进度条
    with tqdm(total=num_crops, ncols=80, desc=f"正在处理 {base_name}",
              bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}") as pbar:
        for idx, (x, y) in enumerate(windows):
            suffix = f"_rand{idx:03d}.tif"
            opt = gdal.TranslateOptions(format='GTiff',
                                        srcWin=(x, y, crop_size, crop_size))
            gdal.Translate(os.path.join(out_img_dir, base_name + suffix), img_ds, options=opt)
            gdal.Translate(os.path.join(out_label_dir, base_name + suffix), label_ds, options=opt)
            pbar.update(1)          # 更新进度
    # 一张图完成自动换行
    print()

# ================= 主程序 =================
img_root   = r"D:\Delaware\train\img"
label_root = r"D:\Delaware\train\label"
out_img    = r"H:\large-scale land cover classification\Delaware\segformer\VOCdevkit\VOC2007\TIFFImages"
out_label  = r"H:\large-scale land cover classification\Delaware\segformer\VOCdevkit\VOC2007\SegmentationClass_TIF"

os.makedirs(out_img,  exist_ok=True)
os.makedirs(out_label, exist_ok=True)

for filename in os.listdir(img_root):
    if not filename.lower().endswith('.tif'):
        continue

    img_path   = os.path.join(img_root,   filename)
    label_path = os.path.join(label_root, filename)

    img_ds   = gdal.Open(img_path)
    label_ds = gdal.Open(label_path)
    if img_ds is None or label_ds is None:
        print(f"❌ 无法打开 img/label：{filename}")
        continue

    base_name = os.path.splitext(filename)[0]
    make_same_random_crops(img_ds, label_ds, out_img, out_label,
                           base_name, crop_size=224, num_crops=50)
    img_ds = label_ds = None

print("🎉 全部 img/label 同步随机裁剪完成！")