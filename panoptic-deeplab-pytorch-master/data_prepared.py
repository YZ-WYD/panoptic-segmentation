from osgeo import gdal
import os

def crop_image(input_image_path, output_image_path, x_offset, y_offset, size=512):
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    gdal.UseExceptions()
    # 使用 gdal.Translate 进行裁剪
    translate_options = gdal.TranslateOptions(
        format='GTiff',
        srcWin=(x_offset, y_offset, size, size)  # 假设每次裁剪的区域大小为size*size
    )
    gdal.Translate(output_image_path, input_image_path, options=translate_options)

def process_images(src_folder, output_base_folder, block_size=512):
    # 获取所有子文件夹并按名称排序
    subfolders = [f for f in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, f))]
    subfolders.sort()  # 按名称排序

    for subfolder in subfolders:
        subfolder_path = os.path.join(src_folder, subfolder)
        if os.path.isdir(subfolder_path):  # 确保是文件夹
            processed = False
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.tif'):
                    src_path = os.path.join(subfolder_path, filename)
                    dst_folder = os.path.join(output_base_folder, subfolder)
                    if not os.path.exists(dst_folder):
                        os.makedirs(dst_folder)
                    processed = True

                    # 获取图像的尺寸
                    src_ds = gdal.Open(src_path)
                    if src_ds is None:
                        print(f"Failed to open {src_path}")
                        continue

                    cols = src_ds.RasterXSize
                    rows = src_ds.RasterYSize

                    # 裁剪图像为小块
                    for i in range(0, rows, block_size):
                        for j in range(0, cols, block_size):
                            x_offset = j
                            y_offset = i
                            part_filename = f"image_{i}_{j}.tif"
                            part_path = os.path.join(dst_folder, part_filename)
                            crop_image(src_path, part_path, x_offset, y_offset, block_size)

            if processed:
                print(f'All images in {subfolder} have been cropped and saved to {dst_folder}')

    print('All images in all subfolders have been cropped.')

# 用户需要输入的路径信息
src_folder = r'D:\原电脑\成果数据\影像'  # 源文件夹路径，包含多个子文件夹，每个子文件夹有一个 TIF 文件
output_base_folder = r'D:\yz\1_yz\data\20241121'  # 输出文件夹的根目录路径

# 用户指定的裁剪大小
block_size = 512  # 裁剪大小，可以根据需要修改

# 开始处理图像
process_images(src_folder, output_base_folder, block_size)