from osgeo import gdal
import os

def crop_image(src_ds, output_image_path, x_offset, y_offset, cols, rows, size=512):
    actual_width  = min(size, cols - x_offset)
    actual_height = min(size, rows - y_offset)
    if actual_width <= 0 or actual_height <= 0:
        return          # 跳过越界小块

    opt = gdal.TranslateOptions(format='GTiff',
                                srcWin=(x_offset, y_offset,
                                        actual_width, actual_height))
    gdal.Translate(output_image_path, src_ds, options=opt)

# 路径
src_folder = r'D:\HTGY\labels_up_align'
dst_folder = r'I:\large-scale land cover classification\deeplabv3+\VOCdevkit\VOC2007\SegmentationClass_TIF'

os.makedirs(dst_folder, exist_ok=True)

for filename in os.listdir(src_folder):
    if not filename.lower().endswith('.tif'):
        continue

    src_path = os.path.join(src_folder, filename)
    src_ds = gdal.Open(src_path)
    if src_ds is None:
        print(f'Failed to open {src_path}')
        continue

    cols, rows = src_ds.RasterXSize, src_ds.RasterYSize

    base_name = os.path.splitext(filename)[0]   # 去掉扩展名
    tile_id = 0                                 # 序号从 0 开始
    block = 512
    for y in range(0, rows, block):
        for x in range(0, cols, block):
            out_name = f'{base_name}_{tile_id}.tif'
            out_path = os.path.join(dst_folder, out_name)
            crop_image(src_ds, out_path, x, y, cols, rows, block)
            tile_id += 1

    src_ds.FlushCache()
    src_ds = None
    print(f'Finished {filename} -> {tile_id} tiles')

print('All images have been cropped.')