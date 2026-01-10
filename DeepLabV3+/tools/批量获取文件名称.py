from pathlib import Path

# ===== 直接修改这里 =====
folder_path = r"H:\large-scale land cover classification\Delaware\segformer\VOCdevkit\VOC2007\TIFFImages"  # 修改成你的实际路径
output_file = r'H:\large-scale land cover classification\Delaware\segformer\VOCdevkit\VOC2007\ImageSets\Segmentation\trainval.txt'

# 获取图片文件（修正版）
image_files = []
folder = Path(folder_path)

# 只遍历一次，不区分大小写
for file in folder.iterdir():
    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png','.tif']:
        image_files.append(file)

# 排序并写入
image_files.sort(key=lambda x: x.name)
with open(output_file, 'w') as f:
    for img in image_files:
        f.write(img.stem + '\n')

print(f"完成！处理了 {len(image_files)} 个文件")