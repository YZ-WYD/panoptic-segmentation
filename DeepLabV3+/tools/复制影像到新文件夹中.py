import shutil
from pathlib import Path

# ===== 在这里修改你的路径 =====
source_folder = r"D:\yz\1_yz\dataset\cocodataset\val2017"  # 原始图片文件夹
target_folder = r"D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\JPEGImages"  # 复制到的目标文件夹

# 创建目标文件夹（如果不存在）
target_path = Path(target_folder)
target_path.mkdir(parents=True, exist_ok=True)

# 获取所有 jpg/png 文件（不区分大小写）
source_path = Path(source_folder)
image_files = [f for f in source_path.iterdir()
               if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]

# 复制文件
for img in image_files:
    shutil.copy2(img, target_path / img.name)

print(f"完成！共复制了 {len(image_files)} 张图片到 {target_folder}")