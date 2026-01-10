import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

# -------------------------------------------------------------------------#
#   配置区域
# -------------------------------------------------------------------------#
# JSON 所在的文件夹
json_dir = "datasets/before"

# 输出标签的文件夹 (会自动创建)
out_dir = "datasets/SegmentationClass_TIF"

# 类别文件 (txt)，一行一个类别，不含背景
classes_path = "model_data/cls_classes.txt"

if __name__ == "__main__":
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 1. 读取类别
    with open(classes_path, "r", encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    # 建立类别映射表: name -> index (1, 2, 3...)
    # 背景默认为 0
    name_to_index = {name: i + 1 for i, name in enumerate(class_names)}
    print("类别映射表:", name_to_index)

    # 2. 获取所有 JSON 文件
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    print(f"找到 {len(json_files)} 个 JSON 文件，开始转换...")

    for json_file in tqdm(json_files):
        json_path = os.path.join(json_dir, json_file)

        try:
            with open(json_path, "r", encoding='utf-8') as f:
                data = json.load(f)

            # 获取图像尺寸
            img_h = data.get("imageHeight")
            img_w = data.get("imageWidth")

            # 如果 JSON 里没有尺寸信息，尝试找对应的图片读取 (这里假设有)
            if img_h is None or img_w is None:
                print(f"Skipping {json_file}: No image size info.")
                continue

            # 创建空白标签图 (背景为0)
            label_img = Image.new('L', (img_w, img_h), 0)
            draw = ImageDraw.Draw(label_img)

            # 遍历 shapes
            for shape in data.get("shapes", []):
                label_name = shape.get("label")
                points = shape.get("points")

                # 获取类别索引
                idx = name_to_index.get(label_name)
                if idx is None:
                    # 如果遇到未定义的类别，可以选择跳过或报错
                    # print(f"Warning: Unknown label '{label_name}' in {json_file}")
                    continue

                # 绘制多边形
                # 注意：Labelme 的 points 可能是 float，ImageDraw 需要 tuple
                points_tuple = [tuple(p) for p in points]
                draw.polygon(points_tuple, outline=idx, fill=idx)

            # 保存为 TIF (单通道 uint8)
            # 文件名与 json 同名
            out_name = os.path.splitext(json_file)[0] + ".tif"
            label_img.save(os.path.join(out_dir, out_name))

        except Exception as e:
            print(f"Error converting {json_file}: {e}")

    print("转换完成！")