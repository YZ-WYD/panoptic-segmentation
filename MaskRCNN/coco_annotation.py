# -------------------------------------------------------------------------
# coco_annotation.py (已修复：支持 Tiff / 多波段，防止数据损坏)
# -------------------------------------------------------------------------
import json
import os
import glob
import os.path as osp
import numpy as np
import PIL.Image
import labelme.utils
import shutil
from tqdm import tqdm

# 从新配置读取类别路径
from utils.config import CLASSES_PATH


def main():
    # ------------------------------------#
    #   参数设置
    # ------------------------------------#
    input_dir = "datasets/before"
    Img_output_dir = "datasets/coco/JPEGImages"
    Json_output_dir = "datasets/coco/Jsons"

    # 训练/验证 比例
    trainval_percent = 0.9
    train_percent = 0.9

    os.makedirs(Img_output_dir, exist_ok=True)
    os.makedirs(Json_output_dir, exist_ok=True)

    # 1. 获取所有 json 文件
    label_files = glob.glob(osp.join(input_dir, '*.json'))

    # 打乱顺序
    np.random.seed(10101)
    np.random.shuffle(label_files)
    np.random.seed(None)

    num_train_val = int(trainval_percent * len(label_files))
    num_train = int(train_percent * num_train_val)

    train_label_files = label_files[:num_train]
    val_label_files = label_files[num_train:num_train_val]
    test_label_files = label_files[num_train_val:]

    # 准备输出
    label_files_list = [train_label_files, val_label_files, test_label_files]
    ann_names = ['train_annotations.json', 'val_annotations.json', 'test_annotations.json']

    # 2. 读取类别
    from utils.utils import get_classes
    class_names, _ = get_classes(CLASSES_PATH)

    # 建立 name -> id 映射 (1-based index for COCO)
    # 注意：Labelme json 里的 label 必须在 classes.txt 里能找到
    class_name_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    print("Class Mapping:", class_name_to_id)

    # 3. 循环处理
    for label_files_index, label_files in enumerate(label_files_list):
        data_dict = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 写入 Categories
        for name, id_ in class_name_to_id.items():
            data_dict['categories'].append({"id": id_, "name": name, "supercategory": "object"})

        ann_id_count = 0
        current_ann_file = ann_names[label_files_index]
        print(f"Processing {current_ann_file} ({len(label_files)} files)...")

        for image_id, label_file in enumerate(tqdm(label_files)):
            with open(label_file, encoding='utf-8') as f:
                label_data = json.load(f)

            # -------------------------------------------------------
            #   核心修复：寻找并复制原图 (支持 Tiff)
            # -------------------------------------------------------
            base_name = osp.splitext(osp.basename(label_file))[0]

            # 尝试寻找对应的图片 (可能是 jpg, png, tif, tiff)
            possible_exts = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
            found_img_path = None
            found_ext = None

            # A. 先看 json 里写的 imagePath
            json_img_path = osp.join(osp.dirname(label_file), label_data.get('imagePath', ''))
            if osp.exists(json_img_path):
                found_img_path = json_img_path
                found_ext = osp.splitext(json_img_path)[1]
            else:
                # B. 遍历后缀查找
                for ext in possible_exts:
                    img_path = osp.join(osp.dirname(label_file), base_name + ext)
                    if osp.exists(img_path):
                        found_img_path = img_path
                        found_ext = ext
                        break

            if not found_img_path:
                print(f"[Warning] Image for {label_file} not found. Skipping.")
                continue

            # 复制图片到 datasets/coco/JPEGImages
            # 注意：如果已经是 Tiff，这里只是复制，不会改变格式
            out_img_name = base_name + found_ext
            out_img_path = osp.join(Img_output_dir, out_img_name)
            if not osp.exists(out_img_path):
                shutil.copy(found_img_path, out_img_path)

            # 读取宽用于写入 json (PIL 打开 Tiff 没问题)
            img_pil = PIL.Image.open(found_img_path)
            h, w = img_pil.size[1], img_pil.size[0]

            data_dict['images'].append({
                "id": image_id,
                "file_name": out_img_name,
                "height": h,
                "width": w
            })

            # -------------------------------------------------------
            #   处理标注
            # -------------------------------------------------------
            for shape in label_data['shapes']:
                label = shape['label']
                # 处理 label (去除 _1, _2 后缀，如 triangle_1 -> triangle)
                clean_label = label.split('_')[0]

                if clean_label not in class_name_to_id:
                    continue

                cls_id = class_name_to_id[clean_label]
                points = shape['points']

                # 计算 Mask / Bbox
                mask = labelme.utils.shape_to_mask((h, w), points, shape_type=shape.get('shape_type', None))
                mask = np.asfortranarray(mask.astype(np.uint8))

                import pycocotools.mask as mask_utils
                rle = mask_utils.encode(mask)
                bbox = mask_utils.toBbox(rle).flatten().tolist()
                area = float(mask_utils.area(rle))

                # 存入 segmentation (polygon)
                # 注意：为了兼容性，这里简单存储 points。如果 mask 有孔洞，这种方式可能不完美，
                # 但对于大多数 Labelme 转 COCO 场景足够了。
                data_dict['annotations'].append({
                    "id": ann_id_count,
                    "image_id": image_id,
                    "category_id": cls_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [np.asarray(points).flatten().tolist()],
                    "iscrowd": 0
                })
                ann_id_count += 1

        # 保存 JSON
        out_path = osp.join(Json_output_dir, current_ann_file)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data_dict, f, indent=4, ensure_ascii=False)
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()