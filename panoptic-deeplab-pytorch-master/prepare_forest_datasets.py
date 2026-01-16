import os
import json
import cv2
import numpy as np
import random
import shutil
from tqdm import tqdm
from PIL import Image

# ================= 1. 用户配置区域 =================
ROOT_DIR = "datasets/original_data"
IMAGE_DIR = os.path.join(ROOT_DIR, "tif_images")
SEM_DIR = os.path.join(ROOT_DIR, "semantic_tif")
INST_DIR = os.path.join(ROOT_DIR, "instance_json")
SEM_SUFFIX = "_pred"

# 输出路径
OUT_ROOT = "datasets/my_forest_panoptic_v12_4band_noaug"  # 改名: noaug 表示无增强

# 配置
VAL_NUM = 5
RANDOM_SEED = 2026

# -----------------------------------------------------------
# 2. 类别定义 (保持不变)
# -----------------------------------------------------------
CATEGORIES = [
    {"id": 0, "name": "vegetation", "isthing": 0, "color": [144, 238, 144]},
    {"id": 1, "name": "bare", "isthing": 0, "color": [139, 69, 19]},
    {"id": 2, "name": "impervious", "isthing": 0, "color": [255, 255, 0]},
    {"id": 3, "name": "tree_a", "isthing": 1, "color": [0, 200, 0]},
    {"id": 4, "name": "tree_b", "isthing": 1, "color": [0, 100, 0]},
    {"id": 5, "name": "tree_c", "isthing": 1, "color": [34, 139, 34]},
    {"id": 6, "name": "tree_d", "isthing": 1, "color": [0, 255, 127]},
]
SEMANTIC_MAPPING = {0: 0, 1: 0, 2: 1, 3: 2}


# ================= 3. 工具函数 =================

def get_category_id_from_label(label_name):
    clean_label = label_name.strip().lower()
    if 'a' in clean_label: return 3
    if 'b' in clean_label: return 4
    if 'c' in clean_label: return 5
    if 'd' in clean_label: return 6
    return 3


def smart_normalize_to_8bit(img):
    """
    智能归一化，支持任意通道数 (3或4)
    """
    if img.dtype == np.uint8: return img
    img_float = img.astype(np.float32)
    min_val, max_val = np.min(img_float), np.max(img_float)

    if max_val <= 2.0:
        img_8bit = img_float * 255.0
    elif max_val > 255:
        if max_val - min_val < 1e-6:
            img_8bit = np.zeros_like(img, dtype=np.uint8)
        else:
            img_8bit = ((img_float - min_val) / (max_val - min_val) * 255.0)
    else:
        img_8bit = img_float

    return np.clip(img_8bit, 0, 255).astype(np.uint8)


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map = id_map.astype(np.int32)
        rgb_map = np.zeros((id_map.shape[0], id_map.shape[1], 3), dtype=np.uint8)
        rgb_map[:, :, 0] = id_map % 256
        rgb_map[:, :, 1] = (id_map // 256) % 256
        rgb_map[:, :, 2] = (id_map // (256 * 256))
        return rgb_map
    return id_map


def save_visualize(panoptic_map, out_path):
    h, w = panoptic_map.shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    vis_img[:] = [50, 50, 50]
    unique_ids = np.unique(panoptic_map)
    for uid in unique_ids:
        mask = (panoptic_map == uid)
        if uid == 0: color = [144, 238, 144]
        elif uid == 1: color = [19, 69, 139]
        elif uid == 2: color = [0, 255, 255]
        else:
            np.random.seed(int(uid))
            color = np.random.randint(50, 255, 3).tolist()
        vis_img[mask] = color
    cv2.imwrite(out_path, vis_img)


# ================= 4. 核心处理逻辑 =================

def process_dataset_split():
    # 0. 准备输出目录
    for sub in ["train", "val"]:
        os.makedirs(os.path.join(OUT_ROOT, sub), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, f"panoptic_{sub}"), exist_ok=True)
        os.makedirs(os.path.join(OUT_ROOT, "visualize_check", sub), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "annotations"), exist_ok=True)

    # 1. 划分
    files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.tif')]
    files.sort()
    random.seed(RANDOM_SEED)
    random.shuffle(files)

    if len(files) <= VAL_NUM:
        raise ValueError(f"图片太少: {len(files)}")

    val_files = files[-VAL_NUM:]
    train_files = files[:-VAL_NUM]

    # 记录
    with open(os.path.join(OUT_ROOT, "split_train.txt"), "w") as f:
        f.write("\n".join(train_files))
    with open(os.path.join(OUT_ROOT, "split_val.txt"), "w") as f:
        f.write("\n".join(val_files))

    print(f"总数: {len(files)} | 训练集: {len(train_files)} | 验证集: {len(val_files)}")

    # 2. 处理函数 (移除 enable_aug 参数)
    def _process_subset(file_list, subset_name):
        images_info = []
        annotations_info = []

        print(f"\n正在生成 {subset_name} 集 (无数据增强)...")

        for file_name in tqdm(file_list):
            file_id_raw = os.path.splitext(file_name)[0]

            # 读取
            img_path = os.path.join(IMAGE_DIR, file_name)
            img_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img_raw is None: continue

            # 归一化 (保留 4 通道)
            img_8bit = smart_normalize_to_8bit(img_raw)
            h, w = img_8bit.shape[:2]

            sem_path = os.path.join(SEM_DIR, f"{file_id_raw}{SEM_SUFFIX}.tif")
            json_path = os.path.join(INST_DIR, f"{file_id_raw}.json")

            # 构建全景Mask
            panoptic_map_raw = np.zeros((h, w), dtype=np.int32)

            if os.path.exists(sem_path):
                sem_mask = cv2.imread(sem_path, cv2.IMREAD_UNCHANGED)
                mapped_sem = np.zeros_like(sem_mask)
                for old, new in SEMANTIC_MAPPING.items():
                    mapped_sem[sem_mask == old] = new
                for cat_id in np.unique(mapped_sem):
                    if cat_id <= 2:
                        panoptic_map_raw[mapped_sem == cat_id] = cat_id

            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                inst_counter = 1
                for shape in label_data.get('shapes', []):
                    cat_id = get_category_id_from_label(shape.get('label', 'a'))
                    pan_id = cat_id * 1000 + inst_counter
                    pts = np.array(shape['points'], dtype=np.int32)
                    inst_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(inst_mask, [pts], 1)
                    panoptic_map_raw[inst_mask == 1] = pan_id
                    inst_counter += 1

            # --- 移除增强循环，直接处理原始数据 ---
            # 定义文件名
            file_name_image = f"{file_id_raw}.tif"
            file_name_png = f"{file_id_raw}.png"

            # 1. 保存图片 (TIF格式, 4波段)
            cv2.imwrite(os.path.join(OUT_ROOT, subset_name, file_name_image), img_8bit)

            # 2. 保存 Mask (ID图 PNG)
            Image.fromarray(id2rgb(panoptic_map_raw)).save(os.path.join(OUT_ROOT, f"panoptic_{subset_name}", file_name_png))

            # 3. 可视化检查 (PNG)
            save_visualize(panoptic_map_raw, os.path.join(OUT_ROOT, "visualize_check", subset_name, f"{file_id_raw}_vis.png"))

            # 4. JSON 信息提取
            segments_info = []
            for uid in np.unique(panoptic_map_raw):
                if uid == 0 and np.sum(panoptic_map_raw == 0) == 0: continue
                mask = (panoptic_map_raw == uid)
                area = int(np.sum(mask))
                if uid <= 2:
                    segments_info.append(
                        {"id": int(uid), "category_id": int(uid), "area": area, "iscrowd": 0, "isthing": 0})
                else:
                    cat_id = uid // 1000
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    bbox = [0, 0, 0, 0]
                    if contours:
                        x, y, w_box, h_box = cv2.boundingRect(contours[0])
                        bbox = [x, y, w_box, h_box]
                    segments_info.append(
                        {"id": int(uid), "category_id": int(cat_id), "area": area, "bbox": bbox, "iscrowd": 0,
                         "isthing": 1})

            images_info.append({"id": file_id_raw, "width": w, "height": h, "file_name": file_name_image})
            annotations_info.append(
                {"image_id": file_id_raw, "file_name": file_name_png, "segments_info": segments_info})

        # 保存JSON
        json_out = os.path.join(OUT_ROOT, "annotations", f"panoptic_forest_{subset_name}.json")
        out_cats = [c.copy() for c in CATEGORIES]
        for c in out_cats:
            if 'color' in c: del c['color']
        with open(json_out, 'w') as f:
            json.dump({"images": images_info, "annotations": annotations_info, "categories": out_cats}, f)

    # 执行
    _process_subset(train_files, "train")
    _process_subset(val_files, "val")

    print(f"\n全部完成！请检查: {OUT_ROOT}/visualize_check")


if __name__ == "__main__":
    process_dataset_split()