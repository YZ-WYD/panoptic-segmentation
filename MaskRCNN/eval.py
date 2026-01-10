# -------------------------------------------------------------------------
# eval.py (已适配：官方模型 + 多波段评估)
# -------------------------------------------------------------------------
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 引入我们写好的核心类
from mask_rcnn import MASK_RCNN
# 引入配置
from utils.config import CLASSES_PATH, IMAGE_DIVISOR
# 引入评估工具 (utils_map.py)
from utils.utils_map import Make_json, prep_metrics
from utils.utils import get_classes

# --------------------------
#  配置区域
# --------------------------
# 0: 预测并计算mAP, 1: 仅预测生成json, 2: 仅计算mAP (前提是已有json)
MAP_MODE = 0

# 指向验证集的标签文件 (由 coco_annotation.py 生成)
JSON_PATH = 'datasets/coco/Jsons/val_annotations.json'
# 图片文件夹
IMAGE_DIR = 'datasets/coco/JPEGImages'
# 结果保存路径
MAP_OUT_PATH = 'map_out'


def main():
    os.makedirs(MAP_OUT_PATH, exist_ok=True)

    # 1. 检查文件是否存在
    if not os.path.exists(JSON_PATH):
        print(f"Error: 找不到标签文件 {JSON_PATH}")
        print("请先运行 coco_annotation.py 生成 COCO 格式标签。")
        return

    coco = COCO(JSON_PATH)
    class_names, _ = get_classes(CLASSES_PATH)

    # 2. 建立映射 (COCO ID <-> Model Index)
    # COCO ID 可能是不连续的，但我们的模型输出索引是连续的 (1, 2, 3...)
    # 这里的逻辑假设 coco_annotation.py 是按顺序生成的 ID
    coco_cat_ids = sorted(coco.getCatIds())

    # 映射：COCO ID -> 1, 2, 3... (用于理解标签)
    coco2model = {c_id: i + 1 for i, c_id in enumerate(coco_cat_ids)}
    # 反向：0, 1, 2... -> COCO ID (用于结果写入，注意这里用的 0-based 索引)
    model2coco = {i: c_id for i, c_id in enumerate(coco_cat_ids)}

    img_ids = list(coco.imgs.keys())

    # 3. 预测阶段 (生成 bbox_detections.json 和 mask_detections.json)
    if MAP_MODE in [0, 1]:
        print("Loading model for evaluation...")
        # 实例化模型 (置信度设低一点，保证召回率)
        mask_rcnn = MASK_RCNN(confidence=0.01, nms_iou=0.5)

        # 初始化结果生成器
        make_json = Make_json(MAP_OUT_PATH, coco2model, model2coco)

        print(f"Start inference on {len(img_ids)} images...")
        for img_id in tqdm(img_ids):
            img_info = coco.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            img_path = os.path.join(IMAGE_DIR, file_name)

            try:
                # 【关键修改】直接打开，不要 convert('RGB')，以支持 Tiff
                image = Image.open(img_path)
            except Exception as e:
                print(f"Read error: {img_path} - {e}")
                continue

            # 调用 mask_rcnn.py 中的专用接口
            # 返回: boxes, scores, labels, masks_bool
            boxes, scores, labels, masks_bool, _ = mask_rcnn.get_map_out(image)

            if boxes is None:
                continue

            # 写入结果
            prep_metrics(boxes, scores, labels, masks_bool, img_id, make_json)

        make_json.dump()
        print("Inference done. Results saved to map_out/")

    # 4. 计算指标阶段
    if MAP_MODE in [0, 2]:
        bbox_json_path = os.path.join(MAP_OUT_PATH, 'bbox_detections.json')
        mask_json_path = os.path.join(MAP_OUT_PATH, 'mask_detections.json')

        if not os.path.exists(bbox_json_path):
            print("No result files found in map_out/.")
            return

        print("\n[Evaluating BBox]")
        cocoDt = coco.loadRes(bbox_json_path)
        cocoEval = COCOeval(coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        print("\n[Evaluating Mask]")
        cocoDt_mask = coco.loadRes(mask_json_path)
        cocoEval_mask = COCOeval(coco, cocoDt_mask, 'segm')
        cocoEval_mask.evaluate()
        cocoEval_mask.accumulate()
        cocoEval_mask.summarize()


if __name__ == '__main__':
    main()