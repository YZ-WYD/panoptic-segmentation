import json
gt = json.load(open(r'D:\yz\1_yz\mask-rcnn-pytorch-master\datasets\coco\Jsons\instances_val2017.json'))
print('GT image_id 示例', sorted({img['id'] for img in gt['images']})[:10])
dt = json.load(open(r'D:\yz\1_yz\mask-rcnn-pytorch-master\map_out\bbox_detections.json'))
print('DT image_id 示例', sorted({d['image_id'] for d in dt})[:10])
print(next(iter(dt))['bbox'])      # 应该类似 [412.3, 183.7, 65.4, 88.1] 浮点