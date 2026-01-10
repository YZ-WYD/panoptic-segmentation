import json
import numpy as np

# 1. 加载刚才生成的检测结果
dt = json.load(open(r'D:\yz\1_yz\mask-rcnn-pytorch-master\map_out\bbox_detections.json'))

# 2. 基本统计
print('总框数:', len(dt))
if not dt:
    print('⚠️  一条检测都没有！')
    exit()

# 3. 抽查前 10 条
bad = 0
for i, d in enumerate(dt[:10]):
    x, y, w, h = d['bbox']
    if w <= 0 or h <= 0:
        bad += 1
        print(f'第{i}条 负宽高: {d["bbox"]}')
    if not isinstance(x, float):
        print(f'第{i}条 非 float: {d["bbox"]}')

if bad == 0:
    print('✅ 前 10 条 bbox 格式正常（无负宽高，均为 float）')
else:
    print(f'❌ 前 10 条里有 {bad} 条负宽高，需要修复 prep_metrics！')