import json
import sys

path = r'D:\yz\1_yz\mask-rcnn-pytorch-master\map_out\bbox_detections.json'
data = json.load(open(path, encoding='utf-8'))
bad = 0
for d in data:
    x, y, w, h = d['bbox']
    if w < 0 or h < 0:          # 发现负宽高
        bad += 1
        d['bbox'][2] = abs(w)
        d['bbox'][3] = abs(h)

if bad:
    print(f'修复了 {bad} 条负宽高')
    json.dump(data, open(path, 'w'), ensure_ascii=False, indent=None)
else:
    print('未发现负宽高，文件保持原样')