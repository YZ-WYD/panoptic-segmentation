import json, os
from tqdm import tqdm   #  pip install tqdm

# ================== 路径自己改 ==================
src_json    = r"D:\yz\1_yz\mask-rcnn-pytorch-master\datasets\coco\Jsons\instances_train2017.json"
src_img_dir = r'D:\yz\1_yz\mask-rcnn-pytorch-master\datasets\coco\JPEGImages'
dst_json    = r'D:\yz\1_yz\mask-rcnn-pytorch-master\datasets\coco\instances_train2017.json'
# ===============================================

# 1. 磁盘上真实文件名（无后缀）
print('1️⃣ 扫描图片文件夹 …')
exist_set = {f[:-4] for f in os.listdir(src_img_dir) if f.lower().endswith('.jpg')}
print(f'   共 {len(exist_set)} 张图')

# 2. 读原始 JSON
print('2️⃣ 加载原始 JSON …')
data = json.load(open(src_json))
images_all = data['images']
anns_all   = data['annotations']

# 3. 过滤 images（带进度条）
print('3️⃣ 过滤 images …')
keep_imgs = []
for img in tqdm(images_all, desc='   images', ncols=80):
    if img['file_name'][:-4] in exist_set:
        keep_imgs.append(img)
keep_img_ids = {img['id'] for img in keep_imgs}

# 4. 过滤 annotations（带进度条）
print('4️⃣ 过滤 annotations …')
keep_anns = []
for ann in tqdm(anns_all, desc='   anns  ', ncols=80):
    if ann['image_id'] in keep_img_ids:
        keep_anns.append(ann)

# 5. 组装新 JSON
data['images'], data['annotations'] = keep_imgs, keep_anns
print('5️⃣ 写入新 JSON …')
json.dump(data, open(dst_json, 'w'), indent=2)

print(f'✅ 完成！保留 {len(keep_imgs)} 张图，{len(keep_anns)} 条标注 → {dst_json}')