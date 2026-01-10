from collections import Counter

txt_path = r"D:\yz\1_yz\deeplabv3-plus-pytorch-main\VOCdevkit\VOC2007\ImageSets\Segmentation\trainval.txt"

# 1. 读取并统计
with open(txt_path, 'r', encoding='utf-8') as f:
    c = Counter(line.strip() for line in f if line.strip())

# 2. 过滤出重复
dup = {name: cnt for name, cnt in c.items() if cnt > 1}

# 3. 打印结果
if dup:
    for name, cnt in dup.items():
        print(f"{name} 重复 {cnt} 次")
    print(f"\n总计重复 {sum(dup.values()) - len(dup)} 条记录")
else:
    print("没有发现重复文件名！")