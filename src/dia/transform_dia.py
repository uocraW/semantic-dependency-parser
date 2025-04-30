# -*- coding: utf8 -*-
#

import json

with open('sdp_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

label_map = {'[PAD]': 0}

# 提取所有 label
for sample in data:
    for triple in sample['relationship']:
        label = triple[1]
        if label not in label_map:
            label_map[label] = len(label_map)

# 打印 label_map
print("===== label_map =====")
for label, idx in sorted(label_map.items(), key=lambda x: x[1]):
    print(f"{idx:3}: {label}")

# 可选：保存为 JSON 文件
with open('label_map_dia.json', 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)
