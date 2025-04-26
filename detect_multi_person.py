from ultralytics import YOLO
import os
import glob
from tqdm import tqdm
import cv2
import torch

# 加載預訓練的 YOLOv11m-pose 模型
model = YOLO('yolo11m-pose.pt')

# 設置檢測的置信度閾值
conf_threshold = 0.25

# Yoga82訓練集路徑
dataset_path = "datasets/yoga82/images/val"

# 輸出結果文件
output_file = "multi_person_images.txt"
summary_file = "multi_person_summary.txt"

# 獲取所有圖片文件
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(dataset_path, ext)))

print(f"找到 {len(image_files)} 張圖片待分析")

# 分析結果統計
total_images = len(image_files)
multi_person_images = []
person_count_dict = {}  # 用於儲存每張圖片中的人數

# 檢測每張圖片中的人數
with open(output_file, 'w') as f:
    f.write("# 多人照片列表 (文件名, 人數)\n\n")
    
    for img_path in tqdm(image_files, desc="分析圖片"):
        # 執行檢測
        results = model(img_path, conf=conf_threshold, verbose=False)
        
        # 獲取檢測到的人數 (class 0 是 'person')
        boxes = results[0].boxes
        persons = len([b for b in boxes if int(b.cls) == 0])
        
        # 更新統計信息
        if persons > 1:
            multi_person_images.append((img_path, persons))
            f.write(f"{img_path}, {persons}\n")
        
        # 更新人數統計
        if persons not in person_count_dict:
            person_count_dict[persons] = 0
        person_count_dict[persons] += 1

# 寫入摘要信息
with open(summary_file, 'w') as f:
    f.write("# Yoga82 數據集多人檢測摘要\n\n")
    f.write(f"分析圖片總數: {total_images}\n")
    f.write(f"多人照片數量: {len(multi_person_images)} ({len(multi_person_images)/total_images*100:.2f}%)\n\n")
    
    f.write("# 人數分佈\n")
    for persons, count in sorted(person_count_dict.items()):
        f.write(f"{persons} 人的照片: {count} 張 ({count/total_images*100:.2f}%)\n")
    
    f.write("\n# 前10張多人照片示例:\n")
    for i, (img_path, persons) in enumerate(multi_person_images[:10]):
        f.write(f"{i+1}. {img_path} - {persons} 人\n")

print(f"分析完成！")
print(f"總圖片數: {total_images}")
print(f"多人照片數: {len(multi_person_images)} ({len(multi_person_images)/total_images*100:.2f}%)")
print(f"詳細結果已保存至 {output_file}")
print(f"摘要信息已保存至 {summary_file}") 