import os
import sys

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ultralytics import YOLO

# Initialize a new model from yaml configuration without pretrained weights
model = YOLO("gde_pose.yaml")

# Train the model with specified parameters
results = model.train(
    data="coco-pose.yaml",
    epochs=100,
    imgsz=640,
    batch=256,
    save_period=1,
    cache="disk",
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    device="0,1"
)
