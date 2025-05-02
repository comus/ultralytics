import os
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import sys
import torch.nn as nn
import torch.distributed as dist
import warnings

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 設置環境變量，確保分佈式訓練使用本地代碼
os.environ["PYTHONPATH"] = f"{current_dir}:{os.environ.get('PYTHONPATH', '')}"
# 忽略 DDP 的 stride 不匹配警告
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
# 忽略除零警告
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

def main():
    # Load a model
    model = YOLO("yolo11n-pose.pt")

    # Train the model
    results = model.train(
        data="coco8-pose.yaml",
        epochs=20,
        imgsz=640,
        # device=[0, 1],

        teacher="yolo11n-pose.pt",
        target_layers=["model.0.conv", 1],

        # freezeAllBN=True,
        # freeze=23,
    )

if __name__ == "__main__":
    main()