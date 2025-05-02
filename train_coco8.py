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

# 設置環境變量，確保所有GPU的日誌都顯示
os.environ["PYTHONIOENCODING"] = "utf-8"  # 確保UTF-8編碼輸出
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 確保使用指定的GPU
# 啟用分佈式訓練的調試信息
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 輸出更詳細的分佈式訓練日誌

def main():
    # Load a model
    model = YOLO("yolo11n-pose.pt")

    print(f"開始訓練模型 - 使用多GPU知識蒸餾...")

    # Train the model
    results = model.train(
        data="coco8-pose.yaml",
        epochs=20,
        imgsz=640,
        device=[0, 1],

        teacher="yolo11n-pose.pt",
        target_layers=[0, 1],  # 簡化目標層設置，確保正確找到
        distill=0.5,  # 蒸餾損失權重

        # freezeAllBN=True,
        # freeze=23,

        verbose=True,  # 啟用詳細日誌
    )

    # 顯示訓練結果
    print(f"訓練完成！最佳模型保存在: {results}")

if __name__ == "__main__":
    main()