from ultralytics import YOLO
import torch

# 加載學生模型
model = YOLO("yolo11n-pose.pt")

# 訓練模型（知識蒸餾）
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11x-pose.pt").model,
    epochs=100,
    imgsz=640,
    batch=64,
    lr0=0.0015,
    lrf=0.005,
    warmup_epochs=5,
    weight_decay=0.0001,
    optimizer="AdamW",
    freeze=[0, 1, 2, 3, 4],
    amp=True,
    close_mosaic=15,
    patience=50,
    save_period=1,
    cos_lr=True,
    cache="disk",
    save=True,
    device=0,
    workers=12,
    project="distill_pose",
    name="yolo11n_distill",
    exist_ok=True,
    pose=12.0,
    kobj=2.0,
    distill=0.4,  # 調整蒸餾損失權重
    
    # 保留官方支持的增強參數
    nbs=64,           # 標準批次大小
    val=True,         # 驗證過程
    plots=True,       # 生成訓練圖表
    label_smoothing=0.01, # 標籤平滑
    mixup=0.1,        # 混合增強概率
    copy_paste=0.1,   # 複製黏貼概率
    degrees=5.0,      # 旋轉範圍
    translate=0.1,    # 平移範圍
    scale=0.1,        # 縮放範圍
    shear=2.0,        # 剪切範圍
    fliplr=0.5,       # 左右翻轉概率
    mosaic=1.0,       # 馬賽克概率
)
