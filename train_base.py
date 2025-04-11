from ultralytics import YOLO

# 載入新模型
model = YOLO("base.yaml")  # 從YAML配置創建全新模型

# 訓練模型，不凍結任何層
results = model.train(
    data="coco-pose.yaml",
    epochs=300,
    imgsz=640,
    batch=128,
    cache=True,
    device=0,
    workers=12,
    patience=50,
    cos_lr=True,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    close_mosaic=15,
    amp=True,
    optimizer="AdamW",
    pose=12.0,
    kobj=1.5,
    plots=True,
    save_period=25,
    project="yolo11-pose-base",
    name="train",
    exist_ok=True,
    multi_scale=True,
    # 無需設置freeze參數
)
