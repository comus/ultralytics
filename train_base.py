from ultralytics import YOLO

# 載入新模型
model = YOLO("base.yaml")

# 訓練模型
results = model.train(
    data="coco-pose.yaml",
    epochs=120,                  # 適當減少但仍保持足夠訓練
    imgsz=640,                   # 保持640尺寸
    batch=64,                    # 降低批次大小
    cache="disk",
    device=0,
    workers=12,
    patience=30,
    cos_lr=True,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    close_mosaic=10,
    amp=True,
    optimizer="AdamW",
    pose=12.0,
    kobj=1.5,
    plots=True,
    save_period=1,              # 每10個epoch保存，節省磁盤空間
    project="yolo11-pose-base",
    name="train",
    exist_ok=True,
    multi_scale=False,           # 關閉多尺度訓練避免OOM風險
)
