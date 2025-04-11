from ultralytics import YOLO

# 載入新模型
model = YOLO("lite.yaml")

# 訓練模型
results = model.train(
    data="coco-pose.yaml",
    epochs=120,                  # 保持120個epochs
    imgsz=640,                   # 保持640尺寸
    batch=64,                   # 增加批次大小至128
    cache="disk",                # 保持磁盤緩存
    device=0,
    workers=16,                  # 增加工作線程數量
    patience=30,
    cos_lr=True,
    lr0=0.02,
    lrf=0.01,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    close_mosaic=10,
    amp=True,
    optimizer="AdamW",
    pose=12.0,
    kobj=1.5,
    plots=True,
    save_period=1,              # 每10個epoch保存一次
    project="yolo11-pose-lite",
    name="train",
    exist_ok=True,
    multi_scale=True,            # 重新啟用多尺度訓練

    teacher=None,
    distill=1.0,
    freezeAllBN=False,
)