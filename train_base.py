from ultralytics import YOLO

# 載入新模型
model = YOLO("base.yaml")  # 從YAML配置創建全新模型

# 訓練模型
results = model.train(
    data="coco-pose.yaml",
    epochs=150,                   # 從300減至150，足夠基礎學習
    imgsz=640,
    batch=128,
    cache="disk",                 # 磁盤緩存沒問題
    device=0,
    workers=12,
    patience=30,                  # 從50減至30，配合較少的epochs
    cos_lr=True,
    lr0=0.001,
    lrf=0.01,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    close_mosaic=10,              # 從15減至10，以適應總epoch的減少
    amp=True,
    optimizer="AdamW",
    pose=12.0,
    kobj=1.5,
    plots=True,
    save_period=1,               # 從25減至15，確保有足夠的檢查點
    project="yolo11-pose-base",
    name="train",
    exist_ok=True,
    multi_scale=True,
)