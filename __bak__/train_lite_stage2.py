from ultralytics import YOLO

# 載入最佳模型
model = YOLO("yolo11-pose-lite/train/weights/best.pt")

# 優化配置：完全解凍所有層，適度學習率
results = model.train(
    data="coco-pose.yaml",
    epochs=10,                # 較長訓練週期以充分優化
    imgsz=640,
    batch=64,
    save=True,
    cache="disk",
    
    # 正確設置學習率
    lr0=0.001,                # 第二次訓練使用較小的初始學習率
    lrf=0.1,                  # 最終學習率為初始值的10%，比標準衰減溫和些
    optimizer="AdamW",
    weight_decay=0.0005,
    cos_lr=True,
    freeze=0,                 # 完全解凍所有層
    
    # 數據增強部分
    multi_scale=True,
    scale=0.3,
    mosaic=0.5,
    degrees=10.0,
    translate=0.15,
    fliplr=0.5,
    hsv_h=0.02,
    hsv_s=0.15,
    hsv_v=0.15,
    
    # 損失權重
    kobj=2.5,
    pose=14.0,
    
    # 訓練穩定性參數
    nbs=64,
    warmup_epochs=1.0,
    warmup_momentum=0.8,
    patience=15,
    save_period=1,
    
    # 環境設置
    device=0,
    workers=16,
    amp=True,
    overlap_mask=True,
    
    # 項目設置
    project="yolo11-pose-lite",
    name="train-stage2",
    exist_ok=True,
    
    # BN層設置
    teacher=None,
    distill=1.0,
    freezeAllBN=True,
)