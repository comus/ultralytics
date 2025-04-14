from ultralytics import YOLO

# 載入新模型
model = YOLO("yolo11n-pose.yaml")

# 訓練模型
results = model.train(
    data="coco-pose.yaml",
    epochs=500,                 # 您要求的500個epochs
    imgsz=640,                  # 您要求的640 image size
    batch=128,                  # 增加到128充分利用RTX 4090顯存
    cache="disk",               # 使用硬碟緩存加速訓練
    device=0,                   # 使用RTX 4090
    workers=12,                 # 增加工作線程數量與CPU核心數匹配
    patience=50,                # 設置早停條件，避免過度擬合
    cos_lr=True,                # 使用余弦學習率調度器
    lr0=0.015,                   # 較高的初始學習率，因為從頭訓練
    lrf=0.005,                   # 最終學習率為初始的1%
    warmup_epochs=5.0,          # 前3個epoch進行warmup
    weight_decay=0.0005,        # 標準的權重衰減
    close_mosaic=20,            # 最後10個epoch關閉mosaic增強
    amp=True,                   # 啟用自動混合精度加速訓練
    optimizer="AdamW",          # 使用AdamW優化器
    plots=True,                 # 生成訓練過程圖表
    save_period=1,              # 每個epoch保存一次模型
    project="yolo11n-pose-pure-distill",     # 項目名稱
    name="train",               # 訓練運行名稱
    exist_ok=True,              # 允許覆蓋現有目錄

    box=0, # (float) box loss gain
    cls=0, # (float) cls loss gain (scale with pixels)
    dfl=0, # (float) dfl loss gain
    pose=0, # (float) pose loss gain
    kobj=0, # (float) keypoint obj loss gain

    # 使用蒸餾，從更大的模型學習
    teacher=YOLO("yolo11m-pose.pt").model,  # 使用YOLO 11x作為教師模型
    distill=8,
    pure_distill=True,
)
