from ultralytics import YOLO

# 第一階段: 結構和特徵學習
model = YOLO("yolo11n-pose.pt")
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11l-pose.pt").model,  # 使用L模型作為教師
    epochs=8,               # 中等輪數
    imgsz=640,
    batch=32,               # 適中批次大小
    save=True,
    save_period=1,
    lr0=0.001,              # 較高起始學習率
    lrf=0.01,               # 較緩和的衰減
    optimizer="AdamW",      
    weight_decay=0.0005,
    warmup_epochs=1.0,      # 預熱階段
    freeze=10,              # 凍結前10層
    augment=True,
    mosaic=0.0,             # 關閉mosaic增強
    degrees=5.0,            # 輕微旋轉
    translate=0.1,          # 適度平移
    val=True,
    device=0,
    workers=8,
    amp=True,               # 混合精度訓練
    kobj=1.2,               # 關鍵點目標損失權重
    pose=12.0,              # 適當姿態權重
    box=0.5,                # 降低邊界框權重
    close_mosaic=0          # 整個訓練過程中關閉mosaic
)