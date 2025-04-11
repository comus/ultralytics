from ultralytics import YOLO

# 使用更穩定的訓練設置
model = YOLO("yolo11n-pose.pt")
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11l-pose.pt").model,
    epochs=8,
    imgsz=640,
    batch=16,               # 降低批次大小
    save=True,
    save_period=1,
    lr0=0.0005,             # 降低學習率
    lrf=0.01,
    optimizer="SGD",        # 改用SGD優化器，更穩定
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=2.0,      # 延長預熱
    freeze=12,              # 凍結更多層
    augment=True,
    mosaic=0.0,
    degrees=0.0,            # 禁用旋轉
    translate=0.1,
    scale=0.5,
    val=True,
    device=0,
    workers=8,
    amp=False,              # 關閉混合精度，避免數值問題
    kobj=1.0,               # 降低權重
    pose=10.0,              # 降低權重
    box=0.5
)
