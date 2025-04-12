from ultralytics import YOLO

# 使用更穩定的訓練設置
model = YOLO("yolo11-pose-lite/train/weights/best.pt")
results = model.train(
    data="coco-pose.yaml",
    epochs=10,
    imgsz=640,
    batch=16,               # 降低批次大小
    save_period=1,
    lr0=0.0005,             # 降低學習率
    lrf=0.01,
    optimizer="SGD",        # 改用SGD優化器，更穩定
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=2.0,      # 延長預熱
    freeze=12,              # 凍結更多層
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
    box=0.5,

    project="yolo11-pose-lite",
    name="train-stage2",
    exist_ok=True,

    teacher=None,
    distill=1.0,
    freezeAllBN=True,
)
