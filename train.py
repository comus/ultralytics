from ultralytics import YOLO

# 加載學生模型
model = YOLO("yolo11n-pose.pt")

# 訓練模型（知識蒸餾）
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11x-pose.pt").model,
    epochs=100,
    imgsz=640,
    batch=8,
    lr0=0.0005,
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
    workers=8,
    project="distill_pose",
    name="yolo11n_distill",
    exist_ok=True,
    pose=12.0,
    kobj=2.0,
    distill=1.0,  # 蒸餾損失權重
)
