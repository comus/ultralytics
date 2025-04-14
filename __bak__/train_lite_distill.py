from ultralytics import YOLO

# 載入新模型
# model = YOLO("lite.yaml")
model = YOLO("yolo11-pose-lite/train-distill-improve/weights/epoch16.pt")

# 訓練模型
results = model.train(
    data="coco-pose.yaml",
    epochs=60,                  # 保持120個epochs
    imgsz=640,                   # 保持640尺寸
    batch=64,                   # 增加批次大小至128
    cache="disk",                # 保持磁盤緩存
    device=0,
    workers=16,                  # 增加工作線程數量
    patience=30,
    cos_lr=True,
    lr0=0.008,
    lrf=0.01,
    warmup_epochs=3.0,
    weight_decay=0.0005,
    close_mosaic=10,
    amp=True,
    optimizer="AdamW",
    plots=True,
    save_period=1,              # 每10個epoch保存一次
    project="yolo11-pose-lite",
    name="train-distill-improve",
    exist_ok=True,
    multi_scale=True,            # 重新啟用多尺度訓練


    kobj=0.0,               # 关键点损失权重
    pose=0.0,              # 姿态损失权重
    box=0.0,               # 框损失权重
    cls=0.0,               # 类别损失权重
    dfl=0.0,               # 目标置信度损失权重


    teacher=YOLO("yolo11m-pose.pt").model,
    distill=8.0,
    freezeAllBN=False,

    resume=True,
)