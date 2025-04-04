from ultralytics import YOLO

# 載入模型
model = YOLO("micropose.yaml")

# 訓練模型 - 修正epochs
results = model.train(
    data="coco-pose.yaml",
    epochs=300,                    # 從100增加到300輪
    patience=25,                   # 提高早停容忍度
    batch=84,
    imgsz=640,
    cache='disk',
    workers=8,
    optimizer='AdamW',
    lr0=0.001,                     # 稍微提高初始學習率
    lrf=0.005,                     # 調整最終學習率係數
    momentum=0.937,
    weight_decay=0.0003,
    warmup_epochs=10,              # 延長預熱期
    cos_lr=True,
    close_mosaic=15,               # 最後15輪關閉馬賽克
    multi_scale=True,
    amp=True,
    val=True,
    dropout=0.1,
    pretrained=False,
    device=0,
    project="micropose",
    name="training_run",
    pose=15.0,
    kobj=2.0,
    save_period=1,
)