from ultralytics import YOLO

# 載入新模型
model = YOLO("yolo11n-pose.pt")

# 訓練模型 - 快速60 epochs
results = model.train(
    data="coco8-pose.yaml",
    epochs=60,                  # 快速訓練60個epochs
    imgsz=640,
    batch=64,                   # 適中的batch size
    cache="disk",
    device=0,
    workers=12,
    patience=20,                # 減少patience以更快停止非改進訓練
    cos_lr=True,
    # lr0=0.02,                   # 較高的初始學習率
    # lrf=0.01,
    warmup_epochs=5.0,
    weight_decay=0.0005,
    close_mosaic=15,            # 在後15個epoch關閉mosaic
    amp=True,
    optimizer="AdamW",
    plots=True,
    save_period=5,              # 每5個epochs保存一次
    project="yolo11n-pose-pure-distill",
    name="train",      # 新名稱以區分此次快速訓練
    exist_ok=True,
    val=True,                   # 啟用驗證
    
    # 損失權重設定為0
    box=0,
    cls=0,
    dfl=0,
    pose=0,
    kobj=0,
    
    # 蒸餾設定
    teacher=YOLO("yolo11m-pose.pt").model,
    distill=12,                 # 提高蒸餾損失權重
    loss_function="pose_loss3",
    
    fraction=0.25,
)
