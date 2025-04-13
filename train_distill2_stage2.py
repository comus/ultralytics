from ultralytics import YOLO

model = YOLO("yolo11n-pose-distill2/train/weights/last.pt")

# 第二階段訓練參數
results = model.train(
    data="coco-pose.yaml",
    epochs=300,                     # 可以減少總輪數
    patience=30,                    # 降低早停耐心值
    batch=32,                       # 增加批次大小
    cos_lr=True,
    lr0=0.001,                      # 降低初始學習率
    lrf=0.0001,                     # 更低的最終學習率
    warmup_epochs=2,                # 減少預熱階段
    save_period=1,
    cache="disk",                     # RAM快取
    close_mosaic=20,                # 提前關閉mosaic
    plots=True,
    mosaic=0.7,                     # 減少擾動
    mixup=0.03,
    copy_paste=0.03,
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
    translate=0.07,
    scale=0.4,
    fliplr=0.5,
    multi_scale=True,
    project="yolo11n-pose-distill2",  # 新專案名稱
    name="train_stage2",
    exist_ok=True,

    # 損失權重微調
    box=5.0,
    cls=0.5,
    dfl=1.5,
    pose=25.0,  # 更強調姿態
    kobj=2.0,

    # 蒸餾設置
    teacher=YOLO("yolo11m-pose.pt").model,
    distill=3.5,                   # 增加蒸餾權重
    loss_function="pose_loss2",
    
    # 新增優化器
    optimizer='AdamW',
    
    # 防止過擬合
    dropout=0.03,
)