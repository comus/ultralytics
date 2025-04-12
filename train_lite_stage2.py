from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo11-pose-lite/train/weights/best.pt")

# 极端保守策略：完全冻结特征提取器，仅训练最后的输出层
results = model.train(
    data="coco-pose.yaml",
    epochs=10,              # 减少轮数，集中训练头部
    imgsz=640,              # 标准图像尺寸
    batch=64,               # 降回稍小的批量避免过度波动
    save=True,
    cache="disk",
    lr0=0.0001,            # 极低的学习率
    lrf=0.001,             # 极低的最终学习率
    warmup_epochs=0.0,      # 无需预热
    optimizer="AdamW",      # 适合微调的优化器  
    weight_decay=0.0,       # 禁用权重衰减，完全保留权重
    cos_lr=True,            # 余弦调度
    freeze=10,              # 冻结整个特征提取网络，只训练最后的Pose输出层(从模型结构可知)
    augment=False,          # 关闭增强
    val=True,               # 验证
    plots=True,             # 性能图表
    device=0,               # RTX 4090
    workers=8,              # 减少工作线程
    amp=True,               # 混合精度
    overlap_mask=True,      # 关键点重叠处理

    # 輕微的數據增強
    multi_scale=True,   # 啟用多尺度訓練，但範圍較小
    scale=0.1,          # 控制尺度變化範圍
    degrees=5.0,        # 輕微旋轉
    translate=0.05,     # 輕微平移

    kobj=2.0,               # 关键点损失权重
    pose=12.0,              # 姿态损失权重

    nbs=64,              # 標準批量大小
    accumulate=2,        # 梯度累積次數

    warmup_epochs=0.5,   # 半個epoch預熱
    warmup_momentum=0.8, # 預熱動量

    patience=20,         # 增加早停耐心值，避免過早停止
    save_period=1,       # 每個epoch保存


    project="yolo11-pose-lite",
    name="train-stage2",
    exist_ok=True,

    teacher=None,
    distill=1.0,
    freezeAllBN=True,
)