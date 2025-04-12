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
    lr0=0.0002,            # 极低的学习率
    lrf=0.001,             # 极低的最终学习率
    optimizer="AdamW",      # 适合微调的优化器  
    weight_decay=0.0003,    # 輕微權重衰減，配合AdamW使用
    cos_lr=True,            # 余弦调度
    freeze=6,              # 冻结整个特征提取网络，只训练最后的Pose输出层(从模型结构可知)
    augment=False,          # 关闭增强
    val=True,               # 验证
    plots=True,             # 性能图表
    device=0,               # RTX 4090
    workers=8,              # 减少工作线程
    amp=True,               # 混合精度
    overlap_mask=True,      # 关键点重叠处理

    # 輕微的數據增強
    multi_scale=True,
    scale=0.2,       # 從0.1增加到0.2
    mosaic=0.3,      # 適度添加mosaic增強(0.0-1.0)
    degrees=8.0,     # 從5度增加到8度
    translate=0.1,   # 從0.05增加到0.1
    fliplr=0.5,      # 添加水平翻轉50%概率
    hsv_h=0.015,     # 輕微色調變化
    hsv_s=0.1,       # 輕微飽和度變化
    hsv_v=0.1,       # 輕微亮度變化

    kobj=2.5,     # 從2.0略增
    pose=14.0,    # 從12.0增加


    nbs=64,              # 標準批量大小

    warmup_epochs=0.5,   # 半個epoch預熱
    warmup_momentum=0.8, # 預熱動量

    patience=10,         # 增加早停耐心值，避免過早停止
    save_period=1,       # 每個epoch保存


    project="yolo11-pose-lite",
    name="train-stage2",
    exist_ok=True,

    teacher=None,
    distill=1.0,
    freezeAllBN=True,
)