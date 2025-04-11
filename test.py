from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo11m-pose.pt")

# 极端保守策略：完全冻结特征提取器，仅训练最后的输出层
results = model.train(
  data="coco8-pose.yaml",
  teacher=YOLO("yolo11n-pose.pt").model,
  epochs=1,              # 减少轮数，集中训练头部
  imgsz=640,              # 标准图像尺寸
  batch=64,               # 降回稍小的批量避免过度波动
  save=True,
  save_period=1,          # 每轮保存检查点
  # cache="disk",
  lr0=0.00005,            # 极低的学习率
  lrf=0.0005,             # 极低的最终学习率
  warmup_epochs=0.0,      # 无需预热
  patience=10,            # 快速早停
  optimizer="AdamW",      # 适合微调的优化器  
  weight_decay=0.0,       # 禁用权重衰减，完全保留权重
  cos_lr=True,            # 余弦调度
  freeze=22,              # 冻结整个特征提取网络，只训练最后的Pose输出层(从模型结构可知)
  augment=False,          # 关闭增强
  val=True,               # 验证
  plots=True,             # 性能图表
  # device=0,               # RTX 4090
  workers=8,              # 减少工作线程
  multi_scale=False,      # 关闭多尺度
  amp=True,               # 混合精度
  overlap_mask=True,      # 关键点重叠处理
  kobj=2.0,               # 关键点损失权重
  pose=12.0,               # 姿态损失权重
  close_mosaic=0,     # 关闭多尺度
  mosaic=0.0,
  mixup=0.0,          # Set to 0 to disable mixup augmentation
  copy_paste=0.0,     # Set to 0 to disable copy-paste augmentation
  degrees=0.0,        # Set to 0 to disable rotation
  translate=0.0,      # Set to 0 to disable translation
  scale=0.0,          # Set to 0 to disable scaling
  shear=0.0,          # Set to 0 to disable shearing
  perspective=0.0,    # Set to 0 to disable perspective warping
  flipud=0.0,         # Set to 0 to disable vertical flipping
  fliplr=0.0,         # Set to 0 to disable horizontal flipping
  hsv_h=0.0,          # Set to 0 to disable hue augmentation
  hsv_s=0.0,          # Set to 0 to disable saturation augmentation
  hsv_v=0.0,           # Set to 0 to disable value augmentation
  bgr=0.0,
  auto_augment=None,
)
