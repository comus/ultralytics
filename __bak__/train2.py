from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo11n-pose.pt")

# 针对已训练过相同数据集的微调策略
results = model.train(
  data="coco-pose.yaml",
  epochs=50,              # 减少训练轮数，避免过拟合
  imgsz=640,              # 标准图像尺寸
  batch=128,               # 大批量提高稳定性
  save=True,
  save_period=1,          # 每轮保存检查点
  cache="disk",
  lr0=0.00008,            # 极低的学习率，避免破坏已学到的知识
  lrf=0.001,              # 极低的最终学习率
  warmup_epochs=0.0,      # 无需预热，模型已经训练过
  patience=15,            # 减少耐心，避免过度训练
  optimizer="AdamW",      # 适合微调的优化器
  weight_decay=0.00001,   # 极小的权重衰减，最小化权重变化
  cos_lr=True,            # 余弦LR调度器实现平滑过渡
  freeze=[0, 1, 2, 3, 4, 5, 6], # 冻结更多层，只微调高层特征和输出层
  augment=False,          # 关闭增强，因为模型已经见过这些数据
  val=True,               # 确保运行验证以监控进度
  plots=True,             # 生成图表跟踪性能
  device=0,               # 使用RTX 4090
  workers=12,             # 优化CPU数据加载
  multi_scale=False,      # 关闭多尺度，保持输入一致性
  amp=True,               # 使用混合精度加速
  overlap_mask=True,      # 确保姿态关键点重叠处理正确
  kobj=2.0,               # 关键点目标性损失权重
  pose=12.0,              # 姿态损失权重
)
