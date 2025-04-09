from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo11n-pose.pt")

# 优化的平衡配置：保持速度的同时最大化精度
results = model.train(
  data="coco-pose.yaml",
  epochs=80,              # 略微增加训练轮数
  imgsz=640,              # 标准图像尺寸
  batch=64,               # 增加批量大小充分利用GPU内存(原为32)
  save=True,
  save_period=1,          # 每轮保存检查点
  cache="disk",
  lr0=0.00025,            # 更精细的学习率
  lrf=0.008,              # 更精细的衰减率
  warmup_epochs=0.5,      # 预训练模型的最小预热
  patience=20,            # 寻找最佳权重的耐心
  optimizer="AdamW",      # 最适合微调的优化器
  weight_decay=0.00004,   # 略微调整权重衰减
  cos_lr=True,            # 余弦LR调度器实现平滑过渡
  freeze=[0, 1, 2, 3],    # 冻结前4层而非5层，允许更多层学习
  augment=True,           # 启用增强以提高泛化能力
  val=True,               # 确保运行验证以监控进度
  plots=True,             # 生成图表跟踪性能
  device=0,               # 使用RTX 4090
  workers=12,             # 增加工作线程，利用16核CPU
  multi_scale=True,       # 启用多尺度训练提高鲁棒性
  amp=True,               # 使用混合精度加速
  overlap_mask=True,      # 确保姿态关键点重叠处理正确
  kobj=2.0,               # 增强关键点目标性损失权重
  pose=12.0               # 增强姿态损失权重
)

# 选择二: 最高精度优先 (去除AMP并进一步优化精度参数)
"""
results = model.train(
  data="coco-pose.yaml",
  epochs=100,             # 增加训练轮数
  imgsz=640,              # 标准图像尺寸
  batch=24,               # 略小的批量大小
  save=True,
  save_period=1,          # 每轮保存检查点
  cache="disk",
  lr0=0.0002,             # 更保守的学习率
  lrf=0.005,              # 更平缓的衰减
  warmup_epochs=1.0,      # 稍长的预热
  patience=25,            # 更多的耐心
  optimizer="AdamW",      # 适合微调的优化器
  weight_decay=0.00002,   # 更小的权重衰减
  cos_lr=True,            # 余弦LR调度器
  freeze=[0, 1, 2, 3, 4], # 冻结早期层
  augment=True,           # 启用增强
  val=True,               # 运行验证
  plots=True,             # 生成性能图表
  device=0,               # 使用RTX 4090
  workers=8,              # 优化数据加载
  multi_scale=True,       # 多尺度训练
  amp=False,              # 禁用AMP以获得最大精度
  overlap_mask=True,      # 确保姿态关键点重叠处理正确
  kobj=2.0,               # 关键点目标性损失权重
  pose=12.0               # 姿态损失权重
)
"""
