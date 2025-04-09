# balanced_distill.py - 平衡任务和蒸馏的训练策略
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# 加载模型
student = YOLO("yolo11n-pose.pt")
teacher = YOLO("yolo11s-pose.pt")

LOGGER.info("初始化平衡式蒸馏训练...")

# 执行平衡蒸馏训练
results = student.train(
    data="coco-pose.yaml",
    teacher=teacher.model,
    
    # 硬件优化设置
    batch=48,  # 减小批次大小，提高稳定性
    workers=8,
    device=0,
    
    # 训练超参数
    epochs=25,  # 延长训练时间
    optimizer="AdamW",
    weight_decay=0.0002,
    
    # 初始参数
    lr0=0.0002,  # 非常小的初始学习率
    lrf=0.05,    # 更低的最终学习率
    
    # 初始任务损失权重
    box=0.5,     # 保持边界框损失
    pose=0.7,    # 强调姿态损失
    kobj=0.5,    # 保持关键点置信度损失
    cls=0.5,     # 保持分类损失
    dfl=0.5,     # 保持分布焦点损失
    distill=0.1, # 轻微蒸馏
    
    # 数据处理
    imgsz=640,
    cache="disk",
    
    # 输出设置
    project="balanced_distillation",
    name="yolo11n_pose_balanced",
    
    # 训练设置
    val=True,
    save_period=1,
    patience=25,
    
    # 预热设置
    warmup_epochs=2,
    warmup_momentum=0.8,
    
    # 关闭Mosaic增强
    close_mosaic=0,

    amp=False
)

LOGGER.info("平衡蒸馏训练完成！")