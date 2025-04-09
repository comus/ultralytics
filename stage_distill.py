# improved_distill.py - 基于结构分析的优化蒸馏脚本
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import os

# 设置环境变量以获取详细日志
os.environ["ULTRALYTICS_DEBUG"] = "1"

# 加载模型
student = YOLO("yolo11n-pose.pt")
teacher = YOLO("yolo11s-pose.pt")

LOGGER.info("基于结构分析优化的姿态蒸馏训练")

# 执行蒸馏训练
results = student.train(
    data="coco-pose.yaml",
    teacher=teacher.model,
    
    # 硬件优化设置
    batch=32,
    workers=8,
    device=0,
    amp=False,  # 禁用混合精度避免数据类型问题
    
    # 训练超参数
    epochs=30,
    optimizer="AdamW",
    weight_decay=0.0002,
    
    # 初始参数
    lr0=0.0002,
    lrf=0.05,
    
    # 输出设置
    project="structure_optimized_distillation",
    name="yolo11n_pose_optimized",
    
    # 训练设置
    val=True,
    save_period=1,
    patience=25,
    
    # 预热设置
    warmup_epochs=2,
    warmup_momentum=0.8,
    close_mosaic=0,
    
    # 蒸馏设置 - 将由PoseTrainer内部管理
    distill=0.1,  # 初始蒸馏权重
    box=0.5,      # 边界框损失权重
    pose=0.7,     # 姿态损失权重
    cls=0.5,      # 分类损失权重
    dfl=0.5,      # DFL损失权重
    kobj=0.5,     # 关键点置信度损失权重
    
    # 使用FGD蒸馏(更适合姿态估计)
    distill_type="fgd",
    
    # 特殊标记(用于传递额外参数)
    distill_layers="19,22",  # 指定蒸馏层
)

LOGGER.info("优化蒸馏训练完成！")