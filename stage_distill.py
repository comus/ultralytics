# safe_distill.py - 安全的姿态蒸馏训练脚本
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import os

# 设置环境变量以获取更详细的日志
os.environ["ULTRALYTICS_DEBUG"] = "1"

# 加载模型
student = YOLO("yolo11n-pose.pt")
teacher = YOLO("yolo11s-pose.pt")

LOGGER.info("初始化安全姿态蒸馏训练...")

# 打印模型信息以便调试
LOGGER.info(f"学生模型: {type(student.model)}")
LOGGER.info(f"教师模型: {type(teacher.model)}")

# 执行蒸馏训练
results = student.train(
    data="coco-pose.yaml",
    teacher=teacher.model,
    
    # 硬件优化设置
    batch=32,  # 减小批次大小提高稳定性
    workers=8,
    device=0,
    amp=False,  # 禁用混合精度
    
    # 训练超参数
    epochs=25,
    optimizer="AdamW",
    weight_decay=0.0002,
    
    # 初始参数
    lr0=0.0002,
    lrf=0.05,
    
    # 初始任务损失权重
    box=0.5,
    pose=0.7,
    kobj=0.5,
    cls=0.5,
    dfl=0.5,
    distill=0.1,
    
    # 数据处理
    imgsz=640,
    cache="disk",
    
    # 输出设置
    project="safe_distillation",
    name="yolo11n_pose_safe",
    
    # 训练设置
    val=True,
    save_period=1,
    patience=25,
    
    # 预热设置
    warmup_epochs=2,
    warmup_momentum=0.8,
    close_mosaic=0,
    
    # 使用FGD蒸馏(而非CWD)
    distill_type="fgd",
)

LOGGER.info("安全姿态蒸馏训练完成！")
