from ultralytics import YOLO
import torch

# 加載學生模型
model = YOLO("yolo11n-pose.pt")

# 凍結所有BN層
for m in model.model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()
        for param in m.parameters():
            param.requires_grad = False

# 訓練模型（知識蒸餾）
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11x-pose.pt").model,
    epochs=100,
    imgsz=640,
    batch=64,
    lr0=0.0005,  # 降低學習率
    lrf=0.005,
    warmup_epochs=3,  # 減少預熱輪數
    weight_decay=0.0001,
    optimizer="AdamW",
    freeze=[0, 1, 2, 3, 4, 5, 6],  # 增加凍結層
    amp=True,
    close_mosaic=30,  # 提前關閉馬賽克
    patience=50,
    save_period=1,
    cos_lr=True,
    cache="disk",
    save=True,
    device=0,
    workers=12,
    project="distill_pose",
    name="yolo11n_distill",
    exist_ok=True,
    pose=15.0,  # 提高姿態損失權重
    kobj=2.0,
    distill=0.2,  # 降低蒸餾損失權重
    
    # 保留官方支持的增強參數，但降低增強強度
    nbs=64,           # 標準批次大小
    val=True,         # 驗證過程
    plots=True,       # 生成訓練圖表
    label_smoothing=0.01, # 標籤平滑
    mixup=0.05,       # 降低混合增強概率
    copy_paste=0.05,  # 降低複製黏貼概率
    degrees=3.0,      # 降低旋轉範圍
    translate=0.07,   # 降低平移範圍
    scale=0.07,       # 降低縮放範圍
    shear=1.0,        # 降低剪切範圍
    fliplr=0.5,       # 保持左右翻轉概率
    mosaic=0.3,       # 進一步降低馬賽克概率
)
