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

# 訓練模型（知識蒸餾）- 最終保守策略
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11s-pose.pt").model,
    epochs=100,
    imgsz=640,
    batch=128,  # 保持小批次大小，穩定訓練
    lr0=0.0005,  # 極度降低學習率
    lrf=0.01,    # 設置更高的lrf以便更快達到較低學習率
    warmup_epochs=0,  # 取消預熱
    weight_decay=0.00001,  # 進一步減少權重衰減
    optimizer="SGD",
    freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8],  # 凍結更多層，只訓練最上層
    amp=True,
    close_mosaic=0,  # 完全關閉馬賽克
    patience=100,  # 保持高耐心值
    save_period=1,
    cos_lr=True,
    cache="disk",
    save=True,
    device=0,
    workers=12,
    project="distill_pose",
    name="yolo11n_distill_ultraconservative",
    exist_ok=True,
    
    # 極度減少數據增強
    nbs=64,             # 標準批次大小
    val=True,           # 驗證過程
    plots=True,         # 生成訓練圖表
    label_smoothing=0.0, # 取消標籤平滑
    mixup=0.0,          # 禁用混合增強
    copy_paste=0.0,     # 禁用複製貼上
    degrees=0.0,        # 禁用旋轉
    translate=0.03,     # 進一步減少平移
    scale=0.03,         # 進一步減少縮放
    shear=0.0,          # 禁用剪切
    fliplr=0.5,         # 保持左右翻轉，這對人體姿態有益
    mosaic=0.0,         # 完全禁用馬賽克


    box=0.0, # (float) box loss gain
    cls=0.0, # (float) cls loss gain (scale with pixels)
    dfl=0.0, # (float) dfl loss gain
    pose=0.0, # (float) pose loss gain
    kobj=0.0, # (float) keypoint obj loss gain
    distill=1.0,  # 極度降低蒸餾損失權重，幾乎只作為輔助
)
