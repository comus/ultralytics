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

# 訓練模型（知識蒸餾）- 高效能策略
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11s-pose.pt").model,
    epochs=150,
    imgsz=640,
    batch=32,  # 大幅降低批次大小，提高性能
    lr0=0.0001,  # 保持低學習率
    lrf=0.01,    # 設置更高的lrf以便更快達到較低學習率
    warmup_epochs=0,  # 取消預熱
    weight_decay=0.0001,  # 適度增加權重衰減防止過擬合
    optimizer="Adam",  # 使用Adam優化器，更適合小批次
    freeze=[0, 1, 2, 3, 4, 5, 6, 7],  # 凍結更多層，只訓練最上層
    amp=True,
    close_mosaic=0,  # 完全關閉馬賽克
    patience=50,  # 減少耐心值加快收斂
    save_period=5,  # 減少保存頻率減少IO負擔
    cos_lr=True,
    cache=False,  # 關閉緩存減少記憶體使用
    save=True,
    device=0,
    workers=8,  # 減少工作線程降低系統負載
    project="distill_pose",
    name="yolo11n_distill_efficient",  # 更新名稱
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

    # 損失權重調整
    box=3.0,   # box loss gain
    cls=0.5,   # cls loss gain
    dfl=0.5,   # dfl loss gain
    pose=2.0,  # 大幅降低pose loss權重，防止數值過大
    kobj=1.0,  # 適度降低keypoint obj loss權重
)
