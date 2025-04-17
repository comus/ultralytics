from ultralytics import YOLO
import torch

# 加載學生模型
model = YOLO("distill_pose/yolo11n_distill_ultraconservative/weights/best.pt")

# 凍結BN層但不凍結參數
for m in model.model.modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        m.eval()  # 只設置為評估模式但不凍結參數

# 訓練模型（知識蒸餾）- 積極突破策略
results = model.train(
    data="coco-pose.yaml",
    teacher=YOLO("yolo11x-pose.pt").model,
    epochs=50,
    imgsz=640,
    batch=32,  # 保持小批次大小，穩定訓練
    lr0=0.0002,  # 適度提高學習率
    lrf=0.01,    # 保持lrf值
    warmup_epochs=2,  # 加入短暫預熱
    weight_decay=0.00005,  # 適度增加權重衰減
    optimizer="AdamW",
    freeze=[0, 1, 2, 3, 4, 5],  # 減少凍結層，允許更多層學習
    amp=True,
    close_mosaic=0,  # 保持關閉馬賽克
    patience=50,  # 減少耐心值
    save_period=1,
    cos_lr=True,
    cache="disk",
    save=True,
    device=0,
    workers=12,
    project="distill_pose",
    name="yolo11n_distill_breakthrough",
    exist_ok=True,
    pose=20.0,  # 降低姿態損失權重
    kobj=3.0,   # 調整關鍵點物體性損失權重
    distill=0.25,  # 顯著增加蒸餾損失權重
    
    # 溫和增加數據增強
    nbs=64,             # 標準批次大小
    val=True,           # 驗證過程
    plots=True,         # 生成訓練圖表
    label_smoothing=0.01, # 輕微標籤平滑
    mixup=0.05,         # 加入輕微混合增強
    copy_paste=0.0,     # 保持禁用複製貼上
    degrees=1.0,        # 加入輕微旋轉
    translate=0.05,     # 增加平移
    scale=0.05,         # 增加縮放
    shear=0.0,          # 保持禁用剪切
    fliplr=0.5,         # 保持左右翻轉
    mosaic=0.0,         # 保持禁用馬賽克
)
