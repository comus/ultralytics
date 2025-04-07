# from ultralytics import YOLO

# teacher_model = YOLO("yolo11l-pose.pt")
# student_model = YOLO("yolo11n-pose.pt")

# student_model.train(
#   data="coco8-pose.yaml",
#   teacher=teacher_model.model,
#   distillation_loss="cwd",
#   distillation_layers=["16", "19", "22"],
#   epochs=20,
#   imgsz=640,
#   workers=4,
#   pure_distill=True,
#   distill=0.1,
# )

from ultralytics import YOLO

teacher_model = YOLO("yolo11l-pose.pt")
student_model = YOLO("yolo11n-pose.pt")  # 使用預訓練模型權重

student_model.train(
    data="coco-pose.yaml",
    teacher=teacher_model.model,
    distillation_loss="cwd",              # 對通道數不同的模型更穩定
    distillation_layers=["16", "19", "22"], # 使用多層特徵蒸餾
    epochs=5,                             # 蒸餾訓練通常收斂更快
    imgsz=640,
    batch=32,                             # RTX 4090能處理更大批次
    workers=8,                            # 利用您的16核CPU
    lr0=0.0001,                           # 保持較低學習率以穩定蒸餾
    lrf=0.01,                             # 學習率衰減到初始的1%
    optimizer="Adam",                     # Adam適合蒸餾任務
    cos_lr=True,                          # 使用餘弦學習率調度
    warmup_epochs=0.3,                    # 縮短預熱時間
    pure_distill=False,                   # 禁用純蒸餾模式，保留任務學習
    patience=3,                           # 啟用早停，避免過擬合
    device=0,
    project="distill_projects",
    name="yolo11l_to_yolo11n_final",
    val=True,                             # 啟用驗證
    save_period=1,                        # 每個epoch保存
    plots=True,                           # 生成訓練圖表便於分析
    close_mosaic=0,                       # 禁用mosaic增強
    augment=False,                        # 關閉增強，保持特徵對齊
    distill=0.01,                         # 適中的蒸餾權重
    fraction=0.3,                         # 蒸餾通常不需要全部數據
    amp=False,                            # 關閉混合精度訓練以提高精度
)