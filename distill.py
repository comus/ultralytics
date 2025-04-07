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
    distillation_layers=["22"],           # 僅蒸餾深層特徵，減少對精確位置的干擾
    epochs=8,                             # 增加輪數以便更好調整
    imgsz=640,
    batch=32,                             # RTX 4090能處理較大批次
    workers=8,                            # 利用您的16核CPU
    lr0=0.00005,                          # 更保守的學習率，避免過度調整
    lrf=0.01,                             # 學習率衰減到初始的1%
    optimizer="Adam",                     # Adam適合蒸餾任務
    cos_lr=True,                          # 使用餘弦學習率調度
    warmup_epochs=0.3,                    # 短暫預熱
    pure_distill=False,                   # 禁用純蒸餾模式，保留任務學習
    patience=5,                           # 增加耐心值，避免過早停止
    device=0,
    project="distill_projects",
    name="yolo11l_to_yolo11n_optimized",  # 新的優化實驗名稱
    val=True,                             # 啟用驗證
    save_period=1,                        # 每個epoch保存
    plots=True,                           # 生成訓練圖表便於分析
    close_mosaic=0,                       # 禁用mosaic增強
    augment=False,                        # 關閉增強，保持特徵對齊
    distill=0.005,                        # 降低蒸餾權重，減少對原任務的干擾
    fraction=0.2,                         # 使用較少數據但增加epochs
    amp=False,                            # 關閉混合精度訓練以提高精度
    pose=15.0,                            # 增加pose損失權重，強化姿態準確性
)