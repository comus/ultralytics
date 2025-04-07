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
    distillation_loss="mgd",              # 對不同通道數模型的蒸餾效果更好
    distillation_layers=["22"], # P3, P4, P5特徵層全部使用
    epochs=10,                            # 5萬+張數據，10個epoch已足夠
    imgsz=640,
    # batch=64,                             # 大批次提高訓練效率
    # workers=12,                           # 充分利用多核CPU
    lr0=0.0001,                            # 提高學習率以加速收斂
    lrf=0.01,                             # 學習率衰減到初始的1%
    optimizer="Adam",                     # Adam優化器適合蒸餾任務
    cos_lr=True,                          # 使用餘弦學習率調度
    warmup_epochs=0.5,                    # 短暫預熱即可
    pure_distill=False,                   # 禁用純蒸餾模式，同時保留任務學習
    # patience=3,                           # 啟用早停參數，避免過度訓練
    device=0,
    project="distill_projects",
    name="yolo11l_to_yolo11n_balanced",   # 新的實驗名稱
    val=True,                             # 啟用驗證
    save_period=2,                        # 每2個epoch保存一次
    # plots=True,                           # 生成訓練圖表，方便分析
    close_mosaic=0,                       # 禁用mosaic增強，蒸餾不需要太多增強
    augment=False,                        # 關閉增強，保持特徵對齊
    distill=0.005,                          # 提高蒸餾權重，增強知識傳遞
    fraction=0.1,                         # 增加訓練數據比例
    amp=False,                            # 關閉混合精度訓練以提高精度
)
