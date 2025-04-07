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
    distillation_loss="cwd",              # 對不同通道數模型的蒸餾效果更好
    distillation_layers=["22"], # P3, P4, P5特徵層
    epochs=10,                            # 5萬+張數據，10個epoch已足夠
    imgsz=640,
    batch=64,                             # 大批次提高訓練效率
    workers=12,                           # 充分利用多核CPU
    lr0=0.0001,                            # 學習率設置適中，考慮到數據集較大
    lrf=0.001,                             # 學習率衰減到初始的1%
    optimizer="Adam",                     # Adam優化器適合蒸餾任務
    cos_lr=True,                          # 使用餘弦學習率調度
    warmup_epochs=0.5,                    # 短暫預熱即可
    pure_distill=True,                    # 啟用純蒸餾模式
    # patience=3,                           # 早停參數，節省時間
    device=0,
    project="distill_projects",
    name="yolo11l_to_yolo11n_pure",
    val=True,                             # 啟用驗證
    save_period=2,                        # 每2個epoch保存一次
    # plots=True,                           # 生成訓練圖表
    close_mosaic=0,                       # 禁用mosaic增強，蒸餾不需要太多增強
    augment=False,                         # 關閉增強，保持特徵對齊
    distill=0.01,
    fraction=0.1,
    amp=False,
)
