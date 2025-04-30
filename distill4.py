# distill_fgd.py
from ultralytics import YOLO

# 載入最佳模型作為起點
student_model = YOLO("distill_projects/yolo11n_final_push/weights/best.pt")
teacher_model = YOLO("yolo11x-pose.pt")

# FGD蒸餾突破訓練
student_model.train(
    data="coco-pose.yaml",
    teacher=teacher_model.model,
    distillation_loss="fgd",        # 關鍵改動：使用FGD方法
    distillation_layers=["13", "16", "19", "22"],  # 增加一層中層特徵
    epochs=15,                      # 延長訓練
    batch=32,
    workers=8,
    lr0=0.00006,                    # 較低學習率以精細調整
    lrf=0.002,                      # 更緩慢的衰減
    optimizer="Adam",
    cos_lr=True,                    # 使用餘弦調度
    pure_distill=False,
    patience=20,
    device=0,
    project="distill_projects",
    name="yolo11n_fgd_breakthrough", # 新實驗名稱
    val=True,
    save_period=1,
    plots=True,
    distill=0.2,                    # 增強蒸餾權重
    pose=30.0,                      # 增強姿態權重
    warmup_epochs=0.0,              # 無需熱身（從良好基礎開始）
    close_mosaic=0,                 # 從開始就關閉mosaic
    amp=False,                      # 禁用混合精度以確保精確度
) 