from ultralytics import YOLO


# 使用當前最佳模型作為基礎
student_model = YOLO("distill_projects/yolo11n_stage2_distill/weights/best.pt")
teacher_model = YOLO("yolo11x-pose.pt")  # 繼續使用強大的教師

student_model.train(
    data="coco-pose.yaml",
    teacher=teacher_model.model,
    distillation_loss="cwd",       # 保持相同的蒸餾方法
    distillation_layers=["16", "19", "22"],  # 專注於關鍵層
    epochs=10,                     # 再來10個epoch
    batch=32,
    workers=8,
    lr0=0.00005,                   # 更低的學習率進行微調
    lrf=0.005,                     # 更緩的學習率衰減
    optimizer="Adam",
    pure_distill=False,
    patience=15,                   # 提高耐心值
    device=0,
    project="distill_projects",
    name="yolo11n_final_push",     # 新實驗名稱
    val=True,
    save_period=1,
    plots=True,
    distill=0.1,                   # 稍微提升蒸餾權重
    pose=25.0,                     # 顯著提高pose權重
    warmup_epochs=0.5,             # 較短的熱身
    close_mosaic=5,                # 提前關閉mosaic增強
)