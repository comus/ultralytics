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

# teacher_model = YOLO("yolo11x-pose.pt")
student_model = YOLO("runs/pose/train12/weights/best.pt")  # 使用預訓練模型權重

student_model.train(
    data="coco-pose.yaml",
    # teacher=teacher_model.model,
    # distillation_loss="cwd",
    # distillation_layers=["6", "8", "13", "16", "19", "22"],  # 使用所有關鍵層
    epochs=40,                            
    batch=32,
    workers=8,
    lr0=0.0003,                           
    lrf=0.01,
    optimizer="Adam",
    cos_lr=True,
    warmup_epochs=1.0,                    
    # pure_distill=False,
    patience=10,
    device=0,
    # project="distill_projects",
    # name="yolo11n_full_spectrum_distill",  # 更新實驗名稱
    val=True,
    save_period=1,
    plots=True,
    close_mosaic=0,
    augment=False,
    # distill=0.08,                         
    fraction=1.0,                         
    amp=False,
    pose=18.0, 
)