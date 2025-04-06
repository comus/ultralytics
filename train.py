from ultralytics import YOLO

teacher_model = YOLO("yolo11l-pose.pt")
student_model = YOLO("yolo11n-pose.pt")

student_model.train(
  data="coco-pose.yaml",
  teacher=teacher_model.model,
  distillation_loss="cwd",
  distillation_layers=["16", "19", "22"],
  epochs=20,
  imgsz=640,
  pure_distill=True,
  distill=0.1,
  workers=2,
)
