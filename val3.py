from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

metrics = model.val(
  data="coco-pose.yaml",
)