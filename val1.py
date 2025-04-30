from ultralytics import YOLO

model = YOLO("yolo11s-pose.pt")

metrics = model.val(
  data="coco-pose.yaml",
)