from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

# Validate the model
metrics = model.val(
  data="yoga82.yaml",
)
