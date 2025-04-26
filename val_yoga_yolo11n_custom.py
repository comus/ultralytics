from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/yolo11n-pose-custom/root/autodl-tmp/ultralytics/runs/pose/train52/weights/best.pt")

# Validate the model
metrics = model.val(
  data="yoga82.yaml",
)
