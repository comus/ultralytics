from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train16/weights/best.pt")

# Validate the model
metrics = model.val(
  data="yoga82.yaml",
)
