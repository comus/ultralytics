from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train58/weights/best.pt")

# Validate the model
metrics = model.val(
  data="coco-pose.yaml",
)
