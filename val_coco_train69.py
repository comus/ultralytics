from ultralytics import YOLO

model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train14/weights/best.pt")

# Validate the model
metrics = model.val(
  data="coco-pose.yaml",
)
