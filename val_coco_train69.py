from ultralytics import YOLO

model = YOLO("train_yoga5_2/train_stage3/weights/best.pt")

# Validate the model
metrics = model.val(
  data="coco-pose.yaml",
)
