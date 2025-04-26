from ultralytics import YOLO

# Load a model
model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train14/weights/best.pt")

# Train the model
results = model.val(data="tiger-pose.yaml")
