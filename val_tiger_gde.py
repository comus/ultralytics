from ultralytics import YOLO

# Load a model
model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train14/weights/best.pt")

# Train the model
results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
