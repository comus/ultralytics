from ultralytics import YOLO

# Load a model
model = YOLO("/root/autodl-tmp/yolo11n-pose-custom/root/autodl-tmp/ultralytics/runs/pose/train52/weights/best.pt")

# Train the model
results = model.train(data="tiger-pose.yaml", epochs=100, imgsz=640)
