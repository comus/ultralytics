from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")

# Train the model
results = model.train(
  data="coco8-pose.yaml",
  epochs=10,
  imgsz=640,

  teacher=YOLO("yolo11n-pose.pt").model,
  target_layers=["model.0.conv", 1],

  # freezeAllBN=True,
  # freeze=23,
)