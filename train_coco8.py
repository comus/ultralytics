from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n-pose.pt")

# Train the model
results = model.train(
  data="coco8-pose.yaml",
  epochs=20,
  imgsz=640,
  # device=[0, 1],

  teacher="yolo11n-pose.pt",
  target_layers=["model.0.conv", 1],

  # freezeAllBN=True,
  # freeze=23,
)