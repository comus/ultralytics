from ultralytics import YOLO

# Initialize a new model from yaml configuration without pretrained weights
model = YOLO("yolo11n-pose.yaml")

# Train the model with specified parameters
results = model.train(
    data="coco-pose.yaml",
    epochs=100,
    imgsz=640,
    batch=128,
    save_period=1,
    cache="disk",
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    device="0,1"
)
