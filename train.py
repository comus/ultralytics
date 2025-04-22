from ultralytics import YOLO

# Initialize a new model from yaml configuration without pretrained weights
model = YOLO("yolo11n-pose.yaml")

# Train the model with specified parameters
results = model.train(
    data="coco-pose.yaml",
    epochs=100,
    imgsz=640,
    batch=128,
    device=0,
    save_period=1,
    workers=8,
    optimizer="auto",
    cos_lr=False,
    amp=True,
    close_mosaic=10,
    patience=100
)
