from ultralytics import YOLO

model = YOLO("yolo11n-pose.yaml")

# Train the model with optimized parameters for RTX 4090 (24GB VRAM)
results = model.train(
    data="coco-pose.yaml",          # Point to your actual dataset YAML
    epochs=500,                     # More epochs for training from scratch
    patience=50,                    # Early stopping patience
    batch=32,                       # Batch size - adjust based on your VRAM
    cos_lr=True,                    # Use cosine learning rate scheduler
    lrf=0.001,                      # Final learning rate as a fraction of initial rate
    warmup_epochs=5,                # Warmup epochs - useful for training from scratch
    save_period=1,                 # Save checkpoint every 10 epochs
    cache="disk",                    # Do not cache images in RAM (large dataset)
    close_mosaic=25,                # Disable mosaic in last 25 epochs for stability
    plots=True,                     # Save plots of training results
    mosaic=1.0,                     # Add back, important!
    mixup=0.1,
    copy_paste=0.1,
    hsv_h=0.015,                     # Color tone enhancement
    hsv_s=0.7,                       # Saturation enhancement
    hsv_v=0.4,                       # Brightness enhancement
    translate=0.1,                   # Translation enhancement
    scale=0.5,                       # Scaling enhancement
    fliplr=0.5,                      # Horizontal flip
    multi_scale=True,
    project="pose_training",
    name="yolo11n_scratch",
    pretrained=False
)