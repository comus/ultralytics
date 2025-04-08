from ultralytics import YOLO

model = YOLO("yolo11n-pose.yaml")

# Train the model with optimized parameters for RTX 4090 (24GB VRAM)
results = model.train(
    data="coco-pose.yaml",          # Point to your actual dataset YAML
    epochs=500,                     # More epochs for training from scratch
    patience=50,                    # Early stopping patience
    batch=32,                       # Batch size - adjust based on your VRAM
    imgsz=640,                      # Image size
    device=0,                       # Use GPU 0
    workers=8,                      # Number of worker threads
    cos_lr=True,                    # Use cosine learning rate scheduler
    lr0=0.01,                       # Initial learning rate
    lrf=0.001,                      # Final learning rate as a fraction of initial rate
    weight_decay=0.0005,            # L2 regularization
    warmup_epochs=5,                # Warmup epochs - useful for training from scratch
    save_period=10,                 # Save checkpoint every 10 epochs
    amp=True,                       # Automatic mixed precision for faster training
    cache="disk",                    # Do not cache images in RAM (large dataset)
    close_mosaic=25,                # Disable mosaic in last 25 epochs for stability
    overlap_mask=True,              # For pose models
    val=True,                       # Run validation during training
    plots=True                      # Save plots of training results
)