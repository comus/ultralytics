from ultralytics import YOLO
import os

# Load the best model from previous training
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")  # Using best weights

# Create a combined dataset YAML configuration
combined_yaml = """
path: ../datasets  # Adjust this path as needed
train: 
  - coco/train2017.json  # Assuming COCO path
  - yoga82/images/train
val:
  - coco/val2017.json    # Assuming COCO path
  - yoga82/images/val

# Keypoints
kpt_shape: [17, 3]  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes
names:
  0: person
"""

# Write the combined YAML file
with open("combined.yaml", "w") as f:
    f.write(combined_yaml)

# Train on the combined dataset
results = model.train(
    data="combined.yaml",
    epochs=20,            # Shorter epochs for fine-tuning
    imgsz=1280,           # High resolution
    batch=24,             # Smaller batch size for mixed dataset
    save_period=1,        # Save every epoch
    cache="disk",         # Use disk caching
    optimizer="AdamW",    # Continue with AdamW optimizer
    lr0=0.000005,         # Even lower learning rate for mixed training
    lrf=0.01,             # Standard final LR factor
    cos_lr=True,          # Cosine LR scheduler
    warmup_epochs=1.0,    # Short warmup
    device="0,1",         # Use available GPUs
    patience=7,           # Earlier stopping
    box=6.0,              # Lower box loss weight
    pose=25.0,            # High pose loss weight
    kobj=7.0,             # High keypoint objectness weight
    
    # Minimal augmentation
    hsv_h=0.01,
    hsv_s=0.1,
    hsv_v=0.1,
    degrees=0.0,
    translate=0.05,
    scale=0.1,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    
    # Additional settings
    overlap_mask=True,
    amp=True,
    val=True,
    freeze=5,             # Freeze fewer layers
    exist_ok=True,
    
    # Class weights to balance dataset importance
    cls_weights=None      # Adjust if needed for class balance
)

# After training on combined dataset, fine-tune on yoga only
yoga_results = model.train(
    data="yoga82.yaml",
    epochs=10,            # Short final yoga-specific fine-tuning
    imgsz=1280,
    batch=32,
    save_period=1,
    cache="disk",
    optimizer="AdamW",
    lr0=0.000001,         # Extremely low learning rate
    lrf=0.01,
    cos_lr=True,
    device="0,1",
    patience=5,
    box=6.0,
    pose=25.0,
    kobj=7.0,
    
    # Minimal augmentation
    hsv_h=0.01,
    hsv_s=0.1,
    hsv_v=0.1,
    degrees=0.0,
    translate=0.05,
    scale=0.1,
    fliplr=0.5,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    
    amp=True,
    val=True,
    freeze=15,            # Freeze more layers in final tuning
    exist_ok=True
) 