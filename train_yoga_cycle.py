from ultralytics import YOLO
import os
import shutil

# Load the best model from previous training
model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")

# Define checkpoint paths
CHECKPOINTS_DIR = "cycle_checkpoints"
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# Save the initial model
INITIAL_MODEL = os.path.join(CHECKPOINTS_DIR, "initial_model.pt")
shutil.copy(model.ckpt_path, INITIAL_MODEL)

# Cyclic training function
def cyclic_train(cycles=3, coco_epochs=5, yoga_epochs=10):
    current_model = INITIAL_MODEL
    
    for cycle in range(1, cycles+1):
        print(f"\n=== Starting cycle {cycle}/{cycles} ===")
        
        # 1. Train on yoga dataset
        print(f"Training on Yoga dataset for {yoga_epochs} epochs")
        model = YOLO(current_model)
        yoga_results = model.train(
            data="yoga82.yaml",
            epochs=yoga_epochs,
            imgsz=1280,
            batch=32,
            save_period=1,
            cache="disk",
            optimizer="AdamW",
            lr0=0.000008 / (cycle * 1.5),  # Gradually decrease learning rate
            lrf=0.01,
            cos_lr=True,
            device="0,1",
            patience=yoga_epochs,  # No early stopping within a cycle
            box=6.0,
            pose=25.0,
            kobj=7.0,
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
            freeze=10,  # Freeze some layers to preserve knowledge
            exist_ok=True,
            project="yoga_cycle",
            name=f"cycle{cycle}_yoga"
        )
        
        # Save yoga model checkpoint
        yoga_ckpt = os.path.join(CHECKPOINTS_DIR, f"cycle{cycle}_yoga.pt")
        shutil.copy(model.best, yoga_ckpt)
        
        if cycle < cycles:  # Skip COCO training on final cycle
            # 2. Train on COCO to prevent forgetting
            print(f"Training on COCO dataset for {coco_epochs} epochs")
            model = YOLO(yoga_ckpt)
            coco_results = model.train(
                data="coco-pose.yaml",
                epochs=coco_epochs,
                imgsz=1280,
                batch=64,
                save_period=1,
                cache="disk",
                optimizer="AdamW",
                lr0=0.000005 / cycle,  # Even lower learning rate for COCO
                lrf=0.01,
                cos_lr=True,
                device="0,1",
                patience=coco_epochs,  # No early stopping within a cycle
                box=6.0,
                pose=25.0,
                kobj=7.0,
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
                freeze=15,  # Freeze more layers for COCO to prevent major changes
                exist_ok=True,
                project="yoga_cycle",
                name=f"cycle{cycle}_coco"
            )
            
            # Save COCO model checkpoint
            coco_ckpt = os.path.join(CHECKPOINTS_DIR, f"cycle{cycle}_coco.pt")
            shutil.copy(model.best, coco_ckpt)
            
            # Use this as starting point for next cycle
            current_model = coco_ckpt
        
    # Return the final model path
    return yoga_ckpt if cycles > 0 else INITIAL_MODEL

# Run the cyclic training
final_model = cyclic_train(cycles=3, coco_epochs=5, yoga_epochs=10)
print(f"Training complete. Final model saved at: {final_model}")

# Optional: Validate the final model on both datasets
model = YOLO(final_model)
print("Validating on COCO dataset...")
model.val(data="coco-pose.yaml")
print("Validating on Yoga dataset...")
model.val(data="yoga82.yaml") 