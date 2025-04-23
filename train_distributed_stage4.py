import os
import sys
import torch.distributed as dist
import subprocess
import warnings

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 設置環境變量，確保分佈式訓練使用本地代碼
os.environ["PYTHONPATH"] = f"{current_dir}:{os.environ.get('PYTHONPATH', '')}"
# 忽略 DDP 的 stride 不匹配警告
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

# 檢查是否為主進程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

from ultralytics import YOLO

def main():    
    # Initialize a new model from yaml configuration without pretrained weights
    model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train54/weights/best.pt")

    # 打印使用的模塊路徑，確認是否正確
    if is_main_process():
        import ultralytics
        print(f"使用的 Ultralytics 模塊路徑: {os.path.dirname(ultralytics.__file__)}")
        
        # 檢查模塊是否包含自定義層
        try:
            import inspect
            from ultralytics.nn.modules.block import C3k2_Ghost, C3k2_DFFM
            print(f"C3k2_Ghost 模塊位置: {inspect.getfile(C3k2_Ghost)}")
            print(f"C3k2_DFFM 模塊位置: {inspect.getfile(C3k2_DFFM)}")
        except (ImportError, AttributeError) as e:
            print(f"警告: 自定義層檢查失敗 - {e}")

    # Train the model with hard-coded parameters
    results = model.train(
        data="coco-pose.yaml",
        epochs=100,
        imgsz=1024,        # Keep high resolution
        batch=48,          # Reduced batch size to prevent CUDA OOM error
        save_period=1,     # Save every epoch as requested
        cache="disk",      # Use disk caching
        optimizer="AdamW", # Continue with AdamW optimizer
        lr0=0.00003,       # Adjusted learning rate for smaller batch size
        lrf=0.01,          # Final LR factor
        cos_lr=True,       # Cosine LR scheduler
        warmup_epochs=3.0, # Increased warmup epochs for stability
        device="0,1",      # Use both GPUs
        patience=25,       # Early stopping patience
        box=12.0,          # Increased box loss weight
        pose=18.0,         # Increased pose loss weight
        kobj=4.0,          # Increased keypoint objectness weight
        multi_scale=True,  # Enable multi-scale training for better generalization
        close_mosaic=10    # Close mosaic in final epochs
    )

if __name__ == "__main__":
    main() 
