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
    model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")

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
        imgsz=1280,        # 提高回1280解析度
        batch=64,          # 增加批次大小以充分利用GPU
        save_period=1,     # 每個epoch保存
        cache="disk",      # 使用磁盤緩存
        optimizer="AdamW", # 繼續使用AdamW優化器
        lr0=0.00002,       # 因增加批次大小而稍微提高學習率
        lrf=0.01,          # 最終學習率因子
        cos_lr=True,       # 餘弦學習率調度
        warmup_epochs=2.0, # 熱身階段
        device="0,1",      # 使用兩個GPU
        patience=30,       # 增加耐心值
        box=15.0,          # 進一步增加框損失權重
        pose=22.0,         # 進一步增加姿態損失權重 
        kobj=5.0,          # 增加關鍵點對象損失權重
        nbs=64,            # 調整標稱批次大小
        degrees=10.0,      # 增加旋轉增強
        translate=0.2,     # 增加平移增強
        scale=0.2,         # 增加縮放增強
        shear=5.0,         # 增加剪切增強
        perspective=0.001, # 增加透視增強
        flipud=0.1,        # 上下翻轉增強
        mosaic=0.5,        # 降低馬賽克增強概率
        mixup=0.15,        # 增加mixup增強
        copy_paste=0.3,    # 增加複製粘貼增強
        auto_augment="randaugment",  # 使用隨機增強
        amp=True,          # 啟用混合精度訓練
        overlap_mask=True  # 重疊口罩
    )

if __name__ == "__main__":
    main() 
