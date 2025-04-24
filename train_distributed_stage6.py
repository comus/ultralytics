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
# 忽略除零警告
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

# 檢查是否為主進程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

from ultralytics import YOLO

def main():    
    # Initialize a new model from yaml configuration without pretrained weights
    model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train9/weights/last.pt")

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
        imgsz=1280,          # 維持高解析度
        batch=64,            # 保持批次大小
        save_period=1,       # 每個epoch保存
        cache="disk",        # 使用磁盤緩存
        optimizer="AdamW",   # 繼續使用AdamW優化器
        lr0=0.000015,        # 再次降低學習率以實現精細微調
        lrf=0.005,           # 更低的最終學習率因子，更精細調整
        cos_lr=True,         # 餘弦學習率調度
        warmup_epochs=2.0,   # 適度的熱身階段
        device="0,1,2,3",        # 使用兩個GPU
        patience=35,         # 增加耐心值，避免過早停止
        box=12.0,            # 降低框損失權重，更專注於姿態
        pose=25.0,           # 進一步增加姿態損失權重
        kobj=6.0,            # 增加關鍵點對象損失權重
        close_mosaic=5,      # 提前關閉馬賽克，讓模型在真實情況下訓練更久
        amp=True,            # 啟用混合精度訓練
        overlap_mask=True,   # 重疊口罩
        augment=True,        # 增強數據增強
        hsv_h=0.015,         # 減少色調增強
        hsv_s=0.2,           # 減少飽和度增強
        hsv_v=0.2,           # 減少亮度增強
        copy_paste=0.0,      # 關閉複製粘貼
        mixup=0.0,           # 關閉mixup
        degrees=0.0,         # 關閉旋轉增強
        translate=0.1,       # 減少平移增強
        scale=0.1,           # 減少縮放增強
        fliplr=0.5,          # 保留水平翻轉
        resume=True
    )

if __name__ == "__main__":
    main() 
