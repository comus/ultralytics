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
        imgsz=1024,        # 恢復與第四次訓練相同的圖像大小
        batch=48,          # 恢復與第四次訓練相同的批次大小
        save_period=1,     # 每個epoch保存
        cache="disk",      # 使用磁盤緩存
        optimizer="SGD",    # 改用SGD優化器
        lr0=0.00005,        # 保持較低學習率
        momentum=0.937,     # 增加標準動量參數
        weight_decay=0.0005, # 增加權重衰減防止過擬合
        lrf=0.01,           # 最終學習率因子
        cos_lr=True,        # 餘弦學習率調度
        warmup_epochs=5.0,  # 增加熱身階段
        device="0,1,2,3",   # 使用全部GPU加速訓練
        patience=50,        # 增加耐心值防止過早停止
        box=12.0,           # 保持與第四次相同的框損失權重
        pose=18.0,          # 保持與第四次相同的姿態損失權重
        kobj=4.0,           # 保持與第四次相同的關鍵點對象損失權重
        multi_scale=True,   # 啟用多尺度訓練
        close_mosaic=10,    # 最後幾個epoch關閉馬賽克增強
        amp=False,          # 關閉混合精度訓練以提高穩定性
        nbs=64,             # 標稱批次大小
        overlap_mask=True,  # 啟用遮罩重疊
        perspective=0.001,   # 重置透視增強強度
        mosaic=0.8,         # 增加馬賽克增強概率
        mixup=0.15,         # 增加mixup增強強度       # 確保從上次的訓練狀態恢復，包括優化器狀態
    )

if __name__ == "__main__":
    main() 
