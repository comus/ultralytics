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
    # 從最佳權重開始進行精調
    # 假設上次訓練的最佳權重路徑，請根據實際情況調整
    model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train9/weights/best.pt")

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

    # 訓練參數設置
    results = model.train(
        data="coco-pose.yaml",
        epochs=50,            # 減少epochs數量以節省訓練時間
        imgsz=1536,           # 進一步提高解析度以提升精度
        batch=96,             # 增加批次大小以充分利用GPU資源
        save_period=1,        # 每個epoch保存
        cache="disk",         # 使用磁盤緩存
        optimizer="AdamW",    # 繼續使用AdamW優化器
        lr0=0.000024,         # 因增加批次大小而調整學習率(原本的3倍)
        lrf=0.001,            # 最終學習率因子
        cos_lr=True,          # 餘弦學習率調度
        warmup_epochs=3.0,    # 較長的熱身期
        device="0,1,2,3",     # 使用全部4個GPU
        patience=40,          # 增加耐心值
        box=10.0,             # 調整框損失權重
        pose=30.0,            # 進一步提高姿態損失權重
        kobj=8.0,             # 增加關鍵點對象損失權重
        close_mosaic=0,       # 完全關閉馬賽克增強
        amp=True,             # 啟用混合精度訓練
        overlap_mask=True,    # 啟用重疊口罩
        
        # 數據增強參數 (根據官方文檔設置)
        hsv_h=0.01,           # 色調變化 (0.0-1.0)
        hsv_s=0.1,            # 飽和度變化 (0.0-1.0)
        hsv_v=0.1,            # 亮度變化 (0.0-1.0)
        degrees=0.0,          # 旋轉增強 (0.0-180.0)
        translate=0.05,       # 平移增強 (0.0-1.0)
        scale=0.05,           # 縮放增強 (>=0.0)
        shear=0.0,            # 剪切增強 (-180.0-180.0)
        perspective=0.0,      # 透視變換 (0.0-0.001)
        flipud=0.0,           # 垂直翻轉概率 (0.0-1.0)
        fliplr=0.5,           # 水平翻轉概率 (0.0-1.0)
        mosaic=0.0,           # 馬賽克增強概率 (0.0-1.0)
        mixup=0.0,            # Mixup增強概率 (0.0-1.0)
        copy_paste=0.0        # 複製粘貼增強概率 (0.0-1.0)
    )

if __name__ == "__main__":
    main() 