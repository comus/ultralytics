import os
import sys
import torch
import torch.distributed as dist
import warnings

# 添加本地路徑到 Python 路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 設置環境變量
os.environ["PYTHONPATH"] = f"{current_dir}:{os.environ.get('PYTHONPATH', '')}"
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

# 檢查是否為主進程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

from ultralytics import YOLO

def main():
    """
    超級第五階段訓練 - 專注於極限關鍵點精度優化
    目標：達到或超過1280分辨率的0.43 mAP
    策略：極端損失權重、精準學習率、專項微調
    """
    # 使用第四階段的最佳權重
    model_path = "runs/pose/stage9_phase4_practical/weights/best.pt"
    model = YOLO(model_path)
    
    if is_main_process():
        print(f"\n=== 超級第五階段：極限關鍵點精度優化 ===")
        print(f"載入模型: {model_path}")
        print("目標：達到或超過1280分辨率的0.43 mAP")
        print("策略：極端權重設定 + 精準學習率控制")
        
        # 檢查模塊路徑
        import ultralytics
        print(f"使用的 Ultralytics 模塊路徑: {os.path.dirname(ultralytics.__file__)}")
    
    # 第一階段：SGD高動量衝擊訓練
    print("\n=== 第一階段：SGD高動量衝擊訓練 ===")
    phase1_results = model.train(
        data="coco-pose.yaml",
        epochs=100,                    # 適度輪數
        imgsz=640,                    # 目標分辨率
        batch=160,                    # 大批次
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="SGD",              # 切換到SGD優化器
        lr0=0.001,                    # 較高學習率
        lrf=0.0005,                   # 適中最終學習率
        momentum=0.98,                # 極高動量
        weight_decay=0.0001,          # 適中權重衰減
        warmup_epochs=3.0,            # 短熱身
        box=1.0,                      # 低框損失權重
        pose=250.0,                   # 超高姿態損失權重
        kobj=100.0,                   # 高關鍵點對象損失權重
        cls=0.005,                    # 極低分類損失權重
        dfl=0.01,                     # 極低分布焦點損失權重
        cos_lr=True,                  # 餘弦學習率調度
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=60,                  # 高耐心值
        val=True,                     # 驗證
        plots=True,                   # 圖表
        label_smoothing=0.1,          # 適度標籤平滑
        
        # 適中數據增強 - SGD需要更多樣化數據
        hsv_h=0.015,                  # 輕微色調
        hsv_s=0.2,                    # 適度飽和度
        hsv_v=0.2,                    # 適度亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.1,                # 適度平移
        scale=0.15,                   # 適度縮放
        fliplr=0.5,                   # 水平翻轉
        mosaic=0.2,                   # 輕度馬賽克
        mixup=0.1,                    # 輕度mixup
        copy_paste=0.0,               # 關閉複製粘貼
        
        project="runs/pose",          # 項目名稱
        name="super_phase5_1_v2",        # 訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
if __name__ == "__main__":
    main() 