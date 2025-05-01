import os
import sys
import torch
import torch.distributed as dist
import subprocess
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
    640分辨率極限優化訓練方案 - 實用版
    目標：達到與1280分辨率相當的mAP (0.43)
    """
    # 使用第八次訓練的最佳權重
    model_path = "/root/autodl-tmp/withcloud/root/autodl-tmp/withcloud/ultralytics/runs/pose/train52/weights/best.pt"
    model = YOLO(model_path)
    
    if is_main_process():
        print(f"載入模型: {model_path}")
        print("=== 640分辨率極限優化訓練方案 - 實用版 ===")
        print("目標：達到與1280分辨率相當的mAP (0.43)")
        
        # 檢查模塊路徑
        import ultralytics
        print(f"使用的 Ultralytics 模塊路徑: {os.path.dirname(ultralytics.__file__)}")
        
        try:
            import inspect
            from ultralytics.nn.modules.block import C3k2_Ghost, C3k2_DFFM
            print(f"C3k2_Ghost 模塊位置: {inspect.getfile(C3k2_Ghost)}")
            print(f"C3k2_DFFM 模塊位置: {inspect.getfile(C3k2_DFFM)}")
        except (ImportError, AttributeError) as e:
            print(f"警告: 自定義層檢查失敗 - {e}")
    
    # 第1階段：凍結backbone，聚焦訓練頭部，高學習率
    print("\n=== 第1階段：凍結backbone，聚焦訓練頭部 ===")
    phase1_results = model.train(
        data="coco-pose.yaml",
        epochs=60,                 # 較長第一階段
        imgsz=640,                 # 目標分辨率
        batch=192,                 # 大批次
        save_period=1,             # 每個epoch保存
        cache="disk",              # 磁盤緩存
        optimizer="AdamW",         # AdamW優化器
        lr0=0.001,                 # 高學習率
        lrf=0.01,                  # 學習率因子
        momentum=0.937,            # 標準動量
        weight_decay=0.0005,       # 標準權重衰減
        warmup_epochs=3.0,         # 熱身階段
        warmup_momentum=0.8,       # 熱身動量
        warmup_bias_lr=0.1,        # 熱身偏置學習率
        box=5.0,                   # 中等框損失權重
        pose=100.0,                # 極高姿態損失權重
        kobj=40.0,                 # 極高關鍵點對象損失權重
        cls=0.1,                   # 低分類損失權重
        dfl=0.5,                   # 低分布焦點損失權重
        cos_lr=True,               # 餘弦學習率調度
        amp=True,                  # 混合精度訓練
        device="0,1,2,3",          # GPU
        overlap_mask=True,         # 重疊口罩
        patience=100,              # 耐心值
        val=True,                  # 驗證
        freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # 凍結backbone
        
        # 數據增強 - 適度
        hsv_h=0.015,               # 色調變化
        hsv_s=0.2,                 # 飽和度變化
        hsv_v=0.2,                 # 亮度變化
        degrees=0.0,               # 關閉旋轉
        translate=0.1,             # 平移
        scale=0.2,                 # 縮放
        fliplr=0.5,                # 水平翻轉
        mosaic=0.2,                # 適度馬賽克
        mixup=0.1,                 # 輕度mixup
        copy_paste=0.0,            # 不使用
        
        project="runs/pose",       # 項目名稱
        name="yolo_stage9_phase1_practical", # 訓練名稱
        exist_ok=True              # 覆蓋已有目錄
    )
    
    # 獲取第一階段最佳權重
    phase1_best = YOLO("runs/pose/yolo_stage9_phase1_practical/weights/best.pt")
    
    # 第2階段：解凍部分backbone，使用較小學習率，添加數據增強
    print("\n=== 第2階段：解凍部分backbone，添加高級數據增強 ===")
    phase2_results = phase1_best.train(
        data="coco-pose.yaml",
        epochs=120,                # 長期訓練
        imgsz=640,                 # 保持目標分辨率
        batch=128,                 # 調整批次
        save_period=1,             # 每個epoch保存
        cache="disk",              # 使用磁盤緩存
        optimizer="AdamW",         # AdamW優化器
        lr0=0.0005,                # 中等學習率
        lrf=0.001,                 # 較小學習率因子
        momentum=0.937,            # 標準動量
        weight_decay=0.0001,       # 較小權重衰減
        warmup_epochs=0.0,         # 無熱身
        box=4.0,                   # 適中框損失權重
        pose=120.0,                # 進一步增加姿態損失權重
        kobj=50.0,                 # 進一步增加關鍵點對象損失權重
        cls=0.05,                  # 低分類損失權重
        dfl=0.2,                   # 低分布焦點損失權重
        cos_lr=True,               # 餘弦學習率調度
        amp=True,                  # 混合精度訓練
        device="0,1,2,3",          # GPU
        overlap_mask=True,         # 重疊口罩
        patience=50,               # 耐心值
        val=True,                  # 驗證
        freeze=[0, 1, 2, 3, 4],    # 只凍結前幾層
        label_smoothing=0.1,       # 標籤平滑
        
        # 增強數據增強
        hsv_h=0.015,               # 色調
        hsv_s=0.2,                 # 飽和度
        hsv_v=0.2,                 # 亮度
        degrees=0.0,               # 關閉旋轉
        translate=0.1,             # 平移
        scale=0.2,                 # 縮放
        shear=0.0,                 # 不剪切
        perspective=0.0,           # 不使用透視變換
        fliplr=0.5,                # 水平翻轉
        mosaic=0.2,                # 適度馬賽克
        mixup=0.1,                 # 輕度mixup
        copy_paste=0.0,            # 不使用
        
        project="runs/pose",       # 項目名稱
        name="yolo_stage9_phase2_practical", # 訓練名稱
        exist_ok=True              # 覆蓋已有目錄
    )
    
    # 獲取第二階段最佳權重
    phase2_best = YOLO("runs/pose/yolo_stage9_phase2_practical/weights/best.pt")
    
    # 第3階段：完全解凍，極低學習率，多尺度訓練
    print("\n=== 第3階段：完全解凍，開啟多尺度訓練 ===")
    phase3_results = phase2_best.train(
        data="coco-pose.yaml",
        epochs=100,                # 適當輪數
        imgsz=640,                 # 基本分辨率
        batch=96,                  # 適中批次
        save_period=1,             # 每個epoch保存
        cache="disk",              # 磁盤緩存
        optimizer="AdamW",         # AdamW優化器
        lr0=0.0001,                # 低學習率
        lrf=0.0001,                # 極低最終學習率
        momentum=0.937,            # 標準動量
        weight_decay=0.00005,      # 極低權重衰減
        warmup_epochs=0.0,         # 無熱身
        box=3.0,                   # 低框損失權重
        pose=150.0,                # 極高姿態損失權重
        kobj=60.0,                 # 極高關鍵點對象損失權重
        cls=0.01,                  # 極低分類損失權重
        dfl=0.1,                   # 極低分布焦點損失權重
        cos_lr=True,               # 餘弦學習率調度
        amp=True,                  # 混合精度訓練
        device="0,1,2,3",          # GPU
        overlap_mask=True,         # 重疊口罩
        patience=35,               # 耐心值
        val=True,                  # 驗證
        plots=True,                # 圖表
        label_smoothing=0.1,       # 標籤平滑
        multi_scale=True,          # 多尺度訓練 - 非常重要
        
        # 輕微數據增強
        hsv_h=0.01,                # 極輕微色調
        hsv_s=0.1,                 # 低飽和度
        hsv_v=0.1,                 # 低亮度
        degrees=0.0,               # 關閉旋轉
        translate=0.05,            # 低平移
        scale=0.1,                 # 低縮放
        fliplr=0.5,                # 水平翻轉
        mosaic=0.1,                # 輕微馬賽克
        mixup=0.0,                 # 關閉mixup
        copy_paste=0.0,            # 關閉
        
        project="runs/pose",       # 項目名稱
        name="yolo_stage9_phase3_practical", # 訓練名稱
        exist_ok=True              # 覆蓋已有目錄
    )
    
    # 獲取第三階段最佳權重
    phase3_best = YOLO("runs/pose/yolo_stage9_phase3_practical/weights/best.pt")
    
    # 第4階段：最終微調 - 極低學習率，無數據增強
    print("\n=== 第4階段：最終微調 - 極低學習率，無數據增強 ===")
    phase4_results = phase3_best.train(
        data="coco-pose.yaml",
        epochs=50,                 # 短期微調
        imgsz=640,                 # 基本分辨率
        batch=64,                  # 較小批次
        save_period=1,             # 每個epoch保存
        cache="disk",              # 磁盤緩存
        optimizer="AdamW",         # AdamW優化器
        lr0=0.00001,               # 極低學習率
        lrf=0.00001,               # 極低最終學習率
        momentum=0.9,              # 較低動量
        weight_decay=0.00001,      # 極低權重衰減
        warmup_epochs=0.0,         # 無熱身
        box=2.0,                   # 最低框損失權重
        pose=180.0,                # 極高姿態損失權重
        kobj=70.0,                 # 極高關鍵點對象損失權重
        cls=0.01,                  # 極低分類損失權重
        dfl=0.1,                   # 極低分布焦點損失權重
        cos_lr=False,              # 關閉餘弦
        amp=True,                  # 混合精度訓練
        device="0,1,2,3",          # GPU
        overlap_mask=True,         # 重疊口罩
        patience=20,               # 較低耐心值
        val=True,                  # 驗證
        plots=True,                # 圖表
        
        # 關閉所有數據增強，專注精確預測
        hsv_h=0.0,                 # 關閉色調
        hsv_s=0.0,                 # 關閉飽和度
        hsv_v=0.05,                # 最小亮度
        degrees=0.0,               # 關閉旋轉
        translate=0.0,             # 關閉平移
        scale=0.05,                # 最小縮放
        fliplr=0.5,                # 保留水平翻轉
        mosaic=0.0,                # 關閉馬賽克
        mixup=0.0,                 # 關閉mixup
        copy_paste=0.0,            # 關閉複製粘貼
        rect=True,                 # 矩形訓練
        
        project="runs/pose",       # 項目名稱
        name="yolo_stage9_phase4_practical", # 訓練名稱
        exist_ok=True              # 覆蓋已有目錄
    )
    
    if is_main_process():
        print("\n=== 訓練完成 ===")
        print("建議使用最終階段的best.pt權重進行評估和部署")

if __name__ == "__main__":
    main() 