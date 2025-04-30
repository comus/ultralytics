import os
import sys
import torch
import torch.nn as nn
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
    # 使用第八次訓練的最佳權重作為起點
    model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train14/weights/best.pt")
    
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

    # 第一階段訓練：凍結backbone，高學習率
    print("=== 第一階段：凍結backbone，高學習率訓練頭部 ===")
    model.train(
        data="coco-pose.yaml",
        epochs=50,                  # 較短的第一階段
        imgsz=640,                  # 設定目標分辨率
        batch=192,                  # 較大批次
        save_period=1,              # 每個epoch保存
        cache="disk",               # 使用磁盤緩存
        optimizer="AdamW",          # 使用AdamW優化器
        lr0=0.0005,                 # 較高的學習率
        lrf=0.01,                   # 學習率衰減
        momentum=0.937,             # 標準動量
        weight_decay=0.0005,        # 標準權重衰減
        warmup_epochs=3.0,          # 熱身階段
        warmup_momentum=0.8,        # 熱身動量
        warmup_bias_lr=0.1,         # 熱身偏置學習率
        box=5.0,                    # 適中框損失權重
        pose=60.0,                  # 大幅增加姿態損失權重
        kobj=25.0,                  # 大幅增加關鍵點對象損失權重
        cls=0.1,                    # 降低分類損失權重
        dfl=0.5,                    # 降低分布焦點損失權重
        cos_lr=True,                # 啟用餘弦學習率調度
        amp=True,                   # 混合精度訓練
        device="0,1,2,3",           # 使用GPU
        overlap_mask=True,          # 重疊口罩
        patience=100,               # 足夠的耐心值
        val=True,                   # 驗證
        plots=True,                 # 生成圖表
        freeze=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], # 凍結backbone所有層
        
        # 適度數據增強
        hsv_h=0.015,                # 輕微色調變化
        hsv_s=0.2,                  # 適度飽和度變化
        hsv_v=0.2,                  # 適度亮度變化
        degrees=0.0,                # 關閉旋轉
        translate=0.1,              # 適度平移
        scale=0.2,                  # 適度縮放
        fliplr=0.5,                 # 水平翻轉
        mosaic=0.3,                 # 適度馬賽克增強
        copy_paste=0.0,             # 關閉複製粘貼
        auto_augment="randaugment", # 使用隨機增強
        erasing=0.2,                # 隨機擦除
        project="runs/pose",        # 項目名稱
        name="stage9_phase1",       # 訓練階段名稱
    )
    
    # 獲取第一階段訓練的最佳權重
    phase1_model = YOLO("runs/pose/stage9_phase1/weights/best.pt")
    
    # 第二階段訓練：解凍backbone，低學習率微調
    print("=== 第二階段：解凍backbone，低學習率微調全網絡 ===")
    phase1_model.train(
        data="coco-pose.yaml",
        epochs=150,                 # 較長的第二階段
        imgsz=640,                  # 保持目標分辨率
        batch=128,                  # 調低批次
        save_period=1,              # 每個epoch保存
        cache="disk",               # 使用磁盤緩存
        optimizer="AdamW",          # 保持AdamW優化器
        lr0=0.0001,                 # 較低的學習率
        lrf=0.001,                  # 更小的衰減
        momentum=0.9,               # 調整動量
        weight_decay=0.0001,        # 降低權重衰減
        warmup_epochs=0.0,          # 關閉熱身
        box=3.0,                    # 降低框損失權重
        pose=70.0,                  # 進一步增加姿態損失權重
        kobj=25.0,                  # 保持高關鍵點對象損失權重
        cls=0.05,                   # 進一步降低分類損失權重
        dfl=0.2,                    # 降低分布焦點損失權重
        cos_lr=True,                # 啟用餘弦學習率調度
        amp=True,                   # 混合精度訓練
        device="0,1,2,3",           # 使用GPU
        dropout=0.1,                # 添加dropout
        overlap_mask=True,          # 重疊口罩
        patience=50,                # 適度耐心值
        val=True,                   # 驗證
        plots=True,                 # 生成圖表
        
        # 減少數據增強
        hsv_h=0.01,                 # 最小色調變化
        hsv_s=0.1,                  # 最小飽和度變化
        hsv_v=0.1,                  # 最小亮度變化
        degrees=0.0,                # 關閉旋轉
        translate=0.05,             # 最小平移
        scale=0.1,                  # 最小縮放
        fliplr=0.5,                 # 保持水平翻轉
        mosaic=0.1,                 # 最小馬賽克增強
        mixup=0.0,                  # 關閉mixup
        copy_paste=0.0,             # 關閉複製粘貼
        project="runs/pose",        # 項目名稱
        name="stage9_phase2",       # 訓練階段名稱
    )
    
    # 獲取第二階段訓練的最佳權重
    phase2_model = YOLO("runs/pose/stage9_phase2/weights/best.pt")
    
    # 第三階段訓練：添加多尺度訓練
    print("=== 第三階段：多尺度訓練，最終微調 ===")
    phase2_model.train(
        data="coco-pose.yaml",
        epochs=100,                 # 適度的第三階段
        imgsz=640,                  # 基本分辨率
        batch=96,                   # 再次調整批次
        save_period=1,              # 每個epoch保存
        cache="disk",               # 使用磁盤緩存
        optimizer="AdamW",          # 保持AdamW優化器
        lr0=0.00005,                # 極低的學習率
        lrf=0.001,                  # 保持小的衰減
        momentum=0.937,             # 標準動量
        weight_decay=0.00005,       # 極低的權重衰減
        warmup_epochs=0.0,          # 關閉熱身
        box=2.0,                    # 最低框損失權重
        pose=50.0,                  # 略微降低姿態損失權重
        kobj=20.0,                  # 略微降低關鍵點對象損失權重
        cls=0.01,                   # 最低分類損失權重
        dfl=0.1,                    # 最低分布焦點損失權重
        cos_lr=True,                # 啟用餘弦學習率調度
        amp=True,                   # 混合精度訓練
        device="0,1,2,3",           # 使用GPU
        dropout=0.0,                # 關閉dropout
        overlap_mask=True,          # 重疊口罩
        patience=25,                # 降低耐心值
        val=True,                   # 驗證
        plots=True,                 # 生成圖表
        label_smoothing=0.1,        # 標籤平滑化
        multi_scale=True,           # 啟用多尺度訓練 (非常重要)
        
        # 最小數據增強
        hsv_h=0.0,                  # 關閉色調變化
        hsv_s=0.0,                  # 關閉飽和度變化
        hsv_v=0.05,                 # 最小亮度變化
        degrees=0.0,                # 關閉旋轉
        translate=0.0,              # 關閉平移
        scale=0.05,                 # 最小縮放
        fliplr=0.5,                 # 保持水平翻轉
        mosaic=0.0,                 # 關閉馬賽克增強
        mixup=0.0,                  # 關閉mixup
        copy_paste=0.0,             # 關閉複製粘貼
        project="runs/pose",        # 項目名稱
        name="stage9_phase3",       # 訓練階段名稱
    )

if __name__ == "__main__":
    main() 