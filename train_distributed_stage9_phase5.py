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
    第五階段訓練 - 640分辨率突破性能極限
    目標：盡可能接近1280分辨率的0.43 mAP
    策略：混合策略、超極端權重、混合分辨率訓練
    """
    # # 使用第四階段的最佳權重作為起點
    # model_path = "runs/pose/stage9_phase4_practical/weights/best.pt"
    # model = YOLO(model_path)
    
    # if is_main_process():
    #     print(f"\n=== 第五階段：極限突破訓練 ===")
    #     print(f"載入模型: {model_path}")
    #     print("目標：盡可能接近1280分辨率的0.43 mAP")
        
    #     # 檢查模塊路徑
    #     import ultralytics
    #     print(f"使用的 Ultralytics 模塊路徑: {os.path.dirname(ultralytics.__file__)}")
    
    # # 第五階段A：混合分辨率策略 - 先使用較大分辨率"預熱"模型
    # print("\n=== 第五階段A：混合分辨率預熱 ===")
    # phase5a_results = model.train(
    #     data="coco-pose.yaml",
    #     epochs=30,                    # 短期訓練
    #     imgsz=720,                    # 稍高分辨率
    #     batch=64,                     # 較小批次以適應更高分辨率
    #     save_period=1,                # 每個epoch保存
    #     cache="disk",                 # 磁盤緩存
    #     optimizer="AdamW",            # AdamW優化器
    #     lr0=0.0001,                   # 低學習率
    #     lrf=0.0001,                   # 極低最終學習率
    #     momentum=0.937,               # 標準動量
    #     weight_decay=0.00001,         # 極低權重衰減
    #     warmup_epochs=3.0,            # 短暫熱身
    #     box=1.0,                      # 最低框損失權重
    #     pose=220.0,                   # 超高姿態損失權重
    #     kobj=90.0,                    # 超高關鍵點對象損失權重
    #     cls=0.005,                    # 極低分類損失權重
    #     dfl=0.05,                     # 極低分布焦點損失權重
    #     cos_lr=True,                  # 餘弦學習率調度
    #     amp=True,                     # 混合精度訓練
    #     device="0,1,2,3",             # GPU
    #     overlap_mask=True,            # 重疊口罩
    #     patience=50,                  # 耐心值
    #     val=True,                     # 驗證
    #     plots=True,                   # 圖表
    #     label_smoothing=0.2,          # 增加標籤平滑
    #     multi_scale=True,             # 多尺度訓練
    #     rect=False,                   # 關閉矩形訓練以便多尺度
        
    #     # 最小化數據增強
    #     hsv_h=0.0,                    # 關閉色調
    #     hsv_s=0.0,                    # 關閉飽和度
    #     hsv_v=0.05,                   # 最小亮度
    #     degrees=0.0,                  # 關閉旋轉
    #     translate=0.0,                # 關閉平移
    #     scale=0.1,                    # 最小縮放
    #     fliplr=0.5,                   # 保留水平翻轉
    #     mosaic=0.0,                   # 關閉馬賽克
    #     mixup=0.0,                    # 關閉mixup
    #     copy_paste=0.0,               # 關閉複製粘貼
        
    #     project="runs/pose",          # 項目名稱
    #     name="stage9_phase5a_new",    # 訓練名稱
    #     exist_ok=True                 # 覆蓋已有目錄
    # )
    
    # 獲取階段5A最佳權重
    phase5a_best = YOLO("runs/pose/stage9_phase5a/weights/best.pt")
    
    # 第五階段B：保留高分辨率，但稍微降低至680，避免過擬合
    print("\n=== 第五階段B：中高分辨率微調 ===")
    phase5b_results = phase5a_best.train(
        data="coco-pose.yaml",
        epochs=40,                    # 減少訓練輪數
        imgsz=680,                    # 使用中高分辨率（不回到640）
        batch=80,                     # 調整批次大小
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="AdamW",            # AdamW優化器
        lr0=0.00005,                  # 更低學習率，避免破壞已有特徵
        lrf=0.00001,                  #
        momentum=0.937,               # 保持標準動量
        weight_decay=0.00001,         # 極低權重衰減，保持現有特徵
        warmup_epochs=0.0,            # 無熱身（已經預熱過）
        box=1.0,                      # 維持框損失權重
        pose=240.0,                   # 略微增加姿態損失權重
        kobj=95.0,                    # 略微增加關鍵點對象損失權重
        cls=0.005,                    # 保持分類損失權重
        dfl=0.05,                     # 保持分布焦點損失權重
        cos_lr=True,                  # 餘弦學習率調度
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=25,                  # 減少耐心值，避免過度訓練
        val=True,                     # 驗證
        plots=True,                   # 圖表
        label_smoothing=0.1,          # 減少標籤平滑
        multi_scale=True,             # 保持多尺度訓練
        rect=False,                   # 關閉矩形訓練以便多尺度
        
        # 維持最小數據增強
        hsv_h=0.0,                    # 關閉色調
        hsv_s=0.0,                    # 關閉飽和度
        hsv_v=0.05,                   # 最小亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.0,                # 關閉平移
        scale=0.1,                    # 最小縮放
        fliplr=0.5,                   # 保留水平翻轉
        mosaic=0.0,                   # 關閉馬賽克
        mixup=0.0,                    # 關閉mixup
        copy_paste=0.0,               # 關閉複製粘貼
        
        project="runs/pose",          # 項目名稱
        name="stage9_phase5b_revised",  # 修改訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
    # 獲取階段5B最佳權重
    phase5b_best = YOLO("runs/pose/stage9_phase5b_revised/weights/best.pt")
    
    # 第五階段C：回到目標分辨率640，最後微調
    print("\n=== 第五階段C：目標分辨率640最終微調 ===")
    phase5c_results = phase5b_best.train(
        data="coco-pose.yaml",
        epochs=30,                    # 短期微調
        imgsz=640,                    # 回到目標分辨率
        batch=96,                     # 適當批次大小
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="AdamW",            # AdamW優化器
        lr0=0.00001,                  # 極低學習率
        lrf=0.000001,                 # 極極低最終學習率
        momentum=0.9,                 # 略微降低動量
        weight_decay=0.000001,        # 極低權重衰減
        warmup_epochs=0.0,            # 無熱身
        box=1.0,                      # 保持框損失權重
        pose=220.0,                   # 與5a階段相同
        kobj=90.0,                    # 與5a階段相同
        cls=0.005,                    # 與5a階段相同
        dfl=0.05,                     # 與5a階段相同
        cos_lr=False,                 # 關閉餘弦調度
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=15,                  # 較低耐心值
        val=True,                     # 驗證
        plots=True,                   # 圖表
        label_smoothing=0.1,          # 輕微標籤平滑
        multi_scale=False,            # 關閉多尺度訓練
        rect=True,                    # 使用矩形訓練
        
        # 極少數據增強
        hsv_h=0.0,                    # 關閉色調
        hsv_s=0.0,                    # 關閉飽和度
        hsv_v=0.02,                   # 最小亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.0,                # 關閉平移
        scale=0.05,                   # 極小縮放
        fliplr=0.5,                   # 保留水平翻轉
        mosaic=0.0,                   # 關閉馬賽克
        mixup=0.0,                    # 關閉mixup
        copy_paste=0.0,               # 關閉複製粘貼
        
        project="runs/pose",          # 項目名稱
        name="stage9_phase5c_revised",  # 訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
    # # 第五階段D：多分辨率測試階段
    # print("\n=== 第五階段D：多分辨率評估 ===")
    # # 獲取階段5C最佳權重
    # phase5c_best = YOLO("runs/pose/stage9_phase5c_revised/weights/best.pt")
    
    # # 640分辨率評估
    # print("評估640分辨率性能...")
    # val_results_640 = phase5c_best.val(data="coco-pose.yaml", imgsz=640)
    
    # # 672分辨率評估
    # print("評估672分辨率性能...")
    # val_results_672 = phase5c_best.val(data="coco-pose.yaml", imgsz=672)
    
    # # 704分辨率評估
    # print("評估704分辨率性能...")
    # val_results_704 = phase5c_best.val(data="coco-pose.yaml", imgsz=704)
    
    # # 720分辨率評估 (與5a階段訓練相同)
    # print("評估720分辨率性能...")
    # val_results_720 = phase5c_best.val(data="coco-pose.yaml", imgsz=720)
    
    # # 評估5a權重在640分辨率下的性能
    # print("\n評估5a模型在640分辨率下的性能...")
    # phase5a_best = YOLO("runs/pose/stage9_phase5a_new/weights/best.pt")
    # val_results_5a_640 = phase5a_best.val(data="coco-pose.yaml", imgsz=640)
    
    # if is_main_process():
    #     print("\n=== 訓練完成 ===")
    #     print("請比較不同分辨率的評估結果，選擇最佳的推理分辨率")
    #     print("建議：如果5a在720分辨率訓練的模型在640分辨率推理時表現良好，可直接使用該模型")

if __name__ == "__main__":
    main() 