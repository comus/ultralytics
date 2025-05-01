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
    # # 使用第四階段的最佳權重
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
    #     name="stage9_phase5a",        # 訓練名稱
    #     exist_ok=True                 # 覆蓋已有目錄
    # )
    
    # 獲取階段5A最佳權重
    phase5a_best = YOLO("runs/pose/stage9_phase5a/weights/best.pt")
    
    # 第五階段B：返回目標分辨率，使用圓形學習率再次優化
    print("\n=== 第五階段B：目標分辨率優化 ===")
    phase5b_results = phase5a_best.train(
        data="coco-pose.yaml",
        epochs=70,                    # 中期訓練
        imgsz=640,                    # 回到目標分辨率
        batch=128,                    # 較大批次
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="AdamW",            # AdamW優化器
        lr0=0.0002,                   # 較高學習率
        lrf=0.00001,                  # 極低最終學習率
        momentum=0.937,               # 標準動量
        weight_decay=0.00005,         # 低權重衰減
        warmup_epochs=5.0,            # 適度熱身
        box=1.0,                      # 最低框損失權重
        pose=250.0,                   # 最高姿態損失權重
        kobj=100.0,                   # 最高關鍵點對象損失權重
        cls=0.001,                    # 極低分類損失權重
        dfl=0.01,                     # 極低分布焦點損失權重
        cos_lr=True,                  # 餘弦學習率調度
        # one_cycle=True,               # 使用one-cycle策略
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=50,                  # 耐心值
        val=True,                     # 驗證
        plots=True,                   # 圖表
        label_smoothing=0.1,          # 標籤平滑
        multi_scale=False,            # 關閉多尺度訓練
        rect=False,                   # 不使用矩形訓練
        
        # 均衡數據增強
        hsv_h=0.015,                  # 輕微色調
        hsv_s=0.1,                    # 輕微飽和度
        hsv_v=0.1,                    # 輕微亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.05,               # 輕微平移
        scale=0.1,                    # 輕微縮放
        fliplr=0.5,                   # 保留水平翻轉
        mosaic=0.1,                   # 輕微馬賽克
        mixup=0.05,                   # 輕微mixup
        copy_paste=0.0,               # 關閉複製粘貼
        
        project="runs/pose",          # 項目名稱
        name="stage9_phase5b",        # 訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
    # 獲取階段5B最佳權重
    phase5b_best = YOLO("runs/pose/stage9_phase5b/weights/best.pt")
    
    # 第五階段C：最終精細調整 - 完全優化損失權重，使用特殊技巧
    print("\n=== 第五階段C：最終精細調整 ===")
    phase5c_results = phase5b_best.train(
        data="coco-pose.yaml",
        epochs=50,                    # 適度輪數
        imgsz=640,                    # 維持目標分辨率
        batch=96,                     # 中等批次
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="AdamW",            # AdamW優化器
        lr0=0.00005,                  # 極低學習率
        lrf=0.000001,                 # 極極低最終學習率
        momentum=0.9,                 # 降低動量
        weight_decay=0.000001,        # 極低權重衰減
        warmup_epochs=0.0,            # 無熱身
        box=0.5,                      # 最低框損失權重
        pose=300.0,                   # 最最高姿態損失權重
        kobj=120.0,                   # 最最高關鍵點對象損失權重
        cls=0.0005,                   # 最低分類損失權重
        dfl=0.005,                    # 最低分布焦點損失權重
        cos_lr=False,                 # 關閉餘弦調度
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=30,                  # 適度耐心值
        val=True,                     # 驗證
        plots=True,                   # 圖表
        label_smoothing=0.0,          # 關閉標籤平滑
        multi_scale=False,            # 關閉多尺度訓練
        rect=True,                    # 使用矩形訓練
        nbs=64,                       # 設置標稱批次大小
        
        # 幾乎無數據增強
        hsv_h=0.0,                    # 關閉色調
        hsv_s=0.0,                    # 關閉飽和度
        hsv_v=0.02,                   # 最小亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.0,                # 關閉平移
        scale=0.02,                   # 極小縮放
        fliplr=0.5,                   # 保留水平翻轉
        mosaic=0.0,                   # 關閉馬賽克
        mixup=0.0,                    # 關閉mixup
        copy_paste=0.0,               # 關閉複製粘貼
        
        # 特殊參數 - 試圖最大化關鍵點精度
        iou=0.7,                      # 提高IoU閾值
        conf=0.001,                   # 降低置信度閾值以增加召回率
        max_det=100,                  # 增加最大檢測數
        
        project="runs/pose",          # 項目名稱
        name="stage9_phase5c",        # 訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
    # 第五階段D：多分辨率測試階段 - 不實際訓練，只進行評估
    print("\n=== 第五階段D：多分辨率評估 ===")
    # 獲取階段5C最佳權重
    phase5c_best = YOLO("runs/pose/stage9_phase5c/weights/best.pt")
    
    # 640分辨率評估
    print("評估640分辨率性能...")
    val_results_640 = phase5c_best.val(data="coco-pose.yaml", imgsz=640)
    
    # 672分辨率評估
    print("評估672分辨率性能...")
    val_results_672 = phase5c_best.val(data="coco-pose.yaml", imgsz=672)
    
    # 704分辨率評估
    print("評估704分辨率性能...")
    val_results_704 = phase5c_best.val(data="coco-pose.yaml", imgsz=704)
    
    # 736分辨率評估 
    print("評估736分辨率性能...")
    val_results_736 = phase5c_best.val(data="coco-pose.yaml", imgsz=736)
    
    # 建議最佳推理分辨率
    if is_main_process():
        print("\n=== 訓練完成 ===")
        print("請比較不同分辨率的評估結果，選擇最佳的推理分辨率")
        print("注意：雖然訓練和評估用640，但在實際部署時可以考慮使用稍高分辨率進行推理")

if __name__ == "__main__":
    main() 