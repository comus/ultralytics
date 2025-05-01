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
    # # 使用第四階段的最佳權重
    # model_path = "runs/pose/stage9_phase4_practical/weights/best.pt"
    # model = YOLO(model_path)
    
    # if is_main_process():
    #     print(f"\n=== 超級第五階段：極限關鍵點精度優化 ===")
    #     print(f"載入模型: {model_path}")
    #     print("目標：達到或超過1280分辨率的0.43 mAP")
    #     print("策略：極端權重設定 + 精準學習率控制")
        
    #     # 檢查模塊路徑
    #     import ultralytics
    #     print(f"使用的 Ultralytics 模塊路徑: {os.path.dirname(ultralytics.__file__)}")
    
    # # 第一階段：SGD高動量衝擊訓練
    # print("\n=== 第一階段：SGD高動量衝擊訓練 ===")
    # phase1_results = model.train(
    #     data="coco-pose.yaml",
    #     epochs=40,                    # 適度輪數
    #     imgsz=640,                    # 目標分辨率
    #     batch=160,                    # 大批次
    #     save_period=1,                # 每個epoch保存
    #     cache="disk",                 # 磁盤緩存
    #     optimizer="SGD",              # 切換到SGD優化器
    #     lr0=0.001,                    # 較高學習率
    #     lrf=0.0005,                   # 適中最終學習率
    #     momentum=0.98,                # 極高動量
    #     weight_decay=0.0001,          # 適中權重衰減
    #     warmup_epochs=3.0,            # 短熱身
    #     box=1.0,                      # 低框損失權重
    #     pose=250.0,                   # 超高姿態損失權重
    #     kobj=100.0,                   # 高關鍵點對象損失權重
    #     cls=0.005,                    # 極低分類損失權重
    #     dfl=0.01,                     # 極低分布焦點損失權重
    #     cos_lr=True,                  # 餘弦學習率調度
    #     amp=True,                     # 混合精度訓練
    #     device="0,1,2,3",             # GPU
    #     overlap_mask=True,            # 重疊口罩
    #     patience=60,                  # 高耐心值
    #     val=True,                     # 驗證
    #     plots=True,                   # 圖表
    #     label_smoothing=0.1,          # 適度標籤平滑
        
    #     # 適中數據增強 - SGD需要更多樣化數據
    #     hsv_h=0.015,                  # 輕微色調
    #     hsv_s=0.2,                    # 適度飽和度
    #     hsv_v=0.2,                    # 適度亮度
    #     degrees=0.0,                  # 關閉旋轉
    #     translate=0.1,                # 適度平移
    #     scale=0.15,                   # 適度縮放
    #     fliplr=0.5,                   # 水平翻轉
    #     mosaic=0.2,                   # 輕度馬賽克
    #     mixup=0.1,                    # 輕度mixup
    #     copy_paste=0.0,               # 關閉複製粘貼
        
    #     project="runs/pose",          # 項目名稱
    #     name="super_phase5_1",        # 訓練名稱
    #     exist_ok=True                 # 覆蓋已有目錄
    # )
    
    # 獲取第一階段最佳權重
    phase1_best = YOLO("runs/pose/super_phase5_1/weights/best.pt")
    
    # 第二階段：極端關鍵點優化
    print("\n=== 第二階段：極端關鍵點優化 ===")
    phase2_results = phase1_best.train(
        data="coco-pose.yaml",
        epochs=60,                    # 中期訓練
        imgsz=640,                    # 目標分辨率
        batch=96,                     # 中等批次
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="AdamW",            # 轉回AdamW
        lr0=0.0003,                   # 中等學習率
        lrf=0.00005,                  # 低最終學習率
        momentum=0.937,               # 標準動量
        weight_decay=0.00005,         # 低權重衰減
        warmup_epochs=0.0,            # 無熱身
        box=0.5,                      # 極低框損失權重
        pose=400.0,                   # 極高姿態損失權重
        kobj=160.0,                   # 極高關鍵點對象損失權重
        cls=0.001,                    # 極低分類損失權重
        dfl=0.005,                    # 極低分布焦點損失權重
        cos_lr=False,                 # 關閉餘弦調度
        # one_cycle=True,               # 使用one-cycle學習率
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=40,                  # 中等耐心值
        val=True,                     # 驗證
        plots=True,                   # 圖表
        
        # 最小化數據增強
        hsv_h=0.0,                    # 關閉色調
        hsv_s=0.0,                    # 關閉飽和度
        hsv_v=0.05,                   # 最小亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.0,                # 關閉平移
        scale=0.05,                   # 最小縮放
        fliplr=0.5,                   # 保留水平翻轉
        mosaic=0.0,                   # 關閉馬賽克
        mixup=0.0,                    # 關閉mixup
        copy_paste=0.0,               # 關閉複製粘貼
        rect=True,                    # 使用矩形訓練
        
        # 特殊設置 - 激進提高關鍵點精度
        single_cls=True,              # 單類別訓練，專注於人體姿態
        
        project="runs/pose",          # 項目名稱
        name="super_phase5_2",        # 訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
    # 獲取第二階段最佳權重
    phase2_best = YOLO("runs/pose/super_phase5_2/weights/best.pt")
    
    # 第三階段：高分辨率知識提取
    print("\n=== 第三階段：高分辨率知識提取 ===")
    phase3_results = phase2_best.train(
        data="coco-pose.yaml",
        epochs=30,                    # 短期訓練
        imgsz=832,                    # 更高分辨率
        batch=32,                     # 小批次適應高分辨率
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="AdamW",            # AdamW優化器
        lr0=0.0001,                   # 低學習率
        lrf=0.00001,                  # 極低最終學習率
        momentum=0.9,                 # 中等動量
        weight_decay=0.000005,        # 極低權重衰減
        warmup_epochs=3.0,            # 短暫熱身
        box=0.2,                      # 極低框損失權重
        pose=500.0,                   # 極極高姿態損失權重
        kobj=200.0,                   # 極極高關鍵點對象損失權重
        cls=0.0005,                   # 極低分類損失權重
        dfl=0.001,                    # 極低分布焦點損失權重
        cos_lr=True,                  # 重啟餘弦調度
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=20,                  # 低耐心值
        val=True,                     # 驗證
        plots=True,                   # 圖表
        multi_scale=True,             # 多尺度訓練
        rect=False,                   # 關閉矩形訓練以適應多尺度
        
        # 無數據增強
        hsv_h=0.0,                    # 關閉色調
        hsv_s=0.0,                    # 關閉飽和度
        hsv_v=0.01,                   # 極小亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.0,                # 關閉平移
        scale=0.0,                    # 關閉縮放
        fliplr=0.5,                   # 保留水平翻轉
        mosaic=0.0,                   # 關閉馬賽克
        mixup=0.0,                    # 關閉mixup
        copy_paste=0.0,               # 關閉複製粘貼
        
        # 推理參數優化
        iou=0.75,                     # 高IoU閾值
        conf=0.0005,                  # 極低置信度閾值
        max_det=300,                  # 增加最大檢測數
        
        project="runs/pose",          # 項目名稱
        name="super_phase5_3",        # 訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
    # 獲取第三階段最佳權重
    phase3_best = YOLO("runs/pose/super_phase5_3/weights/best.pt")
    
    # 第四階段：返回640，最終極致微調
    print("\n=== 第四階段：返回640，最終極致微調 ===")
    phase4_results = phase3_best.train(
        data="coco-pose.yaml",
        epochs=25,                    # 短期訓練
        imgsz=640,                    # 回到目標分辨率
        batch=128,                    # 較大批次
        save_period=1,                # 每個epoch保存
        cache="disk",                 # 磁盤緩存
        optimizer="AdamW",            # AdamW優化器
        lr0=0.000005,                 # 極極低學習率
        lrf=0.0000001,                # 極極低最終學習率
        momentum=0.8,                 # 較低動量，精確優化
        weight_decay=0.0,             # 無權重衰減
        warmup_epochs=0.0,            # 無熱身
        box=0.1,                      # 極低框損失權重
        pose=700.0,                   # 極極極高姿態損失權重
        kobj=300.0,                   # 極極極高關鍵點對象損失權重
        cls=0.0001,                   # 極極低分類損失權重
        dfl=0.0001,                   # 極極低分布焦點損失權重
        cos_lr=False,                 # 關閉餘弦調度
        amp=True,                     # 混合精度訓練
        device="0,1,2,3",             # GPU
        overlap_mask=True,            # 重疊口罩
        patience=15,                  # 極低耐心值
        val=True,                     # 驗證
        plots=True,                   # 圖表
        rect=True,                    # 使用矩形訓練
        
        # 完全關閉數據增強
        hsv_h=0.0,                    # 關閉色調
        hsv_s=0.0,                    # 關閉飽和度
        hsv_v=0.0,                    # 關閉亮度
        degrees=0.0,                  # 關閉旋轉
        translate=0.0,                # 關閉平移
        scale=0.0,                    # 關閉縮放
        fliplr=0.5,                   # 保留水平翻轉 
        mosaic=0.0,                   # 關閉馬賽克
        mixup=0.0,                    # 關閉mixup
        copy_paste=0.0,               # 關閉複製粘貼
        
        # 最終特殊參數
        iou=0.8,                      # 極高IoU閾值
        conf=0.0001,                  # 極極低置信度閾值
        max_det=500,                  # 極高最大檢測數
        
        project="runs/pose",          # 項目名稱
        name="super_phase5_4",        # 訓練名稱
        exist_ok=True                 # 覆蓋已有目錄
    )
    
    # 評估最終模型
    print("\n=== 最終模型評估 ===")
    # 獲取最佳權重
    final_best = YOLO("runs/pose/super_phase5_4/weights/best.pt")
    
    # 多分辨率評估
    resolutions = [640, 672, 704, 736, 768, 800, 832]
    for res in resolutions:
        print(f"評估{res}分辨率性能...")
        try:
            val_results = final_best.val(
                data="coco-pose.yaml", 
                imgsz=res,
                conf=0.001,  # 低置信度閾值
                iou=0.7,     # 高IoU閾值
                max_det=300  # 高檢測數
            )
            if is_main_process():
                print(f"分辨率 {res}x{res} mAP50-95: {val_results.box.map}")
        except Exception as e:
            print(f"分辨率 {res} 評估失敗: {e}")
    
    if is_main_process():
        print("\n=== 訓練與評估完成 ===")
        print("建議：")
        print("1. 對比各分辨率結果，選擇最佳推理分辨率")
        print("2. 考慮使用模型集成 (多個checkpoint平均)")
        print("3. 最終部署時使用量化技術加速推理")
        print("4. 推理時使用 conf=0.15, iou=0.7 獲得更好的精確度/召回率平衡")

if __name__ == "__main__":
    main() 