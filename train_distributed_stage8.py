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

    # 訓練參數設置
    results = model.train(
        data="coco-pose.yaml",
        epochs=100,             # 保持訓練週期
        imgsz=640,             # 使用640解析度
        batch=96,              # 適中的批次大小，確保每張圖片有足夠的計算資源
        save_period=1,         # 每個epoch保存
        cache="disk",          # 使用磁盤緩存
        optimizer="AdamW",     # 使用AdamW優化器
        lr0=0.00005,           # 使用較小的學習率以保持穩定性
        lrf=0.01,              # 標準最終學習率因子
        momentum=0.937,        # 使用官方推薦的動量值
        weight_decay=0.0005,   # 使用官方推薦的權重衰減
        warmup_epochs=5.0,     # 增加熱身週期，幫助模型適應新的圖像尺寸
        warmup_momentum=0.8,   # 使用官方推薦的熱身動量
        warmup_bias_lr=0.1,    # 使用官方推薦的熱身偏置學習率
        box=7.5,               # 使用官方推薦的框損失權重
        pose=20.0,             # 增加姿態損失權重，因為這是主要目標
        kobj=4.0,              # 增加關鍵點對象損失權重
        cls=0.5,               # 使用官方推薦的分類損失權重
        dfl=1.5,               # 使用官方推薦的分布焦點損失權重
        nbs=64,                # 使用官方推薦的標稱批次大小
        cos_lr=True,           # 啟用餘弦學習率調度
        close_mosaic=10,       # 最後10個epoch關閉馬賽克增強
        amp=True,              # 啟用混合精度訓練
        device="0,1,2,3",      # 使用全部4個GPU
        dropout=0.1,           # 適度的dropout以防止過擬合
        overlap_mask=True,     # 啟用重疊口罩
        patience=50,           # 保持耐心值
        val=True,              # 確保每個epoch驗證
        freeze=0,              # 不凍結任何層，充分利用預訓練模型
        
        # 數據增強參數優化
        hsv_h=0.015,           # 使用官方推薦的色調變化
        hsv_s=0.7,             # 使用官方推薦的飽和度變化
        hsv_v=0.4,             # 使用官方推薦的亮度變化
        degrees=0.0,           # 關閉旋轉（因為姿態估計對旋轉敏感）
        translate=0.1,         # 使用官方推薦的平移範圍
        scale=0.5,             # 使用官方推薦的縮放範圍
        fliplr=0.5,            # 使用官方推薦的水平翻轉
        mosaic=1.0,            # 啟用馬賽克增強
        mixup=0.0,             # 關閉mixup（因為姿態估計對mixup敏感）
        copy_paste=0.0         # 關閉複製粘貼
    )

if __name__ == "__main__":
    main() 
