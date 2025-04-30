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
    model = YOLO("/root/autodl-tmp/withcloud/root/autodl-tmp/withcloud/ultralytics/runs/pose/train14/weights/best.pt")

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
        imgsz=640,             # 降低到640解析度
        batch=64,              # 增加批次大小（因為圖像尺寸減小）
        save_period=1,         # 每個epoch保存
        cache="disk",          # 使用磁盤緩存
        optimizer="AdamW",     # 繼續使用AdamW優化器
        lr0=0.00002,           # 略微提高學習率（因為圖像尺寸減小）
        lrf=0.01,              # 保持最終學習率因子
        momentum=0.85,         # 保持動量值
        weight_decay=0.0001,   # 保持權重衰減
        warmup_epochs=2.0,     # 添加短暫熱身
        warmup_momentum=0.8,   # 保持熱身動量
        warmup_bias_lr=0.01,   # 保持熱身偏置學習率
        box=6.0,               # 保持框損失權重
        pose=25.0,             # 保持姿態損失權重
        kobj=7.0,              # 保持關鍵點對象損失權重
        cls=0.2,               # 保持分類損失權重 
        dfl=1.0,               # 保持分布焦點損失權重
        nbs=64,                # 保持標稱批次大小
        cos_lr=True,           # 保持餘弦學習率調度
        close_mosaic=0,        # 保持關閉馬賽克增強
        amp=True,              # 保持混合精度訓練
        device="0,1,2,3",      # 使用全部4個GPU
        dropout=0.2,           # 保持dropout正則化
        overlap_mask=True,     # 保持重疊口罩
        patience=50,           # 保持耐心值
        val=True,              # 確保每個epoch驗證
        freeze=15,             # 保持凍結前15層
        
        # 數據增強參數微調
        hsv_h=0.015,           # 略微增加色調變化
        hsv_s=0.15,            # 略微增加飽和度變化
        hsv_v=0.15,            # 略微增加亮度變化
        degrees=0.0,           # 保持關閉旋轉
        translate=0.05,        # 保持最小平移
        scale=0.1,             # 保持最小縮放
        fliplr=0.5,            # 保持水平翻轉
        mosaic=0.0,            # 保持關閉馬賽克
        mixup=0.0,             # 保持關閉mixup
        copy_paste=0.0         # 保持關閉複製粘貼
    )

if __name__ == "__main__":
    main() 
