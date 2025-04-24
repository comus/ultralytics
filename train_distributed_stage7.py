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
        epochs=50,                 # 減少訓練週期，更適合精調階段
        imgsz=1536,                # 增加解析度以捕捉更多細節
        batch=48,                  # 因為更高解析度，減小批次大小
        save_period=1,             # 每個epoch保存一次
        cache="disk",              # 使用磁盤緩存
        optimizer="AdamW",         # 繼續使用AdamW優化器
        lr0=0.00001,               # 更低的學習率進行精細調整
        lrf=0.002,                 # 更低的最終學習率比例
        cos_lr=True,               # 餘弦學習率調整
        warmup_epochs=2.0,         # 縮短熱身時間，與總週期相適應
        device="0,1,2,3",          # 使用所有GPU
        patience=25,               # 調整耐心值，與總週期相適應
        box=15.0,                  # 增加框損失權重
        pose=30.0,                 # 大幅增加姿態損失權重
        kobj=8.0,                  # 增加關鍵點對象損失權重
        close_mosaic=0,            # 完全關閉馬賽克增強
        amp=True,                  # 自動混合精度
        nbs=48,                    # 標稱批次大小
        overlap_mask=True,         # 掩碼重疊
        weight_decay=0.001,        # 增加權重衰減以提高泛化能力
        dropout=0.03,              # 增加dropout以防過擬合
        val=True,                  # 進行驗證
        plots=True,                # 生成訓練圖表
        degrees=0.0,               # 關閉旋轉增強
        translate=0.1,             # 保留平移增強
        scale=0.1,                 # 保留縮放增強
        hsv_h=0.015,               # 輕微色調增強
        hsv_s=0.2,                 # 適當的飽和度增強
        hsv_v=0.2                  # 適當的亮度增強
    )

if __name__ == "__main__":
    main() 