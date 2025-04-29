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
        data="yoga82.yaml",
        epochs=150,               # 增加訓練週期
        imgsz=640,               
        batch=24,                # 減小批次大小
        save_period=1,           
        cache="disk",            
        optimizer="AdamW",       
        lr0=0.00075,            # 降低學習率
        lrf=0.01,               
        cos_lr=True,            
        warmup_epochs=8.0,      # 增加熱身時間
        device="0,1,2,3",       
        patience=20,            # 增加早停耐心值
        
        # 損失權重調整
        box=7.0,                
        cls=0.5,                
        dfl=1.5,                
        pose=60.0,              # 提高姿態損失權重
        kobj=12.0,              # 提高關鍵點可見性權重
        
        # 優化的數據增強
        hsv_h=0.015,            # 稍微增加色調變化
        hsv_s=0.15,             # 增加飽和度變化
        hsv_v=0.15,             # 增加亮度變化
        degrees=10.0,           # 增加旋轉角度
        translate=0.1,          # 增加平移範圍
        scale=0.25,             # 增加縮放範圍
        fliplr=0.5,             
        perspective=0.0005,     # 稍微增加透視變換
        mosaic=0.15,            # 增加馬賽克增強
        mixup=0.1,              # 增加混合增強
        copy_paste=0.0,         
        
        # 訓練穩定性設置
        overlap_mask=True,      
        amp=True,               
        val=True,               
        freeze=8,               # 減少凍結層數，讓模型更好地適應新任務
        close_mosaic=20,        # 延後關閉馬賽克增強
        weight_decay=0.0001,    
        dropout=0.05,           
        
        # 多GPU訓練設置
        nbs=64,                 
        workers=16,             # 增加工作進程數
    )

if __name__ == "__main__":
    main() 
