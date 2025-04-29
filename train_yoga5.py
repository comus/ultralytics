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

def train_stage1(model):
    """第一階段訓練：使用較大的學習率和強數據增強，快速學習基本特徵"""
    results = model.train(
        data="yoga82.yaml",
        epochs=100,               # 增加訓練週期
        imgsz=640,               
        batch=24,                # 使用較大的批次大小
        save_period=1,           
        cache="disk",            
        optimizer="AdamW",       
        lr0=0.00075,            # 使用較大的學習率
        lrf=0.01,               
        cos_lr=True,            
        warmup_epochs=8.0,      # 較長的熱身期
        device="0,1,2,3",       
        patience=20,            # 增加早停耐心值
        project="train_yoga5",     
        name="train_stage1",     
        exist_ok=True,
        
        # 第一階段使用較高的損失權重
        box=7.0,                
        cls=0.5,                
        dfl=1.5,                
        pose=60.0,              # 較高的姿態損失權重
        kobj=12.0,              # 較高的關鍵點可見性權重
        
        # 強數據增強
        hsv_h=0.015,            
        hsv_s=0.15,             
        hsv_v=0.15,             
        degrees=180.0,           # 較大的旋轉角度
        translate=0.1,          # 較大的平移範圍
        scale=0.25,             # 較大的縮放範圍
        fliplr=0.5,             
        perspective=0.0005,     # 較大的透視變換
        mosaic=0.15,            # 較大的馬賽克增強
        mixup=0.1,              # 較大的混合增強
        copy_paste=0.0,         
        
        # 正則化設置
        overlap_mask=True,      
        amp=True,               
        val=True,               
        freeze=8,               # 適中的凍結層數
        close_mosaic=20,        # 延後關閉馬賽克增強
        weight_decay=0.0001,    # 較大的權重衰減
        dropout=0.05,           # 較大的dropout
        
        # 多GPU訓練設置
        nbs=64,                 
        workers=16,             # 增加工作進程數
    )
    return results

def train_stage2(model):
    """第二階段訓練：使用適中的學習率和數據增強，專注於細節優化"""
    results = model.train(
        data="yoga82.yaml",
        epochs=100,               # 增加訓練週期
        imgsz=640,               
        batch=20,                # 適中的批次大小
        save_period=1,           
        cache="disk",            
        optimizer="AdamW",       
        lr0=0.0005,             # 適中的學習率
        lrf=0.01,               
        cos_lr=True,            
        warmup_epochs=6.0,      # 適中的熱身期
        device="0,1,2,3",       
        patience=20,            # 保持較大的早停耐心值
        project="train_yoga5",     
        name="train_stage2",     
        exist_ok=True,
        
        # 第二階段使用適中的損失權重
        box=7.0,                
        cls=0.5,                
        dfl=1.5,                
        pose=65.0,              # 適中的姿態損失權重
        kobj=13.0,              # 適中的關鍵點可見性權重
        
        # 適中的數據增強
        hsv_h=0.015,            
        hsv_s=0.12,             
        hsv_v=0.12,             
        degrees=90,            # 適中的旋轉角度
        translate=0.08,         # 適中的平移範圍
        scale=0.2,              # 適中的縮放範圍
        fliplr=0.5,             
        perspective=0.0004,     
        mosaic=0.12,            # 適中的馬賽克增強
        mixup=0.08,             # 適中的混合增強
        copy_paste=0.0,         
        
        # 正則化設置
        overlap_mask=True,      
        amp=True,               
        val=True,               
        freeze=6,               # 適度解凍
        close_mosaic=15,        
        weight_decay=0.00008,   # 適中的權重衰減
        dropout=0.04,           # 適中的dropout
        
        # 多GPU訓練設置
        nbs=64,                 
        workers=12,             # 適中的工作進程數
    )
    return results

def train_stage3(model):
    """第三階段訓練：使用較小的學習率和溫和的數據增強，進行精細調整"""
    results = model.train(
        data="yoga82.yaml",
        epochs=100,               # 增加訓練週期
        imgsz=640,               
        batch=16,                # 較小的批次大小
        save_period=1,           
        cache="disk",            
        optimizer="AdamW",       
        lr0=0.0003,             # 較小的學習率
        lrf=0.01,               
        cos_lr=True,            
        warmup_epochs=5.0,      # 較短的熱身期
        device="0,1,2,3",       
        patience=20,            # 保持較大的早停耐心值
        project="train_yoga5",     
        name="train_stage3",     
        exist_ok=True,
        
        # 第三階段使用較高的損失權重
        box=7.0,                
        cls=0.5,                
        dfl=1.5,                
        pose=70.0,              # 較高的姿態損失權重
        kobj=14.0,              # 較高的關鍵點可見性權重
        
        # 溫和的數據增強
        hsv_h=0.015,            
        hsv_s=0.1,              
        hsv_v=0.1,              
        degrees=5.0,            # 較小的旋轉角度
        translate=0.07,         # 較小的平移範圍
        scale=0.15,             # 較小的縮放範圍
        fliplr=0.5,             
        perspective=0.0003,     
        mosaic=0.1,             # 較小的馬賽克增強
        mixup=0.05,             # 較小的混合增強
        copy_paste=0.0,         
        
        # 正則化設置
        overlap_mask=True,      
        amp=True,               
        val=True,               
        freeze=4,               # 較少凍結
        close_mosaic=10,        
        weight_decay=0.00005,   # 較小的權重衰減
        dropout=0.03,           # 較小的dropout
        
        # 多GPU訓練設置
        nbs=64,                 
        workers=8,              # 較少的工作進程數
    )
    return results

def main():    
    # 從最佳權重開始進行精調
    model = YOLO("/root/autodl-tmp/withcloud/train66/weights/best.pt")

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

    # 執行三階段訓練
    print("開始第一階段訓練...")
    results1 = train_stage1(model)
    
    # 使用第一階段的最佳權重
    model = YOLO("train_yoga5/train_stage1/weights/best.pt")
    print("開始第二階段訓練...")
    results2 = train_stage2(model)
    
    # 使用第二階段的最佳權重
    model = YOLO("train_yoga5/train_stage2/weights/best.pt")
    print("開始第三階段訓練...")
    results3 = train_stage3(model)

if __name__ == "__main__":
    main() 
