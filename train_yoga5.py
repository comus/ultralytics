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
    """第一階段訓練：凍結大部分層，使用較小的學習率和溫和的數據增強"""
    results = model.train(
        data="yoga82.yaml",
        epochs=50,                # 第一階段訓練50個epoch
        imgsz=640,               
        batch=16,                
        save_period=1,           
        cache="disk",            
        optimizer="AdamW",       
        lr0=0.0003,             # 較小的學習率
        lrf=0.01,               
        cos_lr=True,            
        warmup_epochs=5.0,       # 增加熱身期
        device="0,1,2,3",       
        patience=15,             # 增加早停耐心值
        project="runs/pose",     # 設置項目目錄
        name="train_stage1",     # 設置階段名稱
        
        # 第一階段使用適中的損失權重
        box=7.0,                
        cls=0.5,                
        dfl=1.5,                
        pose=50.0,              # 適中的姿態損失權重
        kobj=10.0,              # 適中的關鍵點可見性權重
        
        # 溫和的數據增強
        hsv_h=0.015,            
        hsv_s=0.1,              
        hsv_v=0.1,              
        degrees=5.0,            # 適中的旋轉角度
        translate=0.07,         # 適中的平移範圍
        scale=0.15,             # 適中的縮放範圍
        fliplr=0.5,             
        perspective=0.0003,     
        mosaic=0.1,             # 適中的馬賽克增強
        mixup=0.05,             # 適中的混合增強
        copy_paste=0.0,         
        
        # 正則化設置
        overlap_mask=True,      
        amp=True,               
        val=True,               
        freeze=8,               # 適中的凍結層數
        close_mosaic=15,        
        weight_decay=0.00005,   
        dropout=0.03,           
        
        # 多GPU訓練設置
        nbs=64,                 
        workers=8,              
    )
    return results

def train_stage2(model):
    """第二階段訓練：解凍部分層，使用適中的學習率和數據增強"""
    results = model.train(
        data="yoga82.yaml",
        epochs=50,                # 第二階段訓練50個epoch
        imgsz=640,               
        batch=16,                
        save_period=1,           
        cache="disk",            
        optimizer="AdamW",       
        lr0=0.0003,             # 保持學習率
        lrf=0.01,               
        cos_lr=True,            
        warmup_epochs=5.0,       # 保持較長的熱身期
        device="0,1,2,3",       
        patience=15,             # 保持較大的早停耐心值
        project="runs/pose",     # 設置項目目錄
        name="train_stage2",     # 設置階段名稱
        
        # 第二階段使用適中的損失權重
        box=7.0,                
        cls=0.5,                
        dfl=1.5,                
        pose=55.0,              # 適中的姿態損失權重
        kobj=11.0,              # 適中的關鍵點可見性權重
        
        # 適中的數據增強
        hsv_h=0.015,            
        hsv_s=0.1,              
        hsv_v=0.1,              
        degrees=5.0,            # 保持旋轉角度
        translate=0.07,         # 保持平移範圍
        scale=0.15,             # 保持縮放範圍
        fliplr=0.5,             
        perspective=0.0003,     
        mosaic=0.1,             # 保持馬賽克增強
        mixup=0.05,             # 保持混合增強
        copy_paste=0.0,         
        
        # 正則化設置
        overlap_mask=True,      
        amp=True,               
        val=True,               
        freeze=6,               # 適度解凍
        close_mosaic=15,        
        weight_decay=0.00005,   
        dropout=0.03,           
        
        # 多GPU訓練設置
        nbs=64,                 
        workers=8,              
    )
    return results

def train_stage3(model):
    """第三階段訓練：解凍更多層，使用較大的學習率和較強的數據增強"""
    results = model.train(
        data="yoga82.yaml",
        epochs=50,                # 第三階段訓練50個epoch
        imgsz=640,               
        batch=16,                
        save_period=1,           
        cache="disk",            
        optimizer="AdamW",       
        lr0=0.0003,             # 保持學習率
        lrf=0.01,               
        cos_lr=True,            
        warmup_epochs=5.0,       # 保持較長的熱身期
        device="0,1,2,3",       
        patience=15,             # 保持較大的早停耐心值
        project="runs/pose",     # 設置項目目錄
        name="train_stage3",     # 設置階段名稱
        
        # 第三階段使用適中的損失權重
        box=7.0,                
        cls=0.5,                
        dfl=1.5,                
        pose=60.0,              # 適中的姿態損失權重
        kobj=12.0,              # 適中的關鍵點可見性權重
        
        # 適中的數據增強
        hsv_h=0.015,            
        hsv_s=0.1,              
        hsv_v=0.1,              
        degrees=5.0,            # 保持旋轉角度
        translate=0.07,         # 保持平移範圍
        scale=0.15,             # 保持縮放範圍
        fliplr=0.5,             
        perspective=0.0003,     
        mosaic=0.1,             # 保持馬賽克增強
        mixup=0.05,             # 保持混合增強
        copy_paste=0.0,         
        
        # 正則化設置
        overlap_mask=True,      
        amp=True,               
        val=True,               
        freeze=4,               # 適度解凍
        close_mosaic=15,        
        weight_decay=0.00005,   
        dropout=0.03,           
        
        # 多GPU訓練設置
        nbs=64,                 
        workers=8,              
    )
    return results

def main():    
    # 從最佳權重開始進行精調
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

    # 執行三階段訓練
    print("開始第一階段訓練...")
    results1 = train_stage1(model)
    
    # 使用第一階段的最佳權重
    model = YOLO("runs/pose/train_stage1/weights/best.pt")
    print("開始第二階段訓練...")
    results2 = train_stage2(model)
    
    # 使用第二階段的最佳權重
    model = YOLO("runs/pose/train_stage2/weights/best.pt")
    print("開始第三階段訓練...")
    results3 = train_stage3(model)

if __name__ == "__main__":
    main() 
