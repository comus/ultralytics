from ultralytics import YOLO

# 請確認 tinypose.yaml 與 coco-pose.yaml 已就緒
model = YOLO("tinypose.yaml")

results = model.train(
    data="coco-pose.yaml",        
    epochs=300,                   # 1) 延長至 300 epoch
    batch=32,                     
    imgsz=640,                    
    cache="disk",                 
    device=0,                     
    workers=8,                    
    project="tinypose_exp",       
    name="from_scratch_300e",     # 2) 改一個新的名稱
    pretrained=False,             
    optimizer="AdamW",            
    lr0=0.001,                    
    lrf=0.01,                     
    momentum=0.937,               
    weight_decay=0.0005,          
    warmup_epochs=3.0,            
    warmup_momentum=0.8,          
    warmup_bias_lr=0.1,           
    box=7.5,                      
    cls=0.5,                      
    pose=12.0,                    
    kobj=2.0,                     
    dfl=1.5,                      
    patience=50,                  # 3) 提高 early-stop 的耐心度
    seed=42,                      
    deterministic=True,           
    amp=True,                     
    close_mosaic=10,             # 4) 可視需求考慮改大一些，如 20
    resume=False,                 
)
