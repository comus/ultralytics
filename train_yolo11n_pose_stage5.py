from ultralytics import YOLO

# 加載第四階段最佳模型（最好的基礎）
model = YOLO("yolo11n-pose/train_stage4/weights/best.pt")

# 平衡優化策略
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=100,                     
    patience=20,                    
    batch=32,                        # 回到中等批次大小
    imgsz=640,                      
    save_period=1,                  
    
    # 更溫和的優化器設置
    optimizer='SGD',              
    cos_lr=True,                    
    lr0=0.0005,                      # 適中學習率
    lrf=0.01,                        # 更溫和的最終學習率
    weight_decay=0.0005,             # 標準權重衰減
    
    # 更平衡的損失權重
    box=5.0,                         # 適當的框損失權重
    cls=0.3,                         # 適當的分類損失權重
    dfl=1.0,                         # 標準分布焦點損失
    pose=25.0,                       # 高但不極端的姿態損失
    kobj=5.0,                        # 高但不極端的關鍵點目標性
    
    # 更溫和的數據擴充
    mosaic=1.0,                     
    mixup=0.0,                       # 關閉mixup
    copy_paste=0.0,                 
    degrees=0.0,                    
    translate=0.1,                   # 中等平移
    scale=0.1,                       # 中等縮放
    shear=0.0,                      
    perspective=0.0001,              # 極輕微透視變換
    fliplr=0.5,                     
    
    # 適當的正則化
    label_smoothing=0.02,            # 極輕微標籤平滑
    close_mosaic=10,                
    
    # 選擇性凍結（更溫和）
    freeze=[0, 1],                   # 只凍結前兩層
    
    # 標準批次設置
    nbs=64,                         
    
    # 訓練效率設置
    cache="disk",
    rect=False,                     
    amp=True,                       
    
    # 沒有dropout
    dropout=0.0,                    
    
    # 關鍵區別：多尺度訓練
    multi_scale=True,                # 啟用多尺度訓練
    
    # 項目管理
    project="yolo11n-pose",
    name="train_stage5",
    exist_ok=True
)