from ultralytics import YOLO

# 加載第三階段最佳模型
model = YOLO("yolo11n-pose/train_stage3/weights/best.pt")

# 第四階段訓練參數 - 高精度微調階段（固定640圖像尺寸）
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=100,                      # 足夠的訓練週期
    patience=25,                     # 適當的耐心值
    batch=128,                       # 使用較大批次以提高訓練效率
    imgsz=640,                       # 保持640圖像尺寸
    save_period=1,                   # 每個epoch保存一次
    
    # 優化器和學習率
    cos_lr=True,                     
    lr0=0.0002,                      # 極低的學習率進行精細調整
    lrf=0.005,                       # 更低的最終學習率比例
    optimizer='AdamW',               
    weight_decay=0.0015,             # 增加權重衰減防止過擬合
    momentum=0.937,                  
    
    # 損失權重 - 更加專注於姿態精度
    box=4.0,                         # 降低框損失權重
    pose=25.0,                       # 大幅增加姿態損失權重
    kobj=5.0,                        # 增加關鍵點目標性權重
    cls=0.3,                         # 降低分類損失權重
    
    # 效率與穩定性
    cache="disk",                    
    rect=True,                       # 保持矩形訓練
    overlap_mask=True,               
    amp=True,                        
    val=True,                        
    plots=True,                      
    workers=16,                      
    
    # 特殊精調策略
    freeze=[0, 1, 2, 3, 4, 5],       # 凍結早期層
    
    # 數據篩選與處理策略
    fraction=0.95,                   # 使用數據集的95%，過濾掉部分可能存在問題的數據
    
    # 項目管理
    project="yolo11n-pose",
    name="train_stage4",
    exist_ok=True
)
