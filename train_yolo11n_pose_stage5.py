from ultralytics import YOLO

# 加載第四階段最佳模型
model = YOLO("yolo11n-pose/train_stage4/weights/best.pt")

# 第五階段訓練參數 - 突破瓶頸階段
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=80,                       # 相對較短的訓練周期
    patience=15,                     # 較短的早停耐心
    batch=32,                        # 降低批次大小
    imgsz=640,                       # 保持640分辨率
    save_period=1,                   # 每個epoch保存
    
    # 優化器和學習率 - 大幅調整
    cos_lr=False,                    # 嘗試使用step學習率
    lr0=0.00001,                     # 極低的初始學習率
    lrf=0.1,                         # 更高的最終學習率比例
    optimizer='AdamW',               # 保持AdamW
    weight_decay=0.001,              # 維持權重衰減
    momentum=0.937,                  
    
    # 關鍵點：損失函數權重大幅調整
    box=2.0,                         # 大幅降低框檢測權重
    pose=35.0,                       # 極大增加姿態損失權重
    kobj=8.0,                        # 大幅提高關鍵點目標性權重
    cls=0.2,                         # 進一步降低分類權重
    
    # 解凍所有層進行微調
    freeze=None,                     # 解凍所有層
    
    # 完全關閉數據增強
    cache="disk",                    
    rect=True,                       # 保持矩形訓練
    
    # 降低批次大小但增加圖像尺寸 - 關鍵突破點
    overlap_mask=True,               
    amp=True,                        
    
    # 項目管理
    project="yolo11n-pose",
    name="train_stage5",
    exist_ok=True
)
