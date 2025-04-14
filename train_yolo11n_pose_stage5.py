from ultralytics import YOLO

# 加載第四階段最佳模型
model = YOLO("yolo11n-pose/train_stage4/weights/best.pt")

# 第六階段訓練 - 關鍵點精度突破策略
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=50,                       # 較短的訓練周期
    patience=15,                     # 適當的早停耐心
    batch=16,                        # 極小批次大小提高梯度精度
    imgsz=640,                       # 維持640分辨率
    save_period=1,                   # 每個epoch保存
    
    # 優化器和學習率 - 使用高性能優化器
    optimizer='SGD',                # 嘗試Lion優化器或SGD
    cos_lr=True,                     # 恢復余弦學習率
    lr0=0.0004,                      # 中等學習率
    lrf=0.001,                       # 較低的最終學習率比例
    weight_decay=0.0002,             # 調整權重衰減
    
    # 關鍵創新：損失函數權重重新調整
    box=2.0,                         # 極低的框損失權重
    cls=0.1,                         # 極低的分類損失權重
    pose=30.0,                       # 高姿態損失權重
    kobj=10.0,                       # 極高的關鍵點目標性權重
    dfl=0.5,                         # 降低分佈焦點損失權重
    
    # 引入特殊數據處理技術
    perspective=0.0015,              # 輕微透視變換
    degrees=0,                       # 關閉旋轉
    shear=0.0,                       # 關閉剪切
    
    # 重要：使用困難樣本挖掘
    nbs=32,                          # 更小的標稱批次大小
    
    # 針對性模型調整
    dropout=0.05,                    # 輕微的dropout正則化
    
    # 高級訓練設置
    cache="disk",
    rect=True,                       # 保持矩形訓練
    val=True,
    plots=True,
    
    # 項目管理
    project="yolo11n-pose",
    name="train_stage5",
    exist_ok=True
)
