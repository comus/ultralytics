from ultralytics import YOLO

# 加載第四階段最佳模型
model = YOLO("yolo11n-pose/train_stage4/weights/best.pt")

# 最終極致優化策略
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=150,                      # 更長的訓練周期
    patience=30,                     # 更長的早停耐心
    batch=64,                        # 更大的批次以提高統計穩定性
    imgsz=640,                       # 固定640分辨率
    save_period=1,                   # 每epoch保存
    
    # 優化器與學習率策略
    optimizer='AdamW',               # AdamW通常在微調階段表現出色
    cos_lr=True,                     # 使用余弦學習率調度
    lr0=0.0008,                      # 適中的學習率
    lrf=0.001,                       # 較低的最終學習率比例
    weight_decay=0.001,              # 適當增加權重衰減進行正則化
    momentum=0.937,                  # 標準動量
    
    # 關鍵：損失函數權重
    box=1.0,                         # 極低的框損失權重
    cls=0.05,                        # 極低的分類損失權重
    dfl=0.2,                         # 低分佈焦點損失權重
    pose=60.0,                       # 極高的姿態損失權重
    kobj=20.0,                       # 極高的關鍵點目標性權重
    
    # 數據擴充策略
    mosaic=1.0,                      # 充分利用mosaic增強
    mixup=0.1,                       # 輕微mixup增強
    copy_paste=0.0,                  # 關閉copy-paste
    degrees=0.0,                     # 關閉旋轉（避免破壞關鍵點位置）
    translate=0.02,                  # 極輕微平移
    scale=0.05,                      # 極輕微縮放
    shear=0.0,                       # 關閉剪切
    perspective=0.0,                 # 關閉透視變換
    flipud=0.0,                      # 關閉上下翻轉
    fliplr=0.5,                      # 保留水平翻轉（適合人體姿態）
    hsv_h=0.0,                       # 關閉色調變化
    hsv_s=0.0,                       # 關閉飽和度變化
    hsv_v=0.0,                       # 關閉亮度變化
    
    # 關鍵優化技巧
    label_smoothing=0.05,            # 輕微標籤平滑提高泛化能力
    close_mosaic=15,                 # 最後15個epoch關閉mosaic
    
    # 選擇性凍結層
    freeze=[0, 1, 2, 3, 4, 5],       # 凍結前6層保持主幹特徵穩定
    
    # 梯度累積以增大等效批次大小
    nbs=128,                         # 更大的標稱批次大小
    
    # 學習率預熱調整
    warmup_epochs=0,                 # 關閉預熱，因為我們從已訓練的模型開始
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # 訓練效率設置
    overlap_mask=True,              
    rect=False,                      # 關閉矩形訓練以增加關鍵點位置隨機性
    cache=True,                      # 啟用緩存加速訓練
    amp=True,                        # 啟用混合精度
    
    # 數據採樣策略
    fraction=0.95,                   # 使用95%數據，過濾可能存在問題的樣本
    
    # 啟用dropout以增強泛化能力
    dropout=0.02,                    # 輕微dropout
    
    # 加入驗證和可視化
    val=True,
    plots=True,
    
    # 項目管理
    project="yolo11n-pose",
    name="train_stage5",
    exist_ok=True
)
