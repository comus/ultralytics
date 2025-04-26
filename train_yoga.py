from ultralytics import YOLO

# Load the best model from previous training
model = YOLO("/root/autodl-tmp/withcloud/ultralytics/runs/pose/train14/weights/best.pt")  # Using best weights from previous training

# Train the model with yoga dataset fine-tuning
results = model.train(
    data="yoga82.yaml",
    epochs=40,                # 適合小型但高質量的數據集
    imgsz=1280,               # 高解析度有助於精確姿勢識別
    batch=48,                 # 增加批次大小以充分利用4個GPU
    save_period=1,            # 每個epoch保存
    cache="disk",             # 使用磁盤緩存
    optimizer="AdamW",        # 使用AdamW優化器
    lr0=0.00002,              # 調整學習率以適應更大的批次大小
    lrf=0.01,                 # 標準最終學習率因子
    cos_lr=True,              # 餘弦學習率調度
    warmup_epochs=3.0,        # 增加熱身時間以穩定多GPU訓練
    device="0,1,2,3",         # 使用全部四個GPU
    patience=10,              # 適當的早停耐心值
    box=5.0,                  # 適當的框損失權重
    cls=0.3,                  # 適當的分類損失權重
    dfl=1.5,                  # 保持不變
    pose=30.0,                # 高姿態損失權重
    kobj=6.0,                 # 高關鍵點可見性權重
    
    # 數據增強設置，符合官方建議
    hsv_h=0.015,              # 色調變化 (0.0-1.0)
    hsv_s=0.2,                # 飽和度變化 (0.0-1.0)
    hsv_v=0.2,                # 亮度變化 (0.0-1.0)
    degrees=5.0,              # 旋轉角度 (0.0-180.0)
    translate=0.1,            # 平移範圍 (0.0-1.0)
    scale=0.2,                # 縮放範圍 (>=0.0)
    fliplr=0.5,               # 水平翻轉概率 (0.0-1.0)
    perspective=0.0005,       # 透視變換 (0.0-0.001)
    mosaic=0.2,               # 馬賽克增強 (0.0-1.0)
    mixup=0.1,                # 混合增強 (0.0-1.0)
    copy_paste=0.0,           # 無複製粘貼
    
    # 防止災難性遺忘和過擬合
    overlap_mask=True,        # 使用重疊掩碼
    amp=True,                 # 混合精度訓練，加速多GPU訓練
    val=True,                 # 每個epoch驗證
    freeze=5,                 # 減少凍結層數
    close_mosaic=10,          # 在後期關閉馬賽克增強
    weight_decay=0.0005,      # 權重衰減
    dropout=0.1,              # Dropout正則化
    
    # 多GPU訓練加速設置
    sync_bn=True,             # 使用同步批量歸一化，適合多GPU
    nbs=64,                   # 標稱批次大小，用於學習率縮放
    workers=12,               # 數據加載器工作進程數
    
    # 專案設置
    project="yoga_finetune",  # 專案名稱
    name="gde_pose_yoga_hq",  # 標記為高質量數據集實驗
    exist_ok=True             # 覆蓋現有實驗目錄
) 