from ultralytics import YOLO

model = YOLO("yolo11n-pose-distill1/train_stage2/weights/best.pt")

# 第三階段訓練參數 - 精細調整階段，嚴格按照官方支持的參數
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=150,                     # 增加訓練時間以接近官方基準
    patience=50,                    # 適當增加早停耐心值
    batch=64,                      # 大批次提高訓練穩定性
    imgsz=640,                      # 標準輸入分辨率
    save_period=1,                  # 每5個epoch保存一次
    
    # 優化器和學習率
    cos_lr=True,                    # 使用餘弦學習率調度
    lr0=0.0007,                     # 適合微調的學習率
    lrf=0.01,                       # 最終學習率為初始的1%
    optimizer='AdamW',              # 使用AdamW優化器
    weight_decay=0.0005,            # 標準權重衰減
    momentum=0.937,                 # 標準動量值
    warmup_epochs=0,                # 無需預熱
    
    # 損失權重 - 專注提升姿態精度
    box=5.5,                        # 降低框損失權重(默認7.5)
    pose=18.0,                      # 增加姿態損失權重(默認12.0)
    kobj=3.0,                       # 增加關鍵點目標性權重(默認2.0)
    dfl=1.5,                        # 保持分佈焦點損失默認值
    
    # 數據增強設置
    mosaic=0,                       # 關閉mosaic增強
    mixup=0,                        # 關閉mixup增強
    copy_paste=0,                   # 關閉copy_paste增強
    hsv_h=0.01,                     # 最小化色調增強
    hsv_s=0.1,                      # 最小化飽和度增強
    hsv_v=0.1,                      # 最小化亮度增強
    translate=0.1,                  # 保留適度平移增強
    scale=0.2,                      # 保留適度縮放增強
    fliplr=0.5,                     # 保留水平翻轉
    close_mosaic=0,                 # 完全關閉mosaic
    
    # 效率與穩定性
    cache="disk",                   # 使用磁盤緩存
    rect=True,                      # 啟用矩形訓練
    multi_scale=False,              # 關閉多尺度訓練
    amp=True,                       # 使用混合精度訓練
    overlap_mask=True,              # 確保關鍵點遮罩正確處理
    val=True,                       # 每個epoch進行驗證
    plots=True,                     # 生成訓練圖表
    workers=16,                     # 數據加載線程數
    
    # 項目管理
    project="yolo11n-pose-distill1",
    name="train_stage3",
    exist_ok=True,

    # 蒸餾設置
    teacher=YOLO("yolo11m-pose.pt").model,
    distill=3.5,
)
