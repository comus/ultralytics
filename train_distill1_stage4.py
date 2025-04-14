from ultralytics import YOLO

# 載入第三階段最佳模型
model = YOLO("yolo11n-pose-distill1/train_stage3/weights/best.pt")

# 第四階段訓練 - 僅使用官方支持的參數
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=100,                     # 充分的訓練時間
    patience=40,                    # 提高早停耐心值
    batch=128,                      # 保持大批次
    imgsz=640,                      # 標準輸入分辨率
    save_period=1,                  # 每個epoch保存
    
    # 優化器和學習率
    cos_lr=True,                    # 使用餘弦學習率調度
    lr0=0.0003,                     # 極低初始學習率
    lrf=0.001,                      # 最終學習率為初始的0.1%
    optimizer='AdamW',              # 使用AdamW優化器
    weight_decay=0.00025,           # 降低權重衰減
    momentum=0.937,                 # 保持默認動量
    
    # 損失權重
    box=5.0,                        # 降低框損失權重
    pose=20.0,                      # 增加姿態損失權重
    kobj=3.5,                       # 增加關鍵點目標性權重
    cls=0.5,                        # 維持默認分類損失權重
    dfl=1.5,                        # 維持默認DFL權重
    
    # 數據增強
    close_mosaic=0,                 # 完全關閉mosaic
    
    # 效率與穩定性
    cache="disk",                   # 使用磁盤緩存
    rect=True,                      # 啟用矩形訓練
    multi_scale=False,              # 關閉多尺度訓練
    amp=True,                       # 使用混合精度訓練
    val=True,                       # 每個epoch進行驗證
    plots=True,                     # 生成訓練圖表
    workers=16,                     # 數據加載線程數
    
    # 項目管理
    project="yolo11n-pose-distill1",
    name="train_stage4",
    exist_ok=True,
    
    # 蒸餾設置
    teacher=YOLO("yolo11m-pose.pt").model,
    distill=2.0,                    # 降低蒸餾權重
    
    # 正則化
    dropout=0.02,                   # 輕微正則化

    # 特殊精調策略
    freeze=[0, 1, 2, 3, 4, 5],       # 凍結早期層
    
    # 數據篩選與處理策略
    fraction=0.95,                   # 使用數據集的95%，過濾掉部分可能存在問題的數據
)
