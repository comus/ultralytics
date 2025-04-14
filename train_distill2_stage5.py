from ultralytics import YOLO

# 載入第四階段最佳模型
model = YOLO("yolo11n-pose-distill2/train_stage4/weights/best.pt")

# 第五階段訓練 - 專注提升關鍵點精度但保持640尺寸
results = model.train(
    # 基本設置
    data="coco-pose.yaml",          
    epochs=150,                     # 更長的訓練時間
    patience=60,                    # 更大的早停耐心值
    batch=64,                       # 減小批次大小以提高精度
    imgsz=640,                      # 保持標準尺寸
    save_period=1,
    
    # 優化器和學習率
    cos_lr=True,
    lr0=0.00015,                    # 極低的學習率，適合微調
    lrf=0.0005,                     # 最終更低的學習率
    optimizer='AdamW',
    weight_decay=0.0001,            # 降低權重衰減以適應低學習率
    momentum=0.937,
    
    # 損失權重 - 極度強調姿態精度
    box=4.0,                        # 降低框損失權重
    pose=35.0,                      # 大幅增加姿態損失權重
    kobj=6.0,                       # 顯著增加關鍵點目標性權重
    cls=0.4,                        # 輕微降低分類損失
    dfl=1.5,                        # 保持默認DFL權重
    
    # 數據處理和穩定性
    cache="disk",
    rect=True,                      # 保持矩形訓練提高效率
    multi_scale=False,
    amp=True,
    val=True,
    plots=True,
    workers=16,
    
    # 項目管理
    project="yolo11n-pose-distill2",
    name="train_stage5",
    exist_ok=True,
    
    # 蒸餾設置 - 使用pose專家模型
    teacher=YOLO("yolo11m-pose.pt").model,
    distill=4.5,                    # 強化蒸餾權重更專注姿態學習
    loss_function="pose_loss2",
    
    # 凍結層 - 更專注姿態優化
    freeze=2,                       # 輕微凍結，主要讓基礎特徵穩定
)
