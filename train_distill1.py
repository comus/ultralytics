from ultralytics import YOLO

# 載入新模型
model = YOLO("yolo11n-pose.yaml")

# 訓練模型
results = model.train(
    data="coco-pose.yaml",
    epochs=500,                # 500個epochs從頭開始訓練
    imgsz=640,                 # 標準圖片尺寸640
    batch=64,                  # 使用較大batch size充分利用4090顯卡
    cache="disk",              # 使用磁碟快取加速訓練，減少IO瓶頸
    device=0,                  # 使用主GPU
    workers=12,                # CPU有16核，使用12個worker保留一些系統資源
    patience=50,               # 增加耐心值，因為從頭訓練初期可能進展緩慢
    cos_lr=True,               # 使用餘弦學習率調度器
    lr0=0.015,                  # 較高的初始學習率加速收斂
    lrf=0.005,                 # 最終學習率降低到初始值的0.1%
    warmup_epochs=5.0,         # 增加熱身期，從頭訓練需要更長的熱身
    weight_decay=0.0005,       # 標準權重衰減
    close_mosaic=20,           # 在最後15個epoch禁用mosaic增強穩定訓練
    amp=True,                  # 啟用混合精度訓練
    optimizer="AdamW",         # 使用AdamW優化器
    pose=12.0,                 # 姿態損失權重
    kobj=1.5,                  # 關鍵點目標性損失權重
    plots=True,                # 生成訓練圖表
    save_period=1,             # 每個epoch儲存一次
    project="yolo11n-pose-distill1",    # 專案名稱
    name="train",              # 訓練執行名稱
    exist_ok=True,             # 允許覆蓋現有目錄
    
    # 使用蒸餾，從更大的模型學習
    teacher=YOLO("yolo11m-pose.pt").model,  # 使用YOLO 11x作為教師模型
    distill=2.5,               # 蒸餾損失權重
    
    # 額外優化參數
    # multi_scale=True,          # 啟用多尺度訓練增強泛化能力
    freeze=False,              # 不凍結任何層，從頭訓練所有參數
    dropout=0.15,               # 增加dropout減輕過擬合
    nbs=64,                    # 標準batch size，用於損失正規化
    momentum=0.937,            # 動量參數
    val=True,                  # 每個epoch進行驗證
    pretrained=False,          # 確保從頭訓練
)
