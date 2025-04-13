from ultralytics import YOLO

model = YOLO("yolo11n-pose/weights/last.pt")

# 以30 epoch的checkpoint為基礎的二次訓練參數
results = model.train(
    data="coco-pose.yaml",          
    epochs=200,                     # 減少總epoch數
    patience=20,                    # 減少早停耐心值
    batch=64,                       # 增加批次大小以利用GPU性能
    cos_lr=True,                    # 保留餘弦學習率
    lrf=0.01,                       # 增加最終學習率比例
    lr0=0.001,                      # 降低初始學習率，已有基礎模型
    warmup_epochs=1,                # 減少預熱期
    save_period=5,                  # 每5個epoch保存一次
    cache="disk",                    # 如果內存足夠，改用RAM緩存
    close_mosaic=10,                # 最後10個epoch關閉mosaic
    plots=True,                    
    mosaic=0.8,                     # 稍微減少mosaic強度
    mixup=0.05,                     # 降低mixup強度
    copy_paste=0.05,                # 降低copy_paste強度
    hsv_h=0.01,                     # 降低色調增強
    hsv_s=0.5,                      # 降低飽和度增強
    hsv_v=0.3,                      # 降低亮度增強
    translate=0.05,                 # 降低平移增強
    scale=0.3,                      # 降低縮放增強
    fliplr=0.5,                     # 保持水平翻轉
    multi_scale=True,
    amp=True,                       # 確保啟用混合精度訓練
    project="yolo11n-pose",
    name="train_stage2",
    exist_ok=True,
    optimizer='AdamW',              # 嘗試使用AdamW優化器
    weight_decay=0.001,             # 增加權重衰減以提高泛化能力
    workers=16                      # 增加工作線程數
)
