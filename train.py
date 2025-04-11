from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo11n-pose.pt")

# 极端保守策略：完全冻结特征提取器，仅训练最后的输出层
results = model.train(
  data="coco-pose.yaml",
  teacher=YOLO("yolo11x-pose.pt").model,
  epochs=15,              # 適中的訓練輪數
  imgsz=640,              # 標準圖像尺寸
  batch=64,               # 適中的批量
  save=True,
  save_period=1,          # 每輪保存檢查點
  cache="disk",
  lr0=0.0001,             # 較高的學習率
  lrf=0.001,              # 較高的最終學習率
  warmup_epochs=1.0,      # 添加預熱
  patience=5,             # 適當的早停
  optimizer="AdamW",      # 適合微調的優化器  
  weight_decay=0.0001,    # 輕微的權重衰減
  cos_lr=True,            # 餘弦調度
  freeze=10,              # 只凍結前面的層
  augment=True,           # 開啟增強
  mosaic=0.0,             # 關閉mosaic
  mixup=0.0,              # 關閉mixup
  copy_paste=0.0,         # 關閉copy-paste
  degrees=2.0,            # 限制旋轉
  translate=0.02,         # 限制平移
  scale=0.05,             # 限制縮放
  shear=0.0,              # 關閉剪切
  perspective=0.0,        # 關閉透視變換
  val=True,               # 驗證
  plots=True,             # 性能圖表
  device=0,               # GPU設備
  workers=8,              # 工作線程
  multi_scale=False,      # 關閉多尺度
  amp=True,               # 混合精度
  overlap_mask=True,      # 關鍵點重疊處理
  kobj=2.5,               # 增加關鍵點損失權重
  pose=15.0,              # 增加姿態損失權重
  box=0.5                 # 減少邊界框損失權重
)
