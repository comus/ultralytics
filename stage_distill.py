# stage_distill.py - 蒸餾訓練策略
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# 加載模型
student_model = YOLO("yolo11n-pose.pt")  # 學生模型
teacher_model = YOLO("yolo11s-pose.pt")  # 教師模型

LOGGER.info("初始化蒸餾訓練策略...")
LOGGER.info("硬體: RTX 4090 24GB, Xeon Platinum 8352V, 120GB RAM")

# 執行蒸餾訓練
student_model.train(
    data="coco-pose.yaml",
    teacher=teacher_model.model,
    # distillation_loss="enhancedfgd",    # 使用增強版FGD損失
    # distillation_layers=["22"],  # 關鍵層選擇
    # pure_distill=True,
    
    # 硬體優化設置
    batch=128,                           # 充分利用24GB顯存，同時保持精度
    workers=8,                          # 適合Xeon 16核處理器
    # device=0,                           # 使用第一張GPU
    amp=False,                          # 關閉混合精度，提高精度（顯存足夠）
    
    # 訓練超參數
    epochs=18,                          # 階段總輪數: 3+4+11
    optimizer="AdamW",                  # 使用AdamW優化器
    weight_decay=0.001,                 # 較高的權重衰減防止過擬合
    # cos_lr=False,                       # 禁用余弦退火，使用自定義學習率
    
    # # 初始參數 (會被回調動態調整)
    lr0=0.0008,                        # 初始學習率
    lrf=0.01,                       # 最終學習率比例
    # distill=0.0,                       # 蒸餾權重
    # pose=0.0,                           # 姿態權重
    
    # 數據處理
    imgsz=640,                          # 標準輸入大小，平衡速度和精度
    cache="disk",                       # 使用磁盤緩存，考慮到120GB大內存
    
    # 輸出設置
    project="stage_distillation",       # 項目名稱
    name="yolo11n_pose_enhanced",       # 運行名稱
    
    # 訓練設置
    val=True,                           # 每個epoch驗證
    save_period=1,                      # 每個epoch保存一次
    patience=18,                        # 禁用早停，完成全部輪數
    fraction=0.5,                      # 訓練集比例






    warmup_epochs=3,
    warmup_momentum=0.85,
    warmup_bias_lr=0.08,
)

LOGGER.info("蒸餾訓練完成！")
