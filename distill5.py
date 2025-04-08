# distill5.py - 超級終極突破策略 - 極限版
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# 使用最佳FGD模型作為起點
student_model = YOLO("distill_projects/yolo11n_fgd_breakthrough/weights/best.pt")
teacher_model = YOLO("yolo11x-pose.pt")  # 最強教師模型

# 超極限終極訓練設置 - 修改後直接使用內建回調和損失函數
student_model.train(
    data="coco-pose.yaml",
    teacher=teacher_model.model,
    distillation_loss="enhancedfgd",    # 使用增強版FGD損失函數，已在ultralytics/utils/distill.py中實現
    distillation_layers=["4", "6", "8", "11", "13", "16", "19", "22"],  # 更全面的層選擇
    epochs=30,                          # 大幅延長訓練時間
    batch=16,                           # 較小批次提高精度
    workers=8,
    lr0=0.00007,                        # 更高初始學習率
    lrf=0.00001,                        # 更激進的衰減
    optimizer="AdamW",                  # 更先進的優化器
    weight_decay=0.0001,                # 適度增加正則化
    cos_lr=False,                       # 禁用餘弦調度，使用自定義調度
    patience=35,                        # 增加耐心值
    device=0,
    project="distill_projects",
    name="yolo11n_test",
    val=True,
    save_period=1,
    plots=True,
    distill=0.6,                        # 大幅增強蒸餾權重
    pose=50.0,                          # 大幅增強姿態權重
    warmup_epochs=0.0,                  # 無需熱身
    close_mosaic=0,                     # 從頭關閉mosaic
    amp=False,                          # 確保全精度
    overlap_mask=True,                  # 改善遮擋情況
    
    # 進階數據處理設置 - 初始設置，實際會由回調動態調整
    hsv_h=0.15,                         # 初始適中色調增強
    hsv_s=0.15,                         # 初始適中飽和度增強
    hsv_v=0.15,                         # 初始適中亮度增強
    degrees=0.0,                        # 禁用旋轉增強
    translate=0.1,                      # 輕微平移增強
    scale=0.1,                          # 輕微縮放增強
    fliplr=0.5,                         # 啟用水平翻轉以增加多樣性
    mosaic=0.0,                         # 禁用馬賽克增強
    mixup=0.0,                          # 禁用mixup增強
    copy_paste=0.0,                     # 禁用複製粘貼增強
    erasing=0.3,                        # 啟用隨機擦除增強
    
    # 其他設置
    single_cls=False,                   # 維持多類別區分
    nbs=8,                              # 標稱批次大小
    rect=False,                         # 非矩形訓練
    save_json=True,                     # 保存評估JSON
    half=False,                         # 確保全精度訓練
    augment=False,                      # 初始禁用強增強
    fraction=0.01,                       # 使用全部數據
    cache="disk",                       # 使用磁盤緩存加速
    verbose=True,                       # 詳細日誌
    seed=42,                            # 更優的隨機種子
    dropout=0.0,                        # 禁用dropout以保留全部特徵
) 