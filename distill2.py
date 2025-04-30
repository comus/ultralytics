from ultralytics import YOLO

# 加載您第一階段訓練的最佳模型
# 替換為您的實際最佳模型路徑
best_model_path = "distill_projects/yolo11l_to_yolo11n_aggressive/weights/best.pt"

# 加載教師模型
teacher_model = YOLO("yolo11x-pose.pt")  # 使用更強大的YOLO11x作為教師
student_model = YOLO(best_model_path)     # 使用第一階段訓練的最佳模型

# 開始第二階段訓練
student_model.train(
    data="coco-pose.yaml",
    teacher=teacher_model.model,
    distillation_loss="cwd",              # 保持CWD蒸餾方法
    distillation_layers=["16", "19", "22"], # 聚焦在關鍵特徵層
    epochs=10,                            # 10個epoch應該足夠
    imgsz=640,
    batch=32,                             # 保持相同批次大小
    workers=8,                            # 利用多核CPU
    lr0=0.0001,                           # 降低學習率進行精調
    lrf=0.01,                             # 學習率衰減到初始的1%
    optimizer="Adam",                     # 保持Adam優化器
    cos_lr=True,                          # 餘弦學習率調度
    warmup_epochs=0.5,                    # 短暫預熱
    pure_distill=False,                   # 保持任務學習+蒸餾模式
    patience=5,                           # 適當的早停耐心值
    device=0,
    project="distill_projects",
    name="yolo11n_stage2_distill",        # 新的實驗名稱
    val=True,                             # 啟用驗證
    save_period=1,                        # 每個epoch保存
    plots=True,                           # 生成訓練圖表
    close_mosaic=0,                       # 禁用mosaic增強
    augment=False,                        # 關閉增強，保持特徵對齊
    distill=0.05,                         # 適中的蒸餾權重
    fraction=1.0,                         # 使用全部數據
    amp=False,                            # 關閉混合精度訓練
    pose=20.0,                            # 顯著增加pose損失權重
)