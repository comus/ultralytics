# train_from_scratch.py
from ultralytics import YOLO

# 基於GDE架構創建模型
model = YOLO('gde_yolo_initial_layers.pt')  # 使用已複製第0、1層的模型

# 設置更長的訓練
results = model.train(
    data='coco-pose.yaml',
    epochs=300,  # 長時間訓練
    batch=32,
    imgsz=640,
    device=0,
    optimizer='Adam',
    lr0=0.001, 
    name='gde_pose_full_train',
    patience=50  # 較長的耐心等待收斂
)
