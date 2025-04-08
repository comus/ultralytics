from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 设备数量: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0)}")

    # Load a model
    model = YOLO("lite.yaml")
    
    # 使用更保守的训练设置
    results = model.train(
        data="coco-pose.yaml", 
        epochs=500,
        imgsz=640,
        patience=50,
        batch=128,  # 輕量化模型可使用更大批次
        lr0=0.002,  # 輕量化模型可以用略高的學習率
        lrf=0.005,
        warmup_epochs=8,  # 增加預熱時間，幫助穩定初期訓練
        weight_decay=0.0002,  # 輕量模型降低權重衰減防止過擬合
        close_mosaic=15,  # 最後15個epoch關閉mosaic
        cos_lr=True,  # 余弦學習率調度
        
        # 輕量模型對數據增強更敏感，適當增強有助於提高泛化能力
        augment=True,
        degrees=10.0,  # 增加旋轉增強
        translate=0.15,  # 增加平移增強
        scale=0.5,  # 擴大縮放範圍
        shear=5.0,  # 添加剪切變換
        perspective=0.001,  # 添加透視變換
        fliplr=0.5,  # 50%概率水平翻轉
        
        # 其他適合輕量模型的參數
        overlap_mask=False,  # 簡化mask計算
        val=True,
        workers=8,  # 提高數據加載效率
        
        # 保存設置
        save_period=1,  # 每50個epoch保存一次
        project="pose-lite",
        name="pose-lite",
        exist_ok=True,
    )
    