import warnings
import os
import sys
import torch

# 在導入YOLO之前，先設置環境變量
# 設置環境變量，確保所有GPU的日誌都顯示
os.environ["PYTHONIOENCODING"] = "utf-8"  # 確保UTF-8編碼輸出
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 確保使用指定的GPU
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 輸出更詳細的分佈式訓練日誌

# 忽略除零警告
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

def main():
    # 延遲導入YOLO，避免循環導入問題
    from ultralytics import YOLO
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA設備數量: {torch.cuda.device_count()}")
        print(f"當前CUDA設備: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"設備 {i}: {torch.cuda.get_device_name(i)}")

    # Load a model
    model = YOLO("yolo11n-pose.pt")

    print(f"開始訓練模型 - 使用多GPU知識蒸餾...")

    # Train the model with simpler setup
    results = model.train(
        data="coco8-pose.yaml", 
        epochs=20, 
        imgsz=640, 
        device=[0, 1],   # 使用兩個GPU
        
        # 蒸餾設置
        teacher="yolo11n-pose.pt",  # 指定教師模型
        target_layers=[0, 1],       # 目標層，簡化為數字索引
        distill=0.5,                # 蒸餾損失權重
        
        # 其他設置
        verbose=True,               # 啟用詳細日誌
    )

    # 顯示訓練結果
    print(f"訓練完成！最佳模型保存在: {results}")

if __name__ == "__main__":
    main()