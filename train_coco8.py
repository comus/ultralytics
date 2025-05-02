import warnings
import os
import sys
import torch

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 設置環境變量，確保分佈式訓練使用本地代碼
os.environ["PYTHONPATH"] = f"{current_dir}:{parent_dir}:{os.environ.get('PYTHONPATH', '')}"

# 在導入YOLO之前，先設置環境變量
# 設置環境變量，確保所有GPU的日誌都顯示
os.environ["PYTHONIOENCODING"] = "utf-8"  # 確保UTF-8編碼輸出
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 確保使用指定的GPU
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # 輸出更詳細的分佈式訓練日誌

# 忽略除零警告
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
# 忽略 DDP 的 stride 不匹配警告
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")

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
    
    print(f"Python路徑: {sys.path}")
    print(f"當前工作目錄: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '未設置')}")

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