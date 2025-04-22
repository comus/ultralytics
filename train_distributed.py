import os
import sys
import torch.distributed as dist
import subprocess
import argparse

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 設置環境變量，確保分佈式訓練使用本地代碼
os.environ["PYTHONPATH"] = f"{current_dir}:{os.environ.get('PYTHONPATH', '')}"

# 檢查是否為主進程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def parse_args():
    parser = argparse.ArgumentParser(description="GDEPose 分佈式訓練")
    parser.add_argument("--device", type=str, default="0,1", help="要使用的 GPU ID，例如 '0,1'")
    parser.add_argument("--epochs", type=int, default=100, help="訓練的 epoch 數")
    parser.add_argument("--batch-size", type=int, default=256, help="批次大小")
    parser.add_argument("--img-size", type=int, default=640, help="圖像大小")
    parser.add_argument("--yaml", type=str, default="gde_pose.yaml", help="模型配置文件")
    parser.add_argument("--data", type=str, default="coco-pose.yaml", help="數據集配置文件")
    parser.add_argument("--save-period", type=int, default=1, help="每隔多少個 epoch 保存一次模型")
    parser.add_argument("--cache", type=str, default="disk", help="緩存類型")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="優化器類型")
    parser.add_argument("--lr0", type=float, default=0.001, help="初始學習率")
    parser.add_argument("--lrf", type=float, default=0.01, help="最終學習率因子")
    return parser.parse_args()

from ultralytics import YOLO

def main():
    args = parse_args()
    
    # Initialize a new model from yaml configuration without pretrained weights
    model = YOLO(args.yaml)

    # 打印使用的模塊路徑，確認是否正確
    if is_main_process():
        import ultralytics
        print(f"使用的 Ultralytics 模塊路徑: {os.path.dirname(ultralytics.__file__)}")
        
        # 檢查模塊是否包含自定義層
        try:
            import inspect
            from ultralytics.nn.modules.block import C3k2_Ghost, C3k2_DFFM
            print(f"C3k2_Ghost 模塊位置: {inspect.getfile(C3k2_Ghost)}")
            print(f"C3k2_DFFM 模塊位置: {inspect.getfile(C3k2_DFFM)}")
        except (ImportError, AttributeError) as e:
            print(f"警告: 自定義層檢查失敗 - {e}")

    # Train the model with specified parameters
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        save_period=args.save_period,
        cache=args.cache,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        device=args.device
    )

if __name__ == "__main__":
    main() 