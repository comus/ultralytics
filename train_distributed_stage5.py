import os
import sys
import torch.distributed as dist
import subprocess
import warnings

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# 設置環境變量，確保分佈式訓練使用本地代碼
os.environ["PYTHONPATH"] = f"{current_dir}:{os.environ.get('PYTHONPATH', '')}"
# 忽略 DDP 的 stride 不匹配警告
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides")
# 忽略除零警告
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

# 檢查是否為主進程
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

from ultralytics import YOLO

def main():    
    # Initialize a new model from yaml configuration without pretrained weights
    model = YOLO("/root/autodl-tmp/ultralytics/runs/pose/train55/weights/best.pt")

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

    # Train the model with hard-coded parameters
    results = model.train(
        data="coco-pose.yaml",       # 数据集配置文件路径
        epochs=150,                  # 训练轮数
        imgsz=1024,                  # 降低图像尺寸以减少内存使用
        batch=96,                    # 减小批次大小避免OOM
        save_period=1,               # 每5个epoch保存一次
        cache="disk",                # 使用磁盘缓存
        optimizer="AdamW",           # 优化器选择
        lr0=0.00002,                 # 初始学习率
        lrf=0.005,                   # 最终学习率因子
        cos_lr=True,                 # 余弦学习率调度
        warmup_epochs=3.0,           # 预热阶段轮数
        device="0,1,2,3",            # 使用的设备
        patience=30,                 # 早停耐心值
        box=12.0,                    # 框损失权重
        pose=18.0,                   # 姿态损失权重
        kobj=4.0,                    # 关键点对象性权重
        multi_scale=True,            # 多尺度训练
        close_mosaic=10,             # 最后10个epoch关闭马赽克增强
        amp=True,                    # 自动混合精度
        nbs=96,                      # 降低标称批次大小
        overlap_mask=True,           # 掩码重叠
        workers=8,                   # 每个进程的工作线程数
        val=True,                    # 进行验证
        plots=True,                  # 生成训练过程图表
        weight_decay=0.0005,         # 权重衰减
        dropout=0.02,                # dropout率
    )

if __name__ == "__main__":
    main() 
