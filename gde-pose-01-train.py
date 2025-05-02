import os
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import sys
import torch.nn as nn
import torch.distributed as dist
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

def main():
    # ===== 初始設置 =====
    print("="*80)
    print("開始GDE-Pose第0,1層蒸餾實驗 - 使用4x4090 GPU")
    print("="*80)

    # 加載模型
    student_model = YOLO("models/gde-pose-640.pt")
    teacher_model = YOLO("yolo11n-pose.pt")

    # 蒸餾監控回調
    def log_feature_stats(trainer):
        """記錄特徵統計信息"""
        # 每5個epoch或最後一個epoch執行一次特徵分析
        if (trainer.epoch % 5 == 0 or trainer.epoch == trainer.epochs - 1) and trainer.epoch > 0:
            LOGGER.info(f"Epoch {trainer.epoch}: 記錄第0,1層特徵統計...")
            
            # 獲取特徵
            with torch.no_grad():
                # 遍歷所有目標層並檢查其特徵
                for layer_id in trainer.target_layers:
                    # 確保特徵存在於教師和學生模型中
                    if layer_id in trainer.teacher_features and layer_id in trainer.student_features:
                        t_feat = trainer.teacher_features[layer_id]
                        s_feat = trainer.student_features[layer_id]
                        
                        # 計算統計信息
                        t_mean = t_feat.mean().item()
                        s_mean = s_feat.mean().item()
                        t_var = t_feat.var().item()
                        s_var = s_feat.var().item()
                        
                        # 獲取層的顯示名稱 (對於整數索引使用"第N層"，對於字符串路徑直接使用)
                        layer_name = f"第{layer_id}層" if isinstance(layer_id, int) else layer_id
                        
                        # 記錄統計信息
                        LOGGER.info(f"{layer_name}: 教師特徵均值={t_mean:.4f}, 學生特徵均值={s_mean:.4f}")
                        LOGGER.info(f"{layer_name}: 教師特徵方差={t_var:.4f}, 學生特徵方差={s_var:.4f}")

    # 添加回調到模型
    student_model.add_callback("on_train_epoch_end", log_feature_stats)

    # ===== 蒸餾訓練 =====
    print("\n開始第0,1層蒸餾訓練...")
    results = student_model.train(
        # 基本訓練設置
        data="coco-pose.yaml",
        epochs=20,             # 初步實驗用20個epoch
        imgsz=640,
        batch=128,             # 4個4090的大批量
        # device=[0, 1, 2, 3],
        workers=16,
        
        # 蒸餾參數
        teacher=teacher_model.model,
        target_layers=[0, 1],  # 只蒸餾第0,1層
        distill=0.8,           # 較高蒸餾權重
        freezeAllBN=True,
        
        # 優化器設置
        optimizer="Adam",
        lr0=0.001,             # 適中的學習率
        lrf=0.1,               # 學習率可以衰減更多
        weight_decay=0.0005,
        warmup_epochs=2,
        
        # 保存與評估
        save_period=1,
        val=True,
        amp=True,
        
        # 實驗命名
        name="gde_distill_01_layers",
        project="gde_pose_distill",
        exist_ok=True
    )

    # ===== 評估與分析 =====
    print("\n開始蒸餾後評估...")
    # 加載蒸餾後的最佳模型
    distilled_model = YOLO("gde_pose_distill/gde_distill_01_layers/weights/best.pt")

    # 加載原始模型
    original_model = YOLO("models/gde-pose-640.pt")

    # 比較性能
    print("="*50)
    print("模型性能對比:")

    # 原始模型評估
    original_metrics = original_model.val(data="coco-pose.yaml")
    print(f"原始模型 mAP@50: {original_metrics.box.map50:.4f}")
    print(f"原始模型 Pose mAP: {original_metrics.pose.map:.4f}")

    # 蒸餾後模型評估
    distilled_metrics = distilled_model.val(data="coco-pose.yaml")
    print(f"蒸餾後模型 mAP@50: {distilled_metrics.box.map50:.4f}")
    print(f"蒸餾後模型 Pose mAP: {distilled_metrics.pose.map:.4f}")

    # 改善百分比
    if original_metrics.box.map50 > 0:
        box_improvement = (distilled_metrics.box.map50 - original_metrics.box.map50) / original_metrics.box.map50 * 100
        print(f"檢測性能提升: {box_improvement:.1f}%")
    else:
        print("原始檢測性能為零，無法計算百分比提升")

    if original_metrics.pose.map > 0:
        pose_improvement = (distilled_metrics.pose.map - original_metrics.pose.map) / original_metrics.pose.map * 100
        print(f"姿態估計性能提升: {pose_improvement:.1f}%")
    else:
        print("原始姿態性能為零，無法計算百分比提升")

    print("="*50)
    print("GDE-Pose第0,1層蒸餾實驗完成!")
    print("="*50)

    # 保存最終模型
    distilled_model.save('gde_pose_distilled_01_final.pt')
    print("最終模型已保存為 'gde_pose_distilled_01_final.pt'")

if __name__ == "__main__":
    main() 