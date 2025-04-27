#!/usr/bin/env python
# 平衡訓練腳本：同時保持COCO-Pose性能和提升瑜伽姿勢識別能力
# 採用漸進式微調策略的四階段訓練過程

import os
import sys
import torch
import argparse
import torch.distributed as dist
import numpy as np
import warnings
from pathlib import Path
import yaml

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

def train_stage1(model_path, save_dir, device="0,1,2,3", batch=32):
    """
    第一階段: COCO記憶刷新階段
    目標: 刷新模型對COCO數據集的記憶，避免遺忘
    特點: 較低學習率，較短訓練週期，只使用COCO數據
    """
    print("=" * 80)
    print("第一階段：COCO記憶刷新")
    print("=" * 80)
    
    model = YOLO(model_path)
    
    # 僅在COCO上進行短期訓練，避免遺忘
    results = model.train(
        data="coco-pose.yaml",
        epochs=5,                  # 短訓練週期
        imgsz=1600,                # 高解析度
        batch=batch,               # 批次大小
        save_period=1,             # 每個epoch保存
        cache="disk",                # 緩存圖像
        optimizer="AdamW",         # 優化器
        lr0=0.00005,               # 較低學習率避免遺忘原始知識
        lrf=0.2,                   # 學習率衰減因子
        cos_lr=True,               # 餘弦學習率調度
        warmup_epochs=1.0,         # 短熱身
        device=device,             # 設備
        patience=100,              # 關閉早停
        freeze=0,                  # 不凍結層
        box=7.0,                   # 邊界框損失權重
        cls=0.5,                   # 分類損失權重
        pose=12.0,                 # 適中的姿態損失權重
        kobj=3.0,                  # 適中的關鍵點可見性權重
        project=save_dir,          # 保存目錄
        name="stage1_coco_refresh", # 運行名稱
        exist_ok=True,             # 如果目錄存在則覆蓋
        val=True,                  # 每個epoch驗證
    )
    
    # 返回最佳模型路徑
    return Path(save_dir) / "stage1_coco_refresh" / "weights" / "best.pt"

def train_stage2(model_path, save_dir, device="0,1,2,3", batch=32):
    """
    第二階段: 混合數據平衡訓練階段
    目標: 在保持COCO知識的同時，學習瑜伽姿勢
    特點: 使用混合數據集，較高的損失權重，更多凍結層
    """
    print("=" * 80)
    print("第二階段：混合數據平衡訓練")
    print("=" * 80)
    
    model = YOLO(model_path)
    
    # 使用混合數據集進行訓練
    results = model.train(
        data="mixed_coco_yoga.yaml",  # 混合數據集
        epochs=60,                     # 增加訓練週期
        imgsz=1600,                    # 高解析度
        batch=batch,                   # 批次大小
        save_period=1,                 # 每個epoch保存
        cache="disk",                  # 使用磁盤緩存
        optimizer="AdamW",             # 優化器
        lr0=0.0001,                    # 適中學習率
        lrf=0.01,                      # 學習率衰減因子
        cos_lr=True,                   # 餘弦學習率調度
        warmup_epochs=3.0,             # 延長熱身期
        device=device,                 # 設備
        patience=10,                   # 增加早停耐心值
        freeze=8,                      # 增加凍結層數量，保留COCO特徵
        box=7.0,                       # 邊界框損失權重
        cls=0.5,                       # 分類損失權重
        pose=30.0,                     # 大幅增加姿態損失權重
        kobj=6.0,                      # 增加關鍵點可見性權重
        
        # 優化的數據增強設置，更好地保持姿勢完整性
        hsv_h=0.01,                    # 減少色調變化
        hsv_s=0.1,                     # 減少飽和度變化
        hsv_v=0.1,                     # 減少亮度變化
        degrees=4.0,                   # 適度旋轉
        translate=0.1,                 # 適度平移
        scale=0.15,                    # 減少縮放範圍
        fliplr=0.5,                    # 水平翻轉
        perspective=0.0003,            # 減少透視變換
        mosaic=0.15,                   # 適度馬賽克
        mixup=0.1,                     # 適度混合
        
        # 訓練穩定性設置
        overlap_mask=True,             # 使用重疊掩碼
        amp=True,                      # 混合精度訓練
        weight_decay=0.0003,           # 適度權重衰減
        dropout=0.1,                   # 適度Dropout
        
        project=save_dir,              # 保存目錄
        name="stage2_mixed_training",  # 運行名稱
        exist_ok=True,                 # 如果目錄存在則覆蓋
        val=True,                      # 每個epoch驗證
    )
    
    # 返回最佳模型路徑
    return Path(save_dir) / "stage2_mixed_training" / "weights" / "best.pt"

def train_stage3(model_path, save_dir, device="0,1,2,3", batch=32):
    """
    第三階段: 精細微調階段
    目標: 在混合數據上進行最終精細調整
    特點: 適當學習率，更高的姿態損失權重，精細調整
    """
    print("=" * 80)
    print("第三階段：精細微調")
    print("=" * 80)
    
    model = YOLO(model_path)
    
    # 最終精細微調
    results = model.train(
        data="mixed_coco_yoga.yaml",   # 混合數據集
        epochs=40,                     # 增加訓練週期
        imgsz=1600,                    # 高解析度
        batch=batch,                   # 批次大小
        save_period=1,                 # 每個epoch保存
        cache="disk",                  # 使用磁盤緩存
        optimizer="AdamW",             # 優化器
        lr0=0.00005,                   # 提高學習率
        lrf=0.05,                      # 學習率衰減因子
        cos_lr=True,                   # 餘弦學習率衰減
        warmup_epochs=1.0,             # 短熱身
        device=device,                 # 設備
        patience=8,                    # 早停耐心值
        freeze=4,                      # 凍結前幾層，保留特徵
        box=7.0,                       # 邊界框損失權重
        cls=0.5,                       # 分類損失權重
        pose=50.0,            # 極高姿態損失權重
        kobj=10.0,            # 增加關鍵點可見性權重
        
        # 降低數據增強強度，專注於精細調整
        hsv_h=0.01,                    # 減少色調變化
        hsv_s=0.05,                    # 進一步減少飽和度變化
        hsv_v=0.05,                    # 進一步減少亮度變化
        degrees=2.0,                   # 減少旋轉角度
        translate=0.05,                # 減少平移範圍
        scale=0.1,                     # 減少縮放範圍
        fliplr=0.5,                    # 水平翻轉概率
        perspective=0.0001,            # 減少透視變換
        mosaic=0.0,                    # 關閉馬賽克
        mixup=0.0,                     # 關閉混合增強
        copy_paste=0.0,                # 無複製粘貼
        
        # 訓練穩定性設置
        overlap_mask=True,             # 使用重疊掩碼
        amp=True,                      # 混合精度訓練
        weight_decay=0.0001,           # 降低權重衰減，避免過度正則化
        dropout=0.05,                  # 適度Dropout
        
        project=save_dir,              # 保存目錄
        name="stage3_fine_tuning",     # 運行名稱
        exist_ok=True,                 # 如果目錄存在則覆蓋
        val=True,                      # 每個epoch驗證
    )
    
    # 返回最佳模型路徑
    return Path(save_dir) / "stage3_fine_tuning" / "weights" / "best.pt"

def train_stage4(model_path, save_dir, device="0,1,2,3", batch=32):
    """
    第四階段: 瑜伽專注階段
    目標: 專注提升瑜伽姿勢識別能力
    特點: 僅使用瑜伽數據，適度凍結，極高姿態損失權重
    """
    print("=" * 80)
    print("第四階段：瑜伽專注微調")
    print("=" * 80)
    
    model = YOLO(model_path)
    
    # 專注瑜伽精調
    results = model.train(
        data="yoga82.yaml",            # 僅使用瑜伽數據
        epochs=10,                     # 短訓練週期
        imgsz=1600,                    # 高解析度
        batch=batch,                   # 批次大小
        save_period=1,                 # 每個epoch保存
        cache="disk",                  # 使用磁盤緩存
        optimizer="AdamW",             # 優化器
        lr0=0.00003,                   # 低學習率
        lrf=0.01,                      # 學習率衰減因子
        cos_lr=True,                   # 餘弦學習率調度
        warmup_epochs=0.0,             # 無熱身
        device=device,                 # 設備
        patience=5,                    # 早停耐心值
        freeze=8,                      # 凍結前8層保留特徵
        box=7.0,                       # 邊界框損失權重
        cls=0.5,                       # 分類損失權重
        pose=70.0,                     # 極高姿態損失權重
        kobj=15.0,                     # 極高關鍵點可見性權重
        
        # 最小化數據增強
        hsv_h=0.0,                     # 無色調變化
        hsv_s=0.0,                     # 無飽和度變化
        hsv_v=0.0,                     # 無亮度變化
        degrees=0.0,                   # 無旋轉
        translate=0.03,                # 最小平移
        scale=0.03,                    # 最小縮放
        fliplr=0.5,                    # 保留水平翻轉
        perspective=0.0,               # 無透視變換
        mosaic=0.0,                    # 無馬賽克
        mixup=0.0,                     # 無混合增強
        
        project=save_dir,              # 保存目錄
        name="stage4_yoga_focus",      # 運行名稱
        exist_ok=True,                 # 如果目錄存在則覆蓋
        val=True,                      # 每個epoch驗證
    )
    
    return Path(save_dir) / "stage4_yoga_focus" / "weights" / "best.pt"

def validate_on_both(model_path, save_dir):
    """
    在COCO和Yoga數據集上分別驗證模型性能
    """
    print("=" * 80)
    print("在COCO和Yoga數據集上驗證最終模型")
    print("=" * 80)
    
    model = YOLO(model_path)
    
    # 在COCO上驗證
    print("\n在COCO數據集上驗證:")
    coco_metrics = model.val(data="coco-pose.yaml")
    
    # 在Yoga上驗證
    print("\n在Yoga數據集上驗證:")
    yoga_metrics = model.val(data="yoga82.yaml")
    
    # 保存驗證結果摘要
    summary_path = Path(save_dir) / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("平衡訓練模型驗證結果摘要\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("COCO-Pose 數據集結果:\n")
        f.write(f"Box mAP50: {coco_metrics.box.map50:.4f}\n")
        f.write(f"Box mAP50-95: {coco_metrics.box.map:.4f}\n")
        f.write(f"Box mAP75: {coco_metrics.box.map75:.4f}\n")
        f.write(f"Pose mAP50: {coco_metrics.pose.map50:.4f}\n")
        f.write(f"Pose mAP50-95: {coco_metrics.pose.map:.4f}\n")
        f.write(f"Pose mAP75: {coco_metrics.pose.map75:.4f}\n\n")
        
        f.write("Yoga82 數據集結果:\n")
        f.write(f"Box mAP50: {yoga_metrics.box.map50:.4f}\n")
        f.write(f"Box mAP50-95: {yoga_metrics.box.map:.4f}\n")
        f.write(f"Box mAP75: {yoga_metrics.box.map75:.4f}\n")
        f.write(f"Pose mAP50: {yoga_metrics.pose.map50:.4f}\n")
        f.write(f"Pose mAP50-95: {yoga_metrics.pose.map:.4f}\n")
        f.write(f"Pose mAP75: {yoga_metrics.pose.map75:.4f}\n")
    
    print(f"驗證結果摘要已保存到 {summary_path}")
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='平衡訓練COCO和瑜伽姿勢的四階段訓練腳本')
    parser.add_argument('--model', type=str, default='/root/autodl-tmp/withcloud/ultralytics/runs/pose/train14/weights/best.pt', 
                        help='初始模型路徑 (默認: yolov8x-pose.pt)')
    parser.add_argument('--device', type=str, default='0,1,2,3', 
                        help='訓練設備 (默認: 0,1,2,3)')
    parser.add_argument('--batch', type=int, default=32, 
                        help='批次大小 (默認: 32)')
    parser.add_argument('--save-dir', type=str, default='runs/pose/balanced_training', 
                        help='保存目錄 (默認: runs/pose/balanced_training)')
    parser.add_argument('--skip-stage', type=int, default=0, 
                        help='跳過前N個階段 (默認: 0，不跳過)')
    args = parser.parse_args()
    
    # 創建保存目錄
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取初始模型路徑
    model_path = args.model
    
    # 執行四階段訓練
    if args.skip_stage <= 0:
        model_path = train_stage1(model_path, save_dir, device=args.device, batch=args.batch)
        print(f"第一階段完成，最佳模型保存在: {model_path}")
    
    if args.skip_stage <= 1:
        model_path = train_stage2(model_path, save_dir, device=args.device, batch=args.batch)
        print(f"第二階段完成，最佳模型保存在: {model_path}")
    
    if args.skip_stage <= 2:
        model_path = train_stage3(model_path, save_dir, device=args.device, batch=args.batch)
        print(f"第三階段完成，最佳模型保存在: {model_path}")
    
    # if args.skip_stage <= 3:
    #     model_path = train_stage4(model_path, save_dir, device=args.device, batch=args.batch)
    #     print(f"第四階段完成，最佳模型保存在: {model_path}")
    
    # 在兩個數據集上驗證最終模型
    summary_path = validate_on_both(model_path, save_dir)
    
    print("=" * 80)
    print(f"平衡訓練完成！最終模型: {model_path}")
    print(f"驗證結果摘要: {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    main() 