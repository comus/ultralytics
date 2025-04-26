#!/usr/bin/env python
# 瑜伽姿勢模型比較工具
# 此腳本用於比較多個YOLO關鍵點檢測模型的性能差異

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import torch
from tqdm import tqdm
from collections import defaultdict

# 添加本地路徑到 Python 路徑中，確保使用本地版本
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from ultralytics import YOLO


def compare_models(model_paths, data_yaml, output_dir=None, img_size=1280, batch_size=16, device='0'):
    """
    比較多個模型的性能並生成比較報告
    
    參數:
        model_paths: 模型權重路徑列表
        data_yaml: 數據配置文件路徑
        output_dir: 輸出目錄
        img_size: 圖像大小
        batch_size: 批次大小
        device: 運行設備
    """
    if len(model_paths) < 2:
        raise ValueError("至少需要提供兩個模型進行比較")
    
    if output_dir is None:
        output_dir = Path("model_comparison")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"比較 {len(model_paths)} 個模型...")
    
    # 載入所有模型
    models = []
    model_names = []
    validation_results = []
    
    for i, model_path in enumerate(model_paths):
        print(f"載入模型 {i+1}/{len(model_paths)}: {model_path}")
        model = YOLO(model_path)
        models.append(model)
        model_name = Path(model_path).stem
        model_names.append(model_name)
        
        # 執行驗證
        print(f"驗證模型: {model_name}")
        results = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, device=device)
        validation_results.append(results)
    
    # 比較並可視化結果
    compare_metrics(model_names, validation_results, output_dir)
    
    # 尋找範例圖像進行視覺比較
    sample_images = find_sample_images(data_yaml, output_dir)
    
    # 生成視覺化比較結果
    if sample_images:
        generate_visual_comparisons(models, model_names, output_dir)
        
    return models, model_names, validation_results


def compare_metrics(model_names, validation_results, output_dir):
    """
    比較多個模型的性能指標
    
    參數:
        model_names: 模型名稱列表
        validation_results: 驗證結果列表
        output_dir: 輸出目錄
    """
    print("\n比較模型性能指標...")
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # 提取關鍵指標
    box_map50 = []
    box_map50_95 = []
    pose_map50 = []
    pose_map50_95 = []
    
    for result in validation_results:
        # 提取Box mAP
        box_map50.append(result.box.map50)
        box_map50_95.append(result.box.map50_95)
        
        # 提取Pose mAP
        pose_map50.append(result.keypoints.map50 if hasattr(result, 'keypoints') else 0)
        pose_map50_95.append(result.keypoints.map50_95 if hasattr(result, 'keypoints') else 0)
    
    # 創建比較表格
    with open(metrics_dir / "metrics_comparison.txt", "w") as f:
        f.write("模型性能指標比較\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'模型名稱':<20} {'Box mAP50':<10} {'Box mAP50-95':<12} {'Pose mAP50':<10} {'Pose mAP50-95':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, name in enumerate(model_names):
            f.write(f"{name:<20} {box_map50[i]:<10.4f} {box_map50_95[i]:<12.4f} {pose_map50[i]:<10.4f} {pose_map50_95[i]:<12.4f}\n")
    
    # 繪製比較圖表
    # mAP50 比較
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, box_map50, width, label='Box mAP50')
    plt.bar(x + width/2, pose_map50, width, label='Pose mAP50')
    
    plt.xlabel('模型')
    plt.ylabel('mAP50')
    plt.title('模型 mAP50 性能比較')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(metrics_dir / 'map50_comparison.png', dpi=300)
    plt.close()
    
    # mAP50-95 比較
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, box_map50_95, width, label='Box mAP50-95')
    plt.bar(x + width/2, pose_map50_95, width, label='Pose mAP50-95')
    
    plt.xlabel('模型')
    plt.ylabel('mAP50-95')
    plt.title('模型 mAP50-95 性能比較')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(metrics_dir / 'map50_95_comparison.png', dpi=300)
    plt.close()
    
    print(f"指標比較完成，結果已保存到: {metrics_dir}")


def find_sample_images(data_yaml, output_dir, limit=10):
    """
    從驗證數據集中尋找範例圖像進行比較
    
    參數:
        data_yaml: 數據配置文件路徑
        output_dir: 輸出目錄
        limit: 最大圖像數量
        
    返回:
        範例圖像路徑列表
    """
    print("\n尋找範例圖像進行視覺比較...")
    
    # 從yaml中獲取驗證資料集路徑
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(data_yaml)
    val_images = []
    
    # 獲取驗證圖像路徑
    if 'val' in data_dict:
        import glob
        if isinstance(data_dict['val'], str):
            val_path = data_dict['val']
            if os.path.isdir(val_path):
                val_images = glob.glob(os.path.join(val_path, '**/*.jpg'), recursive=True)
                val_images += glob.glob(os.path.join(val_path, '**/*.png'), recursive=True)
            # 如果val指向一個文本文件
            elif os.path.isfile(val_path) and val_path.endswith('.txt'):
                with open(val_path, 'r') as f:
                    lines = f.readlines()
                val_images = [line.strip() for line in lines]
    
    # 如果沒有找到驗證圖像
    if not val_images:
        print("警告: 未找到驗證圖像")
        return []
    
    # 隨機選擇圖像
    import random
    if len(val_images) > limit:
        sample_images = random.sample(val_images, limit)
    else:
        sample_images = val_images
    
    # 保存樣本圖像路徑到文件中
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    with open(samples_dir / "sample_images.txt", "w") as f:
        for img_path in sample_images:
            f.write(f"{img_path}\n")
    
    print(f"已選擇 {len(sample_images)} 張範例圖像進行比較")
    return sample_images


def generate_visual_comparisons(models, model_names, output_dir):
    """
    為範例圖像生成視覺比較結果
    
    參數:
        models: 模型列表
        model_names: 模型名稱列表
        output_dir: 輸出目錄
    """
    print("\n生成視覺化比較結果...")
    
    samples_dir = output_dir / "samples"
    visual_dir = output_dir / "visual_comparison"
    visual_dir.mkdir(exist_ok=True)
    
    # 讀取樣本圖像路徑
    sample_images = []
    if os.path.exists(samples_dir / "sample_images.txt"):
        with open(samples_dir / "sample_images.txt", "r") as f:
            sample_images = [line.strip() for line in f.readlines()]
    
    if not sample_images:
        print("警告: 未找到範例圖像，無法生成視覺比較")
        return
    
    # 對每個樣本圖像運行所有模型
    for i, img_path in enumerate(tqdm(sample_images, desc="生成視覺比較")):
        if not os.path.exists(img_path):
            print(f"警告: 圖像不存在 - {img_path}")
            continue
        
        # 讀取原始圖像
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"警告: 無法讀取圖像 - {img_path}")
            continue
        
        # 為每個模型運行預測
        model_results = []
        for j, model in enumerate(models):
            results = model.predict(img_path, conf=0.25, device='0', verbose=False)
            model_results.append(results[0])
        
        # 創建網格顯示所有模型結果
        n_models = len(models)
        grid_rows = 1 + (n_models // 3) if n_models > 3 else 2  # 至少2行
        grid_cols = min(n_models, 3)  # 每行最多3個模型
        
        # 計算網格圖像大小
        h, w = img_original.shape[:2]
        aspect_ratio = w / h
        grid_width = 1200
        cell_width = grid_width // grid_cols
        cell_height = int(cell_width / aspect_ratio)
        grid_height = cell_height * grid_rows
        
        # 創建網格圖像
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # 第一行放原始圖像
        img_resized = cv2.resize(img_original, (cell_width, cell_height))
        grid_img[0:cell_height, 0:cell_width] = img_resized
        
        # 添加原始圖像標題
        cv2.putText(grid_img, "原始圖像", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 添加每個模型的結果
        for j, result in enumerate(model_results):
            # 計算網格中的位置
            row = (j + 1) // grid_cols
            col = (j + 1) % grid_cols
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = col * cell_width
            x2 = x1 + cell_width
            
            # 獲取帶有預測的圖像
            pred_img = result.plot(conf=0.25, line_width=2, font_size=1, kpt_line=True, 
                                  kpt_radius=4, kpt_line_thickness=2)
            pred_img_resized = cv2.resize(pred_img, (cell_width, cell_height))
            
            # 放入網格
            grid_img[y1:y2, x1:x2] = pred_img_resized
            
            # 添加模型名稱
            cv2.putText(grid_img, model_names[j], (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 保存網格圖像
        output_file = visual_dir / f"comparison_{i+1:03d}.jpg"
        cv2.imwrite(str(output_file), grid_img)
    
    print(f"視覺比較已完成，結果已保存到: {visual_dir}")


def analyze_confidence_differences(models, model_names, data_yaml, output_dir, num_samples=5):
    """
    分析不同模型之間關鍵點置信度的差異
    
    參數:
        models: 模型列表
        model_names: 模型名稱列表
        data_yaml: 數據配置文件路徑
        output_dir: 輸出目錄
        num_samples: 顯示的樣本數量
    """
    print("\n分析模型間關鍵點置信度差異...")
    
    # 創建輸出目錄
    conf_dir = output_dir / "confidence_analysis"
    conf_dir.mkdir(exist_ok=True)
    
    # 從yaml中獲取驗證資料集路徑
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(data_yaml)
    val_images = []
    
    # 獲取驗證圖像路徑
    if 'val' in data_dict:
        import glob
        if isinstance(data_dict['val'], str):
            val_path = data_dict['val']
            if os.path.isdir(val_path):
                val_images = glob.glob(os.path.join(val_path, '**/*.jpg'), recursive=True)
                val_images += glob.glob(os.path.join(val_path, '**/*.png'), recursive=True)
            elif os.path.isfile(val_path) and val_path.endswith('.txt'):
                with open(val_path, 'r') as f:
                    lines = f.readlines()
                val_images = [line.strip() for line in lines]
    
    # 如果沒有找到驗證圖像
    if not val_images:
        print("警告: 未找到驗證圖像，無法分析置信度差異")
        return
    
    # 隨機選擇圖像
    import random
    sample_size = min(len(val_images), 50)  # 隨機分析50張圖像
    selected_images = random.sample(val_images, sample_size)
    
    # 收集每個模型的關鍵點置信度
    confidence_diffs = []
    
    for img_path in tqdm(selected_images, desc="分析置信度差異"):
        if not os.path.exists(img_path):
            continue
        
        # 為每個模型運行預測
        all_kpt_confs = []
        for model in models:
            results = model.predict(img_path, conf=0.25, device='0', verbose=False)
            
            # 如果有檢測到關鍵點
            if len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0]  # 只取第一個檢測對象
                
                # 確保有關鍵點
                if kpts.shape[0] > 0:
                    # 提取置信度
                    conf_values = kpts[:, 2].cpu().numpy()
                    all_kpt_confs.append(conf_values)
                else:
                    all_kpt_confs.append(None)
            else:
                all_kpt_confs.append(None)
        
        # 計算模型間的置信度差異
        if len(all_kpt_confs) == len(models) and all(x is not None for x in all_kpt_confs):
            # 確保所有關鍵點配置相同
            if len(set(x.shape[0] for x in all_kpt_confs)) == 1:
                # 計算每個關鍵點的置信度標準差
                kpt_stds = np.std(all_kpt_confs, axis=0)
                
                # 計算平均標準差
                avg_std = np.mean(kpt_stds)
                
                # 儲存高標準差的案例
                if avg_std > 0.1:  # 只關注標準差較大的案例
                    confidence_diffs.append({
                        'img_path': img_path,
                        'kpt_stds': kpt_stds,
                        'avg_std': avg_std,
                        'confs': all_kpt_confs
                    })
    
    # 按平均標準差排序
    if confidence_diffs:
        confidence_diffs.sort(key=lambda x: x['avg_std'], reverse=True)
        
        # 將結果保存到文件
        with open(conf_dir / "confidence_differences.txt", "w") as f:
            f.write("模型關鍵點置信度差異分析\n")
            f.write("=" * 80 + "\n\n")
            
            for i, diff in enumerate(confidence_diffs[:20]):  # 只顯示前20個差異最大的
                f.write(f"樣本 {i+1}:\n")
                f.write(f"圖像: {diff['img_path']}\n")
                f.write(f"平均置信度標準差: {diff['avg_std']:.4f}\n")
                f.write(f"關鍵點標準差: {diff['kpt_stds']}\n\n")
        
        # 視覺化差異最大的幾個案例
        visualize_confidence_differences(models, model_names, 
                                        [d['img_path'] for d in confidence_diffs[:num_samples]], 
                                        conf_dir, num_samples)
    else:
        print("未找到顯著的置信度差異")
    
    print(f"置信度差異分析完成，結果已保存到: {conf_dir}")


def visualize_confidence_differences(models, model_names, image_paths, output_dir, num_samples=5):
    """
    視覺化模型間關鍵點置信度的顯著差異
    
    參數:
        models: 模型列表
        model_names: 模型名稱列表
        image_paths: 圖像路徑列表
        output_dir: 輸出目錄
        num_samples: 樣本數量
    """
    print(f"\n視覺化 {min(num_samples, len(image_paths))} 個置信度差異顯著的案例...")
    
    for i, img_path in enumerate(image_paths[:num_samples]):
        if not os.path.exists(img_path):
            print(f"警告: 圖像不存在 - {img_path}")
            continue
        
        # 讀取原始圖像
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"警告: 無法讀取圖像 - {img_path}")
            continue
        
        h, w = img_original.shape[:2]
        
        # 為每個模型運行預測
        model_results = []
        all_kpt_confs = []
        
        for j, model in enumerate(models):
            results = model.predict(img_path, conf=0.25, device='0', verbose=False)
            model_results.append(results[0])
            
            # 提取關鍵點置信度
            if len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0]
                if kpts.shape[0] > 0:
                    conf_values = kpts[:, 2].cpu().numpy()
                    all_kpt_confs.append(conf_values)
                else:
                    all_kpt_confs.append(None)
            else:
                all_kpt_confs.append(None)
        
        # 創建網格顯示所有模型結果
        n_models = len(models)
        grid_rows = 1 + n_models  # 第一行放原始圖像
        grid_cols = 1
        
        # 設置網格尺寸
        cell_height = 480
        cell_width = int(cell_height * (w / h))
        grid_height = cell_height * grid_rows
        grid_width = cell_width
        
        # 創建網格圖像
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # 第一行放原始圖像
        img_resized = cv2.resize(img_original, (cell_width, cell_height))
        grid_img[0:cell_height, 0:cell_width] = img_resized
        
        # 添加原始圖像標題
        cv2.putText(grid_img, "原始圖像", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 添加每個模型的結果
        for j, result in enumerate(model_results):
            row = j + 1  # 從第二行開始
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = 0
            x2 = cell_width
            
            # 獲取帶有預測的圖像
            pred_img = result.plot(conf=0.25, line_width=2, font_size=1, kpt_line=True, 
                                  kpt_radius=4, kpt_line_thickness=2)
            pred_img_resized = cv2.resize(pred_img, (cell_width, cell_height))
            
            # 放入網格
            grid_img[y1:y2, x1:x2] = pred_img_resized
            
            # 添加模型名稱和置信度信息
            model_title = f"{model_names[j]}"
            if all_kpt_confs[j] is not None:
                avg_conf = np.mean(all_kpt_confs[j])
                model_title += f" (平均置信度: {avg_conf:.3f})"
            
            cv2.putText(grid_img, model_title, (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # 保存網格圖像
        output_file = output_dir / f"conf_diff_{i+1:03d}.jpg"
        cv2.imwrite(str(output_file), grid_img)
    
    print(f"置信度差異視覺化已完成，結果已保存到: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='比較多個瑜伽姿勢模型並生成比較報告')
    parser.add_argument('--models', nargs='+', required=True, help='模型權重路徑列表（至少2個）')
    parser.add_argument('--data', type=str, required=True, help='數據YAML文件路徑')
    parser.add_argument('--output-dir', type=str, default='model_comparison', help='輸出目錄')
    parser.add_argument('--img-size', type=int, default=1280, help='驗證用圖像大小')
    parser.add_argument('--batch-size', type=int, default=16, help='驗證用批次大小')
    parser.add_argument('--device', type=str, default='0', help='運行設備（例如: 0 或 cpu）')
    parser.add_argument('--analyze-confidence', action='store_true', help='分析模型間關鍵點置信度差異')
    parser.add_argument('--num-samples', type=int, default=5, help='用於置信度差異視覺化的樣本數量')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 檢查是否提供了足夠的模型
    if len(args.models) < 2:
        print("錯誤: 至少需要提供兩個模型進行比較")
        sys.exit(1)
    
    # 進行模型比較
    models, model_names, validation_results = compare_models(
        model_paths=args.models,
        data_yaml=args.data,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # 如果需要分析置信度差異
    if args.analyze_confidence:
        analyze_confidence_differences(
            models=models,
            model_names=model_names,
            data_yaml=args.data,
            output_dir=Path(args.output_dir),
            num_samples=args.num_samples
        )
    
    print("模型比較完成！") 