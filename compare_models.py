#!/usr/bin/env python
# Yoga Pose Model Comparison Tool
# This script compares the performance of multiple YOLO keypoint detection models

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


def compare_models(model_paths, data_yaml, output_dir=None, img_size=640, batch_size=16, device='0'):
    """
    Compare multiple models' performance and generate comparison report
    
    Args:
        model_paths: List of model weight paths
        data_yaml: Data configuration file path
        output_dir: Output directory
        img_size: Image size
        batch_size: Batch size
        device: Running device
    """
    if len(model_paths) < 2:
        raise ValueError("At least two models need to be provided for comparison")
    
    if output_dir is None:
        output_dir = Path("model_comparison")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Comparing {len(model_paths)} models...")
    
    # Load all models
    models = []
    model_names = []
    validation_results = []
    
    for i, model_path in enumerate(model_paths):
        print(f"Loading model {i+1}/{len(model_paths)}: {model_path}")
        model = YOLO(model_path)
        models.append(model)
        model_name = Path(model_path).stem
        model_names.append(model_name)
        
        # Run validation
        print(f"Validating model: {model_name}")
        results = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, device=device)
        validation_results.append(results)
    
    # Compare and visualize results
    compare_metrics(model_names, validation_results, output_dir)
    
    # Find sample images for visual comparison
    sample_images = find_sample_images(data_yaml, output_dir)
    
    # Generate visual comparison results
    if sample_images:
        generate_visual_comparisons(models, model_names, output_dir, device=device)
        
    return models, model_names, validation_results


def compare_metrics(model_names, validation_results, output_dir):
    """
    Compare performance metrics among multiple models
    
    Args:
        model_names: List of model names
        validation_results: List of validation results
        output_dir: Output directory
    """
    print("\nComparing model performance metrics...")
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    # Extract key metrics
    box_map50 = []
    box_map = []
    pose_map50 = []
    pose_map = []
    
    for result in validation_results:
        # Extract Box mAP
        box_map50.append(result.box.map50)
        box_map.append(result.box.map)
        
        # Extract Pose mAP - fix this to properly access pose metrics
        # According to docs: https://docs.ultralytics.com/tasks/pose/#val
        print(f"Result keys: {dir(result)}")
        
        # First try to access pose metrics directly
        if hasattr(result, 'pose'):
            pose_map50.append(result.pose.map50)
            pose_map.append(result.pose.map)
        # Fall back to keypoints if pose not available
        elif hasattr(result, 'keypoints'):
            pose_map50.append(result.keypoints.map50)
            pose_map.append(result.keypoints.map)
        else:
            # If no pose-related attributes are found, print available metrics
            print(f"Warning: No pose metrics found. Available attributes: {dir(result)}")
            if hasattr(result, 'box'):
                print(f"Box metrics: {dir(result.box)}")
            pose_map50.append(0)
            pose_map.append(0)
    
    # Create comparison table
    with open(metrics_dir / "metrics_comparison.txt", "w") as f:
        f.write("Model Performance Metrics Comparison\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model Name':<20} {'Box mAP50':<10} {'Box mAP':<12} {'Pose mAP50':<10} {'Pose mAP':<12}\n")
        f.write("-" * 80 + "\n")
        
        for i, name in enumerate(model_names):
            f.write(f"{name:<20} {box_map50[i]:<10.4f} {box_map[i]:<12.4f} {pose_map50[i]:<10.4f} {pose_map[i]:<12.4f}\n")
    
    # Draw comparison charts
    # mAP50 comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, box_map50, width, label='Box mAP50')
    plt.bar(x + width/2, pose_map50, width, label='Pose mAP50')
    
    plt.xlabel('Model')
    plt.ylabel('mAP50')
    plt.title('Model mAP50 Performance Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(metrics_dir / 'map50_comparison.png', dpi=300)
    plt.close()
    
    # mAP comparison
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, box_map, width, label='Box mAP')
    plt.bar(x + width/2, pose_map, width, label='Pose mAP')
    
    plt.xlabel('Model')
    plt.ylabel('mAP')
    plt.title('Model mAP Performance Comparison')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(metrics_dir / 'map_comparison.png', dpi=300)
    plt.close()
    
    print(f"Metrics comparison completed, results saved to: {metrics_dir}")


def find_sample_images(data_yaml, output_dir, limit=10):
    """
    Find sample images from validation dataset for comparison
    
    Args:
        data_yaml: Data configuration file path
        output_dir: Output directory
        limit: Maximum number of images
        
    Returns:
        List of sample image paths
    """
    print("\nFinding sample images for visual comparison...")
    
    # Get validation dataset path from yaml
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(data_yaml)
    val_images = []
    
    # Get validation image paths
    if 'val' in data_dict:
        import glob
        if isinstance(data_dict['val'], str):
            val_path = data_dict['val']
            if os.path.isdir(val_path):
                val_images = glob.glob(os.path.join(val_path, '**/*.jpg'), recursive=True)
                val_images += glob.glob(os.path.join(val_path, '**/*.png'), recursive=True)
            # If val points to a text file
            elif os.path.isfile(val_path) and val_path.endswith('.txt'):
                with open(val_path, 'r') as f:
                    lines = f.readlines()
                val_images = [line.strip() for line in lines]
    
    # If no validation images found
    if not val_images:
        print("Warning: No validation images found")
        return []
    
    # Randomly select images
    import random
    if len(val_images) > limit:
        sample_images = random.sample(val_images, limit)
    else:
        sample_images = val_images
    
    # Save sample image paths to file
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    
    with open(samples_dir / "sample_images.txt", "w") as f:
        for img_path in sample_images:
            f.write(f"{img_path}\n")
    
    print(f"Selected {len(sample_images)} sample images for comparison")
    return sample_images


def generate_visual_comparisons(models, model_names, output_dir, device='cpu'):
    """
    Generate visual comparison results for sample images
    
    Args:
        models: List of models
        model_names: List of model names
        output_dir: Output directory
        device: Running device
    """
    print("\nGenerating visual comparison results...")
    
    samples_dir = output_dir / "samples"
    visual_dir = output_dir / "visual_comparison"
    visual_dir.mkdir(exist_ok=True)
    
    # Read sample image paths
    sample_images = []
    if os.path.exists(samples_dir / "sample_images.txt"):
        with open(samples_dir / "sample_images.txt", "r") as f:
            sample_images = [line.strip() for line in f.readlines()]
    
    if not sample_images:
        print("Warning: No sample images found, cannot generate visual comparison")
        return
    
    # Run all models on each sample image
    for i, img_path in enumerate(tqdm(sample_images, desc="Generating visual comparison")):
        if not os.path.exists(img_path):
            print(f"Warning: Image does not exist - {img_path}")
            continue
        
        # Read original image
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"Warning: Cannot read image - {img_path}")
            continue
        
        # Run prediction for each model
        model_results = []
        for j, model in enumerate(models):
            results = model.predict(img_path, conf=0.25, device=device, verbose=False)
            model_results.append(results[0])
        
        # Create grid to display all model results
        n_models = len(models)
        grid_rows = 1 + (n_models // 3) if n_models > 3 else 2  # At least 2 rows
        grid_cols = min(n_models, 3)  # Max 3 models per row
        
        # Calculate grid image size
        h, w = img_original.shape[:2]
        aspect_ratio = w / h
        grid_width = 1200
        cell_width = grid_width // grid_cols
        cell_height = int(cell_width / aspect_ratio)
        grid_height = cell_height * grid_rows
        
        # Create grid image
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # First row for original image
        img_resized = cv2.resize(img_original, (cell_width, cell_height))
        grid_img[0:cell_height, 0:cell_width] = img_resized
        
        # Add original image title
        cv2.putText(grid_img, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add each model's results
        for j, result in enumerate(model_results):
            # Calculate position in grid
            row = (j + 1) // grid_cols
            col = (j + 1) % grid_cols
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = col * cell_width
            x2 = x1 + cell_width
            
            # Get image with predictions
            pred_img = result.plot(conf=0.25, line_width=2, font_size=1, kpt_line=True, 
                                  kpt_radius=4)
            pred_img_resized = cv2.resize(pred_img, (cell_width, cell_height))
            
            # Place in grid
            grid_img[y1:y2, x1:x2] = pred_img_resized
            
            # Add model name
            cv2.putText(grid_img, model_names[j], (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save grid image
        output_file = visual_dir / f"comparison_{i+1:03d}.jpg"
        cv2.imwrite(str(output_file), grid_img)
    
    print(f"Visual comparison completed, results saved to: {visual_dir}")


def analyze_confidence_differences(models, model_names, data_yaml, output_dir, num_samples=5, device='cpu'):
    """
    Analyze keypoint confidence differences between models
    
    Args:
        models: List of models
        model_names: List of model names
        data_yaml: Data configuration file path
        output_dir: Output directory
        num_samples: Number of samples to display
        device: Running device
    """
    print("\nAnalyzing keypoint confidence differences between models...")
    
    # Create output directory
    conf_dir = output_dir / "confidence_analysis"
    conf_dir.mkdir(exist_ok=True)
    
    # Get validation dataset path from yaml
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(data_yaml)
    val_images = []
    
    # Get validation image paths
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
    
    # If no validation images found
    if not val_images:
        print("Warning: No validation images found, cannot analyze confidence differences")
        return
    
    # Randomly select images
    import random
    sample_size = min(len(val_images), 50)  # Randomly analyze 50 images
    selected_images = random.sample(val_images, sample_size)
    
    # Collect keypoint confidence for each model
    confidence_diffs = []
    
    for img_path in tqdm(selected_images, desc="Analyzing confidence differences"):
        if not os.path.exists(img_path):
            continue
        
        # Run prediction for each model
        all_kpt_confs = []
        for model in models:
            results = model.predict(img_path, conf=0.25, device=device, verbose=False)
            
            # If keypoints detected
            if len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0]  # Only take first detection object
                
                # Ensure there are keypoints
                if kpts.shape[0] > 0:
                    # Extract confidence values
                    conf_values = kpts[:, 2].cpu().numpy()
                    all_kpt_confs.append(conf_values)
                else:
                    all_kpt_confs.append(None)
            else:
                all_kpt_confs.append(None)
        
        # Calculate confidence differences between models
        if len(all_kpt_confs) == len(models) and all(x is not None for x in all_kpt_confs):
            # Ensure all keypoint configurations are the same
            if len(set(x.shape[0] for x in all_kpt_confs)) == 1:
                # Calculate standard deviation for each keypoint confidence
                kpt_stds = np.std(all_kpt_confs, axis=0)
                
                # Calculate average standard deviation
                avg_std = np.mean(kpt_stds)
                
                # Store cases with high standard deviation
                if avg_std > 0.1:  # Only focus on cases with higher standard deviation
                    confidence_diffs.append({
                        'img_path': img_path,
                        'kpt_stds': kpt_stds,
                        'avg_std': avg_std,
                        'confs': all_kpt_confs
                    })
    
    # Sort by average standard deviation
    if confidence_diffs:
        confidence_diffs.sort(key=lambda x: x['avg_std'], reverse=True)
        
        # Save results to file
        with open(conf_dir / "confidence_differences.txt", "w") as f:
            f.write("Model Keypoint Confidence Difference Analysis\n")
            f.write("=" * 80 + "\n\n")
            
            for i, diff in enumerate(confidence_diffs[:20]):  # Only show top 20 with largest differences
                f.write(f"Sample {i+1}:\n")
                f.write(f"Image: {diff['img_path']}\n")
                f.write(f"Average confidence std dev: {diff['avg_std']:.4f}\n")
                f.write(f"Keypoint std devs: {diff['kpt_stds']}\n\n")
        
        # Visualize cases with largest differences
        visualize_confidence_differences(models, model_names, 
                                        [d['img_path'] for d in confidence_diffs[:num_samples]], 
                                        conf_dir, num_samples, device)
    else:
        print("No significant confidence differences found")
    
    print(f"Confidence difference analysis completed, results saved to: {conf_dir}")


def visualize_confidence_differences(models, model_names, image_paths, output_dir, num_samples=5, device='cpu'):
    """
    Visualize significant keypoint confidence differences between models
    
    Args:
        models: List of models
        model_names: List of model names
        image_paths: List of image paths
        output_dir: Output directory
        num_samples: Number of samples
        device: Running device
    """
    print(f"\nVisualizing {min(num_samples, len(image_paths))} cases with significant confidence differences...")
    
    for i, img_path in enumerate(image_paths[:num_samples]):
        if not os.path.exists(img_path):
            print(f"Warning: Image does not exist - {img_path}")
            continue
        
        # Read original image
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"Warning: Cannot read image - {img_path}")
            continue
        
        h, w = img_original.shape[:2]
        
        # Run prediction for each model
        model_results = []
        all_kpt_confs = []
        
        for j, model in enumerate(models):
            results = model.predict(img_path, conf=0.25, device=device, verbose=False)
            model_results.append(results[0])
            
            # Extract keypoint confidence
            if len(results[0].keypoints) > 0:
                kpts = results[0].keypoints.data[0]
                if kpts.shape[0] > 0:
                    conf_values = kpts[:, 2].cpu().numpy()
                    all_kpt_confs.append(conf_values)
                else:
                    all_kpt_confs.append(None)
            else:
                all_kpt_confs.append(None)
        
        # Create grid to display all model results
        n_models = len(models)
        grid_rows = 1 + n_models  # First row for original image
        grid_cols = 1
        
        # Set grid size
        cell_height = 480
        cell_width = int(cell_height * (w / h))
        grid_height = cell_height * grid_rows
        grid_width = cell_width
        
        # Create grid image
        grid_img = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 255
        
        # First row for original image
        img_resized = cv2.resize(img_original, (cell_width, cell_height))
        grid_img[0:cell_height, 0:cell_width] = img_resized
        
        # Add original image title
        cv2.putText(grid_img, "Original Image", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add each model's results
        for j, result in enumerate(model_results):
            row = j + 1  # Start from second row
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = 0
            x2 = cell_width
            
            # Get image with predictions
            pred_img = result.plot(conf=0.25, line_width=2, font_size=1, kpt_line=True, 
                                  kpt_radius=4)
            pred_img_resized = cv2.resize(pred_img, (cell_width, cell_height))
            
            # Place in grid
            grid_img[y1:y2, x1:x2] = pred_img_resized
            
            # Add model name and confidence info
            model_title = f"{model_names[j]}"
            if all_kpt_confs[j] is not None:
                avg_conf = np.mean(all_kpt_confs[j])
                model_title += f" (Avg Conf: {avg_conf:.3f})"
            
            cv2.putText(grid_img, model_title, (x1 + 10, y1 + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Save grid image
        output_file = output_dir / f"conf_diff_{i+1:03d}.jpg"
        cv2.imwrite(str(output_file), grid_img)
    
    print(f"Confidence difference visualization completed, results saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Compare multiple yoga pose models and generate comparison report')
    parser.add_argument('--models', nargs='+', required=True, help='List of model weight paths (at least 2)')
    parser.add_argument('--data', type=str, required=True, help='Data YAML file path')
    parser.add_argument('--output-dir', type=str, default='model_comparison', help='Output directory')
    parser.add_argument('--img-size', type=int, default=640, help='Image size for validation')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--device', type=str, default='cpu', help='Running device (e.g., 0 or cpu)')
    parser.add_argument('--analyze-confidence', action='store_true', help='Analyze keypoint confidence differences between models')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples for confidence difference visualization')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Check if enough models provided
    if len(args.models) < 2:
        print("Error: At least two models need to be provided for comparison")
        sys.exit(1)
    
    # Perform model comparison
    models, model_names, validation_results = compare_models(
        model_paths=args.models,
        data_yaml=args.data,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # If confidence difference analysis requested
    if args.analyze_confidence:
        analyze_confidence_differences(
            models=models,
            model_names=model_names,
            data_yaml=args.data,
            output_dir=Path(args.output_dir),
            num_samples=args.num_samples,
            device=args.device
        )
    
    print("Model comparison completed!") 