import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add local paths to ensure using local version
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from ultralytics import YOLO
import torch
import cv2
from tqdm import tqdm

def validate_with_multiple_thresholds(model_path, data_yaml, img_size=1280, batch_size=16, device='0'):
    """Validate model with multiple confidence thresholds and generate performance plots"""
    model = YOLO(model_path)
    
    # Test with different confidence thresholds
    conf_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    kp_conf_thresholds = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
    
    results = {}
    print(f"Evaluating model: {model_path}")
    
    # Baseline validation with default settings
    baseline = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, device=device)
    print(f"Baseline mAP50-95: {baseline.box.map50_95:.4f}, mAP50: {baseline.box.map50:.4f}")
    
    # Test with different detection confidence thresholds
    print("\nTesting different detection confidence thresholds:")
    map_results = []
    for conf in conf_thresholds:
        metrics = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, 
                           device=device, conf=conf, verbose=False)
        map50_95 = metrics.box.map50_95
        map50 = metrics.box.map50
        print(f"Conf: {conf:.2f}, mAP50-95: {map50_95:.4f}, mAP50: {map50:.4f}")
        map_results.append((conf, map50_95, map50))
    
    results['conf_thresholds'] = map_results
    
    # Test with different keypoint confidence thresholds
    print("\nTesting different keypoint confidence thresholds:")
    kp_results = []
    for kp_conf in kp_conf_thresholds:
        metrics = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, 
                           device=device, conf=0.2, kpt_conf=kp_conf, verbose=False)
        map50_95 = metrics.box.map50_95
        map50 = metrics.box.map50
        print(f"KP Conf: {kp_conf:.3f}, mAP50-95: {map50_95:.4f}, mAP50: {map50:.4f}")
        kp_results.append((kp_conf, map50_95, map50))
    
    results['kp_conf_thresholds'] = kp_results
    
    # Plot results
    plot_thresholds(results, save_dir=Path(os.path.dirname(model_path)))
    
    return results

def plot_thresholds(results, save_dir):
    """Plot performance metrics against different threshold values"""
    save_dir = Path(save_dir) / 'validation_plots'
    save_dir.mkdir(exist_ok=True)
    
    # Plot detection confidence thresholds vs mAP
    if 'conf_thresholds' in results:
        conf_data = np.array(results['conf_thresholds'])
        plt.figure(figsize=(10, 6))
        plt.plot(conf_data[:, 0], conf_data[:, 1], 'o-', label='mAP50-95')
        plt.plot(conf_data[:, 0], conf_data[:, 2], 's-', label='mAP50')
        plt.xlabel('Confidence Threshold')
        plt.ylabel('mAP')
        plt.title('Detection Confidence Threshold vs mAP')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'detection_conf_map.png', dpi=300, bbox_inches='tight')
        
    # Plot keypoint confidence thresholds vs mAP
    if 'kp_conf_thresholds' in results:
        kp_data = np.array(results['kp_conf_thresholds'])
        plt.figure(figsize=(10, 6))
        plt.plot(kp_data[:, 0], kp_data[:, 1], 'o-', label='mAP50-95')
        plt.plot(kp_data[:, 0], kp_data[:, 2], 's-', label='mAP50')
        plt.xlabel('Keypoint Confidence Threshold')
        plt.ylabel('mAP')
        plt.title('Keypoint Confidence Threshold vs mAP')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir / 'keypoint_conf_map.png', dpi=300, bbox_inches='tight')
    
    print(f"Plots saved to {save_dir}")

def visualize_predictions(model_path, data_yaml, output_dir=None, num_samples=10, conf=0.2, kpt_conf=0.05):
    """Visualize model predictions on validation samples"""
    model = YOLO(model_path)
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(model_path), 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get validation dataset from yaml
    from ultralytics.data.utils import check_det_dataset
    data_dict = check_det_dataset(data_yaml)
    val_images = []
    
    # Get validation image paths
    if 'val' in data_dict:
        import glob
        if isinstance(data_dict['val'], str):
            # If val points to a folder
            val_path = data_dict['val']
            if os.path.isdir(val_path):
                val_images = glob.glob(os.path.join(val_path, '**/*.jpg'), recursive=True)
                val_images += glob.glob(os.path.join(val_path, '**/*.png'), recursive=True)
            # If val points to a text file
            elif os.path.isfile(val_path) and val_path.endswith('.txt'):
                with open(val_path, 'r') as f:
                    lines = f.readlines()
                val_images = [line.strip() for line in lines]
    
    # If no validation images found or fewer than required
    if len(val_images) < num_samples:
        print(f"Warning: Found only {len(val_images)} validation images. Using available images.")
        num_samples = min(num_samples, len(val_images))
    
    # Randomly select images
    import random
    selected_images = random.sample(val_images, num_samples) if val_images else []
    
    for i, img_path in enumerate(selected_images):
        print(f"Processing image {i+1}/{num_samples}: {img_path}")
        
        # Run prediction
        results = model.predict(img_path, conf=conf, kpt_conf=kpt_conf, save=False, device='0')
        
        # Get image with predictions
        for r in results:
            im_array = r.plot(conf=conf, line_width=2, font_size=1, kpt_line=True, 
                             kpt_radius=4, kpt_line_thickness=2)
            
            # Save visualization
            output_file = os.path.join(output_dir, f"pred_{i:03d}.jpg")
            cv2.imwrite(output_file, im_array)
    
    print(f"Visualizations saved to {output_dir}")

def analyze_class_performance(model_path, data_yaml, img_size=1280, batch_size=16, device='0'):
    """Analyze performance by class to identify problematic yoga poses"""
    model = YOLO(model_path)
    
    # Run validation with per-class metrics
    metrics = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, 
                       device=device, verbose=True)
    
    # Extract per-class metrics if available
    if hasattr(metrics, 'per_class') and metrics.per_class:
        from ultralytics.data.utils import check_det_dataset
        data_dict = check_det_dataset(data_yaml)
        class_names = data_dict.get('names', {})
        
        # Get precision and recall by class
        precisions = metrics.per_class.get('precision', [])
        recalls = metrics.per_class.get('recall', [])
        
        print("\nPer-class Performance:")
        print("Class                  Precision    Recall")
        print("-" * 45)
        
        for i, (prec, rec) in enumerate(zip(precisions, recalls)):
            class_name = class_names.get(i, f"Class {i}")
            print(f"{class_name[:20]:<20}    {prec:.4f}    {rec:.4f}")
            
        # Plot per-class performance
        save_dir = Path(os.path.dirname(model_path)) / 'validation_plots'
        save_dir.mkdir(exist_ok=True)
        
        # Convert class indices to names for plotting
        class_labels = [class_names.get(i, f"Class {i}") for i in range(len(precisions))]
        
        # Sort classes by precision for better visualization
        sorted_indices = np.argsort(precisions)
        sorted_classes = [class_labels[i] for i in sorted_indices]
        sorted_precisions = [precisions[i] for i in sorted_indices]
        sorted_recalls = [recalls[i] for i in sorted_indices]
        
        # Plot
        plt.figure(figsize=(12, 8))
        x = np.arange(len(sorted_classes))
        width = 0.35
        
        plt.bar(x - width/2, sorted_precisions, width, label='Precision')
        plt.bar(x + width/2, sorted_recalls, width, label='Recall')
        
        plt.xlabel('Yoga Pose Class')
        plt.ylabel('Score')
        plt.title('Precision and Recall by Yoga Pose Class')
        plt.xticks(x, sorted_classes, rotation=90)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        print(f"Per-class performance plot saved to {save_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Yoga Pose Model with Various Metrics')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, required=True, help='Path to data YAML file')
    parser.add_argument('--img-size', type=int, default=1280, help='Image size for validation')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--device', type=str, default='0', help='Device for validation (e.g., 0 or 0,1)')
    parser.add_argument('--visualize', action='store_true', help='Generate prediction visualizations')
    parser.add_argument('--num-vis', type=int, default=10, help='Number of visualizations to generate')
    parser.add_argument('--analyze-classes', action='store_true', help='Analyze per-class performance')
    
    args = parser.parse_args()
    
    # Run validation with threshold testing
    validate_with_multiple_thresholds(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # Generate visualizations if requested
    if args.visualize:
        visualize_predictions(
            model_path=args.model,
            data_yaml=args.data,
            num_samples=args.num_vis
        )
    
    # Analyze per-class performance if requested
    if args.analyze_classes:
        analyze_class_performance(
            model_path=args.model,
            data_yaml=args.data,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device
        ) 