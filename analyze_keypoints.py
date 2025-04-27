#!/usr/bin/env python
# Keypoint Analysis Script for Yoga Pose Detection Models
# This script analyzes the performance of a YOLO keypoint detection model on yoga poses

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

from ultralytics import YOLO
from ultralytics.utils.metrics import bbox_iou


def analyze_keypoints(model_path, data_yaml, img_size=640, batch_size=16, device='0', 
                     find_difficult=False, num_difficult=10, repredict=False):
    """
    Analyze keypoint detection performance of a YOLO model on a dataset.
    
    Args:
        model_path: Path to the model weights (.pt file)
        data_yaml: Path to the data YAML file
        img_size: Image size for validation
        batch_size: Batch size for validation
        device: Device to run validation on ('cpu' or GPU index)
        find_difficult: Whether to find and visualize difficult poses
        num_difficult: Number of difficult poses to visualize
        repredict: Whether to re-run prediction to match predict.py behavior
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Create output directory
    output_dir = Path('keypoint_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Get validation dataset name
    dataset_name = Path(data_yaml).stem
    
    # Process keypoint detection statistics
    keypoint_stats = process_keypoint_stats(model, data_yaml)
    
    # Visualize keypoint statistics
    visualize_keypoint_stats(keypoint_stats, output_dir, dataset_name)
    
    # Find and visualize difficult poses if requested
    if find_difficult:
        print("Finding difficult poses...")
        difficult_poses = find_difficult_poses(model, data_yaml, img_size, device, num_difficult, output_dir)
        visualize_difficult_poses(difficult_poses, model, output_dir, repredict=repredict)
    
    print(f"Analysis complete. Results saved to {output_dir}")


def process_keypoint_stats(model, data_yaml):
    """
    Process keypoint detection statistics from validation results.
    
    Args:
        model: YOLO model
        data_yaml: Path to the data YAML file
    
    Returns:
        Dictionary containing keypoint detection statistics
    """
    # Define standard keypoint names for pose models
    standard_keypoint_names = {
        0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
        5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
        9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
        13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
    }
    
    # Check if this is a pose model
    is_pose_model = False
    if hasattr(model, 'task') and model.task == 'pose':
        is_pose_model = True
        print("Detected pose model")
    elif hasattr(model, 'model') and hasattr(model.model, 'kpt_shape'):
        is_pose_model = True
        print(f"Detected pose model with kpt_shape: {model.model.kpt_shape}")
    
    # Get model's detection class names
    class_names = model.names
    print(f"Model class names: {class_names}")
    
    # For pose models, get the keypoint shape if available
    num_keypoints = 17  # Default COCO keypoints
    if hasattr(model, 'model') and hasattr(model.model, 'kpt_shape'):
        num_keypoints = model.model.kpt_shape[0]
        print(f"Model has {num_keypoints} keypoints")
    
    # Initialize keypoint statistics dictionary
    keypoint_stats = {
        'detection_rate': np.zeros(num_keypoints),
        'avg_confidence': np.zeros(num_keypoints),
        'names': {i: standard_keypoint_names.get(i, f'kpt_{i}') for i in range(num_keypoints)}
    }
    
    print(f"Collecting stats for {num_keypoints} keypoints")
    
    from pathlib import Path
    import yaml
    
    # Try to load the data YAML
    try:
        data_path = Path(data_yaml)
        if data_path.exists():
            with open(data_path, 'r') as f:
                data = yaml.safe_load(f)
                print(f"Loaded data YAML: {data_path}")
        else:
            print(f"Data YAML not found at {data_path}, using default paths")
            data = {"path": "/Users/region/yolo11-pose-distiller/datasets/yoga82", "val": "images/val"}
    except Exception as e:
        print(f"Error loading data YAML: {e}")
        data = {"path": "/Users/region/yolo11-pose-distiller/datasets/yoga82", "val": "images/val"}
    
    # Find validation images
    try:
        base_path = Path(data.get("path", "."))
        if not base_path.exists() and '../' in str(base_path):
            # Try to handle relative paths
            base_path = Path.cwd().parent / base_path.name
        
        val_dir = base_path / data.get("val", "images/val")
        
        # Fallback options if the specified path doesn't exist
        if not val_dir.exists():
            hardcoded_path = Path("/Users/region/yolo11-pose-distiller/datasets/yoga82/images/val")
            if hardcoded_path.exists():
                val_dir = hardcoded_path
                print(f"Using hardcoded path: {val_dir}")
        
        if val_dir.exists():
            val_imgs = list(val_dir.glob('*.jpg'))[:30]  # Use 30 images for better statistics
            if val_imgs:
                print(f"Running inference on {len(val_imgs)} images to collect keypoint statistics...")
                
                # Run inference with this model on the validation images
                preds = model.predict(val_imgs, verbose=False)
                
                # Initialize counters for statistics
                total_instances = 0
                detected_keypoints = [0] * num_keypoints
                confidence_sums = [0] * num_keypoints
                
                # Process the predictions
                for pred in preds:
                    if hasattr(pred, 'keypoints') and len(pred.keypoints) > 0:
                        kpts_data = pred.keypoints.data
                        
                        # Count instances
                        num_instances = kpts_data.shape[0]
                        total_instances += num_instances
                        
                        # For each keypoint index
                        for k in range(num_keypoints):
                            if k < kpts_data.shape[1]:  # Ensure this keypoint index exists
                                # Get visibility/confidence of this keypoint across all instances
                                if len(kpts_data.shape) == 3:  # [instances, keypoints, coords]
                                    confidences = kpts_data[:, k, 2]
                                    visible_mask = confidences > 0
                                    
                                    # Count detected keypoints and sum confidences
                                    detected = visible_mask.sum().item()
                                    detected_keypoints[k] += detected
                                    
                                    if detected > 0:
                                        confidence_sums[k] += confidences[visible_mask].sum().item()
                
                # Calculate final statistics
                if total_instances > 0:
                    print(f"Processed {total_instances} person instances")
                    
                    for k in range(num_keypoints):
                        # Detection rate: what percentage of instances had this keypoint visible
                        keypoint_stats['detection_rate'][k] = detected_keypoints[k] / total_instances if total_instances > 0 else 0
                        
                        # Average confidence among detected keypoints
                        keypoint_stats['avg_confidence'][k] = confidence_sums[k] / detected_keypoints[k] if detected_keypoints[k] > 0 else 0
                    
                    print("Successfully collected keypoint statistics")
                else:
                    print("No person instances found in the sample images")
            else:
                print("No validation images found")
        else:
            print(f"Validation directory not found at {val_dir}")
            
    except Exception as e:
        print(f"Error collecting keypoint statistics: {e}")
    
    # Print stats for debugging
    print("\nKeypoint Statistics Summary:")
    for i, name in enumerate(keypoint_stats['names'].values()):
        if i < num_keypoints:
            print(f"{name} - Detection Rate: {keypoint_stats['detection_rate'][i]:.4f}, Avg Confidence: {keypoint_stats['avg_confidence'][i]:.4f}")
    
    return keypoint_stats


def visualize_keypoint_stats(keypoint_stats, output_dir, dataset_name):
    """
    Visualize keypoint detection statistics.
    
    Args:
        keypoint_stats: Dictionary containing keypoint detection statistics
        output_dir: Output directory to save visualizations
        dataset_name: Name of the dataset
    """
    names = keypoint_stats['names']
    keypoint_names = [names[i] for i in range(len(names))]
    detection_rates = keypoint_stats['detection_rate']
    avg_confidences = keypoint_stats['avg_confidence']
    
    # Create a vibrant colormap
    import matplotlib.cm as cm
    
    # Create bar plots for detection rates with vibrant colors
    plt.figure(figsize=(16, 8))
    bars = plt.bar(keypoint_names, detection_rates, color=cm.viridis(detection_rates))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Keypoint Detection Rates - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Keypoint', fontsize=12)
    plt.ylabel('Detection Rate', fontsize=12)
    plt.ylim(0, 1.1)  # Add space for the labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'detection_rates.png'), dpi=200)
    plt.close()
    
    # Create bar plots for average confidences with vibrant colors
    plt.figure(figsize=(16, 8))
    bars = plt.bar(keypoint_names, avg_confidences, color=cm.plasma(avg_confidences))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Keypoint Average Confidence - {dataset_name}', fontsize=16, fontweight='bold')
    plt.xlabel('Keypoint', fontsize=12)
    plt.ylabel('Average Confidence', fontsize=12)
    plt.ylim(0, 1.1)  # Add space for the labels
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(str(output_dir / 'avg_confidences.png'), dpi=200)
    plt.close()
    
    # Create a combined visualization with both metrics
    plt.figure(figsize=(18, 10))
    
    x = np.arange(len(keypoint_names))  # the label locations
    width = 0.35  # the width of the bars
    
    fig, ax = plt.subplots(figsize=(18, 10))
    rects1 = ax.bar(x - width/2, detection_rates, width, label='Detection Rate', color=cm.viridis(detection_rates))
    rects2 = ax.bar(x + width/2, avg_confidences, width, label='Average Confidence', color=cm.plasma(avg_confidences))
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(f'Keypoint Performance - {dataset_name}', fontsize=18, fontweight='bold')
    ax.set_xlabel('Keypoint', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(keypoint_names, rotation=45, ha='right', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(str(output_dir / 'keypoint_performance.png'), dpi=200)
    plt.close()
    
    # Save statistics to a text file
    with open(output_dir / 'keypoint_stats.txt', 'w') as f:
        f.write(f"Keypoint Detection Statistics for {dataset_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Detection Rates:\n")
        for i, name in enumerate(keypoint_names):
            f.write(f"{name:15}: {detection_rates[i]:.4f}\n")
        
        f.write("\nAverage Confidences:\n")
        for i, name in enumerate(keypoint_names):
            f.write(f"{name:15}: {avg_confidences[i]:.4f}\n")


def find_difficult_poses(model, data_yaml, img_size, device, num_difficult, output_dir):
    """
    Find difficult poses in the validation dataset based on both confidence scores and ground truth comparison.
    
    Args:
        model: YOLO model
        data_yaml: Path to the data YAML file
        img_size: Image size for validation
        device: Device to run validation on
        num_difficult: Number of difficult poses to visualize
        output_dir: Output directory to save visualizations
    
    Returns:
        List of difficult poses (image paths and detection difficulties)
    """
    import yaml
    from pathlib import Path
    import glob
    from tqdm import tqdm
    import os
    import re
    import numpy as np
    import torch
    import logging
    
    # Set up logging
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'keypoint_predictions.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('keypoint_analysis')
    
    logger.info(f"Starting difficult pose analysis with model: {model}")
    logger.info(f"Data YAML: {data_yaml}")
    
    # Load the data YAML file
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    # Print the data structure to debug
    logger.info(f"Dataset config: {data}")
    
    # Get actual absolute paths from validation cache
    cache_pattern = re.compile(r'Scanning\s+([^\s]+labels/val\.cache)')
    
    # Try to find the base path from validation output
    base_path = None
    
    # Check if we can extract the path from validation cache
    if not base_path:
        # Try environment variable
        try:
            import sys
            for arg in sys.argv:
                if arg.endswith('.yaml') and not arg.endswith(Path(data_yaml).name):
                    with open(arg, 'r') as f:
                        other_data = yaml.safe_load(f)
                        if 'path' in other_data and os.path.exists(other_data['path']):
                            base_path = Path(other_data['path'])
                            print(f"Found base path from other YAML: {base_path}")
                            break
        except Exception as e:
            print(f"Error finding path from other YAML: {e}")
    
    # Try to use the actual cache path from model validation message
    if not base_path:
        try:
            import subprocess
            cmd = f"python -c \"import os; print(os.path.expanduser('~'))\""
            home = subprocess.check_output(cmd, shell=True).decode().strip()
            
            # Try both paths (at home and relative to working directory)
            potential_paths = [
                Path(home) / "yolo11-pose-distiller" / "datasets" / "yoga82",
                Path.cwd().parent / "datasets" / "yoga82",
                Path("/Users/region/yolo11-pose-distiller/datasets/yoga82"),  # Hardcoded from logs
            ]
            
            for p in potential_paths:
                if (p / "labels" / "val.cache").exists():
                    base_path = p
                    print(f"Found base path from validation logs: {base_path}")
                    break
        except Exception as e:
            print(f"Error finding path from validation logs: {e}")
    
    # If still no base path, use YAML path as last resort
    if not base_path:
        # Get the validation image directory from YAML
        data_root = Path(data_yaml).parent if Path(data_yaml).is_file() else Path(data_yaml)
        if 'path' in data:
            base_path = data_root / data['path'].lstrip('..')
            if not base_path.exists() and '../' in str(data['path']):
                # Try to resolve relative path
                base_path = Path(data_root).resolve().parent / data['path'].replace('../', '')
        else:
            base_path = data_root
    
    print(f"Using base path: {base_path} (exists: {base_path.exists() if base_path else False})")
    
    # Determine validation image path
    val_img_paths = []
    
    # First priority: directly check for the cached path we saw in logs
    hardcoded_path = Path("/Users/region/yolo11-pose-distiller/datasets/yoga82/images/val")
    if hardcoded_path.exists():
        print(f"Using hardcoded path from logs: {hardcoded_path}")
        val_img_paths = list(hardcoded_path.glob('*.[jJ][pP][gG]')) + \
                    list(hardcoded_path.glob('*.[jJ][pP][eE][gG]')) + \
                    list(hardcoded_path.glob('*.[pP][nN][gG]')) + \
                    list(hardcoded_path.glob('*.[bB][mM][pP]'))
    
    # If not found via hardcoded path, try with base_path
    if not val_img_paths and base_path:
        if 'val' in data:
            val_path = base_path / data['val']
            print(f"Checking validation path: {val_path} (exists: {val_path.exists()})")
            
            if val_path.is_dir():
                val_img_paths = list(val_path.glob('*.[jJ][pP][gG]')) + \
                              list(val_path.glob('*.[jJ][pP][eE][gG]')) + \
                              list(val_path.glob('*.[pP][nN][gG]')) + \
                              list(val_path.glob('*.[bB][mM][pP]'))
    
    # If still no images, try inference from the cache path
    if not val_img_paths:
        # Search all possible locations for the cache file
        cache_locations = [
            Path("/Users/region/yolo11-pose-distiller/datasets/yoga82/labels/val.cache"),
            base_path / "labels" / "val.cache" if base_path else None
        ]
        
        for cache_path in cache_locations:
            if cache_path and cache_path.exists():
                print(f"Found validation cache at {cache_path}")
                # Try to find matching image directory (replacing 'labels' with 'images')
                img_dir = str(cache_path).replace("labels", "images").replace("val.cache", "val")
                img_path = Path(img_dir)
                
                if img_path.exists() and img_path.is_dir():
                    print(f"Found image directory at {img_path}")
                    val_img_paths = list(img_path.glob('*.[jJ][pP][gG]')) + \
                                  list(img_path.glob('*.[jJ][pP][eE][gG]')) + \
                                  list(img_path.glob('*.[pP][nN][gG]')) + \
                                  list(img_path.glob('*.[bB][mM][pP]'))
                    if val_img_paths:
                        break
    
    # Print first few paths if available to debug
    if val_img_paths:
        print(f"Found {len(val_img_paths)} validation images.")
        print(f"First few image paths: {val_img_paths[:3]}")
    else:
        print("ERROR: Could not find any validation images after trying all possible paths.")
        print("Please specify the correct path to the validation images using the --val-path argument.")
        return []
    
    # Verify the images exist
    val_img_paths = [p for p in val_img_paths if os.path.exists(p)]
    if not val_img_paths:
        print("ERROR: None of the found image paths actually exist on disk.")
        return []
    
    # Function to load ground truth keypoints from label file
    def load_ground_truth(img_path):
        # Convert image path to label path
        label_path = str(img_path).replace('/images/', '/labels/').replace(
            img_path.suffix, '.txt')
        
        if not os.path.exists(label_path):
            print(f"Ground truth label not found: {label_path}")
            return None
            
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            if not lines:
                print(f"Empty label file: {label_path}")
                return None
                
            # Parse the keypoints from the first line (assuming single person per image)
            parts = lines[0].strip().split()
            if len(parts) < 5:  # Need class_id, bbox (4 values), and at least some keypoints
                print(f"Invalid label format in {label_path}: insufficient data")
                return None
                
            # Format: class_id, bbox_center_x, bbox_center_y, bbox_width, bbox_height, kpt1_x, kpt1_y, kpt1_conf, ...
            # Skip the first 5 elements (class + bbox)
            keypoints_flat = [float(x) for x in parts[5:]]
            
            # Sanity check on number of keypoints
            if len(keypoints_flat) % 3 != 0:
                print(f"Warning: Keypoint data in {label_path} is not a multiple of 3")
                # Truncate to nearest multiple of 3
                keypoints_flat = keypoints_flat[:(len(keypoints_flat) // 3) * 3]
            
            # Convert flat array to [x, y, conf] format
            num_keypoints = len(keypoints_flat) // 3
            keypoints = []
            for i in range(num_keypoints):
                idx = i * 3
                # Ensure values are in proper range
                x = max(0.0, min(1.0, keypoints_flat[idx]))      # Normalize to [0,1]
                y = max(0.0, min(1.0, keypoints_flat[idx + 1]))  # Normalize to [0,1]
                conf = max(0.0, min(1.0, keypoints_flat[idx + 2]))  # Normalize to [0,1]
                
                keypoints.append([x, y, conf])
            
            print(f"Loaded {num_keypoints} ground truth keypoints from {label_path}")
            return np.array(keypoints)
        except Exception as e:
            print(f"Error loading ground truth from {label_path}: {e}")
            return None
    
    # Function to calculate keypoint distance between prediction and ground truth
    def calculate_keypoint_error(pred_kpts, gt_kpts, img_width, img_height):
        """
        Calculate the distance error between predicted and ground truth keypoints.
        
        Args:
            pred_kpts: Predicted keypoints in pixel coordinates [x, y, conf]
            gt_kpts: Ground truth keypoints in normalized coordinates [x, y, conf]
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Average normalized error and list of per-keypoint errors
        """
        if pred_kpts is None or gt_kpts is None:
            return float('inf'), []
        
        # Calculate error for each keypoint
        kpt_errors = []
        
        # For each keypoint that exists in both prediction and ground truth
        min_length = min(len(pred_kpts), len(gt_kpts))
        for i in range(min_length):
            # Only consider keypoints with confidence > 0 in both pred and GT
            if pred_kpts[i][2] > 0 and gt_kpts[i][2] > 0:
                # Convert prediction from pixel to normalized coordinates for fair comparison
                pred_x_norm = pred_kpts[i][0] / img_width
                pred_y_norm = pred_kpts[i][1] / img_height
                
                # Ground truth is already in normalized coordinates
                gt_x_norm = gt_kpts[i][0]
                gt_y_norm = gt_kpts[i][1]
                
                # Calculate Euclidean distance in normalized space (0-1 range)
                # This directly gives us a normalized error metric
                norm_dist = np.sqrt(
                    (pred_x_norm - gt_x_norm)**2 +
                    (pred_y_norm - gt_y_norm)**2
                )
                
                kpt_errors.append((i, norm_dist))
        
        # Calculate average error
        if kpt_errors:
            avg_error = sum(err for _, err in kpt_errors) / len(kpt_errors)
            # Clamp the average error to [0,1] range
            avg_error = min(1.0, avg_error)
            return avg_error, kpt_errors
        else:
            return float('inf'), []
    
    # Limit the number of images to process for efficiency
    max_images = min(500, len(val_img_paths))
    val_img_paths = val_img_paths[:max_images]
    print(f"Processing {len(val_img_paths)} validation images to find difficult poses...")
    
    difficult_poses = []
    
    # Process images in batches with progress bar
    batch_size = 4  # Small batch size for memory efficiency
    total_batches = (len(val_img_paths) + batch_size - 1) // batch_size
    
    # Create progress bar
    pbar = tqdm(total=total_batches, desc="Finding difficult poses")
    
    for i in range(0, len(val_img_paths), batch_size):
        batch_paths = val_img_paths[i:i+batch_size]
        
        # Run model inference on batch
        logger.info(f"Running inference on batch {i//batch_size + 1}/{total_batches}")
        
        # Use more consistent prediction settings to match predict.py
        # Including auto sizing, all visualization features, and proper preprocessing
        results = model.predict(
            batch_paths, 
            imgsz=img_size,
            device=device, 
            verbose=False,
            conf=0.25,      # Confidence threshold
            iou=0.45,       # NMS IoU threshold
            half=False,     # Use FP16 half-precision inference
            augment=False,  # Augmented inference
            save=False      # Don't save separately
        )
        
        # Process results
        for j, result in enumerate(results):
            img_path = batch_paths[j]
            filename = os.path.basename(img_path)
            
            # Load ground truth keypoints
            gt_kpts = load_ground_truth(img_path)
            
            if len(result.keypoints) > 0:
                # Get predicted keypoints
                pred_kpts = result.keypoints.data[0].cpu().numpy()
                
                # Log detailed keypoint information for debugging
                logger.info(f"File: {filename}")
                logger.info(f"Image shape: {result.orig_shape}")
                
                # Skip if no keypoints detected
                if pred_kpts.shape[0] == 0:
                    logger.info("No keypoints detected.")
                    continue
                
                # Log each keypoint in detail
                logger.info("Keypoint predictions (idx: x, y, confidence):")
                for k_idx, (x, y, conf) in enumerate(pred_kpts):
                    logger.info(f"  Keypoint {k_idx}: ({x:.1f}, {y:.1f}), conf={conf:.4f}")
                
                # Log ground truth if available
                if gt_kpts is not None:
                    logger.info("Ground truth keypoints (idx: x, y, confidence):")
                    img_width, img_height = result.orig_shape[1], result.orig_shape[0]
                    for k_idx, (x, y, conf) in enumerate(gt_kpts):
                        # Convert normalized coordinates to pixel for easier comparison
                        pixel_x, pixel_y = x * img_width, y * img_height
                        logger.info(f"  Keypoint {k_idx}: ({x:.4f}, {y:.4f}) -> ({pixel_x:.1f}, {pixel_y:.1f}), conf={conf:.4f}")
                
                # Calculate confidence-based difficulty score
                valid_kpts = pred_kpts[:, 2] > 0
                if valid_kpts.sum() > 0 and gt_kpts is not None:
                    avg_conf = pred_kpts[valid_kpts, 2].mean().item()
                    conf_difficulty = 1 - avg_conf
                    
                    # Calculate geometric error compared to ground truth
                    try:
                        # Try to get original image dimensions from result
                        if hasattr(result, 'orig_shape'):
                            img_width, img_height = result.orig_shape[1], result.orig_shape[0]
                        else:
                            # Load image to get dimensions
                            img = cv2.imread(str(img_path))
                            if img is not None:
                                img_height, img_width = img.shape[:2]
                            else:
                                # Default to a standard size if image can't be loaded
                                img_width, img_height = 640, 640
                        
                        avg_error, kpt_errors = calculate_keypoint_error(pred_kpts, gt_kpts, img_width, img_height)
                        
                        # Log error details
                        logger.info(f"Average error: {avg_error:.4f}")
                        logger.info("Per-keypoint errors (idx: error):")
                        for k_idx, err in kpt_errors:
                            logger.info(f"  Keypoint {k_idx}: error={err:.4f}")
                            
                    except Exception as e:
                        logger.error(f"Error calculating keypoint error: {e}")
                        avg_error, kpt_errors = float('inf'), []
                    
                    # Normalize error to a 0-1 scale
                    # No need for arbitrary thresholds since error is already normalized by image diagonal
                    error_score = avg_error  # Already clamped to [0,1] in calculate_keypoint_error
                    
                    # Combine the two scores - give more weight to geometric error
                    # Use a weighted average of confidence difficulty and error score
                    # Lower confidence and higher error both increase difficulty
                    combined_difficulty = 0.3 * conf_difficulty + 0.7 * error_score
                    
                    # Ensure combined difficulty is in [0,1] range
                    combined_difficulty = min(1.0, max(0.0, combined_difficulty))
                    
                    logger.info(f"Difficulty scores - Confidence: {conf_difficulty:.4f}, Error: {error_score:.4f}, Combined: {combined_difficulty:.4f}")
                    
                    # Store image path and difficulty scores
                    difficult_poses.append({
                        'img_path': str(img_path),
                        'difficulty': combined_difficulty,
                        'conf_difficulty': conf_difficulty,
                        'error_score': error_score,
                        'kpts': pred_kpts,
                        'gt_kpts': gt_kpts,
                        'kpt_errors': kpt_errors,
                        'orig_shape': result.orig_shape,  # Store original shape from result
                        'img_size': img_size              # Store model input size
                    })
                    
                    logger.info(f"Added to difficult poses list, current count: {len(difficult_poses)}")
                    logger.info("-" * 50)
        
        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({"found": len(difficult_poses)})
            
        # Limit the number of difficult poses to find (with some margin)
        if len(difficult_poses) >= num_difficult * 3:
            break
    
    # Close progress bar
    pbar.close()
    
    # Sort by difficulty (highest first) and take the top N
    difficult_poses.sort(key=lambda x: x['difficulty'], reverse=True)
    selected_poses = difficult_poses[:num_difficult]
    
    logger.info(f"Selected {len(selected_poses)} most difficult poses for visualization")
    for i, pose in enumerate(selected_poses):
        filename = os.path.basename(pose['img_path'])
        logger.info(f"Difficult pose {i+1}: {filename}, Score: {pose['difficulty']:.4f}")
    
    return selected_poses


def visualize_difficult_poses(difficult_poses, model, output_dir, repredict=False):
    """
    Visualize difficult poses with ground truth comparison.
    
    Args:
        difficult_poses: List of difficult poses
        model: YOLO model
        output_dir: Output directory to save visualizations
        repredict: Whether to run prediction again for consistent results with predict.py
    """
    if not difficult_poses:
        print("No difficult poses found.")
        return
    
    # Create directory for difficult poses
    difficult_dir = output_dir / 'difficult_poses'
    difficult_dir.mkdir(exist_ok=True)
    
    # Create directory for full diagnostics reports
    diagnostics_dir = output_dir / 'diagnostics'
    diagnostics_dir.mkdir(exist_ok=True)
    
    # Set up logging
    import logging
    log_dir = output_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / 'visualization.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('pose_visualization')
    
    # Get keypoint names from model
    try:
        keypoint_names = model.names
    except:
        keypoint_names = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
        }
    
    logger.info(f"Starting visualization of {len(difficult_poses)} difficult poses")
    logger.info(f"Using keypoint names: {keypoint_names}")
    logger.info(f"Repredict option is {'enabled' if repredict else 'disabled'}")
    
    # Set visibility threshold for keypoints - only show keypoints with confidence above this
    visibility_threshold = 0.1  # Adjust this value to filter out low-confidence detections
    logger.info(f"Using visibility threshold: {visibility_threshold}")
    
    # Create colormap for keypoint confidence visualization
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import numpy as np
    import os
    import matplotlib.patches as mpatches
    
    # Create a custom colormap with more vibrant colors
    colors = ['blue', 'cyan', 'lime', 'yellow', 'red']
    cmap_name = 'confidence'
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
    
    # Create error colormap
    error_cmap = plt.cm.Reds
    
    # Write report file
    with open(difficult_dir / 'difficult_poses_report.txt', 'w') as f:
        f.write("Most Difficult Poses Detection Report\n")
        f.write("=============================\n\n")
        f.write("Scoring criteria: This report combines keypoint prediction confidence and error compared to ground truth.\n")
        f.write("Higher difficulty scores indicate poses where accurate keypoint detection is more challenging.\n\n")
        
        for i, pose in enumerate(difficult_poses):
            # Load the image
            img_path = pose['img_path']
            filename = os.path.basename(img_path)
            logger.info(f"Processing difficult pose {i+1}: {filename}")
            
            # Create a file to store detailed diagnostics for this image
            diagnostics_file = diagnostics_dir / f"diagnostics_{filename}.txt"
            with open(diagnostics_file, 'w') as diag_f:
                diag_f.write(f"DETAILED KEYPOINT DIAGNOSTICS FOR {filename}\n")
                diag_f.write("="*80 + "\n\n")
                
                # Load original image for display
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    logger.warning(f"Could not load image at {img_path}, skipping.")
                    continue
                
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = orig_img.shape[:2]
                logger.info(f"Original image dimensions: {orig_w}x{orig_h}")
                diag_f.write(f"Original image dimensions: {orig_w}x{orig_h}\n\n")
                
                # Store original prediction
                original_pred = pose['kpts'].copy()
                diag_f.write("ORIGINAL PREDICTION (from find_difficult_poses):\n")
                for k in range(len(original_pred)):
                    x, y, conf = original_pred[k]
                    name = keypoint_names.get(k, f"Keypoint {k}")
                    diag_f.write(f"  {k}: {name}, ({x:.1f}, {y:.1f}), conf={conf:.4f}\n")
                
                # Add sections for low confidence and zero coordinate points
                zero_coords = [(i, original_pred[i, 2]) for i in range(len(original_pred)) 
                              if original_pred[i, 0] == 0 and original_pred[i, 1] == 0]
                
                if zero_coords:
                    diag_f.write("\nKeypoints with (0,0) coordinates in original prediction:\n")
                    for idx, conf in zero_coords:
                        diag_f.write(f"  Keypoint {idx}: confidence={conf:.4f}\n")
                
                # Optionally re-run prediction to exactly match predict.py behavior
                if repredict:
                    logger.info("Re-running prediction to exactly match predict.py behavior")
                    diag_f.write("\n" + "="*40 + "\n")
                    diag_f.write("RE-PREDICTION (to match predict.py):\n")
                    diag_f.write("="*40 + "\n\n")
                    
                    img_size = 640  # Use standard size that predict.py uses
                    diag_f.write(f"Using img_size: {img_size}\n\n")
                    
                    # Save the exact predict.py code used for reference
                    predict_py_code = """
                    # From predict.py:
                    from ultralytics import YOLO
                    model = YOLO("yolo11n-pose.pt")
                    results = model(img_path, save=True)  # This uses letterboxing internally
                    for result in results:
                        kpts = result.keypoints.data
                        print(kpts)
                    """
                    diag_f.write("\nReference code from predict.py:\n")
                    diag_f.write(predict_py_code + "\n\n")
                    
                    # Log the original image dimensions
                    from PIL import Image
                    img_original = Image.open(img_path)
                    diag_f.write(f"Original PIL dimensions: {img_original.size[0]}x{img_original.size[1]}\n")
                    
                    # Use the model with identical settings as predict.py
                    # The key is to use the exact same size and letterboxing
                    results = model(
                        img_path,
                        imgsz=img_size,  # Square size forces letterboxing
                        conf=0.25,
                        iou=0.45,
                        verbose=False
                    )
                    
                    # For direct comparison, also try predict method
                    predict_results = model.predict(
                        img_path,
                        imgsz=img_size,  # Square size forces letterboxing
                        conf=0.25,
                        iou=0.45,
                        verbose=False
                    )
                    
                    diag_f.write("\nComparing model() vs model.predict() methods:\n")
                    diag_f.write("="*40 + "\n")
                    
                    # Extract dimensions used by the model for debugging
                    if len(results) > 0:
                        diag_f.write(f"Input shape to model(): {results[0].boxes.orig_shape}\n")
                        diag_f.write(f"Preprocessing shape: {img_size}x{img_size}\n")
                        
                    # Extract keypoints exactly as predict.py would
                    if len(results) > 0 and hasattr(results[0], 'keypoints') and len(results[0].keypoints) > 0:
                        # Get the raw keypoint data - this is identical to what predict.py prints
                        new_pred_kpts = results[0].keypoints.data[0].cpu().numpy()
                        logger.info(f"New prediction shape: {results[0].orig_shape}")
                        diag_f.write(f"model() prediction shape: {results[0].orig_shape}\n\n")
                        
                        # Compare with predict.py output
                        logger.info("Direct keypoint data (exactly as predict.py would show):")
                        diag_f.write("DIRECT model() KEYPOINT DATA (as predict.py would show):\n")
                        for k in range(len(new_pred_kpts)):
                            x, y, conf = new_pred_kpts[k]
                            name = keypoint_names.get(k, f"Keypoint {k}")
                            logger.info(f"  {k}: {name}, ({x:.1f}, {y:.1f}), conf={conf:.4f}")
                            diag_f.write(f"  {k}: {name}, ({x:.1f}, {y:.1f}), conf={conf:.4f}\n")
                        
                        # Check if any keypoints are zero despite having non-zero confidence
                        zero_pts = [(i, new_pred_kpts[i, 2]) for i in range(len(new_pred_kpts)) 
                                   if new_pred_kpts[i, 0] == 0 and new_pred_kpts[i, 1] == 0 and new_pred_kpts[i, 2] > 0]
                        if zero_pts:
                            logger.info("Found keypoints with (0,0) coordinates but non-zero confidence:")
                            diag_f.write("\nKeypoints with (0,0) coordinates but non-zero confidence:\n")
                            for idx, conf in zero_pts:
                                logger.info(f"  Keypoint {idx}: confidence={conf:.4f}")
                                diag_f.write(f"  Keypoint {idx}: confidence={conf:.4f}\n")
                        
                        # Now compare with predict results
                        if len(predict_results) > 0 and hasattr(predict_results[0], 'keypoints') and len(predict_results[0].keypoints) > 0:
                            diag_f.write("\nmodel.predict() KEYPOINT DATA:\n")
                            predict_kpts = predict_results[0].keypoints.data[0].cpu().numpy()
                            
                            for k in range(len(predict_kpts)):
                                x, y, conf = predict_kpts[k]
                                name = keypoint_names.get(k, f"Keypoint {k}")
                                diag_f.write(f"  {k}: {name}, ({x:.1f}, {y:.1f}), conf={conf:.4f}\n")
                            
                            # Compare differences between model() and model.predict()
                            diag_f.write("\nDIFFERENCES BETWEEN model() AND model.predict():\n")
                            for k in range(min(len(new_pred_kpts), len(predict_kpts))):
                                x1, y1, conf1 = new_pred_kpts[k]
                                x2, y2, conf2 = predict_kpts[k]
                                
                                if x1 != x2 or y1 != y2 or abs(conf1 - conf2) > 0.001:
                                    diag_f.write(f"  Keypoint {k}: \n")
                                    diag_f.write(f"    model():       ({x1:.1f}, {y1:.1f}), conf={conf1:.4f}\n")
                                    diag_f.write(f"    model.predict: ({x2:.1f}, {y2:.1f}), conf={conf2:.4f}\n")
                        
                        # Replace the original prediction with the new one
                        pred_kpts = new_pred_kpts
                        pred_shape = results[0].orig_shape
                    else:
                        logger.warning("Re-prediction failed, using original prediction")
                        diag_f.write("Re-prediction failed, using original prediction\n")
                        pred_kpts = pose['kpts']
                        pred_shape = pose.get('orig_shape', (orig_h, orig_w))
                else:
                    # Use the original prediction
                    pred_kpts = pose['kpts']
                    pred_shape = pose.get('orig_shape', (orig_h, orig_w))
                
                # Get ground truth and errors
                gt_kpts = pose['gt_kpts']
                kpt_errors = pose.get('kpt_errors', [])
                
                # Log prediction dimensions
                pred_h, pred_w = pred_shape[:2] if pred_shape else (orig_h, orig_w)
                logger.info(f"Prediction dimensions: {pred_w}x{pred_h}")
                diag_f.write(f"\nPrediction dimensions: {pred_w}x{pred_h}\n")
                
                # Calculate scale factors if prediction dimensions differ from original
                scale_x = orig_w / pred_w
                scale_y = orig_h / pred_h
                logger.info(f"Scale factors: width={scale_x:.4f}, height={scale_y:.4f}")
                diag_f.write(f"Scale factors: width={scale_x:.4f}, height={scale_y:.4f}\n")
                
                # Scale predictions to match original image if needed
                if scale_x != 1.0 or scale_y != 1.0:
                    logger.info("Scaling predictions to match original image dimensions")
                    diag_f.write("Scaling predictions to match original image dimensions\n")
                    scaled_pred_kpts = pred_kpts.copy()
                    for k in range(len(pred_kpts)):
                        if pred_kpts[k][2] > 0:  # Only scale valid keypoints
                            scaled_pred_kpts[k][0] = pred_kpts[k][0] * scale_x
                            scaled_pred_kpts[k][1] = pred_kpts[k][1] * scale_y
                    pred_kpts = scaled_pred_kpts
                
                # For verification, calculate mean confidence of predicted keypoints
                valid_conf = [pred_kpts[k][2] for k in range(len(pred_kpts)) if pred_kpts[k][2] > 0]
                mean_conf = sum(valid_conf) / len(valid_conf) if valid_conf else 0
                logger.info(f"Mean confidence of valid keypoints: {mean_conf:.4f}")
                diag_f.write(f"Mean confidence of valid keypoints: {mean_conf:.4f}\n\n")
                
                # Ground truth analysis
                diag_f.write("GROUND TRUTH KEYPOINTS:\n")
                for k in range(len(gt_kpts)):
                    gt_x_norm, gt_y_norm, gt_conf = gt_kpts[k]
                    gt_x, gt_y = gt_x_norm * orig_w, gt_y_norm * orig_h  # Scale to image dimensions
                    name = keypoint_names.get(k, f"Keypoint {k}")
                    diag_f.write(f"  {k}: {name}, ({gt_x:.1f}, {gt_y:.1f}), conf={gt_conf:.4f}\n")
                
                # Write comparison information
                diag_f.write("\nCOMPARISON WITH GROUND TRUTH:\n")
                for k in range(min(len(pred_kpts), len(gt_kpts))):
                    gt_x_norm, gt_y_norm, gt_conf = gt_kpts[k]
                    gt_x, gt_y = gt_x_norm * orig_w, gt_y_norm * orig_h  # Scale to image dimensions
                    
                    pred_x, pred_y, pred_conf = pred_kpts[k]
                    
                    # Calculate error in normalized coordinates
                    pred_x_norm = pred_x / orig_w
                    pred_y_norm = pred_y / orig_h
                    
                    # Only calculate error if both points have confidence
                    if gt_conf > 0 and pred_conf > 0:
                        norm_error = np.sqrt((pred_x_norm - gt_x_norm)**2 + (pred_y_norm - gt_y_norm)**2)
                        error_pixels = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                        
                        diag_f.write(f"  Keypoint {k}: {keypoint_names.get(k, '')}\n")
                        diag_f.write(f"    GT:   ({gt_x:.1f}, {gt_y:.1f}), conf={gt_conf:.4f}\n")
                        diag_f.write(f"    Pred: ({pred_x:.1f}, {pred_y:.1f}), conf={pred_conf:.4f}\n")
                        diag_f.write(f"    Error: {norm_error:.4f} normalized, {error_pixels:.1f} pixels\n")
                
                # For the original image display
                img = orig_img.copy()

                # Sort keypoints by error
                keypoint_difficulties = []
                if kpt_errors:
                    for k_idx, err in kpt_errors:
                        keypoint_name = keypoint_names.get(k_idx, f"Keypoint {k_idx}")
                        keypoint_difficulties.append((k_idx, keypoint_name, err))
                    
                    # Sort keypoints by error
                    keypoint_difficulties.sort(key=lambda x: x[2], reverse=True)
                    
                    logger.info("Keypoints sorted by error:")
                    for k_idx, k_name, k_err in keypoint_difficulties[:5]:
                        logger.info(f"  {k_idx}: {k_name}, error={k_err:.4f}")

                # Create a figure with three subplots - prediction, ground truth, and comparison
                fig = plt.figure(figsize=(18, 10))

                # Define a custom gridspec to control the layout
                from matplotlib import gridspec
                gs = gridspec.GridSpec(1, 3, width_ratios=[5, 5, 5])

                # Create axes for each part
                ax_pred = plt.subplot(gs[0])    # Predictions only
                ax_gt = plt.subplot(gs[1])      # Ground truth only
                ax_comp = plt.subplot(gs[2])    # Comparison

                # Draw keypoints on images
                radius = int(min(orig_h, orig_w) * 0.015)  # Larger radius for better visibility
                thickness = max(2, int(min(orig_h, orig_w) * 0.005))

                # Display the image in all three axes
                ax_pred.imshow(img)
                ax_gt.imshow(img)
                ax_comp.imshow(img)

                # Set titles for each subplot
                ax_pred.set_title(f"Predicted Keypoints\nConfidence: {1-pose['conf_difficulty']:.2f}", fontsize=14)
                ax_gt.set_title(f"Ground Truth Keypoints", fontsize=14)
                ax_comp.set_title(f"Comparison (Error: {pose['error_score']:.3f})", fontsize=14)

                # Turn off axes for cleaner visualization
                ax_pred.axis('off')
                ax_gt.axis('off')
                ax_comp.axis('off')

                # Add legend for different visualizations
                pred_patch = mpatches.Patch(color='red', alpha=0.5, label='Prediction')
                gt_patch = mpatches.Patch(color='green', alpha=0.5, label='Ground Truth')

                # Error color scale for the comparison legend
                error_patches = [
                    mpatches.Patch(color=cm.RdYlGn_r(0.1), alpha=0.7, label='Small Error'),
                    mpatches.Patch(color=cm.RdYlGn_r(0.5), alpha=0.7, label='Medium Error'),
                    mpatches.Patch(color=cm.RdYlGn_r(0.9), alpha=0.7, label='Large Error')
                ]

                # Add legends to appropriate plots
                ax_pred.legend(handles=[pred_patch], loc='lower right', fontsize=9)
                ax_gt.legend(handles=[gt_patch], loc='lower right', fontsize=9)
                ax_comp.legend(handles=[pred_patch, gt_patch] + error_patches, loc='lower right', fontsize=9)

                # Add keypoint annotations to predicted plot
                for k in range(len(pred_kpts)):
                    # For prediction view
                    x, y, conf = pred_kpts[k]
                    
                    # Apply visibility threshold
                    if conf > visibility_threshold:
                        # Get color based on confidence for prediction
                        color = cm.Reds(conf)
                        
                        # Draw a circle for prediction
                        circle = plt.Circle((x, y), radius, color=color, fill=True, alpha=0.7)
                        ax_pred.add_patch(circle)
                        
                        # Add keypoint number with confidence
                        kpt_name = keypoint_names.get(k, f"{k}")
                        ax_pred.annotate(f"{k}:{conf:.2f}", (x+radius, y+radius), 
                                       color='white', fontsize=8, weight='bold',
                                       bbox=dict(facecolor='red', alpha=0.7))

                # Add ground truth keypoints to ground truth plot
                for k in range(len(gt_kpts)):
                    gt_x_norm, gt_y_norm, gt_conf = gt_kpts[k]
                    gt_x, gt_y = gt_x_norm * orig_w, gt_y_norm * orig_h  # Scale to image dimensions
                    
                    if gt_conf > visibility_threshold:
                        # Draw ground truth keypoint
                        gt_circle = plt.Circle((gt_x, gt_y), radius, color='green', fill=True, alpha=0.7)
                        ax_gt.add_patch(gt_circle)
                        
                        # Add keypoint label
                        kpt_name = keypoint_names.get(k, f"{k}")
                        ax_gt.annotate(f"{k}", (gt_x+radius, gt_y+radius), 
                                    color='white', fontsize=8, weight='bold',
                                    bbox=dict(facecolor='green', alpha=0.7))

                # Add comparison visualization
                for k in range(min(len(pred_kpts), len(gt_kpts))):
                    # For ground truth
                    gt_x_norm, gt_y_norm, gt_conf = gt_kpts[k]
                    gt_x, gt_y = gt_x_norm * orig_w, gt_y_norm * orig_h  # Scale to image dimensions
                    
                    # For prediction
                    pred_x, pred_y, pred_conf = pred_kpts[k]
                    
                    # Draw both points if they're above threshold
                    if gt_conf > visibility_threshold:
                        gt_circle = plt.Circle((gt_x, gt_y), radius, color='green', fill=True, alpha=0.5)
                        ax_comp.add_patch(gt_circle)
                        
                    if pred_conf > visibility_threshold:
                        pred_circle = plt.Circle((pred_x, pred_y), radius, color='red', fill=True, alpha=0.5)
                        ax_comp.add_patch(pred_circle)
                        
                        # Connect ground truth and prediction with a line if both exist
                        if gt_conf > visibility_threshold and pred_conf > visibility_threshold:
                            # Calculate error in normalized coordinates (0-1 space)
                            norm_error = np.sqrt((pred_x_norm - gt_x_norm)**2 + (pred_y_norm - gt_y_norm)**2)
                            
                            # Calculate pixel distance for display
                            error_pixels = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                            
                            # Use normalized error for color coding
                            line_width = max(1, min(5, norm_error * 10))
                            
                            # Calculate color based on error severity (green to red)
                            error_color = cm.RdYlGn_r(min(1.0, norm_error * 3))
                            
                            # Draw line with appropriate thickness and color
                            ax_comp.plot([gt_x, pred_x], [gt_y, pred_y], color=error_color, 
                                        linewidth=line_width, alpha=0.7)
                            
                            # Add error annotation at midpoint
                            mid_x = (gt_x + pred_x) / 2
                            mid_y = (gt_y + pred_y) / 2
                            ax_comp.annotate(f"{norm_error:.2f}", (mid_x, mid_y), 
                                           color='white', fontsize=8, weight='bold',
                                           bbox=dict(facecolor=error_color, alpha=0.7))

                # Add main title for the entire figure
                fig.suptitle(f"Difficult Pose {i+1} (Score: {pose['difficulty']:.3f})", fontsize=16)

                # Add summary of difficult keypoints as text at the bottom
                if keypoint_difficulties:
                    summary_text = "Most Difficult Keypoints:\n"
                    for idx, (k_idx, k_name, k_err) in enumerate(keypoint_difficulties[:5]):
                        summary_text += f"{k_name} ({k_idx}): Error {k_err:.3f}\n"
                    
                    # Add text box with difficult keypoints
                    fig.text(0.02, 0.02, summary_text, fontsize=10,
                            verticalalignment='bottom', 
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

                plt.tight_layout()
                output_path = str(difficult_dir / f'difficult_pose_{i+1}.jpg')
                plt.savefig(output_path, dpi=200)
                plt.close()

                logger.info(f"Visualization saved to {output_path}")

                # Extract pose name from file path if possible
                try:
                    filename = os.path.basename(img_path)
                    if '_or_' in filename:
                        pose_name = filename.split('_or_')[0].replace('_', ' ')
                    else:
                        pose_name = filename.split('_image_')[0].replace('_', ' ')
                except:
                    pose_name = "Unknown Pose"

                # Write details to report
                f.write(f"Difficult Pose {i+1}\n")
                f.write(f"Image Path: {img_path}\n")
                f.write(f"Pose Type: {pose_name}\n")
                f.write(f"Difficulty Score: {pose['difficulty']:.3f}\n")
                f.write(f"Confidence Difficulty: {pose['conf_difficulty']:.3f}\n")
                f.write(f"Error Score: {pose['error_score']:.3f}\n")
                f.write("Most Difficult Keypoints (sorted by error):\n")

                for k_idx, k_name, k_err in keypoint_difficulties[:5]:
                    f.write(f"  {k_name} ({k_idx}): Error {k_err:.3f}\n")

                f.write("\n")

                logger.info("Generating summary visualization")

                # Create a summary visualization with small thumbnails of all difficult poses
                num_poses = len(difficult_poses)
                cols = 3
                rows = (num_poses + cols - 1) // cols

                plt.figure(figsize=(15, rows * 4))
                for i, pose in enumerate(difficult_poses):
                    img_path = pose['img_path']
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize for thumbnail
                    h, w = img.shape[:2]
                    aspect = w / h
                    if aspect > 1:
                        new_w, new_h = 300, int(300 / aspect)
                    else:
                        new_w, new_h = int(300 * aspect), 300
                    
                    img_small = cv2.resize(img, (new_w, new_h))
                    
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(img_small)
                    plt.title(f"Pose {i+1}: Score {pose['difficulty']:.3f}\nError: {pose['error_score']:.2f}", fontsize=9)
                    plt.axis('off')

                plt.tight_layout()
                summary_path = str(difficult_dir / 'difficult_poses_summary.jpg')
                plt.savefig(summary_path, dpi=200)
                plt.close()

                logger.info(f"Summary visualization saved to {summary_path}")
                logger.info(f"Successfully visualized {len(difficult_poses)} difficult poses. Saved to {difficult_dir}")

                print(f"Successfully visualized {len(difficult_poses)} difficult poses. Saved to {difficult_dir}")
                print(f"Detailed diagnostics reports saved to {diagnostics_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze keypoint detection performance of a YOLO model on yoga poses')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, required=True, help='Path to data YAML file')
    parser.add_argument('--img-size', type=int, default=1280, help='Image size for validation')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--device', default='0', help='Device to run validation on (e.g., 0, 0,1,2,3 or cpu)')
    parser.add_argument('--find-difficult', action='store_true', help='Find and visualize difficult poses')
    parser.add_argument('--num-difficult', type=int, default=10, help='Number of difficult poses to visualize')
    parser.add_argument('--repredict', action='store_true', help='Re-run prediction to match predict.py behavior')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_keypoints(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=args.device,
        find_difficult=args.find_difficult,
        num_difficult=args.num_difficult,
        repredict=args.repredict
    ) 