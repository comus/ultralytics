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


def analyze_keypoints(model_path, data_yaml, img_size=1280, batch_size=16, device='0', 
                     find_difficult=False, num_difficult=10):
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
    """
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Create output directory
    output_dir = Path('keypoint_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Run validation to get metrics
    print("Running validation...")
    results = model.val(data=data_yaml, imgsz=img_size, batch=batch_size, device=device)
    
    # Get validation dataset name
    dataset_name = Path(data_yaml).stem
    
    # Process keypoint detection statistics
    keypoint_stats = process_keypoint_stats(model, results)
    
    # Visualize keypoint statistics
    visualize_keypoint_stats(keypoint_stats, output_dir, dataset_name)
    
    # Find and visualize difficult poses if requested
    if find_difficult:
        print("Finding difficult poses...")
        difficult_poses = find_difficult_poses(model, data_yaml, img_size, device, num_difficult, output_dir)
        visualize_difficult_poses(difficult_poses, model, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


def process_keypoint_stats(model, results):
    """
    Process keypoint detection statistics from validation results.
    
    Args:
        model: YOLO model
        results: Validation results
    
    Returns:
        Dictionary containing keypoint detection statistics
    """
    # Get keypoint names from model
    try:
        keypoint_names = model.names
    except:
        # Fallback to default keypoint names if not available in model
        keypoint_names = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle'
        }
    
    # Extract keypoint detection stats from results
    stats = results.speed
    
    # Initialize keypoint statistics dictionary
    keypoint_stats = {
        'detection_rate': np.zeros(len(keypoint_names)),
        'avg_confidence': np.zeros(len(keypoint_names)),
        'names': keypoint_names
    }
    
    # Aggregate keypoint statistics from results
    if hasattr(results, 'keypoints') and results.keypoints is not None:
        kpts = results.keypoints
        
        # Calculate detection rate and average confidence for each keypoint
        for i in range(len(keypoint_names)):
            valid_kpts = kpts[:, :, i, 2] > 0
            if valid_kpts.size > 0:
                keypoint_stats['detection_rate'][i] = valid_kpts.sum() / valid_kpts.size
                conf_values = kpts[:, :, i, 2][valid_kpts]
                keypoint_stats['avg_confidence'][i] = conf_values.mean() if len(conf_values) > 0 else 0
    
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
    
    # Create bar plots for detection rates
    plt.figure(figsize=(14, 7))
    plt.bar(keypoint_names, detection_rates)
    plt.title(f'Keypoint Detection Rates - {dataset_name}')
    plt.xlabel('Keypoint')
    plt.ylabel('Detection Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'detection_rates.png'))
    plt.close()
    
    # Create bar plots for average confidences
    plt.figure(figsize=(14, 7))
    plt.bar(keypoint_names, avg_confidences)
    plt.title(f'Keypoint Average Confidence - {dataset_name}')
    plt.xlabel('Keypoint')
    plt.ylabel('Average Confidence')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(str(output_dir / 'avg_confidences.png'))
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
    Find difficult poses in the validation dataset.
    
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
    # Get validation dataset
    val_loader = model.trainer.get_dataloader(data_yaml, img_size, batch_size=1, rank=-1, mode='val')
    
    difficult_poses = []
    
    for batch_i, batch in enumerate(tqdm(val_loader, desc="Analyzing poses")):
        images, targets = batch
        img_paths = targets['img_paths']
        
        # Run model inference
        results = model.predict(images, device=device, verbose=False)
        
        for i, result in enumerate(results):
            if len(result.keypoints) > 0:
                # Calculate per-keypoint difficulty based on confidence
                kpts = result.keypoints.data[0]
                
                # Skip if no keypoints detected
                if kpts.shape[0] == 0:
                    continue
                
                # Calculate difficulty score (1 - average confidence of detected keypoints)
                valid_kpts = kpts[:, 2] > 0
                if valid_kpts.sum() > 0:
                    avg_conf = kpts[valid_kpts, 2].mean().item()
                    difficulty = 1 - avg_conf
                    
                    # Store image path and difficulty score
                    difficult_poses.append({
                        'img_path': img_paths[i],
                        'difficulty': difficulty,
                        'kpts': kpts.cpu().numpy()
                    })
        
        # Limit the number of batches to process for efficiency
        if len(difficult_poses) >= num_difficult * 5:
            break
    
    # Sort by difficulty (highest first) and take the top N
    difficult_poses.sort(key=lambda x: x['difficulty'], reverse=True)
    return difficult_poses[:num_difficult]


def visualize_difficult_poses(difficult_poses, model, output_dir):
    """
    Visualize difficult poses.
    
    Args:
        difficult_poses: List of difficult poses
        model: YOLO model
        output_dir: Output directory to save visualizations
    """
    if not difficult_poses:
        print("No difficult poses found.")
        return
    
    # Create directory for difficult poses
    difficult_dir = output_dir / 'difficult_poses'
    difficult_dir.mkdir(exist_ok=True)
    
    # Get keypoint names from model
    try:
        keypoint_names = model.names
    except:
        keypoint_names = {i: f"kpt_{i}" for i in range(17)}  # Default keypoint names
    
    # Write report file
    with open(difficult_dir / 'difficult_poses_report.txt', 'w') as f:
        f.write("Most Difficult Poses to Detect\n")
        f.write("=============================\n\n")
        
        for i, pose in enumerate(difficult_poses):
            # Load the image
            img_path = pose['img_path']
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get keypoints and confidences
            kpts = pose['kpts']
            
            # Calculate per-keypoint difficulties
            difficult_keypoints = []
            for k in range(kpts.shape[0]):
                if kpts[k, 2] > 0:
                    conf = kpts[k, 2]
                    difficult_keypoints.append((k, 1 - conf))
            
            # Sort keypoints by difficulty
            difficult_keypoints.sort(key=lambda x: x[1], reverse=True)
            
            # Draw keypoints on image
            h, w = img.shape[:2]
            radius = int(min(h, w) * 0.01)
            thickness = max(1, int(min(h, w) * 0.003))
            
            for k in range(kpts.shape[0]):
                x, y, conf = kpts[k]
                if conf > 0:
                    color = plt.cm.jet(1 - conf)
                    color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                    cv2.circle(img, (int(x), int(y)), radius, color, thickness)
            
            # Save the image
            output_path = str(difficult_dir / f'difficult_pose_{i+1}.jpg')
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(f"Difficult Pose {i+1} (Score: {pose['difficulty']:.3f})")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            # Write details to report
            f.write(f"Difficult Pose {i+1}\n")
            f.write(f"Image: {img_path}\n")
            f.write(f"Difficulty Score: {pose['difficulty']:.3f}\n")
            f.write("Most Difficult Keypoints:\n")
            
            for k_idx, k_diff in difficult_keypoints[:5]:
                k_name = keypoint_names.get(k_idx, f"Keypoint {k_idx}")
                f.write(f"  {k_name}: {k_diff:.3f}\n")
            
            f.write("\n")
    
    print(f"Visualized {len(difficult_poses)} difficult poses. Saved to {difficult_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze keypoint detection performance of a YOLO model on yoga poses')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, required=True, help='Path to data YAML file')
    parser.add_argument('--img-size', type=int, default=1280, help='Image size for validation')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--device', default='0', help='Device to run validation on (e.g., 0, 0,1,2,3 or cpu)')
    parser.add_argument('--find-difficult', action='store_true', help='Find and visualize difficult poses')
    parser.add_argument('--num-difficult', type=int, default=10, help='Number of difficult poses to visualize')
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
        num_difficult=args.num_difficult
    ) 