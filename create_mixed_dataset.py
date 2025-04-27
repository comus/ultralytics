#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Create mixed dataset with weighted sampling')
    parser.add_argument('--datasets-path', type=str, default='/root/autodl-tmp/withcloud/datasets',
                        help='Path to datasets directory')
    parser.add_argument('--coco-weight', type=float, default=0.7,
                        help='Weight for COCO-pose samples')
    parser.add_argument('--yoga-weight', type=float, default=0.3,
                        help='Weight for Yoga82 samples')
    parser.add_argument('--output-dir', type=str, default='mixed_coco_yoga',
                        help='Output directory name (will be created under datasets path)')
    return parser.parse_args()

def read_file_paths(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def create_mixed_dataset(args):
    dataset_path = Path(args.datasets_path)
    
    # Create output directories
    output_path = dataset_path / args.output_dir
    output_images_train = output_path / 'images' / 'train'
    output_images_val = output_path / 'images' / 'val'
    output_labels_train = output_path / 'labels' / 'train'
    output_labels_val = output_path / 'labels' / 'val'
    
    for dir_path in [output_images_train, output_images_val, output_labels_train, output_labels_val]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Read COCO paths
    coco_train_txt = dataset_path / 'coco-pose' / 'train2017.txt'
    coco_val_txt = dataset_path / 'coco-pose' / 'val2017.txt'
    
    coco_train_paths = read_file_paths(coco_train_txt)
    coco_val_paths = read_file_paths(coco_val_txt)
    
    # Get Yoga paths
    yoga_train_dir = dataset_path / 'yoga82' / 'images' / 'train'
    yoga_val_dir = dataset_path / 'yoga82' / 'images' / 'val'
    
    yoga_train_images = list(yoga_train_dir.glob('*.jpg')) + list(yoga_train_dir.glob('*.png'))
    yoga_val_images = list(yoga_val_dir.glob('*.jpg')) + list(yoga_val_dir.glob('*.png'))
    
    yoga_train_paths = [str(p.relative_to(dataset_path)) for p in yoga_train_images]
    yoga_val_paths = [str(p.relative_to(dataset_path)) for p in yoga_val_images]
    
    # Calculate sample counts based on weights
    print(f"COCO train samples (total available): {len(coco_train_paths)}")
    print(f"Yoga train samples (total available): {len(yoga_train_paths)}")
    
    # Calculate how many samples to take from each dataset
    # We'll take all yoga samples, and adjust COCO samples to match the ratio
    target_coco_train_samples = int(len(yoga_train_paths) * args.coco_weight / args.yoga_weight)
    
    # Sample training paths
    sampled_coco_train = random.sample(coco_train_paths, min(target_coco_train_samples, len(coco_train_paths)))
    
    print(f"Using {len(sampled_coco_train)} COCO training samples and {len(yoga_train_paths)} Yoga training samples")
    print(f"Targeted ratio - COCO: {args.coco_weight:.2f}, Yoga: {args.yoga_weight:.2f}")
    print(f"Actual train ratio - COCO: {len(sampled_coco_train)/(len(sampled_coco_train)+len(yoga_train_paths)):.2f}, "
          f"Yoga: {len(yoga_train_paths)/(len(sampled_coco_train)+len(yoga_train_paths)):.2f}")
    
    # 計算驗證集的採樣數量 - 使用與訓練集相同的比例
    # 我們使用全部的Yoga驗證樣本，並根據權重比例調整COCO驗證樣本數量
    target_coco_val_samples = int(len(yoga_val_paths) * args.coco_weight / args.yoga_weight)
    
    # 確保採樣數量不超過可用的COCO驗證樣本數量
    sampled_coco_val = random.sample(coco_val_paths, min(target_coco_val_samples, len(coco_val_paths)))
    
    print(f"Using {len(sampled_coco_val)} COCO validation samples and {len(yoga_val_paths)} Yoga validation samples")
    print(f"Actual val ratio - COCO: {len(sampled_coco_val)/(len(sampled_coco_val)+len(yoga_val_paths)):.2f}, "
          f"Yoga: {len(yoga_val_paths)/(len(sampled_coco_val)+len(yoga_val_paths)):.2f}")
    
    # Create train.txt and val.txt
    train_txt_path = output_path / 'train.txt'
    val_txt_path = output_path / 'val.txt'
    
    # Counters to track processed images
    coco_train_processed = 0
    yoga_train_processed = 0
    coco_val_processed = 0
    yoga_val_processed = 0
    
    # Process training images
    with open(train_txt_path, 'w') as f:
        # Process COCO training images
        for path in sampled_coco_train:
            # Handle path format: ./images/train2017/000000123456.jpg
            # or /images/train2017/000000123456.jpg
            if path.startswith('./'):
                path = path[2:]  # Remove leading ./
            elif path.startswith('/'):
                path = path[1:]  # Remove leading /
                
            # Get full image path
            img_path = dataset_path / path
            
            # Some COCO paths might be relative to a different directory
            # Try alternative path if original doesn't exist
            if not img_path.exists():
                # Try with coco-pose prepended
                alt_path = dataset_path / 'coco-pose' / path
                if alt_path.exists():
                    img_path = alt_path
                else:
                    print(f"Warning: Could not find image at {img_path} or {alt_path}")
                    continue
            
            # Get image filename
            img_filename = img_path.name
            
            # Get label path - it should be in 'labels' instead of 'images'
            # and have .txt extension instead of .jpg/.png
            label_rel_path = path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = dataset_path / label_rel_path
            
            # Also try with coco-pose prepended if needed
            if not label_path.exists():
                alt_label_path = dataset_path / 'coco-pose' / label_rel_path
                if alt_label_path.exists():
                    label_path = alt_label_path
                else:
                    print(f"Warning: Could not find label at {label_path} or {alt_label_path}")
                    continue
            
            # Copy files to our dataset
            dest_img = output_images_train / img_filename
            dest_label = output_labels_train / img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            
            try:
                shutil.copy(img_path, dest_img)
                shutil.copy(label_path, dest_label)
                
                # Write to train.txt using the standard format
                f.write(f"./images/train/{img_filename}\n")
                coco_train_processed += 1
            except Exception as e:
                print(f"Error copying files: {e}")
                continue
        
        # Process Yoga training images
        for path in yoga_train_paths:
            img_path = dataset_path / path
            if not img_path.exists():
                print(f"Warning: Could not find Yoga image at {img_path}")
                continue
            
            # Get the corresponding label path
            label_path = str(path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = dataset_path / label_path
            
            if not label_path.exists():
                print(f"Warning: Could not find Yoga label at {label_path}")
                continue
            
            # Copy image and label to our dataset
            dest_img = output_images_train / img_path.name
            dest_label = output_labels_train / label_path.name
            
            try:
                shutil.copy(img_path, dest_img)
                shutil.copy(label_path, dest_label)
                
                # Write to train.txt
                f.write(f"./images/train/{img_path.name}\n")
                yoga_train_processed += 1
            except Exception as e:
                print(f"Error copying Yoga files: {e}")
                continue
    
    # Process validation images
    with open(val_txt_path, 'w') as f:
        # Process COCO validation images (使用計算後的樣本數)
        for path in sampled_coco_val:
            if path.startswith('./'):
                path = path[2:]  # Remove leading ./
            elif path.startswith('/'):
                path = path[1:]  # Remove leading /
            
            # Get full image path
            img_path = dataset_path / path
            
            # Try alternative path if original doesn't exist
            if not img_path.exists():
                alt_path = dataset_path / 'coco-pose' / path
                if alt_path.exists():
                    img_path = alt_path
                else:
                    continue
            
            # Get image filename
            img_filename = img_path.name
            
            # Get label path
            label_rel_path = path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = dataset_path / label_rel_path
            
            # Try alternative label path if needed
            if not label_path.exists():
                alt_label_path = dataset_path / 'coco-pose' / label_rel_path
                if alt_label_path.exists():
                    label_path = alt_label_path
                else:
                    continue
            
            # Copy files
            dest_img = output_images_val / img_filename
            dest_label = output_labels_val / img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
            
            try:
                shutil.copy(img_path, dest_img)
                shutil.copy(label_path, dest_label)
                
                # Write to val.txt
                f.write(f"./images/val/{img_filename}\n")
                coco_val_processed += 1
            except Exception as e:
                print(f"Error copying val files: {e}")
                continue
        
        # Process all Yoga validation images
        for path in yoga_val_paths:
            img_path = dataset_path / path
            if not img_path.exists():
                continue
            
            # Get the corresponding label path
            label_path = str(path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = Path(dataset_path) / label_path
            
            if not label_path.exists():
                continue
            
            # Copy image and label to our dataset
            dest_img = output_images_val / img_path.name
            dest_label = output_labels_val / label_path.name
            
            try:
                shutil.copy(img_path, dest_img)
                shutil.copy(label_path, dest_label)
                
                # Write to val.txt
                f.write(f"./images/val/{img_path.name}\n")
                yoga_val_processed += 1
            except Exception as e:
                print(f"Error copying Yoga val files: {e}")
                continue
    
    # Print summary
    print("\nSummary:")
    print(f"Training images processed - COCO: {coco_train_processed}, Yoga: {yoga_train_processed}, Total: {coco_train_processed + yoga_train_processed}")
    print(f"Validation images processed - COCO: {coco_val_processed}, Yoga: {yoga_val_processed}, Total: {coco_val_processed + yoga_val_processed}")
    
    # Create a new YAML configuration file
    yaml_path = output_path / 'mixed_coco_yoga.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""# Mixed COCO-Pose and Yoga82 dataset configuration
path: {args.datasets_path}/{args.output_dir}  # Base path to dataset
train: train.txt  # Training images
val: val.txt  # Validation images
test: val.txt  # Test images

# Keypoint configuration - same as COCO and Yoga82
kpt_shape: [17, 3]  # Number of keypoints and dimensions (x,y,visibility)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes
names:
  0: person
""")
    
    print(f"\nCreated mixed dataset at {output_path}")
    print(f"Configuration file created at {yaml_path}")
    print(f"Use this configuration for training: --data {args.datasets_path}/{args.output_dir}/mixed_coco_yoga.yaml")

if __name__ == "__main__":
    args = parse_args()
    create_mixed_dataset(args) 