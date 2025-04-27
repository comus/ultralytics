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
    parser.add_argument('--coco-weight', type=float, default=0.3,
                        help='Weight for COCO-pose samples')
    parser.add_argument('--yoga-weight', type=float, default=0.7,
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
    total_train_samples = len(coco_train_paths) + len(yoga_train_paths)
    print(f"COCO train samples: {len(coco_train_paths)}")
    print(f"Yoga train samples: {len(yoga_train_paths)}")
    
    # Calculate how many samples to take from each dataset
    # Adjust the numbers to match the weights while maintaining a reasonable total
    # We'll take all yoga samples, and adjust COCO samples to match the ratio
    target_coco_samples = int(len(yoga_train_paths) * args.coco_weight / args.yoga_weight)
    
    # Sample paths
    sampled_coco_train = random.sample(coco_train_paths, min(target_coco_samples, len(coco_train_paths)))
    
    print(f"Using {len(sampled_coco_train)} COCO samples and {len(yoga_train_paths)} Yoga samples")
    print(f"Actual ratio - COCO: {len(sampled_coco_train)/(len(sampled_coco_train)+len(yoga_train_paths)):.2f}, "
          f"Yoga: {len(yoga_train_paths)/(len(sampled_coco_train)+len(yoga_train_paths)):.2f}")
    
    # Create train.txt and val.txt
    train_txt_path = output_path / 'train.txt'
    val_txt_path = output_path / 'val.txt'
    
    # Process training images
    with open(train_txt_path, 'w') as f:
        # Process COCO training images
        for path in sampled_coco_train:
            # Convert path format
            if path.startswith('/'):
                path = path[1:]  # Remove leading slash if present
                
            img_path = Path(dataset_path) / path
            if not img_path.exists():
                continue
                
            # Get the corresponding label path
            label_path = str(path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = Path(dataset_path) / label_path
            
            if not label_path.exists():
                continue
                
            # Copy image and label to our dataset
            dest_img = output_images_train / img_path.name
            dest_label = output_labels_train / label_path.name
            
            shutil.copy(img_path, dest_img)
            shutil.copy(label_path, dest_label)
            
            # Write to train.txt using the original format (./images/...)
            f.write(f"./images/train/{img_path.name}\n")
        
        # Process Yoga training images
        for path in yoga_train_paths:
            img_path = Path(dataset_path) / path
            if not img_path.exists():
                continue
                
            # Get the corresponding label path
            label_path = str(path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = Path(dataset_path) / label_path
            
            if not label_path.exists():
                continue
                
            # Copy image and label to our dataset
            dest_img = output_images_train / img_path.name
            dest_label = output_labels_train / label_path.name
            
            shutil.copy(img_path, dest_img)
            shutil.copy(label_path, dest_label)
            
            # Write to train.txt using the original format (./images/...)
            f.write(f"./images/train/{img_path.name}\n")
    
    # Process validation images
    with open(val_txt_path, 'w') as f:
        # Process COCO validation images (take a subset)
        sampled_coco_val = random.sample(coco_val_paths, min(len(coco_val_paths), 500))
        
        for path in sampled_coco_val:
            if path.startswith('/'):
                path = path[1:]
                
            img_path = Path(dataset_path) / path
            if not img_path.exists():
                continue
                
            label_path = str(path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = Path(dataset_path) / label_path
            
            if not label_path.exists():
                continue
                
            dest_img = output_images_val / img_path.name
            dest_label = output_labels_val / label_path.name
            
            shutil.copy(img_path, dest_img)
            shutil.copy(label_path, dest_label)
            
            # Write to val.txt using the original format (./images/...)
            f.write(f"./images/val/{img_path.name}\n")
        
        # Process all Yoga validation images
        for path in yoga_val_paths:
            img_path = Path(dataset_path) / path
            if not img_path.exists():
                continue
                
            label_path = str(path).replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label_path = Path(dataset_path) / label_path
            
            if not label_path.exists():
                continue
                
            dest_img = output_images_val / img_path.name
            dest_label = output_labels_val / label_path.name
            
            shutil.copy(img_path, dest_img)
            shutil.copy(label_path, dest_label)
            
            # Write to val.txt using the original format (./images/...)
            f.write(f"./images/val/{img_path.name}\n")
    
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
    
    print(f"Created mixed dataset at {output_path}")
    print(f"Configuration file created at {yaml_path}")
    print(f"Use this configuration for training: --data {args.datasets_path}/{args.output_dir}/mixed_coco_yoga.yaml")

if __name__ == "__main__":
    args = parse_args()
    create_mixed_dataset(args) 