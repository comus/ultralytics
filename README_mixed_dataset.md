# Mixed COCO-Pose and Yoga82 Dataset Creator

This script creates a mixed dataset from COCO-Pose and Yoga82 datasets with weighted sampling to balance the training data according to specified weights.

## Features

- Creates a balanced dataset with configurable weights between COCO-Pose and Yoga82
- Copies images and labels to a new directory structure
- Generates train.txt and val.txt files with the new paths
- Creates a YAML configuration file compatible with YOLOv8

## Usage

```bash
python create_mixed_dataset.py [--datasets-path PATH] [--coco-weight WEIGHT] [--yoga-weight WEIGHT] [--output-dir DIR]
```

### Arguments

- `--datasets-path`: Path to the datasets directory (default: '/root/autodl-tmp/withcloud/datasets')
- `--coco-weight`: Weight for COCO-pose samples (default: 0.3)
- `--yoga-weight`: Weight for Yoga82 samples (default: 0.7)
- `--output-dir`: Output directory name under datasets path (default: 'mixed_coco_yoga')

## Example

```bash
# Run with default settings
python create_mixed_dataset.py

# Run with custom settings
python create_mixed_dataset.py --datasets-path /path/to/datasets --coco-weight 0.4 --yoga-weight 0.6 --output-dir my_mixed_dataset
```

## Training with the Mixed Dataset

After creating the mixed dataset, you can train your YOLOv8 model using:

```bash
yolo pose train data=/path/to/datasets/mixed_coco_yoga/mixed_coco_yoga.yaml
```

The script will output the exact command to use at the end of its execution.

## How It Works

1. The script calculates how many samples to take from each dataset based on the provided weights
2. It samples COCO-Pose images to match the ratio with Yoga82 images
3. It copies the selected images and their corresponding labels to a new directory structure
4. It creates train.txt and val.txt files pointing to the new image locations
5. It generates a YAML configuration file for YOLOv8 training 