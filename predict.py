from ultralytics import YOLO
import cv2
import numpy as np
import json
import os

def draw_text_with_border(img, text, pos, font, scale, color, thickness=2, border_color=(0, 0, 0)):
    # Draw border (black outline)
    border_thickness = thickness + 2
    cv2.putText(img, text, pos, font, scale, border_color, border_thickness)
    # Draw main text
    cv2.putText(img, text, pos, font, scale, color, thickness)

def draw_skeleton(img, keypoints):
    # Define the skeleton connections for COCO format
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], # legs
        [11, 12], # hips
        [5, 11], [6, 12], # spine
        [5, 6], # shoulders
        [5, 7], [7, 9], [6, 8], [8, 10], # arms
        [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6] # face
    ]
    
    # Colors for visualization
    color = (0, 255, 255)  # Yellow color for skeleton
    
    # Draw the skeleton for each person
    for person_kpts in keypoints:
        # Draw the skeleton
        for pair in skeleton:
            # Get coordinates for the pair of keypoints
            if pair[0] < len(person_kpts) and pair[1] < len(person_kpts):
                pt1 = tuple(map(int, person_kpts[pair[0]][:2]))
                pt2 = tuple(map(int, person_kpts[pair[1]][:2]))
                
                # Only draw if both points are valid (x,y > 0) and visible (conf > 0)
                if (pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0 and 
                    person_kpts[pair[0]][2] > 0 and person_kpts[pair[1]][2] > 0):
                    cv2.line(img, pt1, pt2, color, 2)

def draw_results(img, result, display_text):
    img_draw = img.copy()
    
    # Draw bounding boxes and labels
    boxes = result.boxes
    for box in boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Get confidence and class
        conf = float(box.conf[0])
        
        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add label with border
        label = f'Person {conf:.2f}'
        draw_text_with_border(img_draw, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))

    # Draw keypoints and skeleton
    keypoints = result.keypoints
    if keypoints is not None:
        # Get keypoints for all persons
        all_keypoints = keypoints.data
        
        # Draw skeleton first (so it's behind the keypoints)
        draw_skeleton(img_draw, all_keypoints)
        
        # Draw keypoints for all persons
        for person_kpts in all_keypoints:
            for kpt in person_kpts:
                x, y = int(kpt[0]), int(kpt[1])
                conf = float(kpt[2]) if len(kpt) > 2 else 1.0
                
                # Only draw if point is valid (x,y > 0) and visible (conf > 0)
                if x > 0 and y > 0 and conf > 0:
                    # Draw bigger keypoint circle
                    cv2.circle(img_draw, (x, y), 6, (255, 255, 0), -1)  # Filled yellow circle
                    cv2.circle(img_draw, (x, y), 6, (0, 0, 0), 1)      # Black border

    # Draw multi-line display text
    lines = display_text.split('\n')
    y_offset = 30
    for i, line in enumerate(lines):
        draw_text_with_border(img_draw, line, (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255))
    
    return img_draw

# Function to get metrics from best_metrics.json
def get_metrics(model_path):
    # For official model, return empty values
    if 'official' in model_path:
        return ""
    
    # Extract directory path
    model_dir = os.path.dirname(os.path.dirname(model_path))
    metrics_path = os.path.join(model_dir, 'best_metrics.json')
    
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        map50 = metrics['metrics']['metrics/mAP50(P)']
        map50_95 = metrics['metrics']['metrics/mAP50-95(P)']
        
        return f"\nmAP50(P): {map50:.4f}\nmAP50-95(P): {map50_95:.4f}"
    except Exception as e:
        print(f"Error reading metrics from {metrics_path}: {str(e)}")
        return "\nMetrics not available"

# Model names in specified order
model_names = [
    './models/official/yolo11n-pose.pt',  # Official yolo11n-pose
    './models/yolo11n-pose/train/weights/best.pt',  # Our trained version yolo11n-pose-stage1
    './models/yolo11n-pose-distill1/train/weights/best.pt',  # yolo11n-pose-distill1-stage1
    './models/yolo11n-pose-distill2/train/weights/best.pt',   # yolo11n-pose-distill2-stage1

    './models/official/yolo11m-pose.pt',  # Official yolo11m-pose
    './models/yolo11n-pose/train_stage2/weights/best.pt',  # Our trained version yolo11n-pose-stage2
    './models/yolo11n-pose-distill1/train_stage2/weights/best.pt',  # yolo11n-pose-distill1-stage2
    './models/yolo11n-pose-distill2/train_stage2/weights/best.pt',   # yolo11n-pose-distill2-stage2

    './models/official/yolo11s-pose.pt',  # Official yolo11s-pose
    './models/yolo11n-pose/train_stage3/weights/best.pt',  # Our trained version yolo11n-pose-stage3
    './models/yolo11n-pose-distill1/train_stage3/weights/best.pt',  # yolo11n-pose-distill1-stage3
    './models/yolo11n-pose-distill2/train_stage3/weights/best.pt',   # yolo11n-pose-distill2-stage3

    None,  # Empty slot
    './models/yolo11n-pose/train_stage4/weights/best.pt',  # Our trained version yolo11n-pose-stage4
    './models/yolo11n-pose-distill1/train_stage4/weights/best.pt',  # yolo11n-pose-distill1-stage4
    './models/yolo11n-pose-distill2/train_stage4/weights/best.pt'    # yolo11n-pose-distill2-stage4
]

# Display names for each model
display_names = [
    'official yolo11n-pose',
    'yolo11n-pose-stage1',
    'yolo11n-pose-distill1-stage1',
    'yolo11n-pose-distill2-stage1',

    'official yolo11m-pose',
    'yolo11n-pose-stage2',
    'yolo11n-pose-distill1-stage2',
    'yolo11n-pose-distill2-stage2',

    'official yolo11s-pose',
    'yolo11n-pose-stage3',
    'yolo11n-pose-distill1-stage3',
    'yolo11n-pose-distill2-stage3',

    '',  # Empty slot
    'yolo11n-pose-stage4',
    'yolo11n-pose-distill1-stage4',
    'yolo11n-pose-distill2-stage4'
]

# Load local image
original_img = cv2.imread('image.jpg')
if original_img is None:
    raise ValueError("Could not load bus.jpg. Make sure it exists in the current directory.")

# Process with each model
results_images = []
for i, model_path in enumerate(model_names):
    try:
        if model_path is None:
            # Create an empty black image for the empty slot
            empty_img = np.zeros_like(original_img)
            results_images.append(empty_img)
            continue

        print(f"Processing with {display_names[i]}...")
        model = YOLO(model_path)
        results = model(original_img)
        
        # Get display text including metrics
        metrics_text = get_metrics(model_path)
        display_text = display_names[i] + metrics_text
        
        for result in results:
            img_with_results = draw_results(original_img, result, display_text)
            results_images.append(img_with_results)
            
    except Exception as e:
        print(f"Error processing {display_names[i]}: {str(e)}")

# Only create comparison if we have results
if results_images:
    # Combine images into a grid with four rows and 4 columns
    rows = 4
    cols = 4  # 4 models per row
    cell_height, cell_width = original_img.shape[:2]
    grid_img = np.zeros((cell_height * rows, cell_width * cols, 3), dtype=np.uint8)

    # Place images in grid
    for idx, img in enumerate(results_images):
        i, j = idx // cols, idx % cols  # Row, column in a 4×4 grid
        grid_img[i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width] = img

    # Save the comparison grid
    cv2.imwrite('model_comparison.jpg', grid_img)
    print("Comparison results saved to model_comparison.jpg")
else:
    print("No results were generated. Check the error messages above.")