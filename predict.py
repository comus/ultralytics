from ultralytics import YOLO
from ultralytics.utils.dev import describe_var

# Load a model
model = YOLO("yolo11n-pose.pt") 

# Predict with the model
results = model("image.jpg", save=True)  # predict on an image

# Access the results
for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)
    
# Print results structure information
print(describe_var(results))