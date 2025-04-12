from ultralytics import YOLO

# Load a model
model = YOLO("yolo11-pose-lite/train/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val()  # no arguments needed, dataset and settings remembered

# box
print(metrics.box.map)  # map50-95
print(metrics.box.map50)  # map50
print(metrics.box.map75)  # map75
print(metrics.box.maps)  # a list contains map50-95 of each category

# pose
print(metrics.pose.map)  # map50-95
print(metrics.pose.map50)  # map50
print(metrics.pose.map75)  # map75
print(metrics.pose.maps)  # a list contains map50-95 of each category
