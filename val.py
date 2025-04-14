from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n-pose/train_stage2/weights/best.pt")  # load a custom model

# Validate the model
metrics = model.val(
  data="coco-pose.yaml",
  # save_json=True,
)  # no arguments needed, dataset and settings remembered

# box
print("box mAP50-95", metrics.box.map)  # map50-95
print("box mAP50", metrics.box.map50)  # map50
print("box mAP75", metrics.box.map75)  # map75
print("box mAPs", metrics.box.maps)  # a list contains map50-95 of each category

# pose
print("pose mAP50-95", metrics.pose.map)  # map50-95
print("pose mAP50", metrics.pose.map50)  # map50
print("pose mAP75", metrics.pose.map75)  # map75
print("pose mAPs", metrics.pose.maps)  # a list contains map50-95 of each category
