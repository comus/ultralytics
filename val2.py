from ultralytics import YOLO

model = YOLO("distill_pose/yolo11n_distill_ultraconservative/weights/last.pt")

metrics = model.val(
  data="coco-pose.yaml",
)