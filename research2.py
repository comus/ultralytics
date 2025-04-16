import numpy as np
import torch
from ultralytics import YOLO, nn
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.nn.modules.conv import Conv
from ultralytics.utils import ops
from ultralytics.utils.dev import describe_var
from ultralytics.utils.ops import nms_rotated, xywh2xyxy
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

model = YOLO("yolo11n-pose.pt")

results = model("image2.jpg")

for result in results:
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)

    print(describe_var(result, max_items=20, max_depth=10))

    # print("xy:\n", describe_var(xy))
    # print("xyn:\n", describe_var(xyn))
    print("kpts:\n")
    print(describe_var(kpts))
    print(kpts)
    # print(xy)
    # print(xyn)
