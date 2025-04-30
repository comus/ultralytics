import numpy as np
import torch
from process import postprocess, preprocess, transform_bboxes_xywh
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.dev import describe_var
from ultralytics.utils.ops import nms_rotated, xywh2xyxy
import time
import cv2
import os
from visualize import visualize_indices_on_feature_maps
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from analyze import analyze_positive_assignment
from utils import bbox_decode, kpts_decode, setup_model, target_preprocess

# 讀取圖片
images = [
    {
        "path": '/Users/region/ultralytics/image4.jpg',
        "labels": [
            # [0.0, 0.671279,0.617945,0.645759,0.726859],

            # [0.0, 0.535530, 0.308733, 0.206900, 0.317147, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.514000, 0.194667, 2.000000, 0.534000, 0.213333, 2.000000, 0.482000, 0.224000, 2.000000, 0.526000, 0.229333, 2.000000, 0.462000, 0.186667, 2.000000, 0.568000, 0.210667, 2.000000, 0.446000, 0.165333, 2.000000, 0.610000, 0.184000, 2.000000, 0.462000, 0.336000, 2.000000, 0.498000, 0.341333, 2.000000, 0.462000, 0.368000, 2.000000, 0.536000, 0.290667, 2.000000, 0.460000, 0.453333, 2.000000, 0.508000, 0.376000, 2.000000, 2.000000],
            # [0.0, 0.736090, 0.272987, 0.189260, 0.259413, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.762000, 0.184000, 2.000000, 0.000000, 0.000000, 0.000000, 0.754000, 0.178667, 2.000000, 0.710000, 0.176000, 2.000000, 0.730000, 0.178667, 2.000000, 0.674000, 0.189333, 2.000000, 0.780000, 0.202667, 2.000000, 0.652000, 0.218667, 2.000000, 0.806000, 0.205333, 2.000000, 0.660000, 0.245333, 2.000000, 0.680000, 0.250667, 2.000000, 0.708000, 0.304000, 2.000000, 0.736000, 0.293333, 2.000000, 0.722000, 0.365333, 2.000000, 0.714000, 0.373333, 2.000000],
            # [0.0, 0.146660, 0.667293, 0.194000, 0.441093, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.180000, 0.498667, 2.000000, 0.144000, 0.549333, 2.000000, 0.198000, 0.541333, 2.000000, 0.124000, 0.648000, 2.000000, 0.000000, 0.000000, 0.000000, 0.092000, 0.720000, 2.000000, 0.000000, 0.000000, 0.000000, 0.172000, 0.704000, 2.000000, 0.222000, 0.701333, 2.000000, 0.102000, 0.746667, 2.000000, 0.162000, 0.754667, 2.000000, 0.130000, 0.856000, 2.000000, 0.170000, 0.829333, 2.000000],

            # [0, 0.620039, 0.593900, 0.172415, 0.146080, 0.658793, 0.552000, 2.000000, 0.664042, 0.550000, 2.000000, 0.656168, 0.548000, 2.000000, 0.679790, 0.554000, 2.000000, 0.000000, 0.000000, 0.000000, 0.679790, 0.570000, 2.000000, 0.648294, 0.568000, 2.000000, 0.674541, 0.588000, 2.000000, 0.606299, 0.582000, 2.000000, 0.627297, 0.562000, 2.000000, 0.603675, 0.564000, 2.000000, 0.627297, 0.616000, 2.000000, 0.000000, 0.000000, 0.000000, 0.595801, 0.604000, 2.000000, 0.000000, 0.000000, 0.000000, 0.569554, 0.644000, 2.000000, 0.000000, 0.000000, 0.000000],
            # [0, 0.385525, 0.585570, 0.149370, 0.125860, 0.414698, 0.540000, 1.000000, 0.419948, 0.538000, 2.000000, 0.412073, 0.538000, 1.000000, 0.438320, 0.540000, 2.000000, 0.000000, 0.000000, 0.000000, 0.435696, 0.558000, 2.000000, 0.417323, 0.554000, 2.000000, 0.396325, 0.574000, 2.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.414698, 0.608000, 2.000000, 0.398950, 0.604000, 2.000000, 0.372703, 0.592000, 2.000000, 0.000000, 0.000000, 0.000000, 0.346457, 0.638000, 2.000000, 0.000000, 0.000000, 0.000000],

            [0, 0.662641, 0.494385, 0.674719, 0.988771, 0.717187, 0.189583, 2.000000, 0.798438, 0.127083, 2.000000, 0.701562, 0.091667, 2.000000, 0.921875, 0.118750, 2.000000, 0.000000, 0.000000, 0.000000, 0.971875, 0.379167, 2.000000, 0.554688, 0.262500, 2.000000, 0.000000, 0.000000, 0.000000, 0.367188, 0.427083, 2.000000, 0.767188, 0.772917, 2.000000, 0.421875, 0.500000, 2.000000, 0.829688, 0.960417, 1.000000, 0.517188, 0.881250, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0, 0.198031, 0.560677, 0.392687, 0.586521, 0.104688, 0.522917, 2.000000, 0.142187, 0.481250, 2.000000, 0.084375, 0.468750, 2.000000, 0.250000, 0.497917, 2.000000, 0.000000, 0.000000, 0.000000, 0.301563, 0.633333, 2.000000, 0.048438, 0.635417, 2.000000, 0.365625, 0.833333, 1.000000, 0.015625, 0.858333, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0, 0.487922, 0.144948, 0.200563, 0.285396, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.437500, 0.145833, 2.000000, 0.562500, 0.108333, 2.000000, 0.414062, 0.287500, 1.000000, 0.593750, 0.225000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.464062, 0.400000, 1.000000, 0.551562, 0.395833, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0, 0.360898, 0.098250, 0.106328, 0.196500, 0.418750, 0.041667, 1.000000, 0.423438, 0.027083, 1.000000, 0.401562, 0.031250, 2.000000, 0.000000, 0.000000, 0.000000, 0.384375, 0.045833, 2.000000, 0.000000, 0.000000, 0.000000, 0.368750, 0.091667, 2.000000, 0.000000, 0.000000, 0.000000, 0.326562, 0.177083, 2.000000, 0.000000, 0.000000, 0.000000, 0.373437, 0.108333, 2.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0, 0.283820, 0.093937, 0.126391, 0.178792, 0.264062, 0.097917, 2.000000, 0.273438, 0.083333, 2.000000, 0.254688, 0.085417, 2.000000, 0.296875, 0.064583, 2.000000, 0.000000, 0.000000, 0.000000, 0.335938, 0.110417, 2.000000, 0.257812, 0.122917, 2.000000, 0.331250, 0.208333, 1.000000, 0.237500, 0.202083, 1.000000, 0.304688, 0.127083, 2.000000, 0.239063, 0.127083, 2.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0, 0.101781, 0.181698, 0.203563, 0.363396, 0.171875, 0.095833, 2.000000, 0.000000, 0.000000, 0.000000, 0.162500, 0.089583, 2.000000, 0.000000, 0.000000, 0.000000, 0.139063, 0.104167, 2.000000, 0.004687, 0.158333, 2.000000, 0.112500, 0.185417, 2.000000, 0.000000, 0.000000, 0.000000, 0.167187, 0.312500, 1.000000, 0.000000, 0.000000, 0.000000, 0.178125, 0.189583, 2.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0, 0.662703, 0.090156, 0.054250, 0.175229, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.651563, 0.033333, 2.000000, 0.684375, 0.033333, 2.000000, 0.629687, 0.070833, 1.000000, 0.692187, 0.081250, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.634375, 0.206250, 1.000000, 0.681250, 0.208333, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000],
            [0, 0.586672, 0.097323, 0.124219, 0.194646, 0.545312, 0.018750, 2.000000, 0.556250, 0.002083, 2.000000, 0.535937, 0.002083, 2.000000, 0.589063, 0.006250, 2.000000, 0.000000, 0.000000, 0.000000, 0.618750, 0.058333, 2.000000, 0.526563, 0.075000, 1.000000, 0.612500, 0.179167, 2.000000, 0.000000, 0.000000, 0.000000, 0.537500, 0.181250, 1.000000, 0.000000, 0.000000, 0.000000, 0.600000, 0.275000, 1.000000, 0.531250, 0.283333, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000]

        ]
    },
    # {
    #     "path": '/Users/region/ultralytics/image.jpg',
    #     "labels": [
    #         [0.0, 0.671279,0.617945,0.645759,0.726859],
    #         # [0.0, 0.671279,0.617945,0.645759,0.726859],
    #         # [0.0, 0.671279,0.617945,0.645759,0.726859],
    #     ]
    # }
    # 可以添加更多圖片
]

# 修改圖片處理部分，添加batch_idx
im0s = []
original_shapes = []
all_bboxes = []
total_boxes = 0
batch_idxs = []
batch_cls = []

for batch_idx, img_info in enumerate(images):
    img = cv2.imread(img_info["path"])
    im0s.append(img)
    original_shapes.append(img.shape[:2])  # 添加(h, w)
    
    # 對該圖片中的每個標籤進行處理
    for label in img_info["labels"]:
        # 正確分離類別和邊界框
        cls = label[0]
        box = label[1:5]
        
        all_bboxes.append(box)
        batch_cls.append(cls)
        
        # 為每個box創建對應的batch_idx
        batch_idxs.append([batch_idx])
        total_boxes += 1

# 轉換為tensor格式
batch_idx_tensor = torch.tensor(batch_idxs, dtype=torch.float32)
bboxes_tensor = torch.tensor(all_bboxes, dtype=torch.float32)
cls_tensor = torch.tensor(batch_cls, dtype=torch.float32)

print(f"batch_idx_tensor: {batch_idx_tensor}")
print(f"bboxes_tensor: {bboxes_tensor}")
print(f"cls_tensor: {cls_tensor}")


student = setup_model("student", "yolo11n-pose.pt", im0s, original_shapes, batch_idx_tensor, cls_tensor, bboxes_tensor)
teacher = setup_model("teacher", "yolo11x-pose.pt", im0s, original_shapes, batch_idx_tensor, cls_tensor, bboxes_tensor)

def indices_to_coordinates_vectorized(indices, model):
    """
    將錨點索引轉換為物理坐標（向量化版本）
    
    Args:
        indices: 錨點索引 tensor
        model: 模型對象，用於獲取 stride 和特徵圖大小
        
    Returns:
        coordinates: 物理坐標 tensor [N, 2] (x, y)
        levels: 每個索引所屬的層級 tensor
        h: 行座標 tensor
        w: 列座標 tensor
        strides_per_index: 每個索引對應的stride值 tensor
    """
    # 從模型獲取 stride 和特徵圖大小
    strides = model.model.stride
    
    # 獲取模型的輸入大小，如果無法獲取則使用默認值
    if hasattr(model, 'imgsz'):
        input_size = model.imgsz[0]  # 假設是正方形，取寬度
    elif hasattr(model.model, 'imgsz'):
        input_size = model.model.imgsz[0]
    else:
        # 嘗試從模型的args獲取input_size
        if hasattr(model, 'args') and hasattr(model.args, 'imgsz'):
            input_size = model.args.imgsz[0] if isinstance(model.args.imgsz, list) else model.args.imgsz
        else:
            input_size = 640  # 默認輸入大小
    
    print(f"模型輸入大小: {input_size}")
    
    # 估算特徵圖大小
    feature_sizes = []
    level_start_indices = [0]
    level_sizes = []
    
    for i, stride in enumerate(strides):
        size = int(input_size / stride)
        feature_sizes.append((size, size))  # w, h
        level_size = size * size
        level_sizes.append(level_size)
        if i > 0:
            level_start_indices.append(level_start_indices[-1] + level_sizes[i-1])
    
    # 將層級起始索引和層級大小轉換為張量
    level_start_indices_tensor = torch.tensor(level_start_indices, device=indices.device)
    level_sizes_tensor = torch.tensor(level_sizes, device=indices.device)
    strides_tensor = torch.tensor(strides, device=indices.device)
    
    # 計算每個索引所屬的層級
    indices_expanded = indices.unsqueeze(1).expand(-1, len(level_start_indices))
    level_start_expanded = level_start_indices_tensor.unsqueeze(0).expand(indices.size(0), -1)
    level_end_expanded = (level_start_indices_tensor + level_sizes_tensor).unsqueeze(0).expand(indices.size(0), -1)
    
    # 創建掩碼：indices >= level_start 且 indices < level_end
    mask_start = indices_expanded >= level_start_expanded
    mask_end = indices_expanded < level_end_expanded
    level_masks = mask_start & mask_end
    
    # 獲取每個索引的層級 (每行中True的索引)
    levels = torch.argmax(level_masks.float(), dim=1)
    
    # 獲取每個索引在其層級中的相對索引
    relative_indices = indices - level_start_indices_tensor[levels]
    
    # 獲取每個層級的寬度
    widths = torch.tensor([feature_sizes[i][0] for i in range(len(feature_sizes))], device=indices.device)
    
    # 計算每個索引的行和列
    feature_widths = widths[levels]
    h = relative_indices // feature_widths
    w = relative_indices % feature_widths
    
    # 獲取每個索引對應的stride
    strides_per_index = strides_tensor[levels]
    
    # 計算物理坐標
    coordinates = torch.zeros(indices.size(0), 2, device=indices.device)
    coordinates[:, 0] = (w + 0.5) * strides_per_index  # x座標
    coordinates[:, 1] = (h + 0.5) * strides_per_index  # y座標
    
    return coordinates, levels, h, w, strides_per_index

def match_student_teacher_anchors_direct(student, teacher, threshold=1.5):
    """
    直接使用已有的學生未分配錨點和教師高分錨點進行匹配（完全向量化版本）
    
    Args:
        student: 包含學生未分配錨點的字典，有 unassigned_indices 和 unassigned_confidences
        teacher: 包含教師高分點的字典，有 all_selected_indices 和 all_selected_confidences
        threshold: 距離閾值倍數 (相對於stride)
        
    Returns:
        matched_pairs: 匹配的學生-教師錨點對
    """
    batch_size = len(student["unassigned_indices"])
    device = student["unassigned_indices"][0].device
    student_model = student["model"]
    teacher_model = teacher["model"]
    matched_pairs = []
    
    # 從模型獲取 stride
    s_strides = student_model.model.stride
    t_strides = teacher_model.model.stride
    
    # 獲取模型的輸入大小
    if hasattr(student_model, 'imgsz'):
        s_input_size = student_model.imgsz[0]
    elif hasattr(student_model.model, 'imgsz'):
        s_input_size = student_model.model.imgsz[0]
    else:
        s_input_size = 640  # 默認輸入大小
        
    if hasattr(teacher_model, 'imgsz'):
        t_input_size = teacher_model.imgsz[0]
    elif hasattr(teacher_model.model, 'imgsz'):
        t_input_size = teacher_model.model.imgsz[0]
    else:
        t_input_size = 640  # 默認輸入大小
    
    # 估算特徵圖大小
    s_feature_sizes = []
    t_feature_sizes = []
    for stride in s_strides:
        size = int(s_input_size / stride)
        s_feature_sizes.append((size, size))  # w, h
    for stride in t_strides:
        size = int(t_input_size / stride)
        t_feature_sizes.append((size, size))  # w, h
    
    print(f"學生模型 stride: {s_strides}, 輸入大小: {s_input_size}, 特徵圖大小: {s_feature_sizes}")
    print(f"教師模型 stride: {t_strides}, 輸入大小: {t_input_size}, 特徵圖大小: {t_feature_sizes}")
    
    # 用於收集所有批次的匹配信息
    all_match_info = {
        'student_batch': [],
        'student_index': [],
        'student_confidence': [],
        'student_center_x': [],
        'student_center_y': [],
        'student_level': [],
        'student_h': [],
        'student_w': [],
        'student_feature_size': [],
        'teacher_batch': [],
        'teacher_index': [],
        'teacher_confidence': [],
        'teacher_center_x': [],
        'teacher_center_y': [],
        'teacher_level': [],
        'teacher_h': [],
        'teacher_w': [],
        'teacher_feature_size': [],
        'distance': []
    }
    
    total_match_count = 0
    
    # 處理每個批次
    for b in range(batch_size):
        start_time = time.time()
        
        # 獲取學生未分配的錨點
        s_indices = student["unassigned_indices"][b]  # [N_s]
        s_confidences = student["unassigned_confidences"][b]  # [N_s]
        
        # 獲取教師高分錨點
        t_indices = teacher["all_selected_indices"][b]  # [N_t]
        t_confidences = teacher["all_selected_confidences"][b]  # [N_t]
        
        # 如果其中一個為空，跳過這個批次
        if s_indices.size(0) == 0 or t_indices.size(0) == 0:
            continue
        
        # 將學生和教師索引轉換為物理坐標 (向量化)
        s_centers, s_levels, s_h, s_w, s_stride_values = indices_to_coordinates_vectorized(s_indices, student_model)  # [N_s, 2]
        t_centers, t_levels, t_h, t_w, _ = indices_to_coordinates_vectorized(t_indices, teacher_model)  # [N_t, 2]
        
        # 計算每個學生錨點與所有教師錨點之間的距離矩陣
        # [S, 1, 2] - [1, T, 2] -> [S, T, 2]
        diffs = s_centers.unsqueeze(1) - t_centers.unsqueeze(0)
        
        # 計算歐氏距離: [S, T]
        distances = torch.sqrt(torch.sum(diffs ** 2, dim=2))
        
        # 找出每個學生錨點的最近教師錨點
        min_distances, min_indices = torch.min(distances, dim=1)
        
        # 設置每個學生錨點的距離閾值 (基於它所屬層的stride)
        dist_thresholds = s_stride_values * threshold
        
        # 過濾掉距離超過閾值的匹配
        valid_matches = min_distances < dist_thresholds
        
        if not valid_matches.any():
            continue
            
        # 獲取有效匹配的索引
        valid_s_indices = torch.nonzero(valid_matches).squeeze(1)
        valid_t_indices = min_indices[valid_matches]
        
        # 獲取批次匹配數量
        batch_match_count = valid_s_indices.size(0)
        
        # 使用向量化操作填充匹配信息
        all_match_info['student_batch'].extend([b] * batch_match_count)
        all_match_info['student_index'].extend(s_indices[valid_s_indices].tolist())
        all_match_info['student_confidence'].extend(s_confidences[valid_s_indices].tolist())
        all_match_info['student_center_x'].extend(s_centers[valid_s_indices, 0].tolist())
        all_match_info['student_center_y'].extend(s_centers[valid_s_indices, 1].tolist())
        all_match_info['student_level'].extend(s_levels[valid_s_indices].tolist())
        all_match_info['student_h'].extend(s_h[valid_s_indices].tolist())
        all_match_info['student_w'].extend(s_w[valid_s_indices].tolist())
        all_match_info['student_feature_size'].extend([s_feature_sizes[level] for level in s_levels[valid_s_indices].tolist()])
        
        all_match_info['teacher_batch'].extend([b] * batch_match_count)
        all_match_info['teacher_index'].extend(t_indices[valid_t_indices].tolist())
        all_match_info['teacher_confidence'].extend(t_confidences[valid_t_indices].tolist())
        all_match_info['teacher_center_x'].extend(t_centers[valid_t_indices, 0].tolist())
        all_match_info['teacher_center_y'].extend(t_centers[valid_t_indices, 1].tolist())
        all_match_info['teacher_level'].extend(t_levels[valid_t_indices].tolist())
        all_match_info['teacher_h'].extend(t_h[valid_t_indices].tolist())
        all_match_info['teacher_w'].extend(t_w[valid_t_indices].tolist())
        all_match_info['teacher_feature_size'].extend([t_feature_sizes[level] for level in t_levels[valid_t_indices].tolist()])
        
        all_match_info['distance'].extend(min_distances[valid_s_indices].tolist())
        
        total_match_count += batch_match_count
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"批次 {b} 處理時間: {elapsed_time:.4f} 秒, 找到 {batch_match_count} 個匹配")
    
    # 將收集的匹配信息轉換為所需的格式
    for i in range(total_match_count):
        match_info = {
            'student': {
                'batch': all_match_info['student_batch'][i],
                'index': all_match_info['student_index'][i],
                'confidence': all_match_info['student_confidence'][i],
                'center': (all_match_info['student_center_x'][i], all_match_info['student_center_y'][i]),
                'level': all_match_info['student_level'][i],
                'feature_coord': (all_match_info['student_h'][i], all_match_info['student_w'][i]),
                'feature_size': all_match_info['student_feature_size'][i]
            },
            'teacher': {
                'batch': all_match_info['teacher_batch'][i],
                'index': all_match_info['teacher_index'][i],
                'confidence': all_match_info['teacher_confidence'][i],
                'center': (all_match_info['teacher_center_x'][i], all_match_info['teacher_center_y'][i]),
                'level': all_match_info['teacher_level'][i],
                'feature_coord': (all_match_info['teacher_h'][i], all_match_info['teacher_w'][i]),
                'feature_size': all_match_info['teacher_feature_size'][i]
            },
            'distance': all_match_info['distance'][i]
        }
        matched_pairs.append(match_info)
    
    return matched_pairs

# 匹配學生未分配錨點和教師高分錨點
batch_size = student["preds"][0].shape[0]
print("batch_size", batch_size)
matched_pairs = match_student_teacher_anchors_direct(student, teacher)
print("matched_pairs", describe_var(matched_pairs))

# 打印匹配結果
print(f"找到 {len(matched_pairs)} 個匹配對")
if matched_pairs:
    for i, pair in enumerate(matched_pairs[:5]):  # 只打印前5個匹配
        print(f"匹配 #{i+1}:")
        s_level = pair['student']['level']
        s_h, s_w = pair['student']['feature_coord']
        s_size = pair['student']['feature_size']
        
        t_level = pair['teacher']['level']
        t_h, t_w = pair['teacher']['feature_coord']
        t_size = pair['teacher']['feature_size']
        
        print(f"  學生錨點 - 索引: {pair['student']['index']}, 層級: {s_level} ({s_size[0]}x{s_size[1]}), 座標: ({s_w},{s_h}), 置信度: {pair['student']['confidence']:.4f}")
        print(f"  教師錨點 - 索引: {pair['teacher']['index']}, 層級: {t_level} ({t_size[0]}x{t_size[1]}), 座標: ({t_w},{t_h}), 置信度: {pair['teacher']['confidence']:.4f}")
        print(f"  距離: {pair['distance']:.2f} 像素")
        print()

exit()

print("\n" + "=" * 100 + "\n")

# # 使用 visualize_indices_on_feature_maps 來可視化特徵圖索引分佈
# visualize_indices_on_feature_maps(
#     student_preds=student["preds"],
#     student_indices=student["all_selected_indices"],
#     student_confidences=student["all_selected_confidences"],
#     student_results=student["results"],
#     student_pred_scores=student["normalized_pred_scores"],
#     student_fg_mask=student["fg_mask"],
#     student_target_gt_idx=student["target_gt_idx"],
#     student_target_scores=student["target_scores"],
#     student_high_conf_not_assigned=student["high_conf_not_assigned"],

#     teacher_preds=teacher["preds"],
#     teacher_indices=teacher["all_selected_indices"],
#     teacher_confidences=teacher["all_selected_confidences"],
#     teacher_results=teacher["results"],
#     teacher_pred_scores=teacher["normalized_pred_scores"],
#     teacher_fg_mask=teacher["fg_mask"],
#     teacher_target_gt_idx=teacher["target_gt_idx"],
#     teacher_target_scores=teacher["target_scores"],
#     teacher_high_conf_not_assigned=teacher["high_conf_not_assigned"]
# )

# print("\n" + "=" * 100 + "\n")
