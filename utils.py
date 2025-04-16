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


def bbox_decode(anchor_points, pred_dist, proj):
    """Decode predicted object bounding box coordinates from anchor points and distribution."""
    b, a, c = pred_dist.shape  # batch, anchors, channels
    pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj.type(pred_dist.dtype))
    return dist2bbox(pred_dist, anchor_points, xywh=False)

def kpts_decode(anchor_points, pred_kpts):
    """Decode predicted keypoints to image coordinates."""
    y = pred_kpts.clone()
    y[..., :2] *= 2.0
    y[..., 0] += anchor_points[:, [0]] - 0.5
    y[..., 1] += anchor_points[:, [1]] - 0.5
    return y

def target_preprocess(targets, batch_size, scale_tensor):
    """Preprocess targets by converting to tensor format and scaling coordinates."""
    print("targets", describe_var(targets))
    nl, ne = targets.shape
    if nl == 0:
        out = torch.zeros(batch_size, 0, ne - 1)
    else:
        i = targets[:, 0]  # image index
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1)
        for j in range(batch_size):
            matches = i == j
            if n := matches.sum():
                out[j, :n] = targets[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
    return out

def setup_model(model_name, model_path, im0s, original_shapes, batch_idx_tensor, cls_tensor, bboxes_tensor):
    # 載入模型
    model = YOLO(model_path)
    model.model.train()

    # 凍結
    for k, v in model.model.named_parameters():
        v.requires_grad = False
    for m in model.model.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.eval()  # 只有BN層設為評估模式

    # 設置步幅和圖像大小
    stride = torch.Tensor([32])

    print("stride", stride)

    # 預處理圖片
    im = preprocess(im0s, stride=stride, imgsz=(640, 640))
    print("im", describe_var(im, max_depth=10, max_items=100))

    # 推理
    # x:
    # list[3]: [
    #   torch.Tensor(shape=[2, 65, 80, 80], dtype=torch.float32): tensor([[[[ 5.5371e+00,  2.5322e+00,  1.7914e+00,  ...,  1.5071e+00,  1.6543e+00... (truncated)
    #   torch.Tensor(shape=[2, 65, 40, 40], dtype=torch.float32): tensor([[[[ 5.5876e+00,  2.9318e+00,  1.7968e+00,  ...,  1.6331e+00,  3.5969e-01... (truncated)
    #   torch.Tensor(shape=[2, 65, 20, 20], dtype=torch.float32): tensor([[[[ 4.5414e+00,  2.1928e+00,  2.1972e+00,  ...,  3.1243e+00,  3.5783e+00... (truncated)
    # ]
    x, kpt = model.model(im)
    print("x", describe_var(x, max_depth=10, max_items=100))
    print("kpt", describe_var(kpt, max_depth=10, max_items=100))

    # 解碼
    y = model.model.model[-1]._inference(x)

    # 計算batch size
    bs = x[0].shape[0]

    # 解碼關鍵點
    pred_kpt =  model.model.model[-1].kpts_decode(bs, kpt)

    # 合併預測
    # tuple[2]: [
    #   torch.Tensor(shape=[2, 56, 8400], dtype=torch.float32): tensor([[[1.1811e+01, 1.7961e+01, 2.9799e+01,  ..., 5.4256e+02, 5.8482e+02, 6.09... (truncated)
    #   tuple[2]: [
    #     list[3]: [
    #       torch.Tensor(shape=[2, 65, 80, 80], dtype=torch.float32): tensor([[[[ 5.5371e+00,  2.5322e+00,  1.7914e+00,  ...,  1.5071e+00,  1.6543e+00... (truncated)
    #       torch.Tensor(shape=[2, 65, 40, 40], dtype=torch.float32): tensor([[[[ 5.5876e+00,  2.9318e+00,  1.7968e+00,  ...,  1.6331e+00,  3.5969e-01... (truncated)
    #       torch.Tensor(shape=[2, 65, 20, 20], dtype=torch.float32): tensor([[[[ 4.5414e+00,  2.1928e+00,  2.1972e+00,  ...,  3.1243e+00,  3.5783e+00... (truncated)
    #     ]
    #     torch.Tensor(shape=[2, 51, 8400], dtype=torch.float32): tensor([[[ 0.0622, -0.1197,  1.0606,  ...,  0.0992,  0.2038,  0.0138],
    #          ... (truncated)
    #   ]
    # ]
    print("y", describe_var(y, max_depth=10, max_items=100))
    print("x", describe_var(x, max_depth=10, max_items=100))
    preds = (torch.cat([y, pred_kpt], 1), (x, kpt))

    print("preds", describe_var(preds, max_depth=10, max_items=100))

    prediction = preds[0] # shape=[2, 56, 8400]
    xc = prediction[:, 4:5].amax(1) > 0.25  # shape=[2, 8400]

    prediction = prediction.transpose(-1, -2) # shape[2, 56, 8400] to shape[2, 8400, 56]

    # 向量化處理所有批次
    bs = prediction.shape[0]
    max_det = 300

    # 創建存儲結果的列表
    # list[2]: [
    #     torch.Tensor(shape=[10], dtype=torch.int64): tensor([8273, 8293, 8272, 8291, 8292, 8271, 8252, 8312, 8311, 8251])
    #     torch.Tensor(shape=[31], dtype=torch.int64): tensor([8263, 8244, 8242, 8264, 8222, 8243, 8283, 8223, 8262, 8224, 8109, 8129, ... (truncated)
    # ]
    all_selected_indices = []
    # list[2]: [
    #     torch.Tensor(shape=[10], dtype=torch.float32): tensor([0.8996, 0.8878, 0.8867, 0.8832, 0.8799, 0.8748, 0.8517, 0.8494, 0.8412, ... (truncated)
    #     torch.Tensor(shape=[31], dtype=torch.float32): tensor([0.8680, 0.8667, 0.8666, 0.8662, 0.8595, 0.8591, 0.8589, 0.8542, 0.8501, ... (truncated)
    # ]
    all_selected_confidences = []

    # 一次性處理所有批次
    for xi in range(bs):
        # 獲取通過閾值的索引
        indices = torch.nonzero(xc[xi]).squeeze(-1)  # shape=[num_filtered]
        print(f"在第{xi}張圖像中，有{indices.shape[0]}個預測通過了置信度閾值")
        
        # 獲取過濾後的預測和分數
        filtered_x = prediction[xi, xc[xi]]  # shape=[num_filtered, 56]
        scores = filtered_x[:, 4]  # shape=[num_filtered]
        
        # 使用torch.topk直接獲取前k個最高分數的索引，比argsort更高效
        k = min(max_det, scores.shape[0])
        topk_values, topk_indices = torch.topk(scores, k)
        
        # 獲取原始索引和對應的置信度（已按置信度從高到低排序）
        # torch.Tensor(shape=[10], dtype=torch.int64): tensor([8273, 8293, 8272, 8291, 8292, 8271, 8252, 8312, 8311, 8251])
        selected_indices = indices[topk_indices]
        # torch.Tensor(shape=[10], dtype=torch.float32): tensor([0.8996, 0.8878, 0.8867, 0.8832, 0.8799, 0.8748, 0.8517, 0.8494, 0.8412, ... (truncated)
        selected_confidences = topk_values
        
        all_selected_indices.append(selected_indices)
        all_selected_confidences.append(selected_confidences)
        
        print("    selected_indices:")
        print(describe_var(selected_indices, indent_size=4))
        print("    selected_confidences:")
        print(describe_var(selected_confidences, indent_size=4))

    print("\n" + "=" * 100 + "\n")
    print("all_selected_indices:")
    print(describe_var(all_selected_indices, indent_size=4))
    print("all_selected_confidences:")
    print(describe_var(all_selected_confidences, indent_size=4))

    # 後處理
    results = postprocess(preds, im, im0s)
    for i, result in enumerate(results):
        xy = result.keypoints.xy  # x and y coordinates
        xyn = result.keypoints.xyn  # normalized
        kpts = result.keypoints.data  # x, y, visibility (if available)
        print(f"\nResult#{i} kpts:\n")
        print(describe_var(result.keypoints.data, indent_size=4))

    print("\n" + "=" * 100 + "\n")

    no = 65
    reg_max = 16
    nc = 1
    proj = torch.arange(reg_max, dtype=torch.float)

    feats, pred_kpts = preds[1]

    print("feats", describe_var(feats))
    print("pred_kpts", describe_var(pred_kpts))

    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], no, -1) for xi in feats], 2).split(
        (reg_max * 4, nc), 1
    )
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()
    pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
    # torch.Tensor(shape=[1, 8400, 1], dtype=torch.float32): tensor([[[1.5881e-05],
    #          [1.4163e-05],
    #          [8.0755e-06],
    #          ..... (truncated)
    normalized_pred_scores = pred_scores.detach().sigmoid()

    # 使用新创建的batch_idx_tensor和bboxes_tensor来调用transform_bboxes_xywh
    bboxes_tensor = transform_bboxes_xywh(
        bboxes=bboxes_tensor,
        batch_idx=batch_idx_tensor,
        original_shape=original_shapes, 
        imgsz=(640, 640),
        # stride=stride.item()
    )

    print("bboxes_tensor", describe_var(bboxes_tensor))

    # print(f"转换后的边界框: {bboxes}")
    m = model.model.model[-1]
    s = m.stride
    anchor_points, stride_tensor = make_anchors(feats, s, 0.5)
    print("feats", describe_var(feats))
    print("stride", describe_var(s))



    print("anchor_points", describe_var(anchor_points))
    print("stride_tensor", describe_var(stride_tensor))
    print("pred_distri", describe_var(pred_distri))
    print("proj", describe_var(proj))

    pred_bboxes = bbox_decode(anchor_points, pred_distri, proj)  # xyxy, (b, h*w, 4)
    pred_kpts = kpts_decode(anchor_points, pred_kpts.view(bs, -1, 17, 3))  # (b, h*w, 17, 3)

    imgsz = torch.tensor(feats[0].shape[2:]) * s[0]
    targets = torch.cat((batch_idx_tensor, cls_tensor.view(-1, 1), bboxes_tensor), 1)
    print("batch_idx_tensor", describe_var(batch_idx_tensor))
    print("cls_tensor.view(-1, 1)", describe_var(cls_tensor.view(-1, 1)))
    print("bboxes_tensor", describe_var(bboxes_tensor))

    targets = target_preprocess(targets, bs, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    print("targets2", describe_var(targets))
    print("batch_size", bs)
    print("imgsz[[1, 0, 1, 0]]", imgsz[[1, 0, 1, 0]])
    print("cls_tensor.view(-1, 1)", cls_tensor.view(-1, 1))
    print("bboxes_tensor", bboxes_tensor)
    print("batch_idx_tensor", batch_idx_tensor)

    assigner = TaskAlignedAssigner(topk=10, num_classes=nc, alpha=0.5, beta=6.0)

    # target_bboxes: torch.Tensor(shape=[1, 8400, 4]
    # target_scores: torch.Tensor(shape=[1, 8400, 1]
    # fg_mask: torch.Tensor(shape=[1, 8400]
    # target_gt_idx: torch.Tensor(shape=[1, 8400]
    _, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    print("target_bboxes", describe_var(target_bboxes))
    print("target_scores", describe_var(target_scores))
    print("fg_mask", describe_var(fg_mask))
    print("target_gt_idx", describe_var(target_gt_idx))

    # 分析為什麼有些高分預測沒有被分配真實框
    # 使用TAL內部的掩碼和度量來診斷分配問題
    mask_in_gts = TaskAlignedAssigner.select_candidates_in_gts(anchor_points * stride_tensor, gt_bboxes)
    align_metric, overlaps = assigner.get_box_metrics(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        gt_labels,
        gt_bboxes,
        mask_in_gts * mask_gt
    )

    # # 使用我們添加的函數來分析和記錄結果
    # analyze_positive_assignment(
    #     pred_scores.detach(),
    #     (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
    #     anchor_points * stride_tensor,
    #     gt_labels,
    #     gt_bboxes,
    #     mask_gt,
    #     fg_mask,
    #     mask_in_gts,
    #     align_metric,
    #     overlaps,
    #     threshold=0.1  # 設置為希望分析的分數閾值
    # )

    # 計算一下 fg_mask 的數量 (True 的數量)
    fg_mask_count = fg_mask.sum()
    print("fg_mask_count", fg_mask_count)

    # target_scores 的最大值
    target_scores_max = target_scores.max()
    print("target_scores_max", target_scores_max)

    # target_scores 的最小值
    target_scores_min = target_scores.min()
    print("target_scores_min", target_scores_min)

    # print("normalized_pred_scores", describe_var(normalized_pred_scores))

    print("pred_bboxes", describe_var(pred_bboxes))
    print("pred_kpts", describe_var(pred_kpts))

    print("\n" + "=" * 100 + "\n")


    # 創建一個新的掩碼，找出置信值大於0.1的預測
    high_confidence_mask = normalized_pred_scores > 0.1

    # 將形狀從 [1, 8400, 1] 轉換為 [1, 8400]
    high_confidence_mask = high_confidence_mask.squeeze(-1)

    # 計算高置信度預測的數量
    high_confidence_count = high_confidence_mask.sum().item()
    print(f"高置信度(>0.1)預測數量: {high_confidence_count}")

    # 找出高置信度但未被分配為正樣本的預測
    # torch.Tensor(shape=[1, 8400], dtype=torch.bool): tensor([[False, False, False,  ..., False, False, False]])
    high_conf_not_assigned = high_confidence_mask & (~fg_mask)
    high_conf_not_assigned_count = high_conf_not_assigned.sum().item()
    print("high_conf_not_assigned", describe_var(high_conf_not_assigned))
    print(f"高置信度但未被分配為正樣本的預測數量: {high_conf_not_assigned_count}")

    # 找出被分配為正樣本的預測中的置信度
    assigned_scores = normalized_pred_scores[fg_mask].squeeze(-1)
    if len(assigned_scores) > 0:
        print(f"被分配為正樣本的預測置信度 - 最小值: {assigned_scores.min().item():.6f}, 最大值: {assigned_scores.max().item():.6f}, 平均值: {assigned_scores.mean().item():.6f}")

    # 找出高置信度但未被分配的預測中的置信度
    unassigned_high_scores = normalized_pred_scores[high_conf_not_assigned].squeeze(-1)
    if len(unassigned_high_scores) > 0:
        print(f"高置信度但未被分配的預測置信度 - 最小值: {unassigned_high_scores.min().item():.6f}, 最大值: {unassigned_high_scores.max().item():.6f}, 平均值: {unassigned_high_scores.mean().item():.6f}")

    # 獲取未分配錨點的 indices，根據批次分組
    unassigned_indices_list = []
    unassigned_confidences_list = []
    for i in range(normalized_pred_scores.shape[0]):  # 遍歷每個批次
        # 獲取當前批次的高置信度未分配預測索引
        batch_mask = high_conf_not_assigned[i]
        batch_indices = torch.nonzero(batch_mask).squeeze(-1)  # 獲取單一批次內的索引
        unassigned_indices_list.append(batch_indices)
        batch_confidences = normalized_pred_scores[i, batch_indices].squeeze(-1)
        unassigned_confidences_list.append(batch_confidences)

    print(f"高置信度但未被分配的預測 indices 列表:")
    print(describe_var(unassigned_indices_list))
    print(f"第一個批次未分配的 indices 數量: {len(unassigned_indices_list[0])}")
    print(f"第一個批次前 10 個未分配的 indices: {unassigned_indices_list[0][:10]}")
    print(f"第一個批次未分配的 confidences 數量: {len(unassigned_confidences_list[0])}")
    print(f"第一個批次前 10 個未分配的 confidences: {unassigned_confidences_list[0][:10]}")

    return {
        "preds": preds,
        "all_selected_indices": all_selected_indices,
        "all_selected_confidences": all_selected_confidences,
        "results": results,
        "normalized_pred_scores": normalized_pred_scores,
        "fg_mask": fg_mask,
        "target_gt_idx": target_gt_idx,
        "target_scores": target_scores,
        "high_conf_not_assigned": high_conf_not_assigned,
        "unassigned_indices": unassigned_indices_list,
        "unassigned_confidences": unassigned_confidences_list,
        "model": model,
    }
