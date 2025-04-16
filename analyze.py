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


# 新增函數：記錄分配過程中的評估結果
def analyze_positive_assignment(
    pred_scores, 
    pred_bboxes, 
    anchor_points, 
    gt_labels, 
    gt_bboxes, 
    mask_gt, 
    fg_mask, 
    mask_in_gts=None, 
    align_metric=None, 
    overlaps=None, 
    threshold=0.1
):
    """
    分析為什麼某些預測分數較高的錨點沒有被分配真實框
    
    Args:
        pred_scores: 預測分數 (b, n_anchors, num_classes)
        pred_bboxes: 預測框 (b, n_anchors, 4)
        anchor_points: 錨點坐標 (n_anchors, 2)
        gt_labels: 真實標籤 (b, n_max_boxes, 1)
        gt_bboxes: 真實框 (b, n_max_boxes, 4)
        mask_gt: 真實框有效掩碼 (b, n_max_boxes, 1)
        fg_mask: 前景掩碼 (b, n_anchors)
        mask_in_gts: 是否在真實框內的掩碼 (b, n_max_boxes, n_anchors)
        align_metric: 對齊度量 (b, n_max_boxes, n_anchors)
        overlaps: IoU 重疊度 (b, n_max_boxes, n_anchors)
        threshold: 記錄高於此分數但未被分配的預測閾值
    """
    # 創建日誌目錄
    log_dir = "tal_assignment_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 打開日誌文件
    log_file = os.path.join(log_dir, "assignment_analysis.txt")
    with open(log_file, "w") as f:
        f.write("=== 分析正樣本分配過程 ===\n\n")
        
        # 轉為 CPU 和 numpy 進行分析
        normalized_pred_scores = pred_scores.detach().sigmoid().cpu().numpy()
        fg_mask_np = fg_mask.cpu().numpy()
        
        batch_size = pred_scores.shape[0]
        n_anchors = pred_scores.shape[1]
        
        # 對每個批次進行分析
        for b in range(batch_size):
            f.write(f"\n批次 {b+1}:\n")
            f.write(f"  - 真實框數量: {mask_gt[b].sum().item()}\n")
            f.write(f"  - 被分配為正樣本的錨點數: {fg_mask_np[b].sum()}\n")
            
            # 找出分數高但未被分配的預測
            high_score_mask = normalized_pred_scores[b, :, 0] > threshold
            not_assigned_mask = ~fg_mask_np[b]
            high_score_but_not_assigned = high_score_mask & not_assigned_mask
            
            count_high_not_assigned = high_score_but_not_assigned.sum()
            f.write(f"  - 分數>{threshold}但未被分配的預測數: {count_high_not_assigned}\n")
            
            if count_high_not_assigned > 0:
                # 按分數從高到低排序這些未分配的高分預測
                high_scores_not_assigned = normalized_pred_scores[b, high_score_but_not_assigned, 0]
                high_indices_not_assigned = np.where(high_score_but_not_assigned)[0]
                
                sort_idx = np.argsort(-high_scores_not_assigned)  # 降序排列
                sorted_scores = high_scores_not_assigned[sort_idx]
                sorted_indices = high_indices_not_assigned[sort_idx]
                
                # 記錄前20個高分但未分配的預測
                n_to_log = min(20, len(sorted_scores))
                f.write("\n  分數最高的未分配預測:\n")
                
                for i in range(n_to_log):
                    anchor_idx = sorted_indices[i]
                    score = sorted_scores[i]
                    f.write(f"    {i+1}. 索引 {anchor_idx}, 分數: {score:.4f}\n")
                    
                    # 如果有mask_in_gts，分析在真實框內的情況
                    if mask_in_gts is not None:
                        in_any_gt = False
                        for j in range(mask_in_gts.shape[1]):  # 對每個真實框
                            if mask_gt[b, j, 0] and mask_in_gts[b, j, anchor_idx]:
                                in_any_gt = True
                                f.write(f"       - 在真實框 {j} 內\n")
                        
                        if not in_any_gt:
                            f.write(f"       - 不在任何真實框內 (主要原因)\n")
                    
                    # 如果有對齊度量和重疊度，進一步分析
                    if align_metric is not None and overlaps is not None:
                        max_align = 0
                        max_overlap = 0
                        max_align_gt = -1
                        max_overlap_gt = -1
                        
                        for j in range(align_metric.shape[1]):  # 對每個真實框
                            if mask_gt[b, j, 0]:
                                align_val = align_metric[b, j, anchor_idx].item()
                                overlap_val = overlaps[b, j, anchor_idx].item()
                                
                                if align_val > max_align:
                                    max_align = align_val
                                    max_align_gt = j
                                
                                if overlap_val > max_overlap:
                                    max_overlap = overlap_val
                                    max_overlap_gt = j
                        
                        f.write(f"       - 最高對齊度量: {max_align:.6f} (真實框 {max_align_gt})\n")
                        f.write(f"       - 最高重疊度(IoU): {max_overlap:.6f} (真實框 {max_overlap_gt})\n")
                        
                        # 分析未達到TopK的情況
                        if max_align_gt >= 0:
                            # 找出該真實框中對齊度量最高的前K個值
                            topk_values, _ = torch.topk(align_metric[b, max_align_gt], min(10, align_metric.shape[2]))
                            min_topk_value = topk_values[-1].item()
                            
                            if max_align < min_topk_value:
                                f.write(f"       - 對齊度量未達到TopK (最小TopK值: {min_topk_value:.6f})\n")
                            else:
                                f.write(f"       - 對齊度量達到了TopK，但可能在重疊度或其他步驟被篩選\n")
            
            # 額外分析：找出前景掩碼（被分配）中分數最低的錨點
            if fg_mask_np[b].any():
                assigned_scores = normalized_pred_scores[b, fg_mask_np[b], 0]
                assigned_indices = np.where(fg_mask_np[b])[0]
                
                min_score_idx = np.argmin(assigned_scores)
                min_score = assigned_scores[min_score_idx]
                min_score_anchor_idx = assigned_indices[min_score_idx]
                
                f.write(f"\n  被分配的最低分數: {min_score:.6f} (索引 {min_score_anchor_idx})\n")
        
        f.write("\n=== 分析完成 ===\n")
    
    print(f"分析結果已保存至 {log_file}")
