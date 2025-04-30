from datetime import datetime
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from ultralytics.utils.dev import describe_var
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.loss import KeypointLoss

class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)
        self.model = model

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it for pose estimation."""
        loss = torch.zeros(6, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        if "teacher" in batch and batch["teacher"] is not None:
            t_preds = batch["teacher_preds"]
            t_feats, t_pred_kpts = t_preds if isinstance(t_preds[0], list) else t_preds[1]
            t_pred_distri, t_pred_scores = torch.cat([xi.view(t_feats[0].shape[0], self.no, -1) for xi in t_feats], 2).split(
                (self.reg_max * 4, self.nc), 1
            )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        if "teacher" in batch and batch["teacher"] is not None:
            t_pred_scores = t_pred_scores.permute(0, 2, 1).contiguous()
            t_pred_distri = t_pred_distri.permute(0, 2, 1).contiguous()
            t_pred_kpts = t_pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        if "teacher" in batch and batch["teacher"] is not None:
            t_anchor_points, t_stride_tensor = make_anchors(t_feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        if "teacher" in batch and batch["teacher"] is not None:
            t_pred_bboxes = self.bbox_decode(t_anchor_points, t_pred_distri)  # xyxy, (b, h*w, 4)
            t_pred_kpts = self.kpts_decode(t_anchor_points, t_pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        pred_scores_normalized = pred_scores.detach().sigmoid()

        if "teacher" in batch and batch["teacher"] is not None:
            t_pred_scores_normalized = t_pred_scores.detach().sigmoid()

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores_normalized,
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            
            # 有教師模型時用蒸餾替代關鍵點損失
            if "teacher" in batch and batch["teacher"] is not None:
                # 直接用蒸餾損失取代原有的關鍵點損失
                loss[1], loss[2] = self.pose_loss(
                    batch=batch,
                    s_preds=preds,
                    s_fg_mask=fg_mask,
                    s_pred_scores_normalized=pred_scores_normalized,
                    s_pred_kpts=pred_kpts,
                    s_pred_bboxes_real=(pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                    t_preds=t_preds,
                    t_pred_kpts=t_pred_kpts,
                    t_pred_scores_normalized=t_pred_scores_normalized,
                    t_pred_bboxes_real=(t_pred_bboxes.detach() * t_stride_tensor).type(gt_bboxes.dtype),
                    t_pred_bboxes=t_pred_bboxes,
                    teacher=batch["teacher"],
                )
                # 不再使用loss[5]，因為已經合併到loss[1]和loss[2]中
                loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # 沒有教師模型時使用原來的關鍵點損失
                keypoints = batch["keypoints"].to(self.device).float().clone()
                keypoints[..., 0] *= imgsz[1]
                keypoints[..., 1] *= imgsz[0]
                loss[1], loss[2] = self.calculate_keypoints_loss(
                    fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
                )
                loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss[1] = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss[2] = torch.tensor(0.0, device=self.device, requires_grad=True)
            loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        # 不需要應用distill gain，因為已經合併到pose和kobj的損失中
        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss

    def calculate_keypoints_distillation_loss(
        self, valid_s_keypoints, valid_t_keypoints, valid_t_bboxes
    ):
        kpts_loss = 0
        kpts_obj_loss = 0

        area = xyxy2xywh(valid_t_bboxes)[:, 2:].prod(1, keepdim=True)
        
        # 檢查是否有keypoints
        if valid_s_keypoints.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True), torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 原始的二分法mask
        binary_kpt_mask = valid_t_keypoints[..., 2] != 0 if valid_t_keypoints.shape[-1] == 3 else torch.full_like(valid_t_keypoints[..., 0], True)
        
        if valid_t_keypoints.shape[-1] == 3 and valid_s_keypoints.shape[-1] == 3:
            # 使用教師模型的confidence作為權重
            t_conf = valid_t_keypoints[..., 2].sigmoid()  # 確保在0-1範圍
            
            # 計算基於距離的損失
            d = (valid_s_keypoints[..., 0] - valid_t_keypoints[..., 0]).pow(2) + (valid_s_keypoints[..., 1] - valid_t_keypoints[..., 1]).pow(2)
            kpt_loss_factor = binary_kpt_mask.shape[1] / (torch.sum(binary_kpt_mask != 0, dim=1) + 1e-9)
            # 從keypoint_loss中獲取sigmas
            sigmas = self.keypoint_loss.sigmas
            e = d / ((2 * sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
            
            # 將教師confidence作為權重應用於損失計算
            weighted_loss = (1 - torch.exp(-e)) * binary_kpt_mask * t_conf
            kpts_loss = (kpt_loss_factor.view(-1, 1) * weighted_loss).mean()
            
            # confidence loss，使用教師模型的confidence作為目標
            # 只針對有效keypoints（教師confidence > 0）計算
            conf_mask = t_conf > 0
            if conf_mask.sum() > 0:
                kpts_obj_loss = F.binary_cross_entropy_with_logits(
                    valid_s_keypoints[..., 2][conf_mask], 
                    t_conf[conf_mask], 
                    reduction='mean'
                )
            else:
                kpts_obj_loss = torch.tensor(0.0, device=self.device)
        else:
            # 如果沒有confidence維度，退回到原始實現
            kpts_loss = self.keypoint_loss(valid_s_keypoints, valid_t_keypoints, binary_kpt_mask, area)
            kpts_obj_loss = torch.tensor(0.0, device=self.device)

        return kpts_loss, kpts_obj_loss

    def pose_loss(
        self,
        batch,
        s_preds,
        s_fg_mask,
        s_pred_scores_normalized,
        s_pred_kpts,
        s_pred_bboxes_real,
        t_preds,
        t_pred_kpts,
        t_pred_scores_normalized,
        t_pred_bboxes_real,
        t_pred_bboxes,
        teacher,
    ):
        """Calculate the distillation between student and teacher, and return pose and kobj losses separately."""
        # 初始化零損失
        pose_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        kobj_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 只處理批次的第一張圖像，提高速度
        batch_idx = 0
        
        # 設置安全閾值
        score_threshold = 0.3
        iou_threshold = 0.3
        max_loss_value = 10.0
        
        try:
            # 獲取當前批次的預測
            s_scores = s_pred_scores_normalized[batch_idx]
            t_scores = t_pred_scores_normalized[batch_idx]
            
            # 安全檢查 - 確保有效的尺寸和數值
            if (s_scores.shape[0] == 0 or t_scores.shape[0] == 0 or
                torch.isnan(s_scores).any() or torch.isnan(t_scores).any()):
                return pose_loss, kobj_loss
            
            # 找出高置信度預測
            s_high_conf_mask = s_scores.max(1)[0] > score_threshold
            t_high_conf_mask = t_scores.max(1)[0] > score_threshold
            
            if not s_high_conf_mask.any() or not t_high_conf_mask.any():
                return pose_loss, kobj_loss
                
            # 獲取高置信度預測的索引
            s_indices = torch.nonzero(s_high_conf_mask).squeeze(1)
            t_indices = torch.nonzero(t_high_conf_mask).squeeze(1)
            
            # 限制數量以提高效率
            max_indices = 30
            if len(s_indices) > max_indices:
                s_indices = s_indices[:max_indices]
            if len(t_indices) > max_indices:
                t_indices = t_indices[:max_indices]
            
            # 逐個尋找匹配 (避免大矩陣計算)
            matched_pairs = []
            for i, s_idx in enumerate(s_indices):
                s_box = s_pred_bboxes_real[batch_idx, s_idx]
                best_iou = iou_threshold
                best_t_idx = -1
                
                for j, t_idx in enumerate(t_indices):
                    t_box = t_pred_bboxes_real[batch_idx, t_idx]
                    
                    # 計算IoU
                    x1 = max(s_box[0], t_box[0])
                    y1 = max(s_box[1], t_box[1])
                    x2 = min(s_box[2], t_box[2])
                    y2 = min(s_box[3], t_box[3])
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    inter_area = (x2 - x1) * (y2 - y1)
                    s_area = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
                    t_area = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])
                    union_area = s_area + t_area - inter_area
                    iou = inter_area / (union_area + 1e-6)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_t_idx = t_idx
                
                if best_t_idx >= 0:
                    matched_pairs.append((s_idx.item(), best_t_idx.item()))
            
            # 沒有匹配，返回零損失
            if not matched_pairs:
                return pose_loss, kobj_loss
            
            # 限制最大匹配數量
            if len(matched_pairs) > 10:
                # 隨機選擇10個匹配
                indices = torch.randperm(len(matched_pairs))[:10]
                matched_pairs = [matched_pairs[i] for i in indices]
            
            # 計算關鍵點損失
            batch_pose_loss = 0.0
            batch_kobj_loss = 0.0
            valid_pairs = 0
            
            for s_idx, t_idx in matched_pairs:
                # 獲取關鍵點
                s_kpt = s_pred_kpts[batch_idx, s_idx]    # [17, 3]
                t_kpt = t_pred_kpts[batch_idx, t_idx]    # [17, 3]
                
                # 檢查無效值
                if (torch.isnan(s_kpt).any() or torch.isnan(t_kpt).any() or
                    torch.isinf(s_kpt).any() or torch.isinf(t_kpt).any()):
                    continue
                
                # 關鍵點位置損失 (MSE)
                valid_mask = t_kpt[..., 2] > 0.0   # [17]
                if valid_mask.sum() > 0:
                    # 只計算有效關鍵點的位置損失
                    s_pos = s_kpt[valid_mask, :2]   # [num_valid, 2]
                    t_pos = t_kpt[valid_mask, :2]   # [num_valid, 2]
                    
                    # 安全檢查
                    if s_pos.shape[0] == 0 or t_pos.shape[0] == 0:
                        continue
                    
                    # 計算位置MSE損失
                    pos_loss = F.mse_loss(s_pos, t_pos, reduction='mean')
                    
                    # 限制損失上限
                    pos_loss = torch.clamp(pos_loss, 0.0, max_loss_value)
                    batch_pose_loss += pos_loss
                    
                    # 計算置信度BCE損失 (只考慮有效關鍵點)
                    s_conf = s_kpt[valid_mask, 2]   # [num_valid]
                    t_conf = t_kpt[valid_mask, 2]   # [num_valid]
                    
                    # 限制置信度範圍 (防止極端值)
                    t_conf = torch.sigmoid(t_conf)  # 確保在0-1範圍內
                    
                    # 計算BCE損失
                    conf_loss = F.binary_cross_entropy_with_logits(
                        s_conf, t_conf, reduction='mean'
                    )
                    
                    # 限制損失上限
                    conf_loss = torch.clamp(conf_loss, 0.0, max_loss_value)
                    batch_kobj_loss += conf_loss
                    
                    valid_pairs += 1
            
            # 如果有有效匹配，計算平均損失
            if valid_pairs > 0:
                pose_loss = batch_pose_loss / valid_pairs
                kobj_loss = batch_kobj_loss / valid_pairs
            
            # 應用超參數權重
            pose_loss = pose_loss * self.hyp.pose
            kobj_loss = kobj_loss * self.hyp.kobj
            
            # 最後安全檢查
            if torch.isnan(pose_loss) or torch.isinf(pose_loss):
                pose_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            if torch.isnan(kobj_loss) or torch.isinf(kobj_loss):
                kobj_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 確保損失為正
            pose_loss = torch.abs(pose_loss)
            kobj_loss = torch.abs(kobj_loss)
            
            # 最終限制損失大小
            pose_loss = torch.clamp(pose_loss, 0.0, 100.0)
            kobj_loss = torch.clamp(kobj_loss, 0.0, 100.0)
            
        except Exception as e:
            print(f"Error in pose_loss: {e}")
            pose_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            kobj_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return pose_loss, kobj_loss

    def get_box_overlap(self, box1, box2):
        """Calculate IoU between two bounding boxes [x1, y1, x2, y2]"""
        # Calculate intersection area
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area