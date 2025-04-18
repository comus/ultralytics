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
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        if "teacher" in batch and batch["teacher"] is not None:
            dpose, dkobj = self.pose_loss(
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

            loss[5] = dpose * 10.0 + dkobj * 2.0
        else:
            loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)
            
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.distill  # distill gain
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

    def pose_loss(self, batch, s_preds, t_preds, t_pred_kpts, s_fg_mask, s_pred_scores_normalized, s_pred_kpts, teacher, t_pred_scores_normalized, s_pred_bboxes_real, t_pred_bboxes_real, t_pred_bboxes):
        current_epoch = getattr(self.model, 'epoch', 0) if hasattr(self, 'model') else 0
        total_epochs = getattr(self.model, 'epochs', 100) if hasattr(self, 'model') else 100
        is_first_batch_in_epoch = getattr(self.model, 'is_first_batch_in_epoch', False) if hasattr(self, 'model') else False
        
        # 初始化默認零損失，確保即使沒有有效匹配也能返回有效值
        zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # 學生高分預測 mask
        # torch.Tensor(shape=[1, 8400, 1], dtype=torch.bool): tensor([[[False],
        s_high_confidence_mask = s_pred_scores_normalized > 0.45
        # torch.Tensor(shape=[1, 8400], dtype=torch.bool): tensor([[False, False, False,  ..., False, False, False]])
        s_high_confidence_mask = s_high_confidence_mask.squeeze(-1)

        
        # 學生高分預測但未被分配為正樣本的 mask
        s_high_conf_not_assigned_mask = s_high_confidence_mask & (~s_fg_mask)

        # 獲取 confidence_mask
        # torch.Tensor(shape=[2, 8400], dtype=torch.bool): tensor([[False, False, False,  ..., False, False, False],
        t_confidence_mask = t_pred_scores_normalized.amax(2) > 0.7
        
        s_batch_anchor_indices = torch.nonzero(s_high_conf_not_assigned_mask)
        t_batch_anchor_indices = torch.nonzero(t_confidence_mask)
        
        # 檢查是否有學生/教師高置信度但未被分配的錨點
        if s_batch_anchor_indices.numel() == 0 or t_batch_anchor_indices.numel() == 0:
            return zero_loss, zero_loss

        s_batch_indices = s_batch_anchor_indices[:, 0]  # 第一列是批次索引
        s_anchor_indices = s_batch_anchor_indices[:, 1]  # 第二列是錨點索引
        t_batch_indices = t_batch_anchor_indices[:, 0]  # 第一列是批次索引
        t_anchor_indices = t_batch_anchor_indices[:, 1]  # 第二列是錨點索引

        s_centers = batch["t_coords_tensor"][s_anchor_indices]
        t_centers = batch["t_coords_tensor"][t_anchor_indices]

        same_batch_mask = s_batch_indices.view(-1, 1) == t_batch_indices.view(1, -1)

        diffs = s_centers.unsqueeze(1) - t_centers.unsqueeze(0)  # [total_s, total_t, 2]
        squared_diffs = torch.sum(diffs**2, dim=2)  # [total_s, total_t]

        INF = 1e10
        squared_diffs = torch.where(same_batch_mask, squared_diffs, torch.tensor(INF, device=self.device))
        distances = torch.sqrt(squared_diffs)  # [total_s, total_t]

        s_strides = self.model.stride
        s_levels = batch["s_levels_tensor"][s_anchor_indices]  # [total_s]
        t_levels = batch["t_levels_tensor"][t_anchor_indices]  # [total_t]
        s_stride_values = s_strides.to(s_levels.device)[s_levels]
        threshold = 1.5
        dist_thresholds = (s_stride_values * threshold).unsqueeze(1)

        valid_matches = distances < dist_thresholds

        min_distances, min_indices = torch.min(torch.where(valid_matches, distances, 
                                                       torch.tensor(INF, device=self.device)), dim=1)
        
        valid_mask = min_distances < INF
        
        # 檢查是否有有效匹配
        if valid_mask.sum() == 0:
            return zero_loss, zero_loss

        # 記錄原始匹配數量
        original_match_count = valid_mask.sum().item()

        # 限制匹配點數量上限為 100 個
        if valid_mask.sum() > 100:
            # 獲取有效匹配的索引，並隨機選擇 100 個
            valid_indices = torch.nonzero(valid_mask).squeeze(1)
            perm = torch.randperm(len(valid_indices), device=valid_indices.device)[:100]
            selected_indices = valid_indices[perm]
            
            # 創建新的 mask
            new_valid_mask = torch.zeros_like(valid_mask)
            new_valid_mask[selected_indices] = True
            valid_mask = new_valid_mask

        valid_s_positions = torch.nonzero(valid_mask).squeeze(1)  # [n_valid]
        valid_t_positions = min_indices[valid_mask]  # [n_valid]
        
        # 獲取有效批次索引
        # torch.Tensor(shape=[12], dtype=torch.int64): tensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        valid_batch_indices = s_batch_indices[valid_s_positions]  # [n_valid]
        
        # 獲取有效的學生和教師原始索引
        # torch.Tensor(shape=[12], dtype=torch.int64): tensor([1252, 1412, 6650, 6707, 8123, 8141, 1252, 1412, 6650, 6707, 8123, 8141])
        valid_s_indices = s_anchor_indices[valid_s_positions]  # [n_valid]
        # torch.Tensor(shape=[12], dtype=torch.int64): tensor([1252, 6746, 6690, 1253, 6884, 6883, 1252, 6746, 6690, 1253, 6884, 6883])
        valid_t_indices = t_anchor_indices[valid_t_positions]  # [n_valid]

        # Get the keypoints using advanced indexing
        valid_s_keypoints = s_pred_kpts[valid_batch_indices, valid_s_indices]
        valid_t_keypoints = t_pred_kpts[valid_batch_indices, valid_t_indices]

        valid_t_bboxes = t_pred_bboxes[valid_batch_indices, valid_t_indices]

        # 獲取有效的中心坐標
        valid_s_centers = s_centers[valid_s_positions]  # [n_valid, 2]
        valid_t_centers = t_centers[valid_t_positions]  # [n_valid, 2]
        
        # 獲取有效的層級
        valid_s_levels = s_levels[valid_s_positions]  # [n_valid]
        valid_t_levels = t_levels[valid_t_positions]  # [n_valid]
        
        # 獲取有效的距離
        valid_distances = min_distances[valid_mask]  # [n_valid]
        
        if current_epoch % 1 == 0 and is_first_batch_in_epoch:
            print("\n"*2)
            print("="*50)
            print(f"Epoch {current_epoch + 1}/{total_epochs}")
            print("-"*50)

            # 構建基本匹配信息
            matches = torch.stack([
                valid_batch_indices.float(),     # 批次索引
                valid_s_indices.float(),         # 學生原始索引
                valid_t_indices.float(),         # 教師原始索引
                valid_s_centers[:, 0],           # 學生中心 x
                valid_s_centers[:, 1],           # 學生中心 y
                valid_t_centers[:, 0],           # 教師中心 x
                valid_t_centers[:, 1],           # 教師中心 y
                valid_s_levels.float(),          # 學生層級
                valid_t_levels.float(),          # 教師層級
                valid_distances                  # 距離
            ], dim=1)
            
            if matches.numel() > 0:
                n_matches = matches.size(0)
                print(f"找到 {n_matches} 個學生-教師錨點匹配")
                
                # 如果原始匹配數超過 100，顯示已限制的信息
                if original_match_count > 100:
                    print(f"已限制匹配數量為最大 100 個 (原始匹配數: {original_match_count})")
                
                # 只顯示前5個匹配的詳細信息
                num_to_show = min(4, n_matches)
                for i in range(num_to_show):
                    match = matches[i]
                    batch_idx = int(match[0].item())
                    s_idx = int(match[1].item())
                    t_idx = int(match[2].item())
                    s_center_x, s_center_y = match[3].item(), match[4].item()
                    t_center_x, t_center_y = match[5].item(), match[6].item()
                    s_level = int(match[7].item())
                    t_level = int(match[8].item())
                    distance = match[9].item()
                    
                    # Get confidence values
                    s_conf_value = s_pred_scores_normalized[batch_idx, s_idx].item()
                    t_conf_value = t_pred_scores_normalized[batch_idx, t_idx].item()
                    
                    # Get bounding box information
                    s_pred_bbox = s_pred_bboxes_real[batch_idx, s_idx].detach()
                    s_bbox_width = s_pred_bbox[2] - s_pred_bbox[0]
                    s_bbox_height = s_pred_bbox[3] - s_pred_bbox[1]
                    s_bbox_center_x = (s_pred_bbox[0] + s_pred_bbox[2]) / 2
                    s_bbox_center_y = (s_pred_bbox[1] + s_pred_bbox[3]) / 2
                    
                    t_pred_bbox = t_pred_bboxes_real[batch_idx, t_idx].detach()
                    t_bbox_width = t_pred_bbox[2] - t_pred_bbox[0]
                    t_bbox_height = t_pred_bbox[3] - t_pred_bbox[1]
                    t_bbox_center_x = (t_pred_bbox[0] + t_pred_bbox[2]) / 2
                    t_bbox_center_y = (t_pred_bbox[1] + t_pred_bbox[3]) / 2
                    
                    # Calculate level coordinates (grid coordinates)
                    s_stride = self.model.stride.to(s_levels.device)[s_level]
                    t_stride = self.model.stride.to(t_levels.device)[t_level]  # Assuming teacher uses same strides
                    
                    s_level_x, s_level_y = s_center_x / s_stride, s_center_y / s_stride
                    t_level_x, t_level_y = t_center_x / t_stride, t_center_y / t_stride
                    
                    print(f"匹配 #{i+1}:")
                    print(f"  批次: {batch_idx}, 距離: {distance:.2f} 像素")
                    print(f"  學生錨點 - 索引: {s_idx}, 層級: {s_level}, 置信度: {s_conf_value:.4f}, 座標: ({s_center_x:.1f},{s_center_y:.1f}), 層級座標: ({s_level_x:.1f},{s_level_y:.1f})")
                    print(f"    bbox: 寬x高: {s_bbox_width:.1f}x{s_bbox_height:.1f}, 中心點: ({s_bbox_center_x:.1f},{s_bbox_center_y:.1f})")
                    print(f"  教師錨點 - 索引: {t_idx}, 層級: {t_level}, 置信度: {t_conf_value:.4f}, 座標: ({t_center_x:.1f},{t_center_y:.1f}), 層級座標: ({t_level_x:.1f},{t_level_y:.1f})")
                    print(f"    bbox: 寬x高: {t_bbox_width:.1f}x{t_bbox_height:.1f}, 中心點: ({t_bbox_center_x:.1f},{t_bbox_center_y:.1f})")
            else:
                print("未找到任何學生-教師錨點匹配")

            print("-"*50)
            print("\n"*2)
        
        return self.calculate_keypoints_distillation_loss(valid_s_keypoints, valid_t_keypoints, valid_t_bboxes)