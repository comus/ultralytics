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

def update_temperature(current_epoch, total_epochs, initial_T=4.0, min_T=2.0):
    """根據當前epoch調整溫度 - 使用非線性降溫曲線，考慮epoch從0開始"""
    # 確保在最後一個epoch時達到最小溫度
    if total_epochs <= 1:
        return min_T  # 防止只有一個epoch的情況
    
    # 正規化epoch進度(0到1)，考慮epoch從0開始
    progress = current_epoch / (total_epochs - 1) if total_epochs > 1 else 1.0
    
    # 使用調整後的進度曲線 - S型曲線使降溫在中期更快
    if progress < 0.3:
        # 前30%保持較高溫度
        adjusted_progress = progress * 0.2  # 緩慢降溫
    elif progress < 0.7:
        # 中間40%快速降溫
        adjusted_progress = 0.2 + (progress - 0.3) * 1.5
    else:
        # 後30%緩慢降至最低溫度
        adjusted_progress = 0.8 + (progress - 0.7) * 0.67
    
    # 使用指數衰減計算溫度
    T = initial_T * (min_T / initial_T) ** adjusted_progress
    
    return T

def calculate_progress(current_epoch, total_epochs):
    """計算訓練進度
    
    Args:
        current_epoch: 當前epoch索引(從0開始)
        total_epochs: 總epoch數
    
    Returns:
        float: 0到1之間的進度值
    """
    if total_epochs <= 1:
        return 1.0  # 防止只有一個epoch的情況
    
    # 如果總epochs很少(例如<=5)，進行特殊處理
    if total_epochs <= 5:
        # 在少量epochs情況下，使用非線性進度
        # 這讓進度在早期增長較慢，確保初期策略有足夠時間
        normalized_epoch = current_epoch / (total_epochs - 1)
        return normalized_epoch ** 0.7  # 使用冪函數使進度曲線更平滑
    else:
        # 一般情況，標準計算方式
        return min(1.0, current_epoch / (total_epochs * 0.8))

def analyze_pose_alignment(student_outputs, teacher_outputs, conf_threshold=0.5):
    """
    簡化版的姿態預測對齊分析
    """
    # 提取預測張量
    _, student_preds = student_outputs
    _, teacher_preds = teacher_outputs
    
    # 輸出形狀信息
    print(f"學生預測形狀: {student_preds.shape}")
    print(f"教師預測形狀: {teacher_preds.shape}")
    
    # 獲取參數
    batch_size, num_channels, num_points = student_preds.shape
    
    # 假設前2個通道是中心點坐標，之後每3個通道是一個關鍵點(x,y,conf)
    num_keypoints = (num_channels - 2) // 3
    print(f"檢測到 {num_keypoints} 個關鍵點")
    
    # 存儲匹配的關鍵點信息
    matched_keypoints = []
    
    # 遍歷每個批次
    for b in range(batch_size):
        # 獲取單個批次的預測
        s_pred = student_preds[b]  # [num_channels, num_points]
        t_pred = teacher_preds[b]  # [num_channels, num_points]
        
        # 計算所有關鍵點的平均置信度
        s_conf_channels = [2 + i*3 + 2 for i in range(num_keypoints)]
        t_conf_channels = s_conf_channels
        
        s_conf = torch.sigmoid(s_pred[s_conf_channels])  # [num_keypoints, num_points]
        t_conf = torch.sigmoid(t_pred[t_conf_channels])
        
        s_mean_conf = torch.mean(s_conf, dim=0)  # [num_points]
        t_mean_conf = torch.mean(t_conf, dim=0)
        
        # 找出高置信度的點
        s_high_conf = torch.nonzero(s_mean_conf > conf_threshold).squeeze(-1)
        t_high_conf = torch.nonzero(t_mean_conf > conf_threshold).squeeze(-1)
        
        # 確保是向量形式
        if s_high_conf.dim() == 0 and s_high_conf.numel() > 0:
            s_high_conf = s_high_conf.unsqueeze(0)
        if t_high_conf.dim() == 0 and t_high_conf.numel() > 0:
            t_high_conf = t_high_conf.unsqueeze(0)
        
        print(f"批次 {b}: 學生高置信度點 {s_high_conf.numel()}, 教師高置信度點 {t_high_conf.numel()}")
        
        # 如果兩者都有高置信度點，嘗試匹配
        if s_high_conf.numel() > 0 and t_high_conf.numel() > 0:
            # 提取中心點坐標
            s_centers = torch.stack([s_pred[0, s_high_conf], s_pred[1, s_high_conf]])  # [2, n_s]
            t_centers = torch.stack([t_pred[0, t_high_conf], t_pred[1, t_high_conf]])  # [2, n_t]
            
            # 為每個學生點找最近的教師點
            for i in range(s_high_conf.numel()):
                s_idx = s_high_conf[i].item()
                s_center = s_centers[:, i].unsqueeze(1)  # [2, 1]
                
                # 計算到所有教師點的距離
                distances = torch.sqrt(((s_center - t_centers)**2).sum(dim=0))  # [n_t]
                
                # 找最近的教師點
                min_dist, min_idx = torch.min(distances, dim=0)
                
                # 如果距離小於閾值(0.5)，認為匹配成功
                if min_dist < 0.5:
                    t_idx = t_high_conf[min_idx].item()
                    
                    # 收集每個關鍵點的信息
                    for kp in range(num_keypoints):
                        kp_base = 2 + kp * 3
                        
                        s_kp_x = s_pred[kp_base, s_idx].item()
                        s_kp_y = s_pred[kp_base + 1, s_idx].item()
                        s_kp_conf = s_conf[kp, s_idx].item()
                        
                        t_kp_x = t_pred[kp_base, t_idx].item()
                        t_kp_y = t_pred[kp_base + 1, t_idx].item()
                        t_kp_conf = t_conf[kp, t_idx].item()
                        
                        # 只分析雙方都高置信度的關鍵點
                        if s_kp_conf > conf_threshold and t_kp_conf > conf_threshold:
                            matched_keypoints.append({
                                'keypoint': kp,
                                's_coords': (s_kp_x, s_kp_y),
                                't_coords': (t_kp_x, t_kp_y)
                            })
    
    # 如果沒有匹配點，提前返回
    if not matched_keypoints:
        print("未找到匹配的高置信度關鍵點")
        return None
    
    # 分析坐標分布
    s_x_vals = [kp['s_coords'][0] for kp in matched_keypoints]
    s_y_vals = [kp['s_coords'][1] for kp in matched_keypoints]
    t_x_vals = [kp['t_coords'][0] for kp in matched_keypoints]
    t_y_vals = [kp['t_coords'][1] for kp in matched_keypoints]
    
    # 計算坐標範圍
    s_x_min, s_x_max = min(s_x_vals), max(s_x_vals)
    s_y_min, s_y_max = min(s_y_vals), max(s_y_vals)
    t_x_min, t_x_max = min(t_x_vals), max(t_x_vals)
    t_y_min, t_y_max = min(t_y_vals), max(t_y_vals)
    
    # 計算縮放因子
    s_x_range = s_x_max - s_x_min
    s_y_range = s_y_max - s_y_min
    t_x_range = t_x_max - t_x_min
    t_y_range = t_y_max - t_y_min
    
    x_scale = t_x_range / s_x_range if s_x_range > 0 else 1.0
    y_scale = t_y_range / s_y_range if s_y_range > 0 else 1.0
    
    print("\n===== 坐標範圍分析 =====")
    print(f"學生X範圍: [{s_x_min:.4f}, {s_x_max:.4f}], 跨度: {s_x_range:.4f}")
    print(f"學生Y範圍: [{s_y_min:.4f}, {s_y_max:.4f}], 跨度: {s_y_range:.4f}")
    print(f"教師X範圍: [{t_x_min:.4f}, {t_x_max:.4f}], 跨度: {t_x_range:.4f}")
    print(f"教師Y範圍: [{t_y_min:.4f}, {t_y_max:.4f}], 跨度: {t_y_range:.4f}")
    print(f"建議縮放因子: X={x_scale:.4f}, Y={y_scale:.4f}")
    
    # 計算原始誤差
    distances = []
    for kp in matched_keypoints:
        s_x, s_y = kp['s_coords']
        t_x, t_y = kp['t_coords']
        dist = ((s_x - t_x)**2 + (s_y - t_y)**2)**0.5
        distances.append(dist)
    
    avg_dist = sum(distances) / len(distances)
    map50 = sum(1 for d in distances if d < 0.05) / len(distances)
    
    print("\n===== 原始誤差 =====")
    print(f"平均距離: {avg_dist:.4f}")
    print(f"mAP50: {map50:.4f}")
    
    # 應用縮放後的誤差
    scaled_distances = []
    for kp in matched_keypoints:
        s_x, s_y = kp['s_coords']
        t_x, t_y = kp['t_coords']
        
        # 應用縮放
        s_x_scaled = s_x * x_scale
        s_y_scaled = s_y * y_scale
        
        dist = ((s_x_scaled - t_x)**2 + (s_y_scaled - t_y)**2)**0.5
        scaled_distances.append(dist)
    
    avg_scaled_dist = sum(scaled_distances) / len(scaled_distances)
    scaled_map50 = sum(1 for d in scaled_distances if d < 0.05) / len(scaled_distances)
    
    print("\n===== 縮放後誤差 =====")
    print(f"平均距離: {avg_scaled_dist:.4f} (變化: {avg_scaled_dist - avg_dist:.4f})")
    print(f"mAP50: {scaled_map50:.4f} (變化: {scaled_map50 - map50:.4f})")
    
    return {
        'scale_x': x_scale,
        'scale_y': y_scale,
        'original_map50': map50,
        'scaled_map50': scaled_map50
    }
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
        # print("\n" * 10, "=" * 100, "\n")

        print("preds", describe_var(preds, max_depth=10, max_items=100))

        print("batch", batch["im_file"])
        print("batch", batch["img"])

        if "teacher" in batch and batch["teacher"] is not None:
            print("teacher_preds", describe_var(batch["teacher_preds"]))

        loss = torch.zeros(6, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        print("feats", describe_var(feats))
        print("pred_kpts", describe_var(pred_kpts))
        
        # NaN 檢查 - 確保輸入預測沒有NaN
        if torch.isnan(pred_kpts).any():
            print("警告: 偵測到預測關鍵點包含NaN值，將替換為零值")
            pred_kpts = torch.nan_to_num(pred_kpts, nan=0.0)
            
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        print("self.no", describe_var(self.no))
        print("self.reg_max", describe_var(self.reg_max))
        print("self.nc", describe_var(self.nc))
        
        # NaN 檢查 - 檢查分佈和分數
        if torch.isnan(pred_distri).any() or torch.isnan(pred_scores).any():
            print("警告: 偵測到預測分佈或分數包含NaN值，將替換為零值")
            pred_distri = torch.nan_to_num(pred_distri, nan=0.0)
            pred_scores = torch.nan_to_num(pred_scores, nan=0.0)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
        print("feats", describe_var(feats))
        print("stride", describe_var(self.stride))
        print("type(self.stride)", type(self.stride))
        print("self.stride", describe_var(self.stride))

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        print("targets", describe_var(targets))
        print("batch_idx", describe_var(batch_idx))
        print('batch["cls"].view(-1, 1)', describe_var(batch["cls"].view(-1, 1)))
        print('batch["bboxes"]', describe_var(batch["bboxes"]))
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        print("targets2", describe_var(targets))
        print("batch_size", batch_size)
        print("imgsz[[1, 0, 1, 0]]", imgsz[[1, 0, 1, 0]])

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # NaN 檢查 - 確保目標框沒有NaN
        if torch.isnan(gt_bboxes).any():
            print("警告: 偵測到目標框包含NaN值")
            return loss * batch_size, loss.detach()  # 直接返回零損失

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)
        print("anchor_points", describe_var(anchor_points))
        print("stride_tensor", describe_var(stride_tensor))
        print("pred_distri", describe_var(pred_distri))
        print("proj", describe_var(self.proj))
        print("pred_bboxes", describe_var(pred_bboxes))
        print("pred_kpts", describe_var(pred_kpts))
        
        # NaN 檢查 - 解碼後的預測框和關鍵點
        if torch.isnan(pred_bboxes).any():
            print("警告: 偵測到解碼後的預測框包含NaN值，將替換為零值")
            pred_bboxes = torch.nan_to_num(pred_bboxes, nan=0.0)
            
        if torch.isnan(pred_kpts).any():
            print("警告: 偵測到解碼後的預測關鍵點包含NaN值，將替換為零值")
            pred_kpts = torch.nan_to_num(pred_kpts, nan=0.0)

        print("pred_scores", describe_var(pred_scores))
        print("pred_bboxes", describe_var(pred_bboxes))
        print("gt_bboxes", describe_var(gt_bboxes))
        print("anchor_points", describe_var(anchor_points))
        print("stride_tensor", describe_var(stride_tensor))
        print("mask_gt", describe_var(mask_gt))
        print("gt_labels", describe_var(gt_labels), gt_labels)

        try:
            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
            print("pred_scores", describe_var(pred_scores))
            print("pred_scores.detach().sigmoid()", describe_var(pred_scores.detach().sigmoid()))
            
            exit()

            print("target_bboxes", describe_var(target_bboxes))
            print("target_scores", describe_var(target_scores))
            print("fg_mask", describe_var(fg_mask))
            print("target_gt_idx", describe_var(target_gt_idx))
        except Exception as e:
            print(f"警告: assigner出現異常: {str(e)}")
            return loss * batch_size, loss.detach()  # 直接返回零損失

        # 計算一下 fg_mask 的數量 (True 的數量)
        fg_mask_count = fg_mask.sum()
        print("fg_mask_count", fg_mask_count)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            try:
                target_bboxes /= stride_tensor
                
                # 額外檢查防止無限值
                if torch.isnan(target_bboxes).any() or torch.isinf(target_bboxes).any():
                    print("警告: 縮放後的目標框包含NaN或Inf")
                    target_bboxes = torch.nan_to_num(target_bboxes, nan=0.0, posinf=1.0, neginf=-1.0)
                
                loss[0], loss[4] = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
                )
                
                # 檢查box loss是否為NaN
                if torch.isnan(loss[0]) or torch.isinf(loss[0]):
                    print("警告: box_loss包含NaN或Inf，替換為0.0")
                    loss[0] = torch.tensor(0.0, device=self.device)
                
                if torch.isnan(loss[4]) or torch.isinf(loss[4]):
                    print("警告: dfl_loss包含NaN或Inf，替換為0.0")
                    loss[4] = torch.tensor(0.0, device=self.device)
                
                keypoints = batch["keypoints"].to(self.device).float().clone()
                keypoints[..., 0] *= imgsz[1]
                keypoints[..., 1] *= imgsz[0]

                # 確保關鍵點沒有NaN
                if torch.isnan(keypoints).any():
                    print("警告: 關鍵點包含NaN值，替換為0.0")
                    keypoints = torch.nan_to_num(keypoints, nan=0.0)

                loss[1], loss[2] = self.calculate_keypoints_loss(
                    fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
                )
                
                # 檢查關鍵點損失是否為NaN
                if torch.isnan(loss[1]) or torch.isinf(loss[1]):
                    print("警告: kpt_location_loss包含NaN或Inf，替換為0.0")
                    loss[1] = torch.tensor(0.0, device=self.device)
                
                if torch.isnan(loss[2]) or torch.isinf(loss[2]):
                    print("警告: kpt_visibility_loss包含NaN或Inf，替換為0.0")
                    loss[2] = torch.tensor(0.0, device=self.device)
                
            except Exception as e:
                print(f"警告: bbox/kpt損失計算出現異常: {str(e)}")
                import traceback
                traceback.print_exc()
                # 確保損失為零但可以繼續訓練
                loss[0] = torch.tensor(0.0, device=self.device)
                loss[1] = torch.tensor(0.0, device=self.device) 
                loss[2] = torch.tensor(0.0, device=self.device)
                loss[4] = torch.tensor(0.0, device=self.device)

        supervision_weight = 1.0
        distill_weight = 0.0
        
        if "teacher" in batch and batch["teacher"] is not None:
            analyze_pose_alignment(preds, batch["teacher_preds"])

            # 如果 self.model 有 trainer 屬性，則打印 epoch
            epoch = self.model.epoch if hasattr(self.model, 'epoch') else 1
            epochs = self.model.epochs if hasattr(self.model, 'epochs') else 1

            T = update_temperature(epoch, epochs)

            try:
                if "loss_function" in batch and batch["loss_function"] == "pose_loss2":
                    loss[5] = self.pose_distillation_loss_enhanced2(preds, batch["teacher_preds"], T)
                elif "loss_function" in batch and batch["loss_function"] == "pose_loss3":
                    loss[5] = self.grid_aligned_loss(preds, batch["teacher_preds"])
                else:
                    loss[5] = self.pose_distillation_loss_enhanced(preds, batch["teacher_preds"], T)
                
                # 檢查蒸餾損失是否為NaN
                if torch.isnan(loss[5]) or torch.isinf(loss[5]):
                    print("警告: distillation_loss包含NaN或Inf，替換為0.0")
                    loss[5] = torch.tensor(0.0, device=self.device)
                    
            except Exception as e:
                print(f"警告: 蒸餾損失計算出現異常: {str(e)}")
                loss[5] = torch.tensor(0.0, device=self.device, requires_grad=True)

            if "pure_distill" in batch and batch["pure_distill"]:
                supervision_weight = 0.0
                distill_weight = 1.0
            elif hasattr(self.model, 'epoch') and self.model.epoch < 5:  # 0, 1, 2, 3, 4
                supervision_weight = 0.8  # 監督為主
                distill_weight = 0.2      # 蒸餾為輔
            else:  # 5及以上
                supervision_weight = 0.6
                distill_weight = 0.4
        else:
            loss[5] = torch.zeros(1, device=self.device, requires_grad=True)

        loss[0] *= supervision_weight* self.hyp.box  # box gain
        loss[1] *= supervision_weight* self.hyp.pose  # pose gain
        loss[2] *= supervision_weight* self.hyp.kobj  # kobj gain
        loss[3] *= supervision_weight* self.hyp.cls  # cls gain
        loss[4] *= supervision_weight* self.hyp.dfl  # dfl gain
        loss[5] *= distill_weight* self.hyp.distill
        
        # 最終檢查 - 確保沒有NaN損失
        for i in range(len(loss)):
            if torch.isnan(loss[i]) or torch.isinf(loss[i]):
                print(f"警告: 最終損失[{i}]包含NaN或Inf，替換為0.0")
                loss[i] = torch.tensor(0.0, device=self.device)

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def analyze_keypoint_precision(self, student_outputs, teacher_outputs, conf_threshold=0.5):
        """
        分析高置信度點的精度分布
        """
        # 1. 提取預測張量
        _, student_preds = student_outputs
        _, teacher_preds = teacher_outputs
        
        # 2. 提取坐標和置信度
        s_x = student_preds[:, 0::3, :]  # 假設使用交錯格式 [x1,y1,c1,x2,y2,c2,...]
        s_y = student_preds[:, 1::3, :]
        s_conf = student_preds[:, 2::3, :]
        
        t_x = teacher_preds[:, 0::3, :]
        t_y = teacher_preds[:, 1::3, :]
        t_conf = teacher_preds[:, 2::3, :]
        
        # 3. 計算教師置信度和篩選高置信度點
        t_conf_prob = torch.sigmoid(t_conf)
        high_conf_mask = t_conf_prob > conf_threshold
        
        total_points = t_conf.numel()
        high_conf_points = high_conf_mask.sum().item()
        
        print(f"\n===== 置信度篩選 =====")
        print(f"置信度閾值: {conf_threshold}")
        print(f"總點數: {total_points}")
        print(f"高置信度點數: {high_conf_points}")
        print(f"高置信度點比例: {high_conf_points/total_points*100:.2f}%")
        
        # 如果高置信度點太少，降低閾值
        if high_conf_points < 100:
            new_threshold = conf_threshold * 0.5
            print(f"高置信度點太少，降低閾值至 {new_threshold}")
            return self.analyze_keypoint_precision(student_outputs, teacher_outputs, new_threshold)
        
        # 4. 計算坐標差異
        x_diff = torch.abs(s_x - t_x)
        y_diff = torch.abs(s_y - t_y)
        
        # 計算歐氏距離
        combined_diff = torch.sqrt(x_diff**2 + y_diff**2 + 1e-8)
        
        # 5. 計算不同精度閾值的掩碼
        map50_mask = (combined_diff < 0.05)  # mAP50 (誤差 < 5%)
        map75_mask = (combined_diff < 0.025)  # mAP75 (誤差 < 2.5%)
        map90_mask = (combined_diff < 0.01)   # mAP90 (誤差 < 1%)
        map95_mask = (combined_diff < 0.005)  # mAP95 (誤差 < 0.5%)
        
        # 創建mAP50-95區間掩碼 (0.5% < 誤差 < 5%)
        map50_95_mask = map50_mask & ~map95_mask
        
        # 6. 計算高置信度點中的精度分布
        # 只考慮高置信度點
        high_conf_map50 = (map50_mask & high_conf_mask).sum().item()
        high_conf_map50_95 = (map50_95_mask & high_conf_mask).sum().item()
        high_conf_map95 = (map95_mask & high_conf_mask).sum().item()
        
        # 7. 輸出精度分布
        print(f"\n===== 高置信度點精度分布 =====")
        print(f"mAP50 區間 (誤差 < 5%): {high_conf_map50} 點 ({high_conf_map50/high_conf_points*100:.2f}%)")
        print(f"mAP50-95 區間 (0.5% < 誤差 < 5%): {high_conf_map50_95} 點 ({high_conf_map50_95/high_conf_points*100:.2f}%)")
        print(f"mAP95 區間 (誤差 < 0.5%): {high_conf_map95} 點 ({high_conf_map95/high_conf_points*100:.2f}%)")
        
        # 8. 細分精度區間分析
        print(f"\n===== 詳細精度區間分布 =====")
        map05_mask = (combined_diff < 0.005)  # 誤差 < 0.5%
        map10_mask = (combined_diff < 0.01) & ~map05_mask   # 0.5% < 誤差 < 1%
        map25_mask = (combined_diff < 0.025) & ~map10_mask & ~map05_mask  # 1% < 誤差 < 2.5%
        map50_mask_detail = (combined_diff < 0.05) & ~map25_mask & ~map10_mask & ~map05_mask  # 2.5% < 誤差 < 5%
        
        high_conf_map05 = (map05_mask & high_conf_mask).sum().item()
        high_conf_map10 = (map10_mask & high_conf_mask).sum().item()
        high_conf_map25 = (map25_mask & high_conf_mask).sum().item()
        high_conf_map50_detail = (map50_mask_detail & high_conf_mask).sum().item()
        
        print(f"誤差 < 0.5% (最高精度): {high_conf_map05} 點 ({high_conf_map05/high_conf_points*100:.2f}%)")
        print(f"0.5% < 誤差 < 1%: {high_conf_map10} 點 ({high_conf_map10/high_conf_points*100:.2f}%)")
        print(f"1% < 誤差 < 2.5%: {high_conf_map25} 點 ({high_conf_map25/high_conf_points*100:.2f}%)")
        print(f"2.5% < 誤差 < 5%: {high_conf_map50_detail} 點 ({high_conf_map50_detail/high_conf_points*100:.2f}%)")
        print(f"誤差 > 5% (低精度): {high_conf_points - high_conf_map50} 點 ({(high_conf_points - high_conf_map50)/high_conf_points*100:.2f}%)")
        
        # 9. 百分位數分析
        percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        
        # 提取高置信度點的誤差
        high_conf_errors = torch.masked_select(combined_diff, high_conf_mask)
        
        print(f"\n===== 高置信度點誤差分布 =====")
        print("百分位數\t誤差")
        for p in percentiles:
            try:
                error_value = torch.quantile(high_conf_errors, p/100.0).item()
                print(f"{p}%\t{error_value:.6f}")
            except:
                print(f"{p}%\t計算失敗")
        
        # 10. 計算坐標相關性
        # 提取高置信度點的坐標
        s_x_high = torch.masked_select(s_x, high_conf_mask)
        s_y_high = torch.masked_select(s_y, high_conf_mask)
        t_x_high = torch.masked_select(t_x, high_conf_mask)
        t_y_high = torch.masked_select(t_y, high_conf_mask)
        
        # 嘗試計算相關係數
        try:
            x_corr = torch.corrcoef(torch.stack([s_x_high, t_x_high]))[0, 1].item()
            y_corr = torch.corrcoef(torch.stack([s_y_high, t_y_high]))[0, 1].item()
            
            print(f"\n===== 坐標相關性 =====")
            print(f"X坐標相關係數: {x_corr:.4f}")
            print(f"Y坐標相關係數: {y_corr:.4f}")
        except:
            print("\n===== 坐標相關性 =====")
            print("相關係數計算失敗")
        
        return {
            'high_conf_points': high_conf_points,
            'high_conf_map50': high_conf_map50,
            'high_conf_map50_95': high_conf_map50_95, 
            'high_conf_map95': high_conf_map95
        }

    def grid_aligned_loss(self, student_outputs, teacher_outputs):
        """
        針對網格級別對齊的損失函數
        """
        # 提取預測
        student_features, student_preds = student_outputs
        teacher_features, teacher_preds = teacher_outputs
        
        batch_size = student_preds.shape[0]
        
        # 關鍵點提取 - 使用交錯格式
        num_keypoints = 17
        
        # 提取坐標和置信度 - 交錯格式 [x1,y1,c1,x2,y2,c2,...]
        s_x = student_preds[:, 0::3, :]
        s_y = student_preds[:, 1::3, :]
        s_conf = student_preds[:, 2::3, :]
        
        t_x = teacher_preds[:, 0::3, :]
        t_y = teacher_preds[:, 1::3, :]
        t_conf = teacher_preds[:, 2::3, :]
        
        # 計算教師置信度
        t_conf_prob = torch.sigmoid(t_conf)
        valid_mask = t_conf_prob > 0.3  # 只考慮教師認為可能存在的關鍵點
        
        # ====== 關鍵改進: 網格級別對齊 ======
        # 針對每個網格位置，嘗試不同的對齊方式
        
        # 獲取網格總數
        grid_count = s_x.shape[2]  # 8400
        
        # 估計特徵圖尺寸
        # 假設網格順序是 [80x80, 40x40, 20x20] = 6400 + 1600 + 400 = 8400
        stride_indices = [0, 6400, 8000, 8400]  # 不同特徵層的起始索引
        stride_sizes = [8, 16, 32]  # 對應的步長大小
        
        # 創建網格索引和步長映射
        grid_strides = torch.zeros(grid_count, device=s_x.device)
        for i in range(len(stride_indices)-1):
            grid_strides[stride_indices[i]:stride_indices[i+1]] = stride_sizes[i]
        
        # 打印前10個和後10個網格的步長，確認映射正確
        print(f"\n===== 網格步長檢查 =====")
        print(f"前10個網格步長: {grid_strides[:10].tolist()}")
        print(f"後10個網格步長: {grid_strides[-10:].tolist()}")
        
        # 創建存儲細化後誤差的張量
        refined_x_diff = torch.zeros_like(s_x)
        refined_y_diff = torch.zeros_like(s_y)
        
        # 尋找可能的偏移和縮放參數組合
        scale_options = [1.0, 0.9, 1.1, 0.8, 1.2]
        offset_options_x = [0.0, 0.5, -0.5, 1.0, -1.0]
        offset_options_y = [0.0, 0.5, -0.5, 1.0, -1.0]
        
        # 跟踪最佳參數
        best_params = {}
        
        # 對每個關鍵點、每個特徵層分別嘗試
        for kp_idx in range(num_keypoints):
            for stride_idx in range(len(stride_indices)-1):
                start_idx = stride_indices[stride_idx]
                end_idx = stride_indices[stride_idx+1]
                stride = stride_sizes[stride_idx]
                
                # 對當前關鍵點和特徵層，選擇有效的點
                valid_points = valid_mask[:, kp_idx, start_idx:end_idx].sum().item()
                
                if valid_points > 10:  # 至少需要10個有效點才能進行對齊
                    # 提取當前關鍵點、當前特徵層的坐標
                    curr_s_x = s_x[:, kp_idx, start_idx:end_idx]
                    curr_s_y = s_y[:, kp_idx, start_idx:end_idx]
                    curr_t_x = t_x[:, kp_idx, start_idx:end_idx]
                    curr_t_y = t_y[:, kp_idx, start_idx:end_idx]
                    curr_mask = valid_mask[:, kp_idx, start_idx:end_idx]
                    
                    best_error = float('inf')
                    best_scale = 1.0
                    best_offset_x = 0.0
                    best_offset_y = 0.0
                    
                    # 嘗試不同的縮放和偏移組合
                    for scale in scale_options:
                        for offset_x in offset_options_x:
                            for offset_y in offset_options_y:
                                # 應用變換
                                adjusted_x = curr_s_x * scale + offset_x
                                adjusted_y = curr_s_y * scale + offset_y
                                
                                # 計算誤差
                                x_err = torch.abs(adjusted_x - curr_t_x)
                                y_err = torch.abs(adjusted_y - curr_t_y)
                                combined_err = torch.sqrt(x_err**2 + y_err**2 + 1e-8)
                                
                                # 只考慮有效區域的誤差
                                if curr_mask.sum() > 0:
                                    mean_err = (combined_err * curr_mask.float()).sum() / curr_mask.sum()
                                    if mean_err < best_error:
                                        best_error = mean_err
                                        best_scale = scale
                                        best_offset_x = offset_x
                                        best_offset_y = offset_y
                    
                    # 記錄最佳參數
                    param_key = f"kp{kp_idx}_stride{stride}"
                    best_params[param_key] = {
                        "scale": best_scale,
                        "offset_x": best_offset_x,
                        "offset_y": best_offset_y,
                        "error": best_error.item()
                    }
                    
                    # 應用最佳參數到當前區域
                    refined_x_diff[:, kp_idx, start_idx:end_idx] = torch.abs(
                        curr_s_x * best_scale + best_offset_x - curr_t_x
                    )
                    refined_y_diff[:, kp_idx, start_idx:end_idx] = torch.abs(
                        curr_s_y * best_scale + best_offset_y - curr_t_y
                    )
                else:
                    # 對於無效區域，使用原始差異
                    refined_x_diff[:, kp_idx, start_idx:end_idx] = torch.abs(
                        s_x[:, kp_idx, start_idx:end_idx] - t_x[:, kp_idx, start_idx:end_idx]
                    )
                    refined_y_diff[:, kp_idx, start_idx:end_idx] = torch.abs(
                        s_y[:, kp_idx, start_idx:end_idx] - t_y[:, kp_idx, start_idx:end_idx]
                    )
        
        # 打印一些對齊參數
        print(f"\n===== 對齊參數摘要 =====")
        params_to_show = min(5, len(best_params))
        for i, (key, params) in enumerate(list(best_params.items())[:params_to_show]):
            print(f"{key}: 縮放={params['scale']:.2f}, X偏移={params['offset_x']:.2f}, Y偏移={params['offset_y']:.2f}, 誤差={params['error']:.4f}")
        
        # 使用對齊後的差異重新計算精度
        refined_combined_diff = torch.sqrt(refined_x_diff**2 + refined_y_diff**2 + 1e-8)
        
        # 計算精度掩碼
        map50_mask = (refined_combined_diff < 0.05) & valid_mask
        map90_mask = (refined_combined_diff < 0.01) & valid_mask
        map95_mask = (refined_combined_diff < 0.005) & valid_mask
        
        # 計算精度比例
        if valid_mask.sum() > 0:
            refined_map50_ratio = map50_mask.sum().float() / valid_mask.sum()
            refined_map90_ratio = map90_mask.sum().float() / valid_mask.sum()
            refined_map95_ratio = map95_mask.sum().float() / valid_mask.sum()
        else:
            refined_map50_ratio = torch.tensor(0.0, device=s_x.device)
            refined_map90_ratio = torch.tensor(0.0, device=s_x.device)
            refined_map95_ratio = torch.tensor(0.0, device=s_x.device)
        
        print(f"\n===== 網格對齊後精度 =====")
        print(f"對齊前 mAP50: {(torch.abs(s_x - t_x)**2 + torch.abs(s_y - t_y)**2 < 0.05**2).float().mean().item():.6f}")
        print(f"對齊後 mAP50 點比例 (誤差<5%): {refined_map50_ratio.item():.6f}")
        print(f"對齊後 mAP90 點比例 (誤差<1%): {refined_map90_ratio.item():.6f}")
        print(f"對齊後 mAP95 點比例 (誤差<0.5%): {refined_map95_ratio.item():.6f}")
        
        # 計算關鍵點級別的精度提升
        print(f"\n===== 關鍵點精度提升 =====")
        for kp in range(num_keypoints):
            kp_valid = valid_mask[:, kp].sum().float()
            if kp_valid > 0:
                kp_map50_before = ((torch.abs(s_x[:, kp] - t_x[:, kp])**2 + torch.abs(s_y[:, kp] - t_y[:, kp])**2) < 0.05**2) & valid_mask[:, kp]
                kp_ratio_before = kp_map50_before.sum().float() / kp_valid
                
                kp_map50_after = map50_mask[:, kp]
                kp_ratio_after = kp_map50_after.sum().float() / kp_valid
                
                print(f"關鍵點 {kp}: mAP50 前={kp_ratio_before.item():.4f}, 後={kp_ratio_after.item():.4f}, 提升={kp_ratio_after.item()-kp_ratio_before.item():.4f}")
        
        # 損失計算 - 使用對齊後的差異
        # 加權誤差損失 - 基於精度區間
        w_map95 = 12.0    # mAP95 區間權重
        w_map90 = 6.0     # mAP90 區間權重
        w_map50 = 2.0     # mAP50 區間權重
        w_above = 0.5     # 超出 mAP50 的區間權重
        
        # 計算加權誤差損失
        weighted_diff = torch.zeros_like(refined_combined_diff)
        weighted_diff = torch.where(map95_mask, w_map95 * refined_combined_diff, weighted_diff)
        weighted_diff = torch.where(map90_mask & ~map95_mask, w_map90 * refined_combined_diff, weighted_diff)
        weighted_diff = torch.where(map50_mask & ~map90_mask, w_map50 * refined_combined_diff, weighted_diff)
        weighted_diff = torch.where(valid_mask & ~map50_mask, w_above * refined_combined_diff, weighted_diff)
        
        # 應用教師置信度作為額外權重
        conf_weighted_diff = weighted_diff * t_conf_prob
        
        # 計算最終的加權誤差損失
        if valid_mask.sum() > 0:
            coord_loss = conf_weighted_diff.sum() / valid_mask.sum()
        else:
            coord_loss = torch.tensor(0.0, device=s_x.device)
        
        # 置信度損失 - 與之前相同
        # 根據點的精度調整置信度目標
        target_conf = torch.zeros_like(t_conf_prob)
        target_conf = torch.where(map95_mask, torch.ones_like(target_conf) * 0.95, target_conf)
        target_conf = torch.where(map90_mask & ~map95_mask, torch.ones_like(target_conf) * 0.9, target_conf)
        target_conf = torch.where(map50_mask & ~map90_mask, torch.ones_like(target_conf) * 0.5, target_conf)
        
        # 計算學生模型的置信度
        s_conf_prob = torch.sigmoid(s_conf)
        
        # 置信度損失 - 只針對有效點
        if valid_mask.sum() > 0:
            conf_loss = (torch.abs(s_conf_prob - target_conf) * valid_mask.float()).sum() / valid_mask.sum()
        else:
            conf_loss = torch.tensor(0.0, device=s_conf.device)
        
        # 最終組合損失
        final_loss = coord_loss + 0.2 * conf_loss
        
        # 創建一個字典來存儲參數，便於後期使用
        self.best_alignment_params = best_params
        
        print(f"\n===== 對齊後損失值 =====")
        print(f"坐標損失: {coord_loss.item():.6f}")
        print(f"置信度損失: {conf_loss.item():.6f}")
        print(f"總損失: {final_loss.item():.6f}")
        
        return final_loss

    def log_precision_stats(self, x_diff, y_diff):
        """記錄坐標精度統計信息"""
        # 定義精度等級
        precision_thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
        precision_names = ["0.01%", "0.05%", "0.1%", "0.5%", "1%", "5%"]
        
        # 計算每個精度等級下的點比例
        x_stats = []
        y_stats = []
        
        for threshold in precision_thresholds:
            x_ratio = (x_diff.abs() < threshold).float().mean().item()
            y_ratio = (y_diff.abs() < threshold).float().mean().item()
            x_stats.append(f"{x_ratio*100:.2f}%")
            y_stats.append(f"{y_ratio*100:.2f}%")
        
        # 打印統計數據
        print("\n--- 坐標精度統計 ---")
        print(f"精度等級: {', '.join(precision_names)}")
        print(f"X軸精度: {', '.join(x_stats)}")
        print(f"Y軸精度: {', '.join(y_stats)}")
        
        # 計算整體mAP相關統計
        xy_05 = (x_diff.abs() < 0.05) & (y_diff.abs() < 0.05)
        xy_01 = (x_diff.abs() < 0.01) & (y_diff.abs() < 0.01)
        xy_005 = (x_diff.abs() < 0.005) & (y_diff.abs() < 0.005)
        
        print(f"對應mAP50的點比例 (誤差<5%): {xy_05.float().mean().item()*100:.2f}%")
        print(f"對應mAP90的點比例 (誤差<1%): {xy_01.float().mean().item()*100:.2f}%")
        print(f"對應mAP95的點比例 (誤差<0.5%): {xy_005.float().mean().item()*100:.2f}%")


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
