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

def analyze_matched_pose_predictions(student_outputs, teacher_outputs, conf_threshold=0.5, match_threshold=0.5):
    """
    匹配並分析姿態預測的關鍵點
    
    假設輸出格式為 [B, C, 8400]，其中:
    - 前2個通道是中心點坐標
    - 接下來是17個關鍵點，每個3個通道 (x, y, conf)
    """
    _, student_preds = student_outputs
    _, teacher_preds = teacher_outputs
    
    batch_size = student_preds.shape[0]
    total_matched = 0
    total_s_instances = 0
    total_t_instances = 0
    
    # 存儲所有匹配點對的關鍵點誤差
    all_keypoint_errors = []
    
    print(f"\n===== 基於匹配的姿態預測分析 =====")
    print(f"置信度閾值: {conf_threshold}")
    print(f"匹配距離閾值: {match_threshold}")
    
    for b in range(batch_size):
        # 提取單個批次的預測
        s_pred = student_preds[b]  # [C, 8400]
        t_pred = teacher_preds[b]  # [C, 8400]
        
        # 假設前兩個通道是中心點或物體坐標
        num_keypoints = (s_pred.shape[0] - 2) // 3  # 計算關鍵點數量
        
        # 提取置信度 (每3個通道的最後一個)
        s_conf_channels = [2 + i*3 + 2 for i in range(num_keypoints)]
        t_conf_channels = s_conf_channels
        
        s_conf = s_pred[s_conf_channels]  # [num_keypoints, 8400]
        t_conf = t_pred[t_conf_channels]
        
        # 計算平均置信度作為整體置信度
        s_mean_conf = s_conf.mean(dim=0)  # [8400]
        t_mean_conf = t_conf.mean(dim=0)
        
        # 篩選高置信度的預測
        s_high_conf = s_mean_conf > conf_threshold
        t_high_conf = t_mean_conf > conf_threshold
        
        s_indices = torch.nonzero(s_high_conf).squeeze(-1)
        t_indices = torch.nonzero(t_high_conf).squeeze(-1)
        
        # 計數
        s_count = s_indices.shape[0]
        t_count = t_indices.shape[0]
        
        total_s_instances += s_count
        total_t_instances += t_count
        
        # 構建匹配矩陣
        if s_count > 0 and t_count > 0:
            # 提取中心點
            s_centers = torch.stack([s_pred[0, s_indices], s_pred[1, s_indices]])  # [2, s_count]
            t_centers = torch.stack([t_pred[0, t_indices], t_pred[1, t_indices]])  # [2, t_count]
            
            # 計算距離矩陣
            s_centers_expanded = s_centers.unsqueeze(2)  # [2, s_count, 1]
            t_centers_expanded = t_centers.unsqueeze(1)  # [2, 1, t_count]
            
            distance_matrix = torch.sqrt(((s_centers_expanded - t_centers_expanded) ** 2).sum(dim=0))  # [s_count, t_count]
            
            # 匹配最近的點對
            matched_pairs = []
            
            # 簡單貪婪匹配 (每次選最小距離的點對)
            while distance_matrix.numel() > 0:
                # 找到最小距離
                min_dist = distance_matrix.min()
                if min_dist > match_threshold:
                    break
                
                # 找到最小距離的索引
                min_indices = torch.nonzero(distance_matrix == min_dist)[0]
                s_idx = min_indices[0].item()
                t_idx = min_indices[1].item()
                
                # 添加到匹配對
                matched_pairs.append((s_indices[s_idx].item(), t_indices[t_idx].item()))
                
                # 從距離矩陣中移除這些點
                mask = torch.ones_like(distance_matrix, dtype=torch.bool)
                mask[s_idx, :] = False
                mask[:, t_idx] = False
                distance_matrix = distance_matrix[mask].reshape(-1, distance_matrix.shape[1] - 1)
            
            num_matched = len(matched_pairs)
            total_matched += num_matched
            
            print(f"批次 {b}: 學生實例 {s_count}, 教師實例 {t_count}, 成功匹配 {num_matched}")
            
            if num_matched > 0:
                # 分析匹配點對的關鍵點誤差
                batch_keypoint_errors = []
                
                for s_idx, t_idx in matched_pairs:
                    instance_errors = []
                    
                    for kp in range(num_keypoints):
                        # 提取關鍵點坐標
                        s_kp_x = s_pred[2 + kp*3, s_idx].item()
                        s_kp_y = s_pred[2 + kp*3 + 1, s_idx].item()
                        s_kp_conf = s_pred[2 + kp*3 + 2, s_idx].item()
                        
                        t_kp_x = t_pred[2 + kp*3, t_idx].item()
                        t_kp_y = t_pred[2 + kp*3 + 1, t_idx].item()
                        t_kp_conf = t_pred[2 + kp*3 + 2, t_idx].item()
                        
                        # 關鍵點置信度都高才比較
                        if s_kp_conf > conf_threshold and t_kp_conf > conf_threshold:
                            x_diff = abs(s_kp_x - t_kp_x)
                            y_diff = abs(s_kp_y - t_kp_y)
                            distance = math.sqrt(x_diff**2 + y_diff**2)
                            
                            instance_errors.append({
                                'keypoint': kp,
                                'x_diff': x_diff,
                                'y_diff': y_diff,
                                'distance': distance,
                                's_coords': (s_kp_x, s_kp_y),
                                't_coords': (t_kp_x, t_kp_y)
                            })
                    
                    batch_keypoint_errors.append(instance_errors)
                
                all_keypoint_errors.extend(batch_keypoint_errors)
        else:
            print(f"批次 {b}: 學生實例 {s_count}, 教師實例 {t_count}, 無匹配")
    
    print(f"\n===== 總體統計 =====")
    print(f"學生總實例: {total_s_instances}")
    print(f"教師總實例: {total_t_instances}")
    print(f"成功匹配總數: {total_matched}")
    
    if total_matched > 0:
        # 分析關鍵點誤差
        print(f"\n===== 關鍵點誤差分析 =====")
        
        # 按關鍵點統計誤差
        keypoint_stats = {}
        
        for instance_errors in all_keypoint_errors:
            for error in instance_errors:
                kp = error['keypoint']
                if kp not in keypoint_stats:
                    keypoint_stats[kp] = {
                        'count': 0,
                        'x_diffs': [],
                        'y_diffs': [],
                        'distances': [],
                        's_coords': [],
                        't_coords': []
                    }
                
                stats = keypoint_stats[kp]
                stats['count'] += 1
                stats['x_diffs'].append(error['x_diff'])
                stats['y_diffs'].append(error['y_diff'])
                stats['distances'].append(error['distance'])
                stats['s_coords'].append(error['s_coords'])
                stats['t_coords'].append(error['t_coords'])
        
        # 輸出每個關鍵點的統計
        for kp, stats in sorted(keypoint_stats.items()):
            count = stats['count']
            if count > 0:
                x_diffs = stats['x_diffs']
                y_diffs = stats['y_diffs']
                distances = stats['distances']
                
                mean_x_diff = sum(x_diffs) / count
                mean_y_diff = sum(y_diffs) / count
                mean_distance = sum(distances) / count
                max_distance = max(distances)
                
                # 分析坐標分布
                s_x_vals = [coord[0] for coord in stats['s_coords']]
                s_y_vals = [coord[1] for coord in stats['t_coords']]
                t_x_vals = [coord[0] for coord in stats['s_coords']]
                t_y_vals = [coord[1] for coord in stats['t_coords']]
                
                s_x_min, s_x_max = min(s_x_vals), max(s_x_vals)
                s_y_min, s_y_max = min(s_y_vals), max(s_y_vals)
                t_x_min, t_x_max = min(t_x_vals), max(t_x_vals)
                t_y_min, t_y_max = min(t_y_vals), max(t_y_vals)
                
                # 計算最佳縮放因子
                s_x_range = s_x_max - s_x_min
                t_x_range = t_x_max - t_x_min
                s_y_range = s_y_max - s_y_min
                t_y_range = t_y_max - t_y_min
                
                x_scale = t_x_range / s_x_range if s_x_range > 0 else 1.0
                y_scale = t_y_range / s_y_range if s_y_range > 0 else 1.0
                
                # 輸出統計
                print(f"\n關鍵點 {kp}: {count} 個匹配")
                print(f"  X軸誤差: 平均 {mean_x_diff:.4f}")
                print(f"  Y軸誤差: 平均 {mean_y_diff:.4f}")
                print(f"  綜合距離: 平均 {mean_distance:.4f}, 最大 {max_distance:.4f}")
                print(f"  學生坐標範圍: X [{s_x_min:.2f}, {s_x_max:.2f}], Y [{s_y_min:.2f}, {s_y_max:.2f}]")
                print(f"  教師坐標範圍: X [{t_x_min:.2f}, {t_x_max:.2f}], Y [{t_y_min:.2f}, {t_y_max:.2f}]")
                print(f"  估計縮放因子: X {x_scale:.2f}, Y {y_scale:.2f}")
                
                # 計算MAP50/90/95
                map50 = sum(1 for d in distances if d < 0.05) / count
                map90 = sum(1 for d in distances if d < 0.01) / count
                map95 = sum(1 for d in distances if d < 0.005) / count
                
                print(f"  mAP50: {map50:.4f}")
                print(f"  mAP90: {map90:.4f}")
                print(f"  mAP95: {map95:.4f}")
        
        # 計算總體的MAP指標
        all_distances = [error['distance'] for instance in all_keypoint_errors for error in instance]
        total_kps = len(all_distances)
        
        if total_kps > 0:
            overall_map50 = sum(1 for d in all_distances if d < 0.05) / total_kps
            overall_map90 = sum(1 for d in all_distances if d < 0.01) / total_kps
            overall_map95 = sum(1 for d in all_distances if d < 0.005) / total_kps
            
            print(f"\n===== 總體MAP指標 =====")
            print(f"總關鍵點數: {total_kps}")
            print(f"總體 mAP50: {overall_map50:.4f}")
            print(f"總體 mAP90: {overall_map90:.4f}")
            print(f"總體 mAP95: {overall_map95:.4f}")

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

        # print("preds", describe_var(preds, max_depth=10, max_items=100))

        # print("batch", describe_var(batch, max_depth=10, max_items=100))

        # if "teacher" in batch and batch["teacher"] is not None:
        #     print("teacher_preds", describe_var(batch["teacher_preds"]))

        loss = torch.zeros(6, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        
        # NaN 檢查 - 確保輸入預測沒有NaN
        if torch.isnan(pred_kpts).any():
            print("警告: 偵測到預測關鍵點包含NaN值，將替換為零值")
            pred_kpts = torch.nan_to_num(pred_kpts, nan=0.0)
            
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )
        
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

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # NaN 檢查 - 確保目標框沒有NaN
        if torch.isnan(gt_bboxes).any():
            print("警告: 偵測到目標框包含NaN值")
            return loss * batch_size, loss.detach()  # 直接返回零損失

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)
        
        # NaN 檢查 - 解碼後的預測框和關鍵點
        if torch.isnan(pred_bboxes).any():
            print("警告: 偵測到解碼後的預測框包含NaN值，將替換為零值")
            pred_bboxes = torch.nan_to_num(pred_bboxes, nan=0.0)
            
        if torch.isnan(pred_kpts).any():
            print("警告: 偵測到解碼後的預測關鍵點包含NaN值，將替換為零值")
            pred_kpts = torch.nan_to_num(pred_kpts, nan=0.0)

        try:
            _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
                anchor_points * stride_tensor,
                gt_labels,
                gt_bboxes,
                mask_gt,
            )
        except Exception as e:
            print(f"警告: assigner出現異常: {str(e)}")
            return loss * batch_size, loss.detach()  # 直接返回零損失

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
            analyze_matched_pose_predictions(preds, batch["teacher_preds"])

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

    def pose_distillation_loss_enhanced(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.3, pred_weight=0.6):
        """
        增強版姿態蒸餾損失函數：
        1. 重新加入溫度調節機制
        2. 自適應特徵層權重
        3. 置信度加權的坐標損失
        4. 骨架重要性加權
        5. 更加穩健的損失組合
        6. 動態特徵層選擇策略
        7. 添加數值穩定性檢查
        8. 早期停止機制 
        """
        # 降低容忍度，使模型對更小的誤差敏感
        tolerance = 1e-12  # 從1e-10進一步降低到1e-12
        epsilon = 1e-12    # 同步降低

        try:
            # 【新增】早期停止機制
            # 如果上一批次損失已經非常小，可以早期返回
            if hasattr(self, 'last_loss_values') and isinstance(self.last_loss_values, dict):
                last_total_loss = self.last_loss_values.get('total_loss', 1.0)
                if last_total_loss < 1e-5:
                    # 損失已經非常小，可以跳過詳細計算
                    return torch.tensor(last_total_loss, device=student_outputs[1].device, requires_grad=True)
            
            # 直接解包輸出
            student_features, student_preds = student_outputs[0], student_outputs[1]
            teacher_features, teacher_preds = teacher_outputs[0], teacher_outputs[1]
            
            # 快速檢查NaN
            if torch.isnan(student_preds).any() or torch.isnan(teacher_preds).any():
                return torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 一次性重塑並提取所有需要的張量
            batch_size = student_preds.shape[0]
            s_preds = student_preds.reshape(batch_size, 17, 3, -1)
            t_preds = teacher_preds.reshape(batch_size, 17, 3, -1)
            
            s_x, s_y, s_conf = s_preds[:, :, 0], s_preds[:, :, 1], s_preds[:, :, 2]
            t_x, t_y, t_conf = t_preds[:, :, 0], t_preds[:, :, 1], t_preds[:, :, 2]
            
            # 【新增】數值穩定性檢查 - 對坐標差異應用容忍閾值
            # 將非常接近的坐標視為相同，減少浮點誤差影響
            x_exact_match = torch.abs(s_x - t_x) < tolerance
            y_exact_match = torch.abs(s_y - t_y) < tolerance
            if x_exact_match.any() or y_exact_match.any():
                s_x = torch.where(x_exact_match, t_x, s_x)
                s_y = torch.where(y_exact_match, t_y, s_y)
            
            # 使用向量化操作進行限幅
            s_conf = s_conf.clamp(-50.0, 50.0)
            t_conf = t_conf.clamp(-50.0, 50.0)
            
            # 【改進1】重新加入溫度縮放的軟目標 
            t_conf_T = t_conf / T
            s_conf_T = s_conf / T
            
            t_prob = torch.sigmoid(t_conf_T)
            s_prob = torch.sigmoid(s_conf_T)
            
            # 計算教師模型的置信度掩碼，用於後續加權
            teacher_conf_mask = torch.sigmoid(t_conf)
            
            # 獲取當前epoch和訓練進度信息
            current_epoch = getattr(self.model, 'epoch', 0) if hasattr(self, 'model') else 0
            total_epochs = getattr(self.model, 'epochs', 100) if hasattr(self, 'model') else 100
            is_first_batch_in_epoch = getattr(self.model, 'is_first_batch_in_epoch', False) if hasattr(self, 'model') else False

            # 計算訓練進度比例
            progress = calculate_progress(current_epoch, total_epochs)
            
            # 1. 特徵蒸餾損失 - 加入自適應權重與動態選擇策略
            max_len = min(len(student_features), len(teacher_features))
            
            # 【新增】檢查是否需要進行詳細的特徵蒸餾
            # 當進度很高且損失已經很小時，可以簡化特徵蒸餾計算
            simplified_feature_distill = False
            if hasattr(self, 'last_loss_values') and progress > 0.8:
                last_feat_loss = self.last_loss_values.get('feat_loss', 1.0)
                if last_feat_loss < 0.01:  # 特徵損失已經很小
                    simplified_feature_distill = True

            # 在第6-7個epoch擴展到所有特徵層
            if current_epoch < 6:
                indices = [0, 1] if current_epoch >= 3 else [0]
            else:
                indices = [0, 1, 2]  # 使用所有層
            
            # 確保索引唯一且有序
            indices = sorted(list(set(indices)))
            
            # 【改進】動態特徵層權重策略
            # 訓練初期淺層和深層權重相近，後期深層權重更大
            if progress < 0.5:
                # 初期階段 - 權重差異較小
                min_weight = 0.8
                max_weight = 1.2
            else:
                # 後期階段 - 權重差異加大
                min_weight = 0.5
                max_weight = 1.5
            
            # 根據層的深度生成權重，深度越深權重越大
            layer_weights = []
            for idx in indices:
                # 相對深度 (0到1之間)
                rel_depth = idx / (max_len - 1) if max_len > 1 else 0.5
                # 線性插值計算權重
                weight = min_weight + rel_depth * (max_weight - min_weight)
                layer_weights.append(weight)
            
            # 轉換為張量
            layer_importance = torch.tensor(layer_weights, device=student_preds.device)
            
            # 調試信息 (可選)
            if current_epoch % 1 == 0 and is_first_batch_in_epoch:
                print(f"\nEpoch {current_epoch}/{total_epochs} (Progress: {progress:.2f})")
                print(f"Dynamic feature layers: {indices}")
                print(f"Layer weights: {layer_weights}")
                if simplified_feature_distill:
                    print("Using simplified feature distillation")
            
            if len(indices) > 0:
                # 使用索引張量獲取特徵
                s_feats = [student_features[i] for i in indices]
                t_feats = [teacher_features[i] for i in indices]
                
                # 計算特徵損失
                feat_losses = torch.zeros(len(indices), device=student_preds.device)
                valid_feats = torch.ones(len(indices), device=student_preds.device)
                
                for i, (s_f, t_f) in enumerate(zip(s_feats, t_feats)):
                    if s_f.shape != t_f.shape:
                        try:
                            t_f = F.interpolate(t_f, size=s_f.shape[-2:], mode='bilinear', align_corners=False)
                        except:
                            valid_feats[i] = 0
                            continue
                    
                    if torch.isnan(s_f).any() or torch.isnan(t_f).any():
                        valid_feats[i] = 0
                        continue
                    
                    # 【新增】計算自適應L2距離時應用數值穩定性檢查
                    feat_diff = s_f - t_f
                    
                    # 【新增】將微小差異視為零，減少浮點誤差影響
                    feat_diff = torch.where(torch.abs(feat_diff) < tolerance, 
                                        torch.zeros_like(feat_diff), feat_diff)
                    
                    # 【新增】梯度尺度調整 - 為深層特徵應用梯度尺度衰減
                    grad_scale = 1.0 / (2 ** (i * 0.5))  # 深層特徵梯度尺度平滑衰減
                    feat_losses[i] = (feat_diff ** 2).mean() * grad_scale
                
                # 應用層重要性權重
                weighted_sum = (feat_losses * valid_feats * layer_importance).sum()
                weight_sum = (valid_feats * layer_importance).sum() + epsilon
                feat_loss = weighted_sum / weight_sum
            else:
                feat_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)

            if torch.isnan(feat_loss) or torch.isinf(feat_loss):
                print("警告: 特徵損失為NaN或Inf")
                feat_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
          
            # 2. 【改進4】置信度加權的坐標損失
            # 高置信度區域權重更大
            coord_weights = teacher_conf_mask.unsqueeze(-1)  # [B, 17, 1, grid]
            
            x_diff = (s_x - t_x).unsqueeze(-1)  # [B, 17, 1, grid]
            y_diff = (s_y - t_y).unsqueeze(-1)  # [B, 17, 1, grid]
            
            # 【新增】應用數值穩定性檢查
            x_diff = torch.where(torch.abs(x_diff) < tolerance, torch.zeros_like(x_diff), x_diff)
            y_diff = torch.where(torch.abs(y_diff) < tolerance, torch.zeros_like(y_diff), y_diff)
            
            # 【修改】Y軸誤差權重加大1.5倍
            weighted_x_diff = (x_diff ** 2) * coord_weights

            # 根據當前Y/X軸誤差比例動態調整Y軸權重
            # 在計算Y軸損失時添加這段代碼
            x_diff_mean = x_diff.abs().mean().item()
            y_diff_mean = y_diff.abs().mean().item()
            y_x_ratio = y_diff_mean / (x_diff_mean + epsilon)
            y_axis_factor = min(3.0, max(1.5, y_x_ratio * 0.8))  # 將Y軸權重因子限制在1.5-3.0之間

            # 更新Y軸差異權重
            weighted_y_diff = (y_diff ** 2) * coord_weights * y_axis_factor  # 使用動態因子而不是固定的2.0
            
            # 避免分母為零
            total_weight = coord_weights.sum() + epsilon
            
            # 1. 更精細的坐標損失階梯 - 添加進度自適應權重
            # 隨著訓練進行，精細部分權重逐漸提高
            if progress < 0.3:
                # 前30%訓練階段 - 從低權重開始逐漸增加
                tiny_weight_x = 12.0 + progress * 30  # 起始12，最高21
                small_weight_x = 6.0 + progress * 15  # 起始6，最高10.5
                
                # Y軸權重始終高於X軸
                tiny_weight_y = tiny_weight_x * 1.5
                small_weight_y = small_weight_x * 1.5
            else:
                # 後70%訓練階段 - 保持較高權重
                tiny_weight_x = 21.0
                small_weight_x = 10.5
                tiny_weight_y = 31.5  # 1.5倍於X軸
                small_weight_y = 15.75 # 1.5倍於X軸

            # 在添加更細的階梯部分之前添加這些代碼
            # 添加超微小差異層級
            micro_tiny_diff_mask_x = (x_diff**2 < 0.0001)  # 超微小差異(0.0001)
            micro_tiny_diff_mask_y = (y_diff**2 < 0.0001)
            
            # 添加更細的階梯 - 極微小誤差層級
            ultra_tiny_diff_mask_x = (x_diff**2 < 0.0005) & ~micro_tiny_diff_mask_x  # 極微小差異(0.0005)
            tiny_diff_mask_x = (x_diff**2 < 0.001) & ~ultra_tiny_diff_mask_x  # 極小差異(0.001)
            small_diff_mask_x = (x_diff**2 < 0.005) & ~tiny_diff_mask_x & ~ultra_tiny_diff_mask_x  # 小差異(0.005)
            medium_diff_mask_x = (x_diff**2 < 0.02) & ~small_diff_mask_x & ~tiny_diff_mask_x & ~ultra_tiny_diff_mask_x  # 中等差異
            
            ultra_tiny_diff_mask_y = (y_diff**2 < 0.0005) & ~micro_tiny_diff_mask_y
            tiny_diff_mask_y = (y_diff**2 < 0.001) & ~ultra_tiny_diff_mask_y
            small_diff_mask_y = (y_diff**2 < 0.005) & ~tiny_diff_mask_y & ~ultra_tiny_diff_mask_y
            medium_diff_mask_y = (y_diff**2 < 0.02) & ~small_diff_mask_y & ~tiny_diff_mask_y & ~ultra_tiny_diff_mask_y

            # 更新 X 軸精確損失計算
            precise_x_loss = (
                torch.where(micro_tiny_diff_mask_x, torch.abs(x_diff) * tiny_weight_x * 2.5, torch.zeros_like(x_diff)) +  # 新增超微小差異層級
                torch.where(ultra_tiny_diff_mask_x, torch.abs(x_diff) * tiny_weight_x * 1.8, torch.zeros_like(x_diff)) +
                torch.where(tiny_diff_mask_x, torch.abs(x_diff) * tiny_weight_x, torch.zeros_like(x_diff)) +
                torch.where(small_diff_mask_x, torch.abs(x_diff) * small_weight_x, torch.zeros_like(x_diff)) +
                torch.where(medium_diff_mask_x, torch.abs(x_diff) * 2.0, torch.zeros_like(x_diff))
            )

            # 更新 Y 軸精確損失計算 - 提高 Y 軸權重
            precise_y_loss = (
                torch.where(micro_tiny_diff_mask_y, torch.abs(y_diff) * tiny_weight_y * 5.0, torch.zeros_like(y_diff)) +  # 從4.0提高到5.0
                torch.where(ultra_tiny_diff_mask_y, torch.abs(y_diff) * tiny_weight_y * 3.5, torch.zeros_like(y_diff)) +  # 從2.5提高到3.5
                torch.where(tiny_diff_mask_y, torch.abs(y_diff) * tiny_weight_y * 1.5, torch.zeros_like(y_diff)) +        # 提高權重
                torch.where(small_diff_mask_y, torch.abs(y_diff) * small_weight_y * 1.2, torch.zeros_like(y_diff)) +      # 提高權重
                torch.where(medium_diff_mask_y, torch.abs(y_diff) * 3.5, torch.zeros_like(y_diff))                       # 提高權重
            )
            
            # 將精確定位損失添加到加權損失中
            weighted_precise_x = precise_x_loss * coord_weights
            weighted_precise_y = precise_y_loss * coord_weights

            # 根據當前精度比例自動調整精度權重
            precise_x_ratio = tiny_diff_mask_x.float().mean().item()
            precise_y_ratio = tiny_diff_mask_y.float().mean().item()
            current_y_to_x_precise_ratio = precise_y_ratio / (precise_x_ratio + epsilon)
            target_ratio = 1.0  # 目標Y/X精度比例

            # 如果Y軸精度比例低於X軸，增加Y軸權重
            if current_y_to_x_precise_ratio < target_ratio:
                y_precise_factor = 1.0 + (target_ratio - current_y_to_x_precise_ratio) * 0.5
                weighted_precise_y = weighted_precise_y * y_precise_factor

            # 修改coord_loss計算
            coord_loss = (weighted_x_diff.sum() + weighted_y_diff.sum() + 
                        weighted_precise_x.sum() + weighted_precise_y.sum()) / total_weight
            
            if torch.isnan(coord_loss) or torch.isinf(coord_loss):
                print("警告: 坐標損失為NaN或Inf")
                coord_loss = torch.tensor(0.1, device=student_preds.device, requires_grad=True)
            
            
            # 在計算 coord_loss 之後，pred_loss 之前添加
            # 添加針對 Y 軸的特殊處理
            # 針對垂直重要關鍵點的額外損失
            vertical_important_keypoints = [0, 5, 6, 11, 12, 15, 16]  # 頭頂、肩膀、髖部、腳踝
            y_special_loss = torch.tensor(0.0, device=student_preds.device)

            # 提取這些關鍵點並應用特殊權重
            for kp_idx in vertical_important_keypoints:
                # 取絕對差異
                y_kp_diff = torch.abs(s_y[:, kp_idx] - t_y[:, kp_idx]).unsqueeze(-1)
                # 權重 - 基於教師置信度
                y_kp_weight = teacher_conf_mask[:, kp_idx].unsqueeze(-1) * 1.5
                
                # 計算加權差異
                weighted_y_kp_diff = y_kp_diff * y_kp_weight
                y_special_loss = y_special_loss + weighted_y_kp_diff.sum() / (y_kp_weight.sum() + epsilon)

            # 在 Y 軸特別處理中加入精細化損失
            # 更新Y軸特殊處理的係數
            high_conf_mask = teacher_conf_mask > 0.6  # 降低閾值從0.7到0.6
            if high_conf_mask.any():
                high_conf_y = s_y[high_conf_mask]
                high_conf_t_y = t_y[high_conf_mask]
                if high_conf_y.numel() > 0:
                    # 計算高置信度點的Y軸差異
                    hc_y_diff = torch.abs(high_conf_y - high_conf_t_y)
                    
                    # 特別關注小差異 - 提高權重
                    tiny_hc_mask = hc_y_diff < 0.001
                    small_hc_mask = (hc_y_diff >= 0.001) & (hc_y_diff < 0.005)
                    
                    # 應用更高權重
                    if tiny_hc_mask.any():
                        y_special_loss = y_special_loss + hc_y_diff[tiny_hc_mask].sum() * 8.0 / (tiny_hc_mask.sum() + epsilon)  # 從5.0提高到8.0
                    if small_hc_mask.any():
                        y_special_loss = y_special_loss + hc_y_diff[small_hc_mask].sum() * 4.0 / (small_hc_mask.sum() + epsilon)  # 從2.0提高到4.0
            
            # 3. 結構損失 - 優化骨架選擇和權重
            # 【改進5】擴展骨架集合並按重要性加權
            skeleton = torch.tensor([
                [5, 6],    # 左肩-右肩 (軀幹上部)
                [11, 12],  # 左髖-右髖 (骨盆)
                [5, 11],   # 左肩-左髖 (左軀幹)
                [6, 12],   # 右肩-右髖 (右軀幹)
                [5, 7],    # 左肩-左肘 (左上臂)
                [6, 8],    # 右肩-右肘 (右上臂)
                [7, 9],    # 左肘-左腕 (左前臂)
                [8, 10],   # 右肘-右腕 (右前臂)
                [11, 13],  # 左髖-左膝 (左大腿)
                [12, 14],  # 右髖-右膝 (右大腿)
                [13, 15],  # 左膝-左踝 (左小腿)
                [14, 16],  # 右膝-右踝 (右小腿)
            ], device=student_preds.device)
            
            # 骨架重要性權重 - 軀幹與核心部位權重更高
            skeleton_weights = torch.tensor([
                2.5, 2.5,   # 軀幹上部、骨盆 
                1.5, 1.5,   # 左右軀幹
                1.0, 1.0,   # 上臂
                0.8, 0.8,   # 前臂
                1.2, 1.2,   # 大腿
                0.8, 0.8    # 小腿
            ], device=student_preds.device)
            
            a_idx, b_idx = skeleton[:, 0], skeleton[:, 1]
            
            # 提取坐標
            s_a_x, s_a_y = s_x[:, a_idx], s_y[:, a_idx]
            s_b_x, s_b_y = s_x[:, b_idx], s_y[:, b_idx]
            
            t_a_x, t_a_y = t_x[:, a_idx], t_y[:, a_idx]
            t_b_x, t_b_y = t_x[:, b_idx], t_y[:, b_idx]
            
            # 計算骨架長度
            s_bone_len = torch.sqrt((s_a_x - s_b_x)**2 + (s_a_y - s_b_y)**2 + epsilon)
            t_bone_len = torch.sqrt((t_a_x - t_b_x)**2 + (t_a_y - t_b_y)**2 + epsilon)
            
            # 【新增】應用數值穩定性檢查
            bone_len_diff = s_bone_len - t_bone_len
            bone_len_diff = torch.where(torch.abs(bone_len_diff) < tolerance, 
                                    torch.zeros_like(bone_len_diff), bone_len_diff)
            
            # 【改進6】應用骨架權重和置信度掩碼
            # 為每個骨架提取關節點的平均置信度
            a_conf = teacher_conf_mask[:, a_idx]
            b_conf = teacher_conf_mask[:, b_idx]
            bone_conf = (a_conf + b_conf) / 2.0
            
            # 安全檢查
            if torch.isnan(s_bone_len).any() or torch.isnan(t_bone_len).any():
                structure_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            else:
                # 計算加權骨架損失
                bone_diff = bone_len_diff ** 2
                
                # 應用置信度掩碼和骨架權重
                weighted_bone_diff = bone_diff * bone_conf * skeleton_weights.unsqueeze(0).unsqueeze(-1)
                
                # 安全平均
                total_bone_weight = (bone_conf * skeleton_weights.unsqueeze(0).unsqueeze(-1)).sum() + epsilon
                structure_loss = weighted_bone_diff.sum() / total_bone_weight

            if torch.isnan(structure_loss) or torch.isinf(structure_loss):
                print("警告: 結構損失為NaN或Inf")
                structure_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
          
            # 添加相對位置和方向約束
            # 使用上面已經計算的a_idx和b_idx
            rel_pos_loss = torch.tensor(0.0, device=student_preds.device)

            for pair_idx in range(len(a_idx)):
                i, j = a_idx[pair_idx], b_idx[pair_idx]
                
                # 計算相對位置向量
                s_vec_x = s_x[:, i].unsqueeze(-1) - s_x[:, j].unsqueeze(-1)
                s_vec_y = s_y[:, i].unsqueeze(-1) - s_y[:, j].unsqueeze(-1)
                
                t_vec_x = t_x[:, i].unsqueeze(-1) - t_x[:, j].unsqueeze(-1)
                t_vec_y = t_y[:, i].unsqueeze(-1) - t_y[:, j].unsqueeze(-1)
                
                # 向量長度
                s_norm = torch.sqrt(s_vec_x**2 + s_vec_y**2 + epsilon)
                t_norm = torch.sqrt(t_vec_x**2 + t_vec_y**2 + epsilon)
                
                # 歸一化向量
                s_dir_x = s_vec_x / s_norm
                s_dir_y = s_vec_y / s_norm
                t_dir_x = t_vec_x / t_norm
                t_dir_y = t_vec_y / t_norm
                
                # 方向差異
                dir_diff = 1.0 - (s_dir_x * t_dir_x + s_dir_y * t_dir_y)
                
                # 加權
                weighted_dir_diff = dir_diff * bone_conf[:, pair_idx].unsqueeze(-1) * skeleton_weights[pair_idx]
                rel_pos_loss = rel_pos_loss + weighted_dir_diff.sum()

            # 安全平均
            rel_pos_loss = rel_pos_loss / (total_bone_weight + epsilon)

            if torch.isnan(rel_pos_loss) or torch.isinf(rel_pos_loss):
                print("警告: 相對位置損失為NaN或Inf")
                rel_pos_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
          
            # 4. 【改進7】使用KL散度的置信度損失，加入溫度調節
            # 【新增】數值穩定性檢查
            # 如果概率非常接近，視為相同
            prob_exact_match = torch.abs(t_prob - s_prob) < tolerance
            if prob_exact_match.any():
                s_prob = torch.where(prob_exact_match, t_prob, s_prob)
            
            # 使用軟目標KL散度，這是知識蒸餾的核心
            # log(s_prob)計算前先加epsilon防止log(0)
            kl_loss = t_prob * torch.log(t_prob + epsilon) - t_prob * torch.log(s_prob + epsilon)
            
            # 確保損失有效
            kl_loss = torch.where(torch.isnan(kl_loss) | torch.isinf(kl_loss), 
                                torch.zeros_like(kl_loss), kl_loss)
            
            # 平方項模擬硬目標MSE
            mse_loss = ((s_conf - t_conf) ** 2)
            
            # 【改進8】結合KL散度和MSE的混合置信度損失
            conf_loss = (kl_loss.mean() * T * T * 0.5) + (mse_loss.mean() * 0.5)

            if torch.isnan(conf_loss) or torch.isinf(conf_loss):
                print("警告: 置信度損失為NaN或Inf")
                conf_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
         

            # 2. 引入坐標一致性損失
            # 為每個關鍵點計算與鄰近點的相對位置關係
            keypoint_topology = [
                # 頭部連接
                (0, 1), (0, 2), (1, 3), (2, 4),
                # 軀幹主體
                (5, 6), (5, 11), (6, 12), (11, 12),
                # 上肢
                (5, 7), (7, 9), (6, 8), (8, 10),
                # 下肢
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            consistency_loss = torch.tensor(0.0, device=x_diff.device)
            for src, dst in keypoint_topology:
                # 學生模型的相對向量
                s_rel_x = s_x[:, src].unsqueeze(-1) - s_x[:, dst].unsqueeze(-1)
                s_rel_y = s_y[:, src].unsqueeze(-1) - s_y[:, dst].unsqueeze(-1)
                
                # 教師模型的相對向量
                t_rel_x = t_x[:, src].unsqueeze(-1) - t_x[:, dst].unsqueeze(-1)
                t_rel_y = t_y[:, src].unsqueeze(-1) - t_y[:, dst].unsqueeze(-1)
                
                # 計算夾角一致性
                s_norm = torch.sqrt(s_rel_x**2 + s_rel_y**2 + epsilon)
                t_norm = torch.sqrt(t_rel_x**2 + t_rel_y**2 + epsilon)
                
                s_unit_x, s_unit_y = s_rel_x / s_norm, s_rel_y / s_norm
                t_unit_x, t_unit_y = t_rel_x / t_norm, t_rel_y / t_norm
                
                # 1 - cos(角度)
                angle_diff = 1.0 - (s_unit_x * t_unit_x + s_unit_y * t_unit_y)
                
                # 長度比例一致性 (log比例損失) - 更全面的雙向比例
                length_ratio = s_norm / (t_norm + epsilon)
                inv_length_ratio = t_norm / (s_norm + epsilon)
                
                # 取較大值，確保比例差異被雙向捕捉
                length_diff = torch.maximum(
                    torch.abs(torch.log(length_ratio + epsilon)),
                    torch.abs(torch.log(inv_length_ratio + epsilon))
                )
                
                # 組合角度和長度一致性
                pair_conf = (teacher_conf_mask[:, src] + teacher_conf_mask[:, dst]) / 2.0
                weighted_angle_diff = angle_diff * pair_conf.unsqueeze(-1)
                weighted_length_diff = length_diff * pair_conf.unsqueeze(-1)
                
                consistency_loss = consistency_loss + weighted_angle_diff.sum() + weighted_length_diff.sum()
            
            # 安全平均
            total_pairs = len(keypoint_topology)
            consistency_loss = consistency_loss / (total_pairs * total_weight + epsilon)

            if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
                print("警告: 一致性損失為NaN或Inf")
                consistency_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 3. 添加姿態結構空間損失 - 修訂版
            t_pose = torch.cat([t_x.unsqueeze(-1), t_y.unsqueeze(-1)], dim=2)  # [B, 17, 2, grid]
            s_pose = torch.cat([s_x.unsqueeze(-1), s_y.unsqueeze(-1)], dim=2)  # [B, 17, 2, grid]

            # 計算每個姿態的中心
            t_center = t_pose.mean(dim=1, keepdim=True)  # [B, 1, 2, grid]
            s_center = s_pose.mean(dim=1, keepdim=True)  # [B, 1, 2, grid]

            # 中心化坐標
            t_centered = t_pose - t_center  # [B, 17, 2, grid]
            s_centered = s_pose - s_center  # [B, 17, 2, grid]

            # 計算身體尺度 - 使用標準差
            t_std = torch.std(t_centered, dim=1, unbiased=False, keepdim=True) + epsilon  # [B, 1, 2, grid]
            s_std = torch.std(s_centered, dim=1, unbiased=False, keepdim=True) + epsilon  # [B, 1, 2, grid]

            # 標準化坐標
            t_normalized = t_centered / t_std  # [B, 17, 2, grid]
            s_normalized = s_centered / s_std  # [B, 17, 2, grid]

            # 計算形狀差異，使用適當的縮放因子
            shape_diff = (t_normalized - s_normalized)**2
            pose_structure_loss = shape_diff.mean() * 5.0
            pose_structure_loss = torch.clamp(pose_structure_loss, 0.0, 2.0)

            if torch.isnan(pose_structure_loss) or torch.isinf(pose_structure_loss):
                print("警告: 姿態結構損失為NaN或Inf")
                pose_structure_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 4. 修改損失權重 - 動態調整各項權重
            # 加速早期訓練
            if current_epoch < 5:
                # 前5個epoch重視坐標精度
                coord_weight = 3.0
                structure_weight = 1.5
                consistency_weight = 2.0
                pose_structure_weight = 2.0
                conf_weight = 0.5
                rel_pos_weight = 1.5
            else:
                # 後期更均衡
                coord_weight = 2.5
                structure_weight = 1.2
                consistency_weight = 1.5
                pose_structure_weight = 1.0
                conf_weight = 0.5
                rel_pos_weight = 1.0

            # 組合損失前再次進行數值穩定性檢查
            # 為每個損失項添加上限
            coord_weight = min(coord_weight, 2.0)
            structure_weight = min(structure_weight, 1.0)
            consistency_weight = min(consistency_weight, 1.0)
            pose_structure_weight = min(pose_structure_weight, 0.5)  # 降低權重
            
            # 組合所有損失
            pred_loss = (
                coord_weight * coord_loss +
                structure_weight * structure_loss +
                consistency_weight * consistency_loss +
                0.005 * pose_structure_weight * pose_structure_loss +  # 從 0.01 降至 0.005
                conf_weight * conf_loss +
                rel_pos_weight * rel_pos_loss +
                1.5 * y_special_loss  # 從1.0提高到1.5
            )
            
            # 5. 【改進9】自適應損失組合
            # 使用教師模型的平均置信度來調整損失權重
            avg_teacher_conf = teacher_conf_mask.mean().clamp(0.1, 0.9)
            
            # 【修改】自適應權重 - 初期提高特徵權重
            if current_epoch < 3:
                # 前3個epoch給較高特徵權重
                base_feat_weight = feat_weight * 0.3  # 從0.15提高到0.3
            else:
                # 原有邏輯
                base_feat_weight = feat_weight * (0.15 + 0.85 * min(1.0, (current_epoch - 2) / (total_epochs - 2) * 1.5))
                
            # 低置信度時更信任特徵蒸餾，高置信度時更信任輸出蒸餾
            adaptive_feat_weight = base_feat_weight * (1.0 - avg_teacher_conf.item())
            adaptive_pred_weight = pred_weight * avg_teacher_conf.item()

            # 【初期階段特別處理】首個epoch大幅降低特徵損失影響
            if current_epoch == 0:
                feat_loss = feat_loss * 0.05  # 大幅降低特徵損失
            
            # 組合所有損失
            total_loss = min(
                adaptive_feat_weight * feat_loss + adaptive_pred_weight * pred_loss,
                10.0  # 絕對上限
            )

            # 添加梯度裁剪，防止梯度爆炸
            if total_loss > 10.0:
                total_loss = 10.0 + torch.log(1.0 + (total_loss - 10.0))  # 軟上限
            
            # 【新增】早期停止損失計算的額外檢查
            # 如果損失已經很小，進一步減少計算量
            if total_loss < 1e-6:
                # 設置一個非零但極小的損失值，確保梯度不會完全消失
                total_loss = torch.tensor(1e-6, device=student_preds.device, requires_grad=True)
            
            # 最終安全檢查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Warning: 損失計算出現NaN或Inf")
                return torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 記錄訓練階段信息和損失值
            self.last_loss_values = {
                "coord_loss": float(coord_loss.item()) if not torch.isnan(coord_loss) else 0.0,
                "structure_loss": float(structure_loss.item()) if not torch.isnan(structure_loss) else 0.0,
                "conf_loss": float(conf_loss.item()) if not torch.isnan(conf_loss) else 0.0,
                "rel_pos_loss": float(rel_pos_loss.item()) if not torch.isnan(rel_pos_loss) else 0.0,
                "feat_loss": float(feat_loss.item()) if not torch.isnan(feat_loss) else 0.0,
                "total_loss": float(total_loss.item()) if not torch.isnan(total_loss) else 0.0,
                "teacher_conf": float(avg_teacher_conf.item()),
                "temperature": float(T),
                "progress": float(progress),
                "epoch": int(current_epoch)
            }

            # 記錄更多損失指標
            self.last_loss_values.update({
                "consistency_loss": float(consistency_loss.item()) if not torch.isnan(consistency_loss) else 0.0,
                "pose_structure_loss": float(pose_structure_loss.item()) if not torch.isnan(pose_structure_loss) else 0.0,
                "precise_coord_ratio": float(
                    (tiny_diff_mask_x.sum() + tiny_diff_mask_y.sum()) / 
                    (tiny_diff_mask_x.numel() + tiny_diff_mask_y.numel() + epsilon)
                )
            })
            
            # 輸出日誌信息
            if hasattr(self, 'model') and hasattr(self.model, 'epoch') and (
            current_epoch % 1 == 0) and is_first_batch_in_epoch:
                print(f"\n--- Loss Values (Epoch {current_epoch}/{total_epochs}, T={T:.2f}) ---")
                print(f"coord_loss: {float(coord_loss.item()):.4f}")
                print(f"structure_loss: {float(structure_loss.item()):.4f}")
                print(f"conf_loss: {float(conf_loss.item()):.4f}")
                print(f"rel_pos_loss: {float(rel_pos_loss.item()):.4f}")
                print(f"feat_loss: {float(feat_loss.item()):.4f}" + (f" (raw: {float(feat_loss.item() / (0.05 if current_epoch == 0 else 1.0)):.4f})" if current_epoch == 0 else ""))
                print(f"total_loss: {float(total_loss.item()):.4f}")
                print(f"teacher_conf: {float(avg_teacher_conf.item()):.4f}")
                print(f"feat_weight: {adaptive_feat_weight:.4f}, pred_weight: {adaptive_pred_weight:.4f}")

                print(f"consistency_loss: {float(consistency_loss.item()):.4f}")
                print(f"pose_structure_loss: {float(pose_structure_loss.item()):.4f}")
                
                # 添加精確定位統計
                precise_x_ratio = tiny_diff_mask_x.float().mean().item()
                precise_y_ratio = tiny_diff_mask_y.float().mean().item()
                print(f"極小誤差比例 (x<0.001): {precise_x_ratio:.4f}, (y<0.001): {precise_y_ratio:.4f}")
                print(f"小誤差比例 (x<0.005): {small_diff_mask_x.float().mean().item():.4f}, (y<0.005): {small_diff_mask_y.float().mean().item():.4f}")
                
                # 計算坐標誤差分布
                x_diff_mean = x_diff.abs().mean().item()
                y_diff_mean = y_diff.abs().mean().item()
                x_diff_std = x_diff.abs().std().item()
                y_diff_std = y_diff.abs().std().item()
                print(f"坐標誤差統計 - X軸: 均值={x_diff_mean:.6f}, 標準差={x_diff_std:.6f}")
                print(f"坐標誤差統計 - Y軸: 均值={y_diff_mean:.6f}, 標準差={y_diff_std:.6f}")
                
                # 調用精度統計函數
                self.log_precision_stats(x_diff.flatten(), y_diff.flatten())

                # 添加微小誤差統計
                micro_x_ratio = micro_tiny_diff_mask_x.float().mean().item()
                micro_y_ratio = micro_tiny_diff_mask_y.float().mean().item()
                print(f"超微小誤差比例 (x<0.0001): {micro_x_ratio:.4f}, (y<0.0001): {micro_y_ratio:.4f}")
                
                # Y軸與X軸誤差比例
                if x_diff_mean > 0:
                    y_x_ratio = y_diff_mean / x_diff_mean
                    print(f"Y/X軸誤差比例: {y_x_ratio:.4f}")
            
            return total_loss
            
        except Exception as e:
            print(f"蒸餾損失計算異常: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印詳細錯誤堆棧
            return torch.tensor(0.1, device=student_outputs[1].device, requires_grad=True)

    def pose_distillation_loss_enhanced2(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.3, pred_weight=0.6):
        """
        增強版姿態蒸餾損失函數：
        1. 重新加入溫度調節機制
        2. 自適應特徵層權重
        3. 置信度加權的坐標損失
        4. 骨架重要性加權
        5. 更加穩健的損失組合
        6. 動態特徵層選擇策略
        7. 添加數值穩定性檢查
        8. 早期停止機制 
        """
        # 降低容忍度，使模型對更小的誤差敏感
        tolerance = 1e-12  # 從1e-10進一步降低到1e-12
        epsilon = 1e-12    # 同步降低

        try:
            # 【新增】早期停止機制
            # 如果上一批次損失已經非常小，可以早期返回
            if hasattr(self, 'last_loss_values') and isinstance(self.last_loss_values, dict):
                last_total_loss = self.last_loss_values.get('total_loss', 1.0)
                if last_total_loss < 1e-5:
                    # 損失已經非常小，可以跳過詳細計算
                    return torch.tensor(last_total_loss, device=student_outputs[1].device, requires_grad=True)
            
            # 直接解包輸出
            student_features, student_preds = student_outputs[0], student_outputs[1]
            teacher_features, teacher_preds = teacher_outputs[0], teacher_outputs[1]
            
            # 快速檢查NaN
            if torch.isnan(student_preds).any() or torch.isnan(teacher_preds).any():
                return torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 一次性重塑並提取所有需要的張量
            batch_size = student_preds.shape[0]
            s_preds = student_preds.reshape(batch_size, 17, 3, -1)
            t_preds = teacher_preds.reshape(batch_size, 17, 3, -1)
            
            s_x, s_y, s_conf = s_preds[:, :, 0], s_preds[:, :, 1], s_preds[:, :, 2]
            t_x, t_y, t_conf = t_preds[:, :, 0], t_preds[:, :, 1], t_preds[:, :, 2]
            
            # 【新增】數值穩定性檢查 - 對坐標差異應用容忍閾值
            # 將非常接近的坐標視為相同，減少浮點誤差影響
            x_exact_match = torch.abs(s_x - t_x) < tolerance
            y_exact_match = torch.abs(s_y - t_y) < tolerance
            if x_exact_match.any() or y_exact_match.any():
                s_x = torch.where(x_exact_match, t_x, s_x)
                s_y = torch.where(y_exact_match, t_y, s_y)
            
            # 使用向量化操作進行限幅
            s_conf = s_conf.clamp(-50.0, 50.0)
            t_conf = t_conf.clamp(-50.0, 50.0)
            
            # 【改進1】重新加入溫度縮放的軟目標 
            t_conf_T = t_conf / T
            s_conf_T = s_conf / T
            
            # 对温度缩放后的置信度进行裁剪
            t_conf_T = torch.clamp(t_conf_T, min=-15.0, max=15.0)
            s_conf_T = torch.clamp(s_conf_T, min=-15.0, max=15.0)

            t_prob = torch.sigmoid(t_conf_T)
            s_prob = torch.sigmoid(s_conf_T)

            # 确保概率不会太接近0或1
            t_prob = torch.clamp(t_prob, min=epsilon, max=1.0-epsilon)
            s_prob = torch.clamp(s_prob, min=epsilon, max=1.0-epsilon)
            
            # 計算教師模型的置信度掩碼，用於後續加權
            teacher_conf_mask = torch.sigmoid(t_conf)
            
            # 獲取當前epoch和訓練進度信息
            current_epoch = getattr(self.model, 'epoch', 0) if hasattr(self, 'model') else 0
            total_epochs = getattr(self.model, 'epochs', 100) if hasattr(self, 'model') else 100
            is_first_batch_in_epoch = getattr(self.model, 'is_first_batch_in_epoch', False) if hasattr(self, 'model') else False

            # 計算訓練進度比例
            progress = calculate_progress(current_epoch, total_epochs)
            
            # 1. 特徵蒸餾損失 - 加入自適應權重與動態選擇策略
            max_len = min(len(student_features), len(teacher_features))
            
            # 【新增】檢查是否需要進行詳細的特徵蒸餾
            # 當進度很高且損失已經很小時，可以簡化特徵蒸餾計算
            simplified_feature_distill = False
            if hasattr(self, 'last_loss_values') and progress > 0.8:
                last_feat_loss = self.last_loss_values.get('feat_loss', 1.0)
                if last_feat_loss < 0.01:  # 特徵損失已經很小
                    simplified_feature_distill = True

            # 在第6-7個epoch擴展到所有特徵層
            if current_epoch < 6:
                indices = [0, 1] if current_epoch >= 3 else [0]
            else:
                indices = [0, 1, 2]  # 使用所有層
            
            # 確保索引唯一且有序
            indices = sorted(list(set(indices)))
            
            # 【改進】動態特徵層權重策略
            # 訓練初期淺層和深層權重相近，後期深層權重更大
            if progress < 0.5:
                # 初期階段 - 權重差異較小
                min_weight = 0.8
                max_weight = 1.2
            else:
                # 後期階段 - 權重差異加大
                min_weight = 0.5
                max_weight = 1.5
            
            # 根據層的深度生成權重，深度越深權重越大
            layer_weights = []
            for idx in indices:
                # 相對深度 (0到1之間)
                rel_depth = idx / (max_len - 1) if max_len > 1 else 0.5
                # 線性插值計算權重
                weight = min_weight + rel_depth * (max_weight - min_weight)
                layer_weights.append(weight)
            
            # 轉換為張量
            layer_importance = torch.tensor(layer_weights, device=student_preds.device)
            
            # 調試信息 (可選)
            if current_epoch % 1 == 0 and is_first_batch_in_epoch:
                print(f"\nEpoch {current_epoch}/{total_epochs} (Progress: {progress:.2f})")
                print(f"Dynamic feature layers: {indices}")
                print(f"Layer weights: {layer_weights}")
                if simplified_feature_distill:
                    print("Using simplified feature distillation")
            
            if len(indices) > 0:
                # 使用索引張量獲取特徵
                s_feats = [student_features[i] for i in indices]
                t_feats = [teacher_features[i] for i in indices]
                
                # 計算特徵損失
                feat_losses = torch.zeros(len(indices), device=student_preds.device)
                valid_feats = torch.ones(len(indices), device=student_preds.device)
                
                for i, (s_f, t_f) in enumerate(zip(s_feats, t_feats)):
                    if s_f.shape != t_f.shape:
                        try:
                            t_f = F.interpolate(t_f, size=s_f.shape[-2:], mode='bilinear', align_corners=False)
                        except:
                            valid_feats[i] = 0
                            continue
                    
                    if torch.isnan(s_f).any() or torch.isnan(t_f).any():
                        valid_feats[i] = 0
                        continue
                    
                    # 【新增】計算自適應L2距離時應用數值穩定性檢查
                    feat_diff = s_f - t_f
                    
                    # 【新增】將微小差異視為零，減少浮點誤差影響
                    feat_diff = torch.where(torch.abs(feat_diff) < tolerance, 
                                        torch.zeros_like(feat_diff), feat_diff)
                    
                    # 【新增】梯度尺度調整 - 為深層特徵應用梯度尺度衰減
                    grad_scale = 1.0 / (2 ** (i * 0.5))  # 深層特徵梯度尺度平滑衰減
                    feat_losses[i] = (feat_diff ** 2).mean() * grad_scale
                
                # 應用層重要性權重
                weighted_sum = (feat_losses * valid_feats * layer_importance).sum()
                weight_sum = (valid_feats * layer_importance).sum() + epsilon
                feat_loss = weighted_sum / weight_sum
            else:
                feat_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 2. 【改進4】置信度加權的坐標損失
            # 高置信度區域權重更大
            coord_weights = teacher_conf_mask.unsqueeze(-1)  # [B, 17, 1, grid]
            
            x_diff = (s_x - t_x).unsqueeze(-1)  # [B, 17, 1, grid]
            y_diff = (s_y - t_y).unsqueeze(-1)  # [B, 17, 1, grid]
            # 添加裁剪以避免极端值
            x_diff = torch.clamp(x_diff, min=-5.0, max=5.0)
            y_diff = torch.clamp(y_diff, min=-5.0, max=5.0)
            
            # 【新增】應用數值穩定性檢查
            x_diff = torch.where(torch.abs(x_diff) < tolerance, torch.zeros_like(x_diff), x_diff)
            y_diff = torch.where(torch.abs(y_diff) < tolerance, torch.zeros_like(y_diff), y_diff)
            
            # 【修改】Y軸誤差權重加大1.5倍
            weighted_x_diff = (x_diff ** 2) * coord_weights

            # 根據當前Y/X軸誤差比例動態調整Y軸權重
            # 在計算Y軸損失時添加這段代碼
            x_diff_mean = x_diff.abs().mean().item()
            y_diff_mean = y_diff.abs().mean().item()
            y_x_ratio = y_diff_mean / (x_diff_mean + epsilon)
            y_axis_factor = min(3.0, max(1.5, y_x_ratio * 0.8))  # 將Y軸權重因子限制在1.5-3.0之間

            # 更新Y軸差異權重
            weighted_y_diff = (y_diff ** 2) * coord_weights * y_axis_factor  # 使用動態因子而不是固定的2.0
            
            # 避免分母為零
            total_weight = coord_weights.sum() + epsilon
            
            # 1. 更精細的坐標損失階梯 - 添加進度自適應權重
            # 隨著訓練進行，精細部分權重逐漸提高
            if progress < 0.3:
                # 前30%訓練階段 - 從低權重開始逐漸增加
                tiny_weight_x = 12.0 + progress * 30  # 起始12，最高21
                small_weight_x = 6.0 + progress * 15  # 起始6，最高10.5
                
                # Y軸權重始終高於X軸
                tiny_weight_y = tiny_weight_x * 1.5
                small_weight_y = small_weight_x * 1.5
            else:
                # 後70%訓練階段 - 保持較高權重
                tiny_weight_x = 21.0
                small_weight_x = 10.5
                tiny_weight_y = 31.5  # 1.5倍於X軸
                small_weight_y = 15.75 # 1.5倍於X軸

            # 在添加更細的階梯部分之前添加這些代碼
            # 添加超微小差異層級
            micro_tiny_diff_mask_x = (x_diff**2 < 0.0001)  # 超微小差異(0.0001)
            micro_tiny_diff_mask_y = (y_diff**2 < 0.0001)
            
            # 添加更細的階梯 - 極微小誤差層級
            ultra_tiny_diff_mask_x = (x_diff**2 < 0.0005) & ~micro_tiny_diff_mask_x  # 極微小差異(0.0005)
            tiny_diff_mask_x = (x_diff**2 < 0.001) & ~ultra_tiny_diff_mask_x  # 極小差異(0.001)
            small_diff_mask_x = (x_diff**2 < 0.005) & ~tiny_diff_mask_x & ~ultra_tiny_diff_mask_x  # 小差異(0.005)
            medium_diff_mask_x = (x_diff**2 < 0.02) & ~small_diff_mask_x & ~tiny_diff_mask_x & ~ultra_tiny_diff_mask_x  # 中等差異
            
            ultra_tiny_diff_mask_y = (y_diff**2 < 0.0005) & ~micro_tiny_diff_mask_y
            tiny_diff_mask_y = (y_diff**2 < 0.001) & ~ultra_tiny_diff_mask_y
            small_diff_mask_y = (y_diff**2 < 0.005) & ~tiny_diff_mask_y & ~ultra_tiny_diff_mask_y
            medium_diff_mask_y = (y_diff**2 < 0.02) & ~small_diff_mask_y & ~tiny_diff_mask_y & ~ultra_tiny_diff_mask_y

            # 更新 X 軸精確損失計算 - 增強 micro_tiny 和 ultra_tiny 權重
            precise_x_loss = (
                torch.where(micro_tiny_diff_mask_x, torch.abs(x_diff) * tiny_weight_x * 4.0, torch.zeros_like(x_diff)) +  # 從2.5提高到4.0
                torch.where(ultra_tiny_diff_mask_x, torch.abs(x_diff) * tiny_weight_x * 2.5, torch.zeros_like(x_diff)) +  # 從1.8提高到2.5
                torch.where(tiny_diff_mask_x, torch.abs(x_diff) * tiny_weight_x * 1.2, torch.zeros_like(x_diff)) +        # 增加1.2倍
                torch.where(small_diff_mask_x, torch.abs(x_diff) * small_weight_x, torch.zeros_like(x_diff)) +
                torch.where(medium_diff_mask_x, torch.abs(x_diff) * 2.0, torch.zeros_like(x_diff))
            )

            # 更新 Y 軸精確損失計算 - 大幅提高超精細區域權重
            precise_y_loss = (
                torch.where(micro_tiny_diff_mask_y, torch.abs(y_diff) * tiny_weight_y * 8.0, torch.zeros_like(y_diff)) +  # 從5.0提高到8.0
                torch.where(ultra_tiny_diff_mask_y, torch.abs(y_diff) * tiny_weight_y * 5.0, torch.zeros_like(y_diff)) +  # 從3.5提高到5.0
                torch.where(tiny_diff_mask_y, torch.abs(y_diff) * tiny_weight_y * 2.0, torch.zeros_like(y_diff)) +        # 從1.5提高到2.0
                torch.where(small_diff_mask_y, torch.abs(y_diff) * small_weight_y * 1.5, torch.zeros_like(y_diff)) +      # 從1.2提高到1.5
                torch.where(medium_diff_mask_y, torch.abs(y_diff) * 4.0, torch.zeros_like(y_diff))                        # 從3.5提高到4.0
            )

            # 添加專門針對 mAP50-95 的優化 - 新增
            # 定義更細的誤差區間，對應不同IoU閾值
            map75_mask_x = (x_diff**2 < 0.002)  # 對應約IoU 0.75的精度要求
            map80_mask_x = (x_diff**2 < 0.0015) # 對應約IoU 0.80的精度要求
            map90_mask_x = (x_diff**2 < 0.001)  # 對應約IoU 0.90的精度要求
            map95_mask_x = (x_diff**2 < 0.0005) # 對應約IoU 0.95的精度要求

            map75_mask_y = (y_diff**2 < 0.002)
            map80_mask_y = (y_diff**2 < 0.0015)
            map90_mask_y = (y_diff**2 < 0.001)
            map95_mask_y = (y_diff**2 < 0.0005)

            # 添加專門的mAP梯度損失
            map_grad_x_loss = (
                torch.where(map95_mask_x, torch.abs(x_diff) * 10.0, torch.zeros_like(x_diff)) +  # 最高精度區域
                torch.where(map90_mask_x & ~map95_mask_x, torch.abs(x_diff) * 7.0, torch.zeros_like(x_diff)) +
                torch.where(map80_mask_x & ~map90_mask_x, torch.abs(x_diff) * 5.0, torch.zeros_like(x_diff)) +
                torch.where(map75_mask_x & ~map80_mask_x, torch.abs(x_diff) * 3.0, torch.zeros_like(x_diff))
            )

            map_grad_y_loss = (
                torch.where(map95_mask_y, torch.abs(y_diff) * 15.0, torch.zeros_like(y_diff)) +  # Y軸更高權重
                torch.where(map90_mask_y & ~map95_mask_y, torch.abs(y_diff) * 10.0, torch.zeros_like(y_diff)) +
                torch.where(map80_mask_y & ~map90_mask_y, torch.abs(y_diff) * 7.0, torch.zeros_like(y_diff)) +
                torch.where(map75_mask_y & ~map80_mask_y, torch.abs(y_diff) * 4.0, torch.zeros_like(y_diff))
            )

            # 添加進權重精確損失
            weighted_map_grad_x = map_grad_x_loss * coord_weights
            weighted_map_grad_y = map_grad_y_loss * coord_weights
            
            # 將精確定位損失添加到加權損失中
            weighted_precise_x = precise_x_loss * coord_weights
            weighted_precise_y = precise_y_loss * coord_weights

            # 根據當前精度比例自動調整精度權重
            precise_x_ratio = tiny_diff_mask_x.float().mean().item()
            precise_y_ratio = tiny_diff_mask_y.float().mean().item()
            current_y_to_x_precise_ratio = precise_y_ratio / (precise_x_ratio + epsilon)
            target_ratio = 1.0  # 目標Y/X精度比例

            # 如果Y軸精度比例低於X軸，增加Y軸權重
            if current_y_to_x_precise_ratio < target_ratio:
                y_precise_factor = 1.0 + (target_ratio - current_y_to_x_precise_ratio) * 0.5
                weighted_precise_y = weighted_precise_y * y_precise_factor

            # 修改coord_loss計算，加入新的mAP梯度損失
            coord_loss = (weighted_x_diff.sum() + weighted_y_diff.sum() + 
                        weighted_precise_x.sum() + weighted_precise_y.sum() +
                        weighted_map_grad_x.sum() + weighted_map_grad_y.sum()) / total_weight
            
            # 在計算 coord_loss 之後，pred_loss 之前添加
            # 添加針對 Y 軸的特殊處理
            # 針對垂直重要關鍵點的額外損失
            vertical_important_keypoints = [0, 5, 6, 11, 12, 15, 16]  # 頭頂、肩膀、髖部、腳踝
            y_special_loss = torch.tensor(0.0, device=student_preds.device)

            # 提取這些關鍵點並應用特殊權重
            for kp_idx in vertical_important_keypoints:
                # 取絕對差異
                y_kp_diff = torch.abs(s_y[:, kp_idx] - t_y[:, kp_idx]).unsqueeze(-1)
                # 權重 - 基於教師置信度
                y_kp_weight = teacher_conf_mask[:, kp_idx].unsqueeze(-1) * 1.5
                
                # 計算加權差異
                weighted_y_kp_diff = y_kp_diff * y_kp_weight
                y_special_loss = y_special_loss + weighted_y_kp_diff.sum() / (y_kp_weight.sum() + epsilon)

            # 在 Y 軸特別處理中加入精細化損失
            # 更新Y軸特殊處理的係數
            high_conf_mask = teacher_conf_mask > 0.6  # 降低閾值從0.7到0.6
            if high_conf_mask.any():
                high_conf_y = s_y[high_conf_mask]
                high_conf_t_y = t_y[high_conf_mask]
                if high_conf_y.numel() > 0:
                    # 計算高置信度點的Y軸差異
                    hc_y_diff = torch.abs(high_conf_y - high_conf_t_y)
                    
                    # 特別關注小差異 - 提高權重
                    tiny_hc_mask = hc_y_diff < 0.001
                    small_hc_mask = (hc_y_diff >= 0.001) & (hc_y_diff < 0.005)
                    
                    # 應用更高權重
                    if tiny_hc_mask.any():
                        y_special_loss = y_special_loss + hc_y_diff[tiny_hc_mask].sum() * 8.0 / (tiny_hc_mask.sum() + epsilon)  # 從5.0提高到8.0
                    if small_hc_mask.any():
                        y_special_loss = y_special_loss + hc_y_diff[small_hc_mask].sum() * 4.0 / (small_hc_mask.sum() + epsilon)  # 從2.0提高到4.0
            
            # 3. 結構損失 - 優化骨架選擇和權重
            # 【改進5】擴展骨架集合並按重要性加權
            skeleton = torch.tensor([
                [5, 6],    # 左肩-右肩 (軀幹上部)
                [11, 12],  # 左髖-右髖 (骨盆)
                [5, 11],   # 左肩-左髖 (左軀幹)
                [6, 12],   # 右肩-右髖 (右軀幹)
                [5, 7],    # 左肩-左肘 (左上臂)
                [6, 8],    # 右肩-右肘 (右上臂)
                [7, 9],    # 左肘-左腕 (左前臂)
                [8, 10],   # 右肘-右腕 (右前臂)
                [11, 13],  # 左髖-左膝 (左大腿)
                [12, 14],  # 右髖-右膝 (右大腿)
                [13, 15],  # 左膝-左踝 (左小腿)
                [14, 16],  # 右膝-右踝 (右小腿)
            ], device=student_preds.device)
            
            # 骨架重要性權重 - 軀幹與核心部位權重更高
            skeleton_weights = torch.tensor([
                3.5, 3.5,   # 軀幹上部、骨盆 (從2.5提高到3.5)
                2.5, 2.5,   # 左右軀幹 (從1.5提高到2.5)
                1.5, 1.5,   # 上臂 (從1.0提高到1.5)
                1.0, 1.0,   # 前臂 (從0.8提高到1.0)
                2.0, 2.0,   # 大腿 (從1.2提高到2.0)
                1.2, 1.2    # 小腿 (從0.8提高到1.2)
            ], device=student_preds.device)
            
            a_idx, b_idx = skeleton[:, 0], skeleton[:, 1]
            
            # 提取坐標
            s_a_x, s_a_y = s_x[:, a_idx], s_y[:, a_idx]
            s_b_x, s_b_y = s_x[:, b_idx], s_y[:, b_idx]
            
            t_a_x, t_a_y = t_x[:, a_idx], t_y[:, a_idx]
            t_b_x, t_b_y = t_x[:, b_idx], t_y[:, b_idx]
            
            # 計算骨架長度
            # 增加epsilon，确保长度计算稳定
            s_bone_len = torch.sqrt((s_a_x - s_b_x)**2 + (s_a_y - s_b_y)**2 + 1e-5)
            t_bone_len = torch.sqrt((t_a_x - t_b_x)**2 + (t_a_y - t_b_y)**2 + 1e-5)

            # 确保骨架长度不会太小
            s_bone_len = torch.clamp(s_bone_len, min=1e-4)
            t_bone_len = torch.clamp(t_bone_len, min=1e-4)
            
            # 【新增】應用數值穩定性檢查
            bone_len_diff = s_bone_len - t_bone_len
            bone_len_diff = torch.where(torch.abs(bone_len_diff) < tolerance, 
                                    torch.zeros_like(bone_len_diff), bone_len_diff)
            
            # 【改進6】應用骨架權重和置信度掩碼
            # 為每個骨架提取關節點的平均置信度
            a_conf = teacher_conf_mask[:, a_idx]
            b_conf = teacher_conf_mask[:, b_idx]
            bone_conf = (a_conf + b_conf) / 2.0
            
            # 安全檢查
            if torch.isnan(s_bone_len).any() or torch.isnan(t_bone_len).any():
                structure_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            else:
                # 計算加權骨架損失
                bone_diff = bone_len_diff ** 2
                
                # 應用置信度掩碼和骨架權重
                weighted_bone_diff = bone_diff * bone_conf * skeleton_weights.unsqueeze(0).unsqueeze(-1)
                
                # 安全平均
                total_bone_weight = (bone_conf * skeleton_weights.unsqueeze(0).unsqueeze(-1)).sum() + epsilon
                structure_loss = weighted_bone_diff.sum() / total_bone_weight

            rel_pos_loss = torch.tensor(0.0, device=student_preds.device)  # 在循環前初始化
            for pair_idx in range(len(a_idx)):
                i, j = a_idx[pair_idx], b_idx[pair_idx]
                
                # 計算相對位置向量
                s_vec_x = s_x[:, i].unsqueeze(-1) - s_x[:, j].unsqueeze(-1)
                s_vec_y = s_y[:, i].unsqueeze(-1) - s_y[:, j].unsqueeze(-1)
                
                t_vec_x = t_x[:, i].unsqueeze(-1) - t_x[:, j].unsqueeze(-1)
                t_vec_y = t_y[:, i].unsqueeze(-1) - t_y[:, j].unsqueeze(-1)
                
                # 向量長度
                s_norm = torch.sqrt(s_vec_x**2 + s_vec_y**2 + epsilon)
                t_norm = torch.sqrt(t_vec_x**2 + t_vec_y**2 + epsilon)
                
                # 歸一化向量
                # 确保范数足够大
                s_norm = torch.clamp(s_norm, min=1e-5)
                t_norm = torch.clamp(t_norm, min=1e-5)

                s_dir_x = s_vec_x / s_norm
                s_dir_y = s_vec_y / s_norm
                t_dir_x = t_vec_x / t_norm
                t_dir_y = t_vec_y / t_norm
                
                # 方向差異
                dir_diff = 1.0 - (s_dir_x * t_dir_x + s_dir_y * t_dir_y)

                # 添加高精度方向差異處理
                high_precision_dir_mask = dir_diff < 0.05  # 非常小的角度差異
                weighted_dir_diff = torch.where(
                    high_precision_dir_mask,
                    dir_diff * bone_conf[:, pair_idx].unsqueeze(-1) * skeleton_weights[pair_idx] * 2.5, # 高精度區域權重提高
                    dir_diff * bone_conf[:, pair_idx].unsqueeze(-1) * skeleton_weights[pair_idx]
                )
                rel_pos_loss = rel_pos_loss + weighted_dir_diff.sum()

            # 安全平均
            rel_pos_loss = rel_pos_loss / (total_bone_weight + epsilon)

            
            # 4. 【改進7】使用KL散度的置信度損失，加入溫度調節
            # 【新增】數值穩定性檢查
            # 如果概率非常接近，視為相同
            prob_exact_match = torch.abs(t_prob - s_prob) < tolerance
            if prob_exact_match.any():
                s_prob = torch.where(prob_exact_match, t_prob, s_prob)
            
            # 使用軟目標KL散度，這是知識蒸餾的核心
            # log(s_prob)計算前先加epsilon防止log(0)
            # 用更安全的方式计算KL散度
            kl_loss = t_prob * (torch.log(t_prob) - torch.log(s_prob))
            kl_loss = torch.where(t_prob > 0, kl_loss, torch.zeros_like(kl_loss))
            
            # 確保損失有效
            kl_loss = torch.where(torch.isnan(kl_loss) | torch.isinf(kl_loss), 
                                torch.zeros_like(kl_loss), kl_loss)
            
            # 平方項模擬硬目標MSE
            mse_loss = ((s_conf - t_conf) ** 2)
            
            # 【改進8】結合KL散度和MSE的混合置信度損失
            conf_loss = (kl_loss.mean() * T * T * 0.5) + (mse_loss.mean() * 0.5)

            # 2. 引入坐標一致性損失
            # 為每個關鍵點計算與鄰近點的相對位置關係
            keypoint_topology = [
                # 頭部連接
                (0, 1), (0, 2), (1, 3), (2, 4),
                # 軀幹主體
                (5, 6), (5, 11), (6, 12), (11, 12),
                # 上肢
                (5, 7), (7, 9), (6, 8), (8, 10),
                # 下肢
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            consistency_loss = torch.tensor(0.0, device=x_diff.device)
            for src, dst in keypoint_topology:
                # 學生模型的相對向量
                s_rel_x = s_x[:, src].unsqueeze(-1) - s_x[:, dst].unsqueeze(-1)
                s_rel_y = s_y[:, src].unsqueeze(-1) - s_y[:, dst].unsqueeze(-1)
                
                # 教師模型的相對向量
                t_rel_x = t_x[:, src].unsqueeze(-1) - t_x[:, dst].unsqueeze(-1)
                t_rel_y = t_y[:, src].unsqueeze(-1) - t_y[:, dst].unsqueeze(-1)
                
                # 計算夾角一致性
                s_norm = torch.sqrt(s_rel_x**2 + s_rel_y**2 + epsilon)
                t_norm = torch.sqrt(t_rel_x**2 + t_rel_y**2 + epsilon)
                
                s_unit_x, s_unit_y = s_rel_x / s_norm, s_rel_y / s_norm
                t_unit_x, t_unit_y = t_rel_x / t_norm, t_rel_y / t_norm
                
                # 1 - cos(角度)
                angle_diff = 1.0 - (s_unit_x * t_unit_x + s_unit_y * t_unit_y)
                
                # 長度比例一致性 (log比例損失) - 更全面的雙向比例
                length_ratio = s_norm / (t_norm + epsilon)
                inv_length_ratio = t_norm / (s_norm + epsilon)
                
                # 取較大值，確保比例差異被雙向捕捉
                length_diff = torch.maximum(
                    torch.abs(torch.log(length_ratio + epsilon)),
                    torch.abs(torch.log(inv_length_ratio + epsilon))
                )
                
                # 組合角度和長度一致性
                pair_conf = (teacher_conf_mask[:, src] + teacher_conf_mask[:, dst]) / 2.0
                weighted_angle_diff = angle_diff * pair_conf.unsqueeze(-1)
                weighted_length_diff = length_diff * pair_conf.unsqueeze(-1)
                
                consistency_loss = consistency_loss + weighted_angle_diff.sum() + weighted_length_diff.sum()
            
            # 安全平均
            total_pairs = len(keypoint_topology)
            consistency_loss = consistency_loss / (total_pairs * total_weight + epsilon)

            if torch.isnan(consistency_loss) or torch.isinf(consistency_loss):
                print("警告: 一致性損失為NaN或Inf")
                consistency_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 3. 添加姿態結構空間損失 - 修訂版
            t_pose = torch.cat([t_x.unsqueeze(-1), t_y.unsqueeze(-1)], dim=2)  # [B, 17, 2, grid]
            s_pose = torch.cat([s_x.unsqueeze(-1), s_y.unsqueeze(-1)], dim=2)  # [B, 17, 2, grid]

            # 計算每個姿態的中心
            t_center = t_pose.mean(dim=1, keepdim=True)  # [B, 1, 2, grid]
            s_center = s_pose.mean(dim=1, keepdim=True)  # [B, 1, 2, grid]

            # 中心化坐標
            t_centered = t_pose - t_center  # [B, 17, 2, grid]
            s_centered = s_pose - s_center  # [B, 17, 2, grid]

            # 計算身體尺度 - 使用標準差
            # 提高标准差计算的稳定性
            t_std = torch.std(t_centered, dim=1, unbiased=False, keepdim=True) + 1e-4  # 更大的值
            s_std = torch.std(s_centered, dim=1, unbiased=False, keepdim=True) + 1e-4  # 更大的值

            # 确保标准差不会太小
            t_std = torch.clamp(t_std, min=1e-3)
            s_std = torch.clamp(s_std, min=1e-3)

            # 標準化坐標
            t_normalized = t_centered / t_std  # [B, 17, 2, grid]
            s_normalized = s_centered / s_std  # [B, 17, 2, grid]

            # 更新姿態結構空間損失計算，添加精度階梯

            # 修改姿態結構空間損失的權重和細節
            shape_diff = (t_normalized - s_normalized)**2

            # 識別高精度區域
            high_precision_shape_mask = shape_diff < 0.01
            medium_precision_shape_mask = (shape_diff >= 0.01) & (shape_diff < 0.05)

            # 精度階梯權重
            weighted_shape_diff = torch.where(
                high_precision_shape_mask,
                shape_diff * 10.0,  # 高精度區域加大權重
                torch.where(
                    medium_precision_shape_mask,
                    shape_diff * 5.0,
                    shape_diff * 2.0
                )
            )

            pose_structure_loss = weighted_shape_diff.mean()
            
            # 4. 修改損失權重 - 動態調整各項權重
            # 加速早期訓練
            # 4. 修改損失權重 - 加重精度相關部分的權重
            if current_epoch < 5:
                # 前5個epoch重視坐標精度
                coord_weight = 3.5       # 從3.0提高到3.5
                structure_weight = 2.0   # 從1.5提高到2.0
                consistency_weight = 2.5 # 從2.0提高到2.5
                pose_structure_weight = 3.0 # 從2.0提高到3.0 - 顯著提高
                conf_weight = 0.5
                rel_pos_weight = 2.0     # 從1.5提高到2.0
            else:
                # 後期更均衡，但仍保持高權重
                coord_weight = 3.0       # 從2.5提高到3.0
                structure_weight = 1.8   # 從1.2提高到1.8
                consistency_weight = 2.0 # 從1.5提高到2.0
                pose_structure_weight = 2.0 # 從1.0提高到2.0
                conf_weight = 0.5
                rel_pos_weight = 1.5     # 從1.0提高到1.5
    
            # 組合所有損失
            pred_loss = (
                coord_weight * coord_loss +
                structure_weight * structure_loss +
                consistency_weight * consistency_loss +
                0.01 * pose_structure_weight * pose_structure_loss +  # 從0.005提高回0.01
                conf_weight * conf_loss +
                rel_pos_weight * rel_pos_loss +
                2.0 * y_special_loss  # 從1.5提高到2.0
            )
            
            # 5. 【改進9】自適應損失組合
            # 使用教師模型的平均置信度來調整損失權重
            avg_teacher_conf = teacher_conf_mask.mean().clamp(0.1, 0.9)
            
            # 【修改】自適應權重 - 初期提高特徵權重
            if current_epoch < 3:
                # 前3個epoch給較高特徵權重
                base_feat_weight = feat_weight * 0.3  # 從0.15提高到0.3
            else:
                # 原有邏輯
                base_feat_weight = feat_weight * (0.15 + 0.85 * min(1.0, (current_epoch - 2) / (total_epochs - 2) * 1.5))
                
            # 低置信度時更信任特徵蒸餾，高置信度時更信任輸出蒸餾
            adaptive_feat_weight = base_feat_weight * (1.0 - avg_teacher_conf.item())
            adaptive_pred_weight = pred_weight * avg_teacher_conf.item()

            # 【初期階段特別處理】首個epoch大幅降低特徵損失影響
            if current_epoch == 0:
                feat_loss = feat_loss * 0.05  # 大幅降低特徵損失

            # 添加專門針對高IoU檢測的損失項
            # 計算高置信度點的精確定位增強
            high_conf_mask = teacher_conf_mask > 0.7  # 回到較高的閾值要求
            if high_conf_mask.any():
                # 提取高置信度點
                high_conf_x = s_x[high_conf_mask]
                high_conf_y = s_y[high_conf_mask]
                high_conf_t_x = t_x[high_conf_mask]
                high_conf_t_y = t_y[high_conf_mask]
                
                if high_conf_x.numel() > 0:
                    # 計算高置信度點的誤差
                    hc_x_diff = torch.abs(high_conf_x - high_conf_t_x)
                    hc_y_diff = torch.abs(high_conf_y - high_conf_t_y)
                    
                    # 針對不同IoU閾值的精度要求
                    hc_map95_x_mask = hc_x_diff < 0.0005
                    hc_map90_x_mask = (hc_x_diff >= 0.0005) & (hc_x_diff < 0.001)
                    hc_map80_x_mask = (hc_x_diff >= 0.001) & (hc_x_diff < 0.0015)
                    
                    hc_map95_y_mask = hc_y_diff < 0.0005
                    hc_map90_y_mask = (hc_y_diff >= 0.0005) & (hc_y_diff < 0.001)
                    hc_map80_y_mask = (hc_y_diff >= 0.001) & (hc_y_diff < 0.0015)
                    
                    # 計算高置信度點的IoU損失
                    hc_iou_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
                    
                    # X軸IoU損失
                    if hc_map95_x_mask.any():
                        hc_iou_loss = hc_iou_loss + hc_x_diff[hc_map95_x_mask].sum() * 12.0 / (hc_map95_x_mask.sum() + epsilon)
                    if hc_map90_x_mask.any():
                        hc_iou_loss = hc_iou_loss + hc_x_diff[hc_map90_x_mask].sum() * 8.0 / (hc_map90_x_mask.sum() + epsilon)
                    if hc_map80_x_mask.any():
                        hc_iou_loss = hc_iou_loss + hc_x_diff[hc_map80_x_mask].sum() * 5.0 / (hc_map80_x_mask.sum() + epsilon)
                    
                    # Y軸IoU損失 - 更高權重
                    if hc_map95_y_mask.any():
                        hc_iou_loss = hc_iou_loss + hc_y_diff[hc_map95_y_mask].sum() * 18.0 / (hc_map95_y_mask.sum() + epsilon)
                    if hc_map90_y_mask.any():
                        hc_iou_loss = hc_iou_loss + hc_y_diff[hc_map90_y_mask].sum() * 12.0 / (hc_map90_y_mask.sum() + epsilon)
                    if hc_map80_y_mask.any():
                        hc_iou_loss = hc_iou_loss + hc_y_diff[hc_map80_y_mask].sum() * 8.0 / (hc_map80_y_mask.sum() + epsilon)
                    
                    # 將高置信度IoU損失添加到總損失
                    pred_loss = pred_loss + 0.5 * hc_iou_loss  # 適度權重

            # 添加進度自適應的精細定位權重增強
            if current_epoch >= 10:  # 訓練後期特別關注精細定位
                # 計算訓練進度比例影響因子
                progress_factor = min(1.0, (current_epoch - 10) / 20.0)  # 從0到1緩慢提升
                
                # 計算超精細誤差比例
                ultra_precision_x_ratio = (map95_mask_x.float().mean() + map90_mask_x.float().mean()) / 2
                ultra_precision_y_ratio = (map95_mask_y.float().mean() + map90_mask_y.float().mean()) / 2
                
                # 目標比例 - 隨進度逐步提高
                target_ultra_precision = 0.02 + progress_factor * 0.04  # 從2%提高到6%
                
                # 如果當前比例低於目標，增加相應的損失權重
                if ultra_precision_x_ratio < target_ultra_precision:
                    boost_factor_x = 1.0 + min(1.0, (target_ultra_precision - ultra_precision_x_ratio) * 10.0)
                    pred_loss = pred_loss + coord_weight * weighted_map_grad_x.sum() * boost_factor_x / total_weight
                
                if ultra_precision_y_ratio < target_ultra_precision:
                    boost_factor_y = 1.0 + min(1.5, (target_ultra_precision - ultra_precision_y_ratio) * 15.0)  # Y軸更積極
                    pred_loss = pred_loss + coord_weight * weighted_map_grad_y.sum() * boost_factor_y / total_weight
                
                # 記錄訓練統計
                if hasattr(self, 'model') and hasattr(self.model, 'epoch') and is_first_batch_in_epoch:
                    print(f"目標超精細比例: {target_ultra_precision:.4f}, 當前X: {ultra_precision_x_ratio:.4f}, Y: {ultra_precision_y_ratio:.4f}")
                    if ultra_precision_x_ratio < target_ultra_precision or ultra_precision_y_ratio < target_ultra_precision:
                        print(f"超精細定位權重提升 - X: {boost_factor_x if 'boost_factor_x' in locals() else 0.0:.2f}, Y: {boost_factor_y if 'boost_factor_y' in locals() else 0.0:.2f}")
            
            # 組合所有損失
            # 确保各个损失项不是NaN
            feat_loss = torch.nan_to_num(feat_loss, nan=0.0, posinf=1.0, neginf=0.0)
            pred_loss = torch.nan_to_num(pred_loss, nan=0.0, posinf=1.0, neginf=0.0)

            # 组合损失并增加额外的安全检查
            total_loss = adaptive_feat_weight * feat_loss + adaptive_pred_weight * pred_loss

            # 应用软上限
            total_loss = torch.clamp(total_loss, max=10.0)

            # 添加梯度裁剪，防止梯度爆炸
            if total_loss > 10.0:
                total_loss = 10.0 + torch.log(1.0 + (total_loss - 10.0))  # 軟上限
            
            # 【新增】早期停止損失計算的額外檢查
            # 如果損失已經很小，進一步減少計算量
            if total_loss < 1e-6:
                # 設置一個非零但極小的損失值，確保梯度不會完全消失
                total_loss = torch.tensor(1e-6, device=student_preds.device, requires_grad=True)
            
            # 最終安全檢查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Warning: 損失計算出現NaN或Inf")
                return torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 記錄訓練階段信息和損失值
            self.last_loss_values = {
                "coord_loss": float(coord_loss.item()) if not torch.isnan(coord_loss) else 0.0,
                "structure_loss": float(structure_loss.item()) if not torch.isnan(structure_loss) else 0.0,
                "conf_loss": float(conf_loss.item()) if not torch.isnan(conf_loss) else 0.0,
                "rel_pos_loss": float(rel_pos_loss.item()) if not torch.isnan(rel_pos_loss) else 0.0,
                "feat_loss": float(feat_loss.item()) if not torch.isnan(feat_loss) else 0.0,
                "total_loss": float(total_loss.item()) if not torch.isnan(total_loss) else 0.0,
                "teacher_conf": float(avg_teacher_conf.item()),
                "temperature": float(T),
                "progress": float(progress),
                "epoch": int(current_epoch)
            }

            # 記錄更多損失指標
            self.last_loss_values.update({
                "consistency_loss": float(consistency_loss.item()) if not torch.isnan(consistency_loss) else 0.0,
                "pose_structure_loss": float(pose_structure_loss.item()) if not torch.isnan(pose_structure_loss) else 0.0,
                "precise_coord_ratio": float(
                    (tiny_diff_mask_x.sum() + tiny_diff_mask_y.sum()) / 
                    (tiny_diff_mask_x.numel() + tiny_diff_mask_y.numel() + epsilon)
                )
            })
            
            # 輸出日誌信息
            if hasattr(self, 'model') and hasattr(self.model, 'epoch') and (
            current_epoch % 1 == 0) and is_first_batch_in_epoch:
                print(f"\n--- Loss Values (Epoch {current_epoch}/{total_epochs}, T={T:.2f}) ---")
                print(f"coord_loss: {float(coord_loss.item()):.4f}")
                print(f"structure_loss: {float(structure_loss.item()):.4f}")
                print(f"conf_loss: {float(conf_loss.item()):.4f}")
                print(f"rel_pos_loss: {float(rel_pos_loss.item()):.4f}")
                print(f"feat_loss: {float(feat_loss.item()):.4f}" + (f" (raw: {float(feat_loss.item() / (0.05 if current_epoch == 0 else 1.0)):.4f})" if current_epoch == 0 else ""))
                print(f"total_loss: {float(total_loss.item()):.4f}")
                print(f"teacher_conf: {float(avg_teacher_conf.item()):.4f}")
                print(f"feat_weight: {adaptive_feat_weight:.4f}, pred_weight: {adaptive_pred_weight:.4f}")

                print(f"consistency_loss: {float(consistency_loss.item()):.4f}")
                print(f"pose_structure_loss: {float(pose_structure_loss.item()):.4f}")
                
                # 添加精確定位統計
                precise_x_ratio = tiny_diff_mask_x.float().mean().item()
                precise_y_ratio = tiny_diff_mask_y.float().mean().item()
                print(f"極小誤差比例 (x<0.001): {precise_x_ratio:.4f}, (y<0.001): {precise_y_ratio:.4f}")
                print(f"小誤差比例 (x<0.005): {small_diff_mask_x.float().mean().item():.4f}, (y<0.005): {small_diff_mask_y.float().mean().item():.4f}")
                
                # 計算坐標誤差分布
                x_diff_mean = x_diff.abs().mean().item()
                y_diff_mean = y_diff.abs().mean().item()
                x_diff_std = x_diff.abs().std().item()
                y_diff_std = y_diff.abs().std().item()
                print(f"坐標誤差統計 - X軸: 均值={x_diff_mean:.6f}, 標準差={x_diff_std:.6f}")
                print(f"坐標誤差統計 - Y軸: 均值={y_diff_mean:.6f}, 標準差={y_diff_std:.6f}")
                
                # 調用精度統計函數
                self.log_precision_stats(x_diff.flatten(), y_diff.flatten())

                # 添加微小誤差統計
                micro_x_ratio = micro_tiny_diff_mask_x.float().mean().item()
                micro_y_ratio = micro_tiny_diff_mask_y.float().mean().item()
                print(f"超微小誤差比例 (x<0.0001): {micro_x_ratio:.4f}, (y<0.0001): {micro_y_ratio:.4f}")
                
                # Y軸與X軸誤差比例
                if x_diff_mean > 0:
                    y_x_ratio = y_diff_mean / x_diff_mean
                    print(f"Y/X軸誤差比例: {y_x_ratio:.4f}")

                print("\n--- 高IoU精度統計 ---")
                map95_x_ratio = map95_mask_x.float().mean().item()
                map90_x_ratio = map90_mask_x.float().mean().item() 
                map80_x_ratio = map80_mask_x.float().mean().item()
                map75_x_ratio = map75_mask_x.float().mean().item()
                
                map95_y_ratio = map95_mask_y.float().mean().item()
                map90_y_ratio = map90_mask_y.float().mean().item()
                map80_y_ratio = map80_mask_y.float().mean().item()
                map75_y_ratio = map75_mask_y.float().mean().item()
                
                print(f"IoU閾值對應精度 - IoU75/80/90/95")
                print(f"X軸精度比例: {map75_x_ratio:.4f}/{map80_x_ratio:.4f}/{map90_x_ratio:.4f}/{map95_x_ratio:.4f}")
                print(f"Y軸精度比例: {map75_y_ratio:.4f}/{map80_y_ratio:.4f}/{map90_y_ratio:.4f}/{map95_y_ratio:.4f}")
                print(f"綜合精度目標達成率: {((map95_x_ratio + map95_y_ratio) / 0.06):.2f}")
            
            return total_loss
            
        except Exception as e:
            print(f"蒸餾損失計算異常: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印詳細錯誤堆棧
            return torch.tensor(0.1, device=student_outputs[1].device, requires_grad=True)


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
