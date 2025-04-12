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

initial_T = 4.0
min_T = 2.0

def update_temperature(current_epoch, total_epochs, initial_T=8.0, min_T=2.0):
    """根據當前epoch調整溫度
    
    Args:
        current_epoch: 當前epoch索引(從0開始)
        total_epochs: 總epoch數
        initial_T: 初始溫度
        min_T: 最小溫度
    
    Returns:
        float: 當前epoch的溫度值
    """
    # 確保在最後一個epoch時達到最小溫度
    if total_epochs <= 1:
        return min_T  # 防止只有一個epoch的情況
    
    # 正規化epoch進度(0到1)
    progress = current_epoch / (total_epochs - 1) if total_epochs > 1 else 1.0
    
    # 使用指數衰減計算溫度
    T = initial_T * (min_T / initial_T) ** progress
    
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
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

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

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
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
            # 如果 self.model 有 trainer 屬性，則打印 epoch
            epoch = self.model.epoch if hasattr(self.model, 'epoch') else 1
            epochs = self.model.epochs if hasattr(self.model, 'epochs') else 1

            T = update_temperature(epoch, epochs)

            loss[5] = self.pose_distillation_loss_enhanced(preds, batch["teacher_preds"], T)
        else:
            loss[5] = torch.zeros(1, device=self.device, requires_grad=True)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.distill

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def pose_distillation_loss_enhanced(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.2, pred_weight=0.4):
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
        epsilon = 1e-8
        tolerance = 1e-6  # 數值穩定性容忍閾值
        
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

            # if is_first_batch_in_epoch:
            #     print("\n=== Feature Maps Shapes ===")
            #     for i, (s_f, t_f) in enumerate(zip(student_features, teacher_features)):
            #         print(f"Layer {i}: Student shape {s_f.shape}, Teacher shape {t_f.shape}")

            # if is_first_batch_in_epoch:
            #     print("\n=== Channel Value Ranges ===")
            #     print(f"X channels - min: {s_x.min().item():.4f}, max: {s_x.max().item():.4f}")
            #     print(f"Y channels - min: {s_y.min().item():.4f}, max: {s_y.max().item():.4f}")
            #     print(f"Conf channels - min: {s_conf.min().item():.4f}, max: {s_conf.max().item():.4f}")
                
            #     # 檢查置信度通道是否符合預期
            #     conf_sigmoid = torch.sigmoid(s_conf)
            #     print(f"Sigmoid(Conf) - min: {conf_sigmoid.min().item():.4f}, max: {conf_sigmoid.max().item():.4f}")
            
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
            
            # # 動態調整選擇的特徵層數量 - 從多到少
            # if max_len >= 4 and not simplified_feature_distill:
            #     # 初期選擇更多層，後期集中於關鍵層
            #     start_layers = min(max_len, 5)  # 初始階段最多選5層
            #     end_layers = 3                  # 最終階段選3層
                
            #     # 隨著訓練進行逐漸減少層數
            #     num_layers = int(start_layers - (start_layers - end_layers) * progress)
                
            #     if progress < 0.3:  # 訓練初期 - 均勻選取多層
            #         # 均勻選擇層
            #         indices = np.linspace(0, max_len-1, num_layers, dtype=int).tolist()
            #     elif progress < 0.7:  # 訓練中期 - 傾向選取中間層和深層
            #         # 選擇一個淺層，其餘選擇較深的層
            #         indices = [0]  # 始終包含第一層
            #         deep_indices = np.linspace(max_len//3, max_len-1, num_layers-1, dtype=int).tolist()
            #         indices.extend(deep_indices)
            #     else:  # 訓練後期 - 專注於關鍵層
            #         # 固定選擇首層、中間層和末層
            #         indices = [0, max_len//2, max_len-1]
                    
            #         # 如果需要更多層，在深層區域增加
            #         if num_layers > 3:
            #             extra_deep = np.linspace(max_len//2, max_len-2, num_layers-3, dtype=int).tolist()
            #             indices = sorted(list(set(indices + extra_deep)))  # 去重並排序
            # else:
            #     # 如果層數較少或使用簡化蒸餾，選擇關鍵層
            #     if simplified_feature_distill and max_len >= 3:
            #         # 簡化模式：只使用首層、中間層和末層
            #         indices = [0, max_len//2, max_len-1]
            #     else:
            #         # 層數少時全部使用
            #         indices = list(range(max_len))

            # # 動態調整選擇的特徵層數量 - 初期只使用淺層
            # if current_epoch < 2:
            #     # 初期只使用淺層特徵
            #     indices = [0]
            # elif current_epoch < total_epochs // 2:
            #     # 中期使用兩層特徵
            #     indices = [0, 1] if max_len > 1 else [0]
            # else:
            #     # 後期使用全部特徵層
            #     indices = list(range(max_len))

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
            
            # 【新增】應用數值穩定性檢查
            x_diff = torch.where(torch.abs(x_diff) < tolerance, torch.zeros_like(x_diff), x_diff)
            y_diff = torch.where(torch.abs(y_diff) < tolerance, torch.zeros_like(y_diff), y_diff)
            
            weighted_x_diff = (x_diff ** 2) * coord_weights
            weighted_y_diff = (y_diff ** 2) * coord_weights
            
            # 避免分母為零
            total_weight = coord_weights.sum() + epsilon
            coord_loss = (weighted_x_diff.sum() + weighted_y_diff.sum()) / total_weight
            
            # 3. 結構損失 - 優化骨架選擇和權重
            # 【改進5】擴展骨架集合並按重要性加權
            skeleton = torch.tensor([
                [5, 6],    # 左肩-右肩 (軀幹上部)
                [11, 12],  # 左髖-右髖 (骨盆)
                [5, 11],   # 左肩-左髖 (左軀幹)
                [6, 12],   # 右肩-右髖 (右軀幹)
                [5, 7],    # 左肩-左肘 (左上臂)
                [6, 8],    # 右肩-右肘 (右上臂)
            ], device=student_preds.device)
            
            # 骨架重要性權重 - 軀幹骨架權重更高
            skeleton_weights = torch.tensor([1.5, 1.5, 1.0, 1.0, 0.7, 0.7], device=student_preds.device)
            
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
            
            # 5. 【改進9】自適應損失組合
            # 使用教師模型的平均置信度來調整損失權重
            avg_teacher_conf = teacher_conf_mask.mean().clamp(0.1, 0.9)
            
            # 【修改】自適應權重 - 初期大幅降低特徵損失權重
            if current_epoch < 2:
                # 初期極低特徵權重，避免過度蒸餾
                base_feat_weight = feat_weight * 0.1
            else:
                # 隨著訓練進行逐漸恢復權重
                base_feat_weight = feat_weight * (0.1 + 0.9 * min(1.0, (current_epoch - 2) / (total_epochs - 2)))
                
            # 低置信度時更信任特徵蒸餾，高置信度時更信任輸出蒸餾
            adaptive_feat_weight = base_feat_weight * (1.0 - avg_teacher_conf.item())
            adaptive_pred_weight = pred_weight * avg_teacher_conf.item()

            # 【初期階段特別處理】首個epoch大幅降低特徵損失影響
            if current_epoch == 0:
                feat_loss = feat_loss * 0.05  # 大幅降低特徵損失
            
            # 組合所有損失
            pred_loss = coord_loss + 0.5 * structure_loss + 0.5 * conf_loss
            total_loss = adaptive_feat_weight * feat_loss + adaptive_pred_weight * pred_loss
            
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
                "feat_loss": float(feat_loss.item()) if not torch.isnan(feat_loss) else 0.0,
                "total_loss": float(total_loss.item()) if not torch.isnan(total_loss) else 0.0,
                "teacher_conf": float(avg_teacher_conf.item()),
                "temperature": float(T),
                "progress": float(progress),
                "epoch": int(current_epoch)
            }
            
            # 輸出日誌信息
            if hasattr(self, 'model') and hasattr(self.model, 'epoch') and (
            current_epoch % 1 == 0) and is_first_batch_in_epoch:
                print(f"\n--- Loss Values (Epoch {current_epoch}/{total_epochs}, T={T:.2f}) ---")
                print(f"coord_loss: {float(coord_loss.item()):.4f}")
                print(f"structure_loss: {float(structure_loss.item()):.4f}")
                print(f"conf_loss: {float(conf_loss.item()):.4f}")
                print(f"feat_loss: {float(feat_loss.item()):.4f}" + (f" (raw: {float(feat_loss.item() / (0.05 if current_epoch == 0 else 1.0)):.4f})" if current_epoch == 0 else ""))
                print(f"total_loss: {float(total_loss.item()):.4f}")
                print(f"teacher_conf: {float(avg_teacher_conf.item()):.4f}")
                print(f"feat_weight: {adaptive_feat_weight:.4f}, pred_weight: {adaptive_pred_weight:.4f}")
            
            return total_loss
            
        except Exception as e:
            print(f"蒸餾損失計算異常: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印詳細錯誤堆棧
            return torch.tensor(0.1, device=student_outputs[1].device, requires_grad=True)

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
