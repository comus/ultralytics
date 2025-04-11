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

class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()

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
            loss[5] = self.pose_distillation_loss_optimized(preds, batch["teacher_preds"])
        else:
            loss[5] = torch.zeros(1, device=self.device, requires_grad=True)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= 12.0

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def pose_distillation_loss_optimized(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.5, pred_weight=1.0):
        """
        高度優化的姿態蒸餾損失函數，最大化GPU利用率並減少冗餘計算
        """
        epsilon = 1e-8
        
        try:
            # 直接解包輸出，減少中間變量
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
            
            # 使用向量化操作進行限幅
            s_conf = s_conf.clamp(-50.0, 50.0)
            t_conf = t_conf.clamp(-50.0, 50.0)
            
            # 1. 特徵蒸餾損失 - 使用mask張量操作替代循環
            # 預先選擇特徵層
            max_len = min(len(student_features), len(teacher_features))
            indices = torch.tensor([i for i in range(max_len) if (i % 3 == 0 or i == max_len - 1)], 
                                device=student_preds.device)
            
            if len(indices) > 0:  # 確保有特徵層可用
                # 使用索引張量一次性獲取所有特徵
                s_feats = [student_features[i] for i in indices]
                t_feats = [teacher_features[i] for i in indices]
                
                # 並行計算所有特徵層的損失
                feat_loss_sum = 0.0
                valid_count = 0
                
                for s_f, t_f in zip(s_feats, t_feats):
                    # 檢查形狀和NaN
                    if s_f.shape != t_f.shape:
                        try:
                            t_f = F.interpolate(t_f, size=s_f.shape[-2:], mode='bilinear', align_corners=False)
                        except:
                            continue
                    
                    if not (torch.isnan(s_f).any() or torch.isnan(t_f).any()):
                        # 使用單次操作計算均方誤差
                        feat_loss_sum += ((s_f - t_f) ** 2).mean()
                        valid_count += 1
                
                # 安全平均
                feat_loss = feat_loss_sum / max(valid_count, 1)
            else:
                feat_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # 2. 坐標損失 - 使用單一操作計算
            # 合併x和y的MSE計算，減少操作數
            coord_diff = torch.cat([(s_x - t_x).unsqueeze(-1), (s_y - t_y).unsqueeze(-1)], dim=-1)
            coord_loss = (coord_diff ** 2).mean()
            
            # 3. 結構損失 - 使用矩陣操作
            # 定義骨架關節對
            skeleton = torch.tensor([
                [5, 6],    # 左肩-右肩
                [11, 12],  # 左髖-右髖
                [5, 11],   # 左肩-左髖
                [6, 12],   # 右肩-右髖
            ], device=student_preds.device)
            
            # 使用高效索引獲取坐標
            a_idx, b_idx = skeleton[:, 0], skeleton[:, 1]
            
            # 優化骨架長度計算 - 避免多餘的維度擴展
            # 直接從原始坐標提取
            s_a_x, s_a_y = s_x[:, a_idx], s_y[:, a_idx]
            s_b_x, s_b_y = s_x[:, b_idx], s_y[:, b_idx]
            
            t_a_x, t_a_y = t_x[:, a_idx], t_y[:, a_idx]
            t_b_x, t_b_y = t_x[:, b_idx], t_y[:, b_idx]
            
            # 使用單一操作計算所有骨架長度
            s_bone_len = torch.sqrt((s_a_x - s_b_x)**2 + (s_a_y - s_b_y)**2 + epsilon)
            t_bone_len = torch.sqrt((t_a_x - t_b_x)**2 + (t_a_y - t_b_y)**2 + epsilon)
            
            # 安全計算結構損失
            if torch.isnan(s_bone_len).any() or torch.isnan(t_bone_len).any():
                structure_loss = torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            else:
                structure_loss = ((s_bone_len - t_bone_len) ** 2).mean()
            
            # 4. 置信度損失 - 直接計算
            conf_loss = ((s_conf - t_conf) ** 2).mean()
            
            # 5. 一次性組合所有損失
            total_loss = feat_weight * feat_loss + pred_weight * (coord_loss + 0.5 * structure_loss + 0.3 * conf_loss)
            
            # 最終安全檢查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                return torch.tensor(0.0, device=student_preds.device, requires_grad=True)
            
            # # 輸出損失值
            # loss_values = {
            #     "coord_loss": float(coord_loss.item()) if not torch.isnan(coord_loss) else 0.0,
            #     "structure_loss": float(structure_loss.item()) if not torch.isnan(structure_loss) else 0.0,
            #     "conf_loss": float(conf_loss.item()) if not torch.isnan(conf_loss) else 0.0,
            #     "feat_loss": float(feat_loss.item()) if not torch.isnan(feat_loss) else 0.0,
            #     "total_loss": float(total_loss.item()) if not torch.isnan(total_loss) else 0.0
            # }
            
            # print("\n--- Loss Values ---")
            # for k, v in loss_values.items():
            #     print(f"{k}: {v:.4f}")

            return total_loss
            
        except Exception as e:
            return torch.tensor(0.1, device=student_preds.device, requires_grad=True)
    # def fixed_distillation_loss(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.5, pred_weight=1.0):
    #     """
    #     最終修正版蒸餾損失函數 - 使用MSE而非BCE作為置信度損失
    #     """
    #     # 獲取學生和教師模型的特徵圖和預測
    #     student_features = student_outputs[0]
    #     teacher_features = teacher_outputs[0]
        
    #     student_preds = student_outputs[1]
    #     teacher_preds = teacher_outputs[1]
        
    #     batch_size = student_preds.shape[0]
    #     num_keypoints = 17
        
    #     # 重塑預測張量
    #     s_preds = student_preds.reshape(batch_size, num_keypoints, 3, -1)
    #     t_preds = teacher_preds.reshape(batch_size, num_keypoints, 3, -1)
        
    #     # 提取x、y座標和置信度
    #     s_x = s_preds[:, :, 0, :]
    #     s_y = s_preds[:, :, 1, :]
    #     s_conf = s_preds[:, :, 2, :]
        
    #     t_x = t_preds[:, :, 0, :]
    #     t_y = t_preds[:, :, 1, :]
    #     t_conf = t_preds[:, :, 2, :]
        
    #     # 1. 特徵蒸餾損失
    #     feat_loss = 0.0
    #     for s_feat, t_feat in zip(student_features, teacher_features):
    #         feat_loss += F.mse_loss(s_feat, t_feat)
    #     feat_loss = feat_loss / len(student_features)
        
    #     # 2. 坐標損失
    #     coord_loss = F.mse_loss(s_x, t_x) + F.mse_loss(s_y, t_y)
        
    #     # 3. 結構損失
    #     skeleton = torch.tensor([
    #         [5, 6], [11, 12], [5, 11], [6, 12]
    #     ], device=student_preds.device)
        
    #     structure_loss = 0.0
    #     for (a, b) in skeleton:
    #         s_bone_len = torch.sqrt((s_x[:, a, :] - s_x[:, b, :])**2 + (s_y[:, a, :] - s_y[:, b, :])**2)
    #         t_bone_len = torch.sqrt((t_x[:, a, :] - t_x[:, b, :])**2 + (t_y[:, a, :] - t_y[:, b, :])**2)
    #         structure_loss += F.mse_loss(s_bone_len, t_bone_len)
        
    #     structure_loss = structure_loss / len(skeleton)
        
    #     # 4. 修正的置信度損失 - 使用MSE而非BCE
    #     # 對於相同的輸入，MSE始終為零
    #     conf_loss = F.mse_loss(s_conf, t_conf)
        
    #     # 5. 組合損失
    #     pred_loss = coord_loss + 0.5 * structure_loss + 0.3 * conf_loss
    #     total_loss = feat_weight * feat_loss + pred_weight * pred_loss
        
    #     # 輸出損失值
    #     loss_values = {
    #         "coord_loss": coord_loss.item(),
    #         "structure_loss": structure_loss.item(),
    #         "conf_loss": conf_loss.item(),
    #         "feat_loss": feat_loss.item(),
    #         "total_loss": total_loss.item()
    #     }
        
    #     print("\n--- Loss Values ---")
    #     for k, v in loss_values.items():
    #         print(f"{k}: {v:.4f}")
        
    #     return total_loss
    
    # def debug_confidence_loss_detailed(self, student_outputs, teacher_outputs):
    #     """
    #     更詳細的置信度損失調試
    #     """
    #     # 獲取置信度值
    #     student_preds = student_outputs[1]
    #     teacher_preds = teacher_outputs[1]
        
    #     batch_size = student_preds.shape[0]
    #     num_keypoints = 17
        
    #     s_preds = student_preds.reshape(batch_size, num_keypoints, 3, -1)
    #     t_preds = teacher_preds.reshape(batch_size, num_keypoints, 3, -1)
        
    #     s_conf = s_preds[:, :, 2, :]  # [B, 17, A]
    #     t_conf = t_preds[:, :, 2, :]  # [B, 17, A]
        
    #     # 1. 基本檢查
    #     print("基本檢查:")
    #     are_equal = torch.allclose(s_conf, t_conf)
    #     print(f"  學生和教師置信度是否相等: {are_equal}")
        
    #     # 2. 轉換為概率空間
    #     s_prob = torch.sigmoid(s_conf)
    #     t_prob = torch.sigmoid(t_conf)
        
    #     are_probs_equal = torch.allclose(s_prob, t_prob)
    #     print(f"  學生和教師概率是否相等: {are_probs_equal}")
        
    #     # 3. 檢查值的分布
    #     print("\n值分布:")
    #     s_conf_flat = s_conf.flatten()
    #     t_conf_flat = t_conf.flatten()
        
    #     # 基本統計
    #     print(f"  學生置信度: min={s_conf_flat.min().item():.4f}, max={s_conf_flat.max().item():.4f}, mean={s_conf_flat.mean().item():.4f}")
    #     print(f"  教師置信度: min={t_conf_flat.min().item():.4f}, max={t_conf_flat.max().item():.4f}, mean={t_conf_flat.mean().item():.4f}")
        
    #     # 4. 計算不同的損失值
    #     print("\n不同損失計算:")
        
    #     # PyTorch BCE實現
    #     bce_loss = F.binary_cross_entropy_with_logits(s_conf, t_conf)
    #     print(f"  PyTorch BCE損失: {bce_loss.item():.6f}")
        
    #     # 手動BCE實現1 - 使用sigmoid後的概率
    #     epsilon = 1e-12
    #     manual_bce_1 = -torch.mean(t_prob * torch.log(s_prob + epsilon) + (1 - t_prob) * torch.log(1 - s_prob + epsilon))
    #     print(f"  手動BCE損失1 (概率空間): {manual_bce_1.item():.6f}")
        
    #     # 手動BCE實現2 - 使用logits直接計算
    #     max_val = torch.clamp(s_conf, min=None, max=20)
    #     log_sum_exp = max_val + torch.log(torch.exp(-max_val) + torch.exp(t_conf - max_val))
    #     manual_bce_2 = torch.mean(log_sum_exp - s_conf * t_conf)
    #     print(f"  手動BCE損失2 (logits空間): {manual_bce_2.item():.6f}")
        
    #     # MSE損失比較
    #     mse_loss = F.mse_loss(s_conf, t_conf)
    #     print(f"  MSE損失: {mse_loss.item():.6f}")
        
    #     # 5. 逐元素BCE計算
    #     # 抽樣一些元素進行詳細檢查
    #     print("\n抽樣元素BCE計算:")
        
    #     # 隨機選擇5個元素
    #     indices = torch.randint(0, s_conf.numel(), (5,))
        
    #     for i, idx in enumerate(indices):
    #         # 獲取扁平索引對應的多維索引
    #         batch_idx = idx // (num_keypoints * s_conf.shape[-1])
    #         remainder = idx % (num_keypoints * s_conf.shape[-1])
    #         kp_idx = remainder // s_conf.shape[-1]
    #         anchor_idx = remainder % s_conf.shape[-1]
            
    #         s_val = s_conf[batch_idx, kp_idx, anchor_idx].item()
    #         t_val = t_conf[batch_idx, kp_idx, anchor_idx].item()
            
    #         s_p = s_prob[batch_idx, kp_idx, anchor_idx].item()
    #         t_p = t_prob[batch_idx, kp_idx, anchor_idx].item()
            
    #         # 計算該元素的BCE
    #         elem_bce = -(t_p * math.log(s_p + epsilon) + (1 - t_p) * math.log(1 - s_p + epsilon))
            
    #         print(f"  元素 {i+1}:")
    #         print(f"    位置: batch={batch_idx}, keypoint={kp_idx}, anchor={anchor_idx}")
    #         print(f"    學生logit={s_val:.6f}, 概率={s_p:.6f}")
    #         print(f"    教師logit={t_val:.6f}, 概率={t_p:.6f}")
    #         print(f"    元素BCE={elem_bce:.6f}")
        
    #     # 6. 探索PyTorch BCE實現
    #     print("\nPyTorch BCE實現探索:")
        
    #     # 嘗試使用BCEWithLogitsLoss類
    #     bce_with_logits = torch.nn.BCEWithLogitsLoss()
    #     bce_class_loss = bce_with_logits(s_conf, t_conf)
    #     print(f"  使用BCEWithLogitsLoss類: {bce_class_loss.item():.6f}")
        
    #     # 嘗試單獨batch元素
    #     batch_losses = []
    #     for b in range(min(batch_size, 3)):  # 只取前3個批次
    #         batch_loss = F.binary_cross_entropy_with_logits(s_conf[b], t_conf[b])
    #         batch_losses.append(batch_loss.item())
    #         print(f"  批次 {b} BCE損失: {batch_loss.item():.6f}")
        
    #     return {
    #         "torch_bce": bce_loss.item(),
    #         "manual_bce_1": manual_bce_1.item(),
    #         "manual_bce_2": manual_bce_2.item(),
    #         "mse": mse_loss.item()
    #     }
    
    # def debug_confidence_loss(self, student_outputs, teacher_outputs):
    #     """
    #     調試置信度損失計算
    #     """
    #     # 獲取學生和教師模型的預測
    #     student_preds = student_outputs[1]  # [batch_size, 51, num_anchors]
    #     teacher_preds = teacher_outputs[1]  # [batch_size, 51, num_anchors]
        
    #     batch_size = student_preds.shape[0]
    #     num_keypoints = 17
        
    #     # 重塑預測張量
    #     s_preds = student_preds.reshape(batch_size, num_keypoints, 3, -1)  # [B, 17, 3, A]
    #     t_preds = teacher_preds.reshape(batch_size, num_keypoints, 3, -1)  # [B, 17, 3, A]
        
    #     # 提取置信度
    #     s_conf = s_preds[:, :, 2, :]  # [B, 17, A]
    #     t_conf = t_preds[:, :, 2, :]  # [B, 17, A]
        
    #     # 檢查學生和教師置信度是否真的相同
    #     exact_match = torch.all(s_conf == t_conf).item()
    #     print(f"Exact confidence match: {exact_match}")
        
    #     # 檢查差異
    #     conf_diff = torch.abs(s_conf - t_conf)
    #     max_diff = torch.max(conf_diff).item()
    #     mean_diff = torch.mean(conf_diff).item()
    #     print(f"Max confidence difference: {max_diff:.6f}")
    #     print(f"Mean confidence difference: {mean_diff:.6f}")
        
    #     # 計算不同的損失函數值
    #     bce_loss = F.binary_cross_entropy_with_logits(s_conf, t_conf).item()
    #     print(f"BCE loss: {bce_loss:.6f}")
        
    #     mse_loss = F.mse_loss(s_conf, t_conf).item()
    #     print(f"MSE loss: {mse_loss:.6f}")
        
    #     # 檢查sigmoid後的差異
    #     s_sigmoid = torch.sigmoid(s_conf)
    #     t_sigmoid = torch.sigmoid(t_conf)
    #     sigmoid_diff = torch.abs(s_sigmoid - t_sigmoid)
    #     max_sigmoid_diff = torch.max(sigmoid_diff).item()
    #     mean_sigmoid_diff = torch.mean(sigmoid_diff).item()
    #     print(f"Max sigmoid difference: {max_sigmoid_diff:.6f}")
    #     print(f"Mean sigmoid difference: {mean_sigmoid_diff:.6f}")
        
    #     # 檢查是否有任何NaN或Inf值
    #     has_nan_s = torch.isnan(s_conf).any().item()
    #     has_inf_s = torch.isinf(s_conf).any().item()
    #     has_nan_t = torch.isnan(t_conf).any().item()
    #     has_inf_t = torch.isinf(t_conf).any().item()
    #     print(f"Student conf has NaN: {has_nan_s}, has Inf: {has_inf_s}")
    #     print(f"Teacher conf has NaN: {has_nan_t}, has Inf: {has_inf_t}")
        
    #     # 檢查兩個模型輸出是否使用的是同一張量對象
    #     same_object = (id(s_conf) == id(t_conf))
    #     print(f"Same tensor object: {same_object}")
        
    #     # 檢查極端值
    #     s_min, s_max = torch.min(s_conf).item(), torch.max(s_conf).item()
    #     t_min, t_max = torch.min(t_conf).item(), torch.max(t_conf).item()
    #     print(f"Student conf range: [{s_min:.2f}, {s_max:.2f}]")
    #     print(f"Teacher conf range: [{t_min:.2f}, {t_max:.2f}]")
        
    #     # 檢查分布差異
    #     s_mean, s_std = torch.mean(s_conf).item(), torch.std(s_conf).item()
    #     t_mean, t_std = torch.mean(t_conf).item(), torch.std(t_conf).item()
    #     print(f"Student conf stats: mean={s_mean:.4f}, std={s_std:.4f}")
    #     print(f"Teacher conf stats: mean={t_mean:.4f}, std={t_std:.4f}")
        
    #     # 計算KL散度
    #     s_log_softmax = F.log_softmax(s_conf.reshape(-1, s_conf.shape[-1]), dim=-1)
    #     t_softmax = F.softmax(t_conf.reshape(-1, t_conf.shape[-1]), dim=-1)
    #     kl_loss = F.kl_div(s_log_softmax, t_softmax, reduction='batchmean').item()
    #     print(f"KL divergence loss: {kl_loss:.6f}")
        
    #     # 返回不同損失計算方法的結果
    #     loss_values = {
    #         "BCE Loss": bce_loss,
    #         "MSE Loss": mse_loss,
    #         "KL Loss": kl_loss
    #     }
        
    #     return loss_values

    # def distillation_loss(self, student_outputs, teacher_outputs, T=2.0):
    #     """
    #     蒸餾損失函數，帶有詳細日誌輸出
    #     """
    #     # 記錄輸入結構
    #     print("\n=== Distillation Loss Function Debug ===")
        
    #     # 檢查student_outputs和teacher_outputs的結構
    #     print(f"student_outputs type: {type(student_outputs)}, length: {len(student_outputs)}")
    #     print(f"teacher_outputs type: {type(teacher_outputs)}, length: {len(teacher_outputs)}")
        
    #     # 獲取特徵圖和預測
    #     student_features = student_outputs[0]
    #     teacher_features = teacher_outputs[0]
    #     student_preds = student_outputs[1]
    #     teacher_preds = teacher_outputs[1]
        
    #     # 檢查特徵圖結構
    #     print("\n--- Feature Maps Structure ---")
    #     print(f"Number of feature maps: {len(student_features)}")
    #     for i, feat in enumerate(student_features):
    #         print(f"Feature map {i} shape: {feat.shape}")
        
    #     # 檢查預測張量結構
    #     print("\n--- Prediction Tensor Structure ---")
    #     print(f"Student preds shape: {student_preds.shape}")
    #     print(f"Teacher preds shape: {teacher_preds.shape}")
        
    #     # 檢查通道分配
    #     print("\n--- Channel Value Distribution ---")

    #     for i in range(51):
    #         print(f"channel{i} - shape: {student_preds[:, i, :].shape}")
    #         print(f"channel{i} - min: {student_preds[:, i, :].min().item():.4f}, max: {student_preds[:, i, :].max().item():.4f}, mean: {student_preds[:, i, :].mean().item():.4f}")
        
    #     # 檢查邊界框部分（假設前4個通道）
    #     box_channels = student_preds[:, :4, :]
    #     print(f"Box channels (first 4) - shape: {box_channels.shape}")
    #     print(f"Box channels - min: {box_channels.min().item():.4f}, max: {box_channels.max().item():.4f}, mean: {box_channels.mean().item():.4f}")
        
    #     # 檢查置信度部分（假設第5個通道）
    #     conf_channel = student_preds[:, 4, :]
    #     print(f"Conf channel (5th) - shape: {conf_channel.shape}")
    #     print(f"Conf channel - min: {conf_channel.min().item():.4f}, max: {conf_channel.max().item():.4f}, mean: {conf_channel.mean().item():.4f}")
    #     print(f"Conf channel - values above 0.5: {(conf_channel > 0.5).sum().item()}")
    #     print(f"Conf channel - histogram: {torch.histc(conf_channel, bins=5, min=-1, max=1)}")
        
    #     # 檢查可能的關鍵點部分
    #     kpt_channels = student_preds[:, 5:, :]
    #     print(f"Keypoint channels (rest) - shape: {kpt_channels.shape}")
    #     print(f"Keypoint channels - min: {kpt_channels.min().item():.4f}, max: {kpt_channels.max().item():.4f}, mean: {kpt_channels.mean().item():.4f}")
        
    #     # 檢查3個通道為一組的模式（如果是關鍵點x,y,v）
    #     if (kpt_channels.shape[1] % 3 == 0):
    #         num_keypoints = kpt_channels.shape[1] // 3
    #         print(f"\nAssuming {num_keypoints} keypoints with (x,y,v) format")
            
    #         # 檢查每個關鍵點組的摘要統計
    #         for i in range(num_keypoints):
    #             kpt_group = kpt_channels[:, i*3:(i+1)*3, :]
    #             print(f"Keypoint {i} - shape: {kpt_group.shape}, mean: {kpt_group.mean().item():.4f}")
                
    #             # 抽樣檢查前10個預測的關鍵點值
    #             sample_idx = 0
    #             print(f"  Sample values (batch={sample_idx}, first 3 predictions):")
    #             for j in range(min(3, kpt_group.shape[2])):
    #                 print(f"    Pred {j}: {kpt_group[sample_idx, :, j].tolist()}")
        
    #     # 特徵圖蒸餾
    #     feat_loss = 0
    #     for s_feat, t_feat in zip(student_features, teacher_features):
    #         feat_loss += F.mse_loss(s_feat, t_feat)
    #     feat_loss = feat_loss / len(student_features)
        
    #     # 邊界框損失 (前4個通道)
    #     box_loss = F.smooth_l1_loss(student_preds[:, :4, :], teacher_preds[:, :4, :])
        
    #     # 關鍵點損失 (第5-55通道)
    #     kpt_loss = F.mse_loss(student_preds[:, 5:, :], teacher_preds[:, 5:, :])
        
    #     # 置信度損失 (第4通道)
    #     conf_loss = 0
    #     batch_size = student_preds.shape[0]
    #     for i in range(batch_size):
    #         s_conf = student_preds[i, 4, :] / T
    #         t_conf = teacher_preds[i, 4, :] / T
            
    #         s_conf_log = F.log_softmax(s_conf, dim=0)
    #         t_conf_soft = F.softmax(t_conf, dim=0)
            
    #         conf_loss += F.kl_div(s_conf_log, t_conf_soft, reduction='sum') * (T**2) / student_preds.shape[2]
        
    #     conf_loss = conf_loss / batch_size
        
    #     # 總預測損失
    #     pred_loss = box_loss + 1.5 * kpt_loss + 2.0 * conf_loss
        
    #     # 總損失
    #     total_loss = 0.5 * feat_loss + 1.0 * pred_loss
        
    #     # 輸出損失值
    #     print("\n--- Loss Values ---")
    #     print(f"Feature map loss: {feat_loss.item():.4f}")
    #     print(f"Box loss: {box_loss.item():.4f}")
    #     print(f"Keypoint loss: {kpt_loss.item():.4f}")
    #     print(f"Confidence loss: {conf_loss.item():.4f}")
    #     print(f"Total loss: {total_loss.item():.4f}")
        
    #     return total_loss

    # def distillation_loss(self, student_outputs, teacher_outputs, T=2.0, feat_weight=0.5, pred_weight=1.0):
    #     """
    #     YOLO-Pose蒸餾損失函數，專為全關鍵點輸出設計
        
    #     參數:
    #         student_outputs: 學生模型輸出，格式為(特徵圖列表, 預測張量)
    #         teacher_outputs: 教師模型輸出，格式為(特徵圖列表, 預測張量)
    #         T: 溫度參數
    #         feat_weight: 特徵蒸餾權重
    #         pred_weight: 預測蒸餾權重
            
    #     返回:
    #         total_loss: 總蒸餾損失
    #     """
    #     # 獲取學生和教師模型的特徵圖和預測
    #     student_features = student_outputs[0]
    #     teacher_features = teacher_outputs[0]
        
    #     student_preds = student_outputs[1]  # [batch_size, 51, num_anchors]
    #     teacher_preds = teacher_outputs[1]  # [batch_size, 51, num_anchors]
        
    #     # 1. 特徵圖蒸餾損失
    #     feat_loss = 0
    #     for s_feat, t_feat in zip(student_features, teacher_features):
    #         feat_loss += F.mse_loss(s_feat, t_feat)
    #     feat_loss = feat_loss / len(student_features)
        
    #     # 2. 關鍵點預測蒸餾損失
    #     num_keypoints = 17
        
    #     # 2.1 關鍵點坐標損失 (x和y座標)
    #     coord_loss = 0
    #     for i in range(num_keypoints):
    #         # 每個關鍵點的x和y座標索引
    #         x_idx = i * 3
    #         y_idx = i * 3 + 1
            
    #         # 使用平滑L1損失進行坐標蒸餾
    #         x_loss = F.smooth_l1_loss(student_preds[:, x_idx, :], teacher_preds[:, x_idx, :])
    #         y_loss = F.smooth_l1_loss(student_preds[:, y_idx, :], teacher_preds[:, y_idx, :])
            
    #         coord_loss += x_loss + y_loss
        
    #     coord_loss = coord_loss / (2 * num_keypoints)  # 平均到每個坐標
        
    #     # 2.2 關鍵點置信度損失
    #     conf_loss = 0
    #     batch_size = student_preds.shape[0]
        
    #     for i in range(num_keypoints):
    #         conf_idx = i * 3 + 2  # 置信度索引
            
    #         # 使用KL散度進行置信度蒸餾
    #         for b in range(batch_size):
    #             s_conf = student_preds[b, conf_idx, :] / T
    #             t_conf = teacher_preds[b, conf_idx, :] / T
                
    #             # 軟化後的置信度分佈
    #             s_conf_log = F.log_softmax(s_conf, dim=0)
    #             t_conf_soft = F.softmax(t_conf, dim=0)
                
    #             # KL散度損失
    #             kl_loss = F.kl_div(
    #                 s_conf_log, 
    #                 t_conf_soft, 
    #                 reduction='sum'
    #             ) * (T**2) / student_preds.shape[2]
                
    #             conf_loss += kl_loss
        
    #     conf_loss = conf_loss / (batch_size * num_keypoints)  # 平均到每個關鍵點的每個批次
        
    #     # 2.3 關注高置信度區域的特殊蒸餾
    #     # 獲取教師模型中所有關鍵點的平均置信度
    #     teacher_conf_mean = torch.zeros((batch_size, student_preds.shape[2]), device=student_preds.device)
        
    #     for i in range(num_keypoints):
    #         conf_idx = i * 3 + 2
    #         teacher_conf_mean += teacher_preds[:, conf_idx, :]
        
    #     teacher_conf_mean = teacher_conf_mean / num_keypoints
    #     high_conf_mask = teacher_conf_mean > 0  # 只關注正置信度區域
        
    #     if torch.any(high_conf_mask):
    #         # 計算高置信度區域的位置蒸餾損失
    #         weighted_coord_loss = 0
    #         count = 0
            
    #         for b in range(batch_size):
    #             batch_mask = high_conf_mask[b]
                
    #             if torch.any(batch_mask):
    #                 for i in range(num_keypoints):
    #                     x_idx = i * 3
    #                     y_idx = i * 3 + 1
    #                     conf_idx = i * 3 + 2
                        
    #                     # 獲取此關鍵點的置信度作為權重
    #                     weights = torch.sigmoid(teacher_preds[b, conf_idx, batch_mask])
                        
    #                     # 計算加權L1損失
    #                     x_diff = torch.abs(student_preds[b, x_idx, batch_mask] - teacher_preds[b, x_idx, batch_mask])
    #                     y_diff = torch.abs(student_preds[b, y_idx, batch_mask] - teacher_preds[b, y_idx, batch_mask])
                        
    #                     weighted_x_loss = (weights * x_diff).sum() / (weights.sum() + 1e-8)
    #                     weighted_y_loss = (weights * y_diff).sum() / (weights.sum() + 1e-8)
                        
    #                     weighted_coord_loss += weighted_x_loss + weighted_y_loss
    #                     count += 2
            
    #         if count > 0:
    #             weighted_coord_loss = weighted_coord_loss / count
    #             # 加入加權損失
    #             coord_loss = 0.5 * coord_loss + 0.5 * weighted_coord_loss
        
    #     # 組合損失
    #     pred_loss = 1.5 * coord_loss + 2.0 * conf_loss
        
    #     # 組合特徵損失和預測損失
    #     total_loss = feat_weight * feat_loss + pred_weight * pred_loss

    #     # # 輸出損失值
    #     # print("\n--- Loss Values ---")
    #     # print(f"feat loss: {feat_loss.item():.4f}")
    #     # print(f"coord loss: {coord_loss.item():.4f}")
    #     # print(f"conf loss: {conf_loss.item():.4f}")
    #     # print(f"pred loss: {pred_loss.item():.4f}")
    #     # print(f"total loss: {total_loss.item():.4f}")
        
    #     return total_loss, {
    #         'feat_loss': feat_loss.item(),
    #         'coord_loss': coord_loss.item(),
    #         'conf_loss': conf_loss.item(),
    #         'pred_loss': pred_loss.item(),
    #         'total_loss': total_loss.item()
    #     }

    # def distillation_loss(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.5, pred_weight=1.0):
    #     """
    #     YOLO-Pose蒸餾損失函數 - 效能優化版本
        
    #     參數:
    #         student_outputs: 學生模型輸出，格式為(特徵圖列表, 預測張量)
    #         teacher_outputs: 教師模型輸出，格式為(特徵圖列表, 預測張量)
    #         T: 溫度參數
    #         feat_weight: 特徵蒸餾權重
    #         pred_weight: 預測蒸餾權重
            
    #     返回:
    #         total_loss: 總蒸餾損失
    #     """
    #     # 獲取學生和教師模型的特徵圖和預測
    #     student_features = student_outputs[0]
    #     teacher_features = teacher_outputs[0]
        
    #     student_preds = student_outputs[1]  # [batch_size, 51, num_anchors]
    #     teacher_preds = teacher_outputs[1]  # [batch_size, 51, num_anchors]
        
    #     batch_size = student_preds.shape[0]
    #     num_keypoints = 17
        
    #     # 1. 特徵圖蒸餾損失 - 向量化操作
    #     feat_loss = torch.tensor(0.0, device=student_preds.device)
    #     for s_feat, t_feat in zip(student_features, teacher_features):
    #         feat_loss += F.mse_loss(s_feat, t_feat)
    #     feat_loss = feat_loss / len(student_features)
        
    #     # 2. 關鍵點預測蒸餾損失 - 向量化處理
        
    #     # 2.1 關鍵點坐標損失 (x和y座標) - 不使用迴圈
    #     # 創建x和y索引掩碼
    #     x_indices = torch.tensor([i * 3 for i in range(num_keypoints)], device=student_preds.device)
    #     y_indices = torch.tensor([i * 3 + 1 for i in range(num_keypoints)], device=student_preds.device)
        
    #     # 使用索引一次性選取所有x座標
    #     s_x_coords = student_preds[:, x_indices, :]  # [batch_size, num_keypoints, num_anchors]
    #     t_x_coords = teacher_preds[:, x_indices, :]
        
    #     # 使用索引一次性選取所有y座標
    #     s_y_coords = student_preds[:, y_indices, :]
    #     t_y_coords = teacher_preds[:, y_indices, :]
        
    #     # 計算x和y座標的損失
    #     x_loss = F.smooth_l1_loss(s_x_coords, t_x_coords)
    #     y_loss = F.smooth_l1_loss(s_y_coords, t_y_coords)
        
    #     coord_loss = (x_loss + y_loss) / 2
        
    #     # 2.2 關鍵點置信度損失 - 簡化版本
    #     conf_indices = torch.tensor([i * 3 + 2 for i in range(num_keypoints)], device=student_preds.device)
        
    #     # 獲取所有置信度通道
    #     s_conf = student_preds[:, conf_indices, :]  # [batch_size, num_keypoints, num_anchors]
    #     t_conf = teacher_preds[:, conf_indices, :]
        
    #     # 直接使用MSE或BCE損失進行置信度蒸餾，避免使用KL散度的多重迴圈
    #     conf_loss = F.mse_loss(s_conf, t_conf)
        
    #     # 針對高置信度區域的額外蒸餾 - 簡化版本
    #     # 計算每個位置的平均置信度
    #     t_conf_mean = torch.mean(t_conf, dim=1)  # [batch_size, num_anchors]
    #     high_conf_mask = t_conf_mean > 0.5  # 只關注高置信度區域
        
    #     # 初始化加權損失
    #     weighted_coord_loss = torch.tensor(0.0, device=student_preds.device)
    #     valid_samples = 0
        
    #     # 檢查是否有高置信度預測
    #     if torch.any(high_conf_mask):
    #         # 批次維度的掩碼處理
    #         for b in range(batch_size):
    #             batch_mask = high_conf_mask[b]
    #             if torch.any(batch_mask):
    #                 # 提取高置信度區域的坐標
    #                 s_x_high = s_x_coords[b, :, batch_mask]  # [num_keypoints, num_high_conf]
    #                 t_x_high = t_x_coords[b, :, batch_mask]
    #                 s_y_high = s_y_coords[b, :, batch_mask]
    #                 t_y_high = t_y_coords[b, :, batch_mask]
                    
    #                 # 計算坐標損失
    #                 high_x_loss = F.smooth_l1_loss(s_x_high, t_x_high)
    #                 high_y_loss = F.smooth_l1_loss(s_y_high, t_y_high)
                    
    #                 weighted_coord_loss += high_x_loss + high_y_loss
    #                 valid_samples += 1
            
    #         if valid_samples > 0:
    #             weighted_coord_loss = weighted_coord_loss / valid_samples
    #             # 合併普通坐標損失和加權坐標損失
    #             coord_loss = 0.5 * coord_loss + 0.5 * weighted_coord_loss
        
    #     # 組合損失
    #     pred_loss = 1.5 * coord_loss + 2.0 * conf_loss
        
    #     # 組合特徵損失和預測損失
    #     total_loss = feat_weight * feat_loss + pred_weight * pred_loss

    #     # 輸出損失值
    #     # print("\n--- Loss Values ---")
    #     # print(f"feat loss: {feat_loss.item():.4f}")
    #     # print(f"coord loss: {coord_loss.item():.4f}")
    #     # print(f"conf loss: {conf_loss.item():.4f}")
    #     # print(f"pred loss: {pred_loss.item():.4f}")
    #     # print(f"total loss: {total_loss.item():.4f}")
        
    #     return total_loss

    # def distillation_loss(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.5, pred_weight=1.0):
    #     """
    #     YOLO-Pose蒸餾損失函數 - 效能優化版本 (修正版)
        
    #     參數:
    #         student_outputs: 學生模型輸出，格式為(特徵圖列表, 預測張量)
    #         teacher_outputs: 教師模型輸出，格式為(特徵圖列表, 預測張量)
    #         T: 溫度參數 - 現在會正確使用
    #         feat_weight: 特徵蒸餾權重
    #         pred_weight: 預測蒸餾權重
            
    #     返回:
    #         total_loss: 總蒸餾損失
    #     """
    #     # 獲取學生和教師模型的特徵圖和預測
    #     student_features = student_outputs[0]
    #     teacher_features = teacher_outputs[0]
        
    #     student_preds = student_outputs[1]  # [batch_size, 51, num_anchors]
    #     teacher_preds = teacher_outputs[1]  # [batch_size, 51, num_anchors]
        
    #     batch_size = student_preds.shape[0]
    #     num_keypoints = 17
        
    #     # 1. 特徵圖蒸餾損失 - 向量化操作
    #     feat_loss = torch.tensor(0.0, device=student_preds.device)
    #     for s_feat, t_feat in zip(student_features, teacher_features):
    #         feat_loss += F.mse_loss(s_feat, t_feat)
    #     feat_loss = feat_loss / len(student_features)
        
    #     # 2. 關鍵點預測蒸餾損失 - 向量化處理
        
    #     # 2.1 關鍵點坐標損失 (x和y座標) - 不使用迴圈
    #     # 創建x和y索引掩碼
    #     x_indices = torch.tensor([i * 3 for i in range(num_keypoints)], device=student_preds.device)
    #     y_indices = torch.tensor([i * 3 + 1 for i in range(num_keypoints)], device=student_preds.device)
        
    #     # 使用索引一次性選取所有x座標
    #     s_x_coords = student_preds[:, x_indices, :]  # [batch_size, num_keypoints, num_anchors]
    #     t_x_coords = teacher_preds[:, x_indices, :]
        
    #     # 使用索引一次性選取所有y座標
    #     s_y_coords = student_preds[:, y_indices, :]
    #     t_y_coords = teacher_preds[:, y_indices, :]
        
    #     # 計算x和y座標的損失
    #     x_loss = F.smooth_l1_loss(s_x_coords, t_x_coords)
    #     y_loss = F.smooth_l1_loss(s_y_coords, t_y_coords)
        
    #     coord_loss = (x_loss + y_loss) / 2
        
    #     # 2.2 關鍵點置信度損失 - 使用溫度參數
    #     conf_indices = torch.tensor([i * 3 + 2 for i in range(num_keypoints)], device=student_preds.device)
        
    #     # 獲取所有置信度通道
    #     s_conf = student_preds[:, conf_indices, :]  # [batch_size, num_keypoints, num_anchors]
    #     t_conf = teacher_preds[:, conf_indices, :]
        
    #     # 使用溫度軟化置信度分佈
    #     s_conf_T = s_conf / T
    #     t_conf_T = t_conf / T
        
    #     # 計算軟化後的KL散度損失
    #     kl_loss = 0.0
    #     for i in range(num_keypoints):
    #         for b in range(batch_size):
    #             # 對每個關鍵點的置信度計算KL散度
    #             s_conf_logits = s_conf_T[b, i]
    #             t_conf_logits = t_conf_T[b, i]
                
    #             # 應用softmax獲取概率分佈
    #             s_conf_log_softmax = F.log_softmax(s_conf_logits, dim=0)
    #             t_conf_softmax = F.softmax(t_conf_logits, dim=0)
                
    #             # 計算KL散度
    #             kl = F.kl_div(s_conf_log_softmax, t_conf_softmax, reduction='batchmean') * (T**2)
    #             kl_loss += kl
        
    #     # 平均每個批次和關鍵點的KL散度
    #     conf_loss = kl_loss / (batch_size * num_keypoints)
        
    #     # 針對高置信度區域的額外蒸餾 - 簡化版本
    #     # 計算每個位置的平均置信度
    #     t_conf_sigmoid = torch.sigmoid(t_conf)  # 轉換為概率
    #     t_conf_mean = torch.mean(t_conf_sigmoid, dim=1)  # [batch_size, num_anchors]
    #     high_conf_mask = t_conf_mean > 0.5  # 只關注高置信度區域
        
    #     # 初始化加權損失
    #     weighted_coord_loss = torch.tensor(0.0, device=student_preds.device)
    #     valid_samples = 0
        
    #     # 檢查是否有高置信度預測
    #     if torch.any(high_conf_mask):
    #         # 批次維度的掩碼處理
    #         for b in range(batch_size):
    #             batch_mask = high_conf_mask[b]
    #             if torch.any(batch_mask):
    #                 # 提取高置信度區域的坐標
    #                 s_x_high = s_x_coords[b, :, batch_mask]  # [num_keypoints, num_high_conf]
    #                 t_x_high = t_x_coords[b, :, batch_mask]
    #                 s_y_high = s_y_coords[b, :, batch_mask]
    #                 t_y_high = t_y_coords[b, :, batch_mask]
                    
    #                 # 計算坐標損失
    #                 high_x_loss = F.smooth_l1_loss(s_x_high, t_x_high)
    #                 high_y_loss = F.smooth_l1_loss(s_y_high, t_y_high)
                    
    #                 weighted_coord_loss += high_x_loss + high_y_loss
    #                 valid_samples += 1
            
    #         if valid_samples > 0:
    #             weighted_coord_loss = weighted_coord_loss / valid_samples
    #             # 合併普通坐標損失和加權坐標損失
    #             coord_loss = 0.5 * coord_loss + 0.5 * weighted_coord_loss
        
    #     # 增加關鍵點結構一致性損失 (新增)
    #     structure_loss = 0.0
        
    #     # 選擇一些重要的關鍵點對
    #     # 例如：左右肩、左右髖、肩髖連接等
    #     keypoint_pairs = [
    #         (5, 6),    # 左右肩
    #         (11, 12),  # 左右髖
    #         (5, 11),   # 左肩到左髖
    #         (6, 12),   # 右肩到右髖
    #         (5, 7),    # 左肩到左肘
    #         (6, 8),    # 右肩到右肘
    #         (7, 9),    # 左肘到左腕
    #         (8, 10),   # 右肘到右腕
    #         (11, 13),  # 左髖到左膝
    #         (12, 14),  # 右髖到右膝
    #         (13, 15),  # 左膝到左踝
    #         (14, 16)   # 右膝到右踝
    #     ]
        
    #     for a, b in keypoint_pairs:
    #         # 計算關鍵點對之間的相對距離
    #         s_dist = torch.sqrt((s_x_coords[:, a, :] - s_x_coords[:, b, :])**2 + 
    #                             (s_y_coords[:, a, :] - s_y_coords[:, b, :])**2)
    #         t_dist = torch.sqrt((t_x_coords[:, a, :] - t_x_coords[:, b, :])**2 + 
    #                             (t_y_coords[:, a, :] - t_y_coords[:, b, :])**2)
            
    #         # 使用平滑L1損失比較距離
    #         pair_loss = F.smooth_l1_loss(s_dist, t_dist)
    #         structure_loss += pair_loss
        
    #     # 平均所有對的損失
    #     structure_loss = structure_loss / len(keypoint_pairs)
        
    #     # 組合損失 - 添加結構損失
    #     pred_loss = 1.2 * coord_loss + 1.5 * conf_loss + 1.0 * structure_loss
        
    #     # 組合特徵損失和預測損失
    #     total_loss = feat_weight * feat_loss + pred_weight * pred_loss

    #     # 輸出損失值
    #     # print("\n--- Loss Values ---")
    #     # print(f"Feat loss: {feat_loss.item():.4f}")
    #     # print(f"Coord loss: {coord_loss.item():.4f}")
    #     # print(f"Conf loss: {conf_loss.item():.4f}")
    #     # print(f"Structure loss: {structure_loss.item():.4f}")
    #     # print(f"Pred loss: {pred_loss.item():.4f}")
    #     # print(f"Total loss: {total_loss.item():.4f}")
        
    #     return total_loss

    # def distillation_loss(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.5, pred_weight=1.0):
    #     """
    #     YOLO-Pose蒸餾損失函數 - 高效率版本
        
    #     參數:
    #         student_outputs: 學生模型輸出，格式為(特徵圖列表, 預測張量)
    #         teacher_outputs: 教師模型輸出，格式為(特徵圖列表, 預測張量)
    #         T: 溫度參數
    #         feat_weight: 特徵蒸餾權重
    #         pred_weight: 預測蒸餾權重
            
    #     返回:
    #         total_loss: 總蒸餾損失
    #     """
    #     # 獲取學生和教師模型的特徵圖和預測
    #     student_features = student_outputs[0]
    #     teacher_features = teacher_outputs[0]
        
    #     student_preds = student_outputs[1]  # [batch_size, 51, num_anchors]
    #     teacher_preds = teacher_outputs[1]  # [batch_size, 51, num_anchors]
        
    #     batch_size = student_preds.shape[0]
    #     num_keypoints = 17
        
    #     # 1. 特徵圖蒸餾損失 - 加速版本
    #     # 只選擇一部分重要層進行蒸餾，而不是所有層
    #     feat_loss = 0.0
    #     selected_features = [0, -3, -2, -1]  # 只選最重要的幾層：首層和最後幾層
    #     for idx in selected_features:
    #         s_feat = student_features[idx]
    #         t_feat = teacher_features[idx]
    #         # 降低特徵圖分辨率以加速計算
    #         if s_feat.numel() > 1000000:  # 如果特徵元素數量太多
    #             s_feat = F.avg_pool2d(s_feat, kernel_size=2)
    #             t_feat = F.avg_pool2d(t_feat, kernel_size=2)
    #         feat_loss += F.mse_loss(s_feat, t_feat)
    #     feat_loss = feat_loss / len(selected_features)
        
    #     # 2. 關鍵點預測蒸餾損失 - 高效率版本
        
    #     # 重塑預測張量以加速計算
    #     # 重新整理張量形狀，使關鍵點通道在最後一維
    #     s_preds = student_preds.reshape(batch_size, num_keypoints, 3, -1)  # [B, 17, 3, A]
    #     t_preds = teacher_preds.reshape(batch_size, num_keypoints, 3, -1)  # [B, 17, 3, A]
        
    #     # 提取x、y座標和置信度
    #     s_x = s_preds[:, :, 0, :]  # [B, 17, A]
    #     s_y = s_preds[:, :, 1, :]  # [B, 17, A]
    #     s_conf = s_preds[:, :, 2, :]  # [B, 17, A]
        
    #     t_x = t_preds[:, :, 0, :]  # [B, 17, A]
    #     t_y = t_preds[:, :, 1, :]  # [B, 17, A]
    #     t_conf = t_preds[:, :, 2, :]  # [B, 17, A]
        
    #     # 2.1 計算坐標損失 - 向量化版本
    #     # 使用平滑L1損失計算坐標差異
    #     coord_loss = (F.smooth_l1_loss(s_x, t_x) + F.smooth_l1_loss(s_y, t_y)) / 2
        
    #     # 2.2 計算置信度損失 - 高效率版本
    #     # 使用均方誤差替代KL散度，顯著提高效率
    #     conf_loss = F.mse_loss(s_conf / T, t_conf / T) * (T**2)
        
    #     # 2.3 計算結構損失 - 高效率版本
    #     # 預定義骨架連接，每行代表一對關鍵點索引
    #     skeleton = torch.tensor([
    #         [5, 6],    # 左右肩
    #         [11, 12],  # 左右髖
    #         [5, 11],   # 左肩到左髖
    #         [6, 12]    # 右肩到右髖
    #     ], device=student_preds.device)
        
    #     # 批量計算所有骨架部分的長度
    #     s_lengths = torch.zeros((batch_size, len(skeleton)), device=student_preds.device)
    #     t_lengths = torch.zeros((batch_size, len(skeleton)), device=student_preds.device)
        
    #     for i, (a, b) in enumerate(skeleton):
    #         # 計算歐式距離
    #         s_lengths[:, i] = torch.sqrt(((s_x[:, a, :] - s_x[:, b, :])**2 + 
    #                                     (s_y[:, a, :] - s_y[:, b, :])**2).mean(dim=1))
    #         t_lengths[:, i] = torch.sqrt(((t_x[:, a, :] - t_x[:, b, :])**2 + 
    #                                     (t_y[:, a, :] - t_y[:, b, :])**2).mean(dim=1))
        
    #     # 計算比例一致性，而不是絕對距離
    #     structure_loss = F.mse_loss(s_lengths, t_lengths)
        
    #     # 3. 根據教師置信度選擇性地加權損失 - 向量化版本
    #     # 計算教師模型的平均置信度
    #     t_conf_sigmoid = torch.sigmoid(t_conf)  # [B, 17, A]
        
    #     # 計算加權係數 - 無需循環
    #     # 使用教師置信度作為權重，但避免過高權重
    #     weight_mask = torch.clamp(t_conf_sigmoid, min=0.1, max=0.9)  # [B, 17, A]
        
    #     # 計算加權坐標損失 - 完全向量化
    #     weighted_x_loss = ((s_x - t_x)**2 * weight_mask).mean()
    #     weighted_y_loss = ((s_y - t_y)**2 * weight_mask).mean()
    #     weighted_coord_loss = (weighted_x_loss + weighted_y_loss) / 2
        
    #     # 組合損失 - 簡化權重以提高計算效率
    #     coord_loss = 0.7 * coord_loss + 0.3 * weighted_coord_loss  # 混合普通和加權坐標損失
    #     pred_loss = coord_loss + 0.8 * conf_loss + 0.5 * structure_loss
        
    #     # 總損失
    #     total_loss = feat_weight * feat_loss + pred_weight * pred_loss
        
    #     return total_loss

    def distillation_loss_fixed(self, student_outputs, teacher_outputs, T=3.0, feat_weight=0.5, pred_weight=1.0):
        """
        修正版蒸餾損失函數 - 解決同模型蒸餾問題
        """
        # 獲取學生和教師模型的特徵圖和預測
        student_features = student_outputs[0]
        teacher_features = teacher_outputs[0]
        
        student_preds = student_outputs[1]  # [batch_size, 51, num_anchors]
        teacher_preds = teacher_outputs[1]  # [batch_size, 51, num_anchors]
        
        batch_size = student_preds.shape[0]
        num_keypoints = 17
        
        # 重塑預測張量
        s_preds = student_preds.reshape(batch_size, num_keypoints, 3, -1)  # [B, 17, 3, A]
        t_preds = teacher_preds.reshape(batch_size, num_keypoints, 3, -1)  # [B, 17, 3, A]
        
        # 提取x、y座標和置信度
        s_x = s_preds[:, :, 0, :]  # [B, 17, A]
        s_y = s_preds[:, :, 1, :]  # [B, 17, A]
        s_conf = s_preds[:, :, 2, :]  # [B, 17, A]
        
        t_x = t_preds[:, :, 0, :]  # [B, 17, A]
        t_y = t_preds[:, :, 1, :]  # [B, 17, A]
        t_conf = t_preds[:, :, 2, :]  # [B, 17, A]
        
        # 1. 特徵蒸餾損失 - 向量化操作
        feat_loss = 0.0
        for s_feat, t_feat in zip(student_features, teacher_features):
            feat_loss += F.mse_loss(s_feat, t_feat)
        feat_loss = feat_loss / len(student_features)
        
        # 2. 坐標損失 - 簡單MSE
        coord_loss = F.mse_loss(s_x, t_x) + F.mse_loss(s_y, t_y)
        
        # 3. 結構損失 - 計算骨架結構一致性
        # 定義關鍵點對
        skeleton = torch.tensor([
            [5, 6],    # 左右肩
            [11, 12],  # 左右髖
            [5, 11],   # 左肩到左髖
            [6, 12]    # 右肩到右髖
        ], device=student_preds.device)
        
        structure_loss = 0.0
        for (a, b) in skeleton:
            # 計算骨骼長度
            s_bone_len = torch.sqrt((s_x[:, a, :] - s_x[:, b, :])**2 + (s_y[:, a, :] - s_y[:, b, :])**2)
            t_bone_len = torch.sqrt((t_x[:, a, :] - t_x[:, b, :])**2 + (t_y[:, a, :] - t_y[:, b, :])**2)
            
            # 比較長度
            structure_loss += F.mse_loss(s_bone_len, t_bone_len)
        
        structure_loss = structure_loss / len(skeleton)
        
        # 4. 修正的置信度損失 - 使用BCE損失而非KL散度
        # 修正為直接使用BCE，避免溫度縮放問題
        conf_loss = F.binary_cross_entropy_with_logits(s_conf, torch.sigmoid(t_conf))
        
        # 5. 組合損失 - 調整權重
        pred_loss = coord_loss + 0.5 * structure_loss + 0.3 * conf_loss
        total_loss = feat_weight * feat_loss + pred_weight * pred_loss
        
        # 輸出損失值
        loss_values = {
            "coord_loss": coord_loss.item(),
            "structure_loss": structure_loss.item(),
            "conf_loss": conf_loss.item(),
            "feat_loss": feat_loss.item(),
            "total_loss": total_loss.item()
        }
        
        print("\n--- Loss Values ---")
        for k, v in loss_values.items():
            print(f"{k}: {v:.4f}")

        return total_loss

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
