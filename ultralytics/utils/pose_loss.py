from datetime import datetime
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

from .metrics import bbox_iou, probiou
from .tal import bbox2dist

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
        self.model = model

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it for pose estimation."""

        # if "teacher" in batch and batch["teacher"] is not None:
        #     # 创建输出目录
        #     output_dir = Path('outputs/visualization')
        #     output_dir.mkdir(parents=True, exist_ok=True)
            
        #     # 处理预测结果用于可视化
        #     feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
            
        #     # 先进行预处理
        #     pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
        #         (self.reg_max * 4, self.nc), 1
        #     )
            
        #     # B, grids, ..
        #     pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        #     pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        #     pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        #     dtype = pred_scores.dtype
        #     imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        #     anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)
            
        #     # 解码预测的边界框和关键点
        #     batch_size = pred_scores.shape[0]
            
        #     with torch.no_grad():  # 用于可视化的计算都不需要梯度
        #         pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        #         student_pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)
            
        #     # 获取教师模型的预测（如果可用）
        #     teacher_pred_kpts = None
        #     teacher_pred_bboxes = None
            
        #     with torch.no_grad():  # 教师模型预测不需要梯度
        #         teacher_preds = batch["teacher_pred"]
        #         # 提取教师模型的预测
        #         teacher_feats, teacher_kpts = teacher_preds if isinstance(teacher_preds[0], list) else teacher_preds[1]
        #         teacher_pred_distri, teacher_pred_scores = torch.cat([xi.view(teacher_feats[0].shape[0], self.no, -1) for xi in teacher_feats], 2).split(
        #             (self.reg_max * 4, self.nc), 1
        #         )
                
        #         teacher_pred_scores = teacher_pred_scores.permute(0, 2, 1).contiguous()
        #         teacher_pred_distri = teacher_pred_distri.permute(0, 2, 1).contiguous()
        #         teacher_kpts = teacher_kpts.permute(0, 2, 1).contiguous()
                
        #         teacher_anchor_points, teacher_stride_tensor = make_anchors(teacher_feats, self.stride, 0.5)
        #         teacher_pred_bboxes = self.bbox_decode(teacher_anchor_points, teacher_pred_distri)
        #         teacher_pred_kpts = self.kpts_decode(teacher_anchor_points, teacher_kpts.view(batch_size, -1, *self.kpt_shape))

        if "teacher" in batch and batch["teacher"] is not None:
            teacher_preds = batch["teacher_preds"]

            # print("\n")
            # print("=" * 60)
            # print("student_preds:\n\n", describe_var(preds))
            
            # print("=" * 60)

            # print("teacher_preds:\n\n", describe_var(teacher_preds))

            # print("=" * 60)


        loss = torch.zeros(7, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility, dpose, dkobj

        # 學生模型
        # 這行從模型預測結果中提取兩部分數據：特徵圖(feats)和關鍵點預測(pred_kpts)
        # 根據輸入格式的不同選擇不同的解包方式（處理直接輸入或嵌套輸入的情況）
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        # 將所有特徵圖層的預測結果合併處理：
        # xi.view(feats[0].shape[0], self.no, -1)：將每個特徵圖層重塑為統一格式
        # torch.cat([...], 2)：沿第3維度（網格點維度）連接所有特徵圖層
        # .split((self.reg_max * 4, self.nc), 1)：將連接後的張量分割成兩部分：
        #   - pred_distri：邊界框分佈預測（用於精確定位物體位置）
        #   - pred_scores：類別得分預測（用於物體分類）
        # 這些處理步驟將網絡的原始輸出轉換為後續損失計算和目標檢測所需的格式。
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        # 這段代碼主要是對預測數據進行維度重排，具體作用是：
        # 1. permute(0, 2, 1)：對張量的維度順序進行調整，
        # 將原本的 [batch_size, channels, grid_points] 格式轉換為 [batch_size, grid_points, channels] 格式。
        # 2. .contiguous()：確保張量在記憶體中是連續存儲的，這樣後續操作（如維度變形、索引等）可以更高效地執行。
        # 這種維度轉換在目標檢測和姿態估計模型中很常見，主要目的是將特徵圖的通道維度
        # （表示不同預測類型，如類別得分、邊界框參數、關鍵點座標等）調整到最後一個維度，
        # 使得每個網格點的所有預測信息被組織在一起，便於後續處理和損失計算。
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        # if "teacher" in batch and batch["teacher"] is not None:
        #     # 教师模型
        #     teacher_feats, teacher_pred_kpts = teacher_preds if isinstance(teacher_preds[0], list) else teacher_preds[1]
        #     teacher_pred_distri, teacher_pred_scores = torch.cat([xi.view(teacher_feats[0].shape[0], self.no, -1) for xi in teacher_feats], 2).split(
        #         (self.reg_max * 4, self.nc), 1
        #     )

        #     teacher_pred_scores = teacher_pred_scores.permute(0, 2, 1).contiguous()
        #     teacher_pred_distri = teacher_pred_distri.permute(0, 2, 1).contiguous()
        #     teacher_pred_kpts = teacher_pred_kpts.permute(0, 2, 1).contiguous()

            # 增加日志: 特征图形状
        # if "teacher" in batch and batch["teacher"] is not None:
        #     # 增加日志: 特征图形状
        #     print("\n========== 學生特征图和预测信息 ==========")
        #     print(f"學生特征图数量: {len(feats)}")
        #     for i, feat in enumerate(feats):
        #         print(f"學生特征图 {i} 形状: {feat.shape}")
        #     print(f"學生预测分布形状: {pred_distri.shape}")  
        #     print(f"學生预测分数形状: {pred_scores.shape}")
        #     print(f"學生预测关键点形状: {pred_kpts.shape}")

        #     print("\n========== 教师特征图和预测信息 ==========")
        #     print(f"教师特征图数量: {len(teacher_feats)}")
        #     for i, feat in enumerate(teacher_feats):
        #         print(f"教师特征图 {i} 形状: {feat.shape}")
        #     print(f"教师预测分布形状: {teacher_pred_distri.shape}")  
        #     print(f"教师预测分数形状: {teacher_pred_scores.shape}")
        #     print(f"教师预测关键点形状: {teacher_pred_kpts.shape}")

        # if "teacher" in batch and batch["teacher"] is not None:
        #     print("=" * 60)
        #     print("pred_scores:\n", describe_var(pred_scores), "\n\n")
        #     print("pred_distri:\n", describe_var(pred_distri), "\n\n")
        #     print("pred_kpts:\n", describe_var(pred_kpts), "\n\n")
        #     print("dtype:\n", dtype, "\n\n")
        #     print("imgsz:\n", imgsz, "\n\n")
        #     print("anchor_points:\n", anchor_points, "\n\n")
        #     print("stride_tensor:\n", stride_tensor, "\n\n")
        #     print("=" * 60)

        # 獲取預測分數的數據類型，確保後續計算使用一致的數據類型
        dtype = pred_scores.dtype
        # 計算原始輸入圖像的尺寸
        # feats[0].shape[2:] 獲取特徵圖的高度和寬度
        # 乘以 self.stride[0] 將特徵圖的尺寸還原到原始輸入圖像的尺寸
        # 注釋 # image size (h,w) 說明這是存儲高度和寬度的張量
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        # 調用 make_anchors 函數生成以下內容：
        # anchor_points：所有特徵圖層上的錨點坐標，用於預測定位
        # stride_tensor：對應每個錨點的步長信息
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # 錨點 (Anchor Points)
        # 錨點是特徵圖上的參考位置，類似於一個網格上的點。它們均勻分布在特徵圖上，用來預測圖像中物體的位置。
        # 想像你在看一張照片，並在上面畫了一個網格，每個網格交叉點就是一個「錨點」。模型會從這些錨點出發，預測附近是否有物體，以及物體的邊界框和關鍵點位置。
        
        # 步長 (Stride)
        # 步長表示從原始圖像到特徵圖的縮小比例。例如：
        #   - 如果步長是8，表示原始圖像上的8×8像素區域在特徵圖上對應1個像素
        #   - 如果步長是16，表示原始圖像上的16×16像素區域在特徵圖上對應1個像素
        # 步長越大，特徵圖就越小，每個錨點覆蓋的原始圖像區域就越大。
        # YOLO通常使用多個不同步長的特徵圖層，小步長用於檢測小物體，大步長用於檢測大物體。
        # stride_tensor記錄了每個錨點對應的步長值，用於將特徵圖上的預測轉換回原始圖像坐標。

        # if "teacher" in batch and batch["teacher"] is not None:
        #     teacher_dtype = teacher_pred_scores.dtype
        #     teacher_imgsz = torch.tensor(teacher_feats[0].shape[2:], device=self.device, dtype=teacher_dtype) * self.stride[0]  # image size (h,w)
        #     teacher_anchor_points, teacher_stride_tensor = make_anchors(teacher_feats, self.stride, 0.5)

        # # 教师模型
        # if "teacher" in batch and batch["teacher"] is not None:
        #     # 增加日志: 锚点信息
        #     print("\n========== 學生锚点和步长信息 ==========")
        #     print(f"學生输入图像尺寸: {imgsz}")
        #     print(f"學生锚点数量: {anchor_points.shape[0]}")
        #     print(f"學生锚点前5个: {anchor_points[:5]}")
        #     print(f"學生步长张量形状: {stride_tensor.shape}")
        #     print(f"學生步长前5个: {stride_tensor[:5]}")

        #     print("\n========== 教师锚点和步长信息 ==========")
        #     print(f"教师输入图像尺寸: {teacher_imgsz}")
        #     print(f"教师锚点数量: {teacher_anchor_points.shape[0]}")
        #     print(f"教师锚点前5个: {teacher_anchor_points[:5]}")
        #     print(f"教师步长张量形状: {teacher_stride_tensor.shape}")
        #     print(f"教师步长前5个: {teacher_stride_tensor[:5]}")

        # Targets
        # 這段代碼在處理訓練數據中的標籤(Ground Truth)，整理成模型需要的格式
        # 獲取當前批次中的樣本數量
        batch_size = pred_scores.shape[0]
        # print(f"batch_size: {describe_var(batch_size)}")
        # 獲取每個標籤對應的批次索引，並重塑為列向量
        batch_idx = batch["batch_idx"].view(-1, 1)
        # print(f"batch_idx: {describe_var(batch_idx)}")
        if "teacher" in batch and batch["teacher"] is not None:
            batch_idx2 = self.create_batch_idx_tensor(teacher_preds)
            # print(f"batch_idx2: {describe_var(batch_idx2)}")
        # 將批次索引、類別標籤和邊界框信息合併成一個完整的標籤張量
        # 格式為：[批次索引, 類別標籤, 邊界框座標]
        # print(f"batch['bboxes']: {describe_var(batch['bboxes'])}")
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        # 對標籤數據進行預處理，移動到指定設備(GPU/CPU)
        # scale_tensor=imgsz[[1, 0, 1, 0]] 調整邊界框座標比例，使其與輸入圖像尺寸匹配
        # imgsz[[1, 0, 1, 0]] 相當於 [寬度, 高度, 寬度, 高度]，用於分別縮放 x1, y1, x2, y2 座標
        # print("targets0: ", describe_var(targets), targets)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # 將預處理後的標籤拆分為類別標籤和邊界框
        # (1, 4) 表示分別取1列(類別)和4列(邊界框)
        # 2 表示在第3維進行拆分
        # 注釋 # cls, xyxy 說明格式為類別標籤和xyxy格式的邊界框
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        # 創建有效標籤的掩碼，過濾掉無效標籤
        # gt_bboxes.sum(2, keepdim=True) 計算每個邊界框座標的總和
        # .gt_(0.0) 判斷總和是否大於0，即邊界框是否有效
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        
        # print("targets1: ", describe_var(targets), targets)
        # print("gt_labels: ", describe_var(gt_labels), gt_labels)
        # print("gt_bboxes: ", describe_var(gt_bboxes), gt_bboxes)
        # print("mask_gt: ", describe_var(mask_gt), mask_gt)

        if "teacher" in batch and batch["teacher"] is not None: 
            targets2 = self.preprocess2(teacher_preds)
            gt_labels2, gt_bboxes2 = targets2.split((1, 4), 2)  # cls, xyxy
            mask_gt2 = gt_bboxes2.sum(2, keepdim=True).gt_(0.0)
            # print("targets2: ", describe_var(targets2), targets2)

        # if "teacher" in batch and batch["teacher"] is not None:
        #     teacher_batch_size = teacher_pred_scores.shape[0]
        #     teacher_batch_idx = batch["teacher_batch_idx"].view(-1, 1)
        #     teacher_targets = torch.cat((teacher_batch_idx, batch["cls"].view(-1, 1), batch["teacher_bboxes"]), 1)
        #     teacher_targets = self.preprocess(teacher_targets.to(self.device), teacher_batch_size, scale_tensor=teacher_imgsz[[1, 0, 1, 0]])
        #     teacher_gt_labels, teacher_gt_bboxes = teacher_targets.split((1, 4), 2)  # cls, xyxy
        #     teacher_mask_gt = teacher_gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        # 邊界框解碼
        # 將模型輸出的原始分佈預測 pred_distri 轉換為實際的邊界框座標
        # 利用錨點位置 anchor_points 作為參考，計算出相對於原圖的邊界框座標
        # 結果格式為 (batch_size, 網格點數量, 4)，每個邊界框用 [x1, y1, x2, y2] 表示
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        # 關鍵點解碼
        # 將預測的關鍵點偏移量轉換為實際的關鍵點座標
        # 同樣基於錨點位置進行解碼
        # 結果格式為 (batch_size, 網格點數量, 17, 3)，每個關鍵點用 [x, y, 可見度] 表示
        # 17表示COCO數據集的17個人體關鍵點
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # if "teacher" in batch and batch["teacher"] is not None:
        #     teacher_pred_bboxes = self.bbox_decode(teacher_anchor_points, teacher_pred_distri)  # xyxy, (b, h*w, 4)
        #     teacher_pred_kpts = self.kpts_decode(teacher_anchor_points, teacher_pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # 使用任務對齊分配器(Task-Aligned Assigner)將地面真相(ground truth)與預測結果匹配
        # 決定哪些預測應該負責檢測哪些真實物體
        # 輸入參數包括：
        #   - 預測的分類分數(經過sigmoid激活)
        #   - 預測的邊界框(縮放到原始圖像尺寸)
        #   - 錨點位置(縮放到原始圖像尺寸)
        #   - 真實標籤、邊界框和有效性掩碼
        # 輸出結果包括：
        #   - target_bboxes: 每個預測應該匹配的目標邊界框
        #   - target_scores: 每個預測的目標分類分數
        #   - fg_mask: 前景掩碼，指示哪些預測是前景(有物體)
        #   - target_gt_idx: 每個預測對應的真實標籤索引
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        # print("target_labels", describe_var(target_labels))
        # print("target_bboxes", describe_var(target_bboxes))

        if "teacher" in batch and batch["teacher"] is not None:
            _, target_bboxes2, target_scores2, fg_mask2, target_gt_idx2 = self.assigner(
                pred_scores.detach().sigmoid(),
                (pred_bboxes.detach() * stride_tensor).type(gt_bboxes2.dtype),
                anchor_points * stride_tensor,
                gt_labels2,
                gt_bboxes2,
                mask_gt2,
            )

        # if "teacher" in batch and batch["teacher"] is not None:
        #     teacher_fg_mask = self.assigner2(
        #         pred_scores.detach().sigmoid(),
        #         (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        #         teacher_pred_scores.detach().sigmoid(),
        #         (teacher_pred_bboxes.detach() * teacher_stride_tensor).type(gt_bboxes.dtype),
        #     )

        #     print(f"teacher_fg_mask: {describe_var(teacher_fg_mask)}")

        # print(f"gt_bboxes: {gt_bboxes}")
        # print(f"target_bboxes: {target_bboxes}")
        # print(f"fg_mask shape: {fg_mask.shape}")
        # print(f"fg_mask: {fg_mask}")
        # print(f"pred_bboxes shape: {pred_bboxes.shape}")
        # print(f"pred_bboxes[fg_mask] shape: {pred_bboxes[fg_mask].shape}")

        # 這個分配器負責決定哪些預測應該負責檢測真實物體：
        #   - 計算每個預測框與真實框的IoU(交並比)
        #   - 結合分類得分和IoU計算匹配質量
        #   - 選擇匹配質量最高的預測框作為正樣本(fg_mask)
        #   - 其餘預測框被視為負樣本(背景)
        # 前景掩碼(fg_mask)：
        #   - 只有被標記為前景的預測框(fg_mask=True)才會計算邊界框和關鍵點的損失
        #   - 即使有很多預測框，通常只有少數(如topk=10個)被選為負責檢測這個人

        # 計算目標分數總和
        # 計算所有目標分數的總和，用於後續損失計算的正規化
        # 使用 max(..., 1) 確保總和至少為1，避免除以0的情況
        target_scores_sum = max(target_scores.sum(), 1)

        if "teacher" in batch and batch["teacher"] is not None:
            target_scores_sum2 = max(target_scores2.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        # 這段代碼是在計算損失函數，只有當存在前景物體時(fg_mask.sum() 為真)才執行
        # 什麼為之前景物體? 前景物體是指在圖像中需要被檢測和定位的物體
        # 在目標檢測任務中，前景物體通常是圖像中需要被檢測和定位的物體，例如人、車、動物等
        # 在姿態估計任務中，前景物體通常是圖像中需要被檢測和定位的人體關鍵點
        if fg_mask.sum():
            # # 未乘步長前的預測框
            # fg_pred_boxes = pred_bboxes[fg_mask]
            # print("\n" + "="*50)
            # print("前景預測框（未乘步長）:")
            # print(f"形狀: {fg_pred_boxes.shape}")
            # if fg_pred_boxes.shape[0] > 0:
            #     print("前5個預測框:")
            #     for i in range(min(5, fg_pred_boxes.shape[0])):
            #         print(f"預測框 {i}: {fg_pred_boxes[i].detach().cpu().numpy().round(3)}")
            
            # # 先將所有預測框乘以步長
            # scaled_pred_bboxes = pred_bboxes.detach() * stride_tensor
            
            # # 然後再選取前景預測框
            # fg_pred_boxes_scaled = scaled_pred_bboxes[fg_mask]
            # print("\n前景預測框（乘上步長後）:")
            # print(f"形狀: {fg_pred_boxes_scaled.shape}")
            # if fg_pred_boxes_scaled.shape[0] > 0:
            #     print("前5個預測框:")
            #     for i in range(min(5, fg_pred_boxes_scaled.shape[0])):
            #         print(f"預測框 {i}: {fg_pred_boxes_scaled[i].detach().cpu().numpy().round(3)}")
            
            # # 顯示一些步長信息
            # print("\n步長張量信息:")
            # print(f"形狀: {stride_tensor.shape}")
            # print(f"值範圍: {stride_tensor.min().item()}-{stride_tensor.max().item()}")
            # print(f"唯一值: {torch.unique(stride_tensor).detach().cpu().numpy()}")
            
            # print("="*50)
            
            # 邊界框座標調整
            # 將目標邊界框座標除以步長，轉換到特徵圖的尺度上
            target_bboxes /= stride_tensor

            if "teacher" in batch and batch["teacher"] is not None:
                target_bboxes2 /= stride_tensor

            # print(f"target_bboxes2: {target_bboxes}")
            # 計算邊界框損失
            # 計算邊界框位置的損失(loss[0])和分佈焦點損失(loss[4])
            # 使用預測的分佈、預測的邊界框、錨點位置、目標邊界框等參數計算損失
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

            if "teacher" in batch and batch["teacher"] is not None:
                dbox, ddfl = self.bbox_loss(
                    pred_distri, pred_bboxes, anchor_points, target_bboxes2, target_scores2, target_scores_sum2, fg_mask2
                )

            # 只有fg_mask標記為True的預測框參與邊界框損失計算
            # 損失根據目標分數(target_scores)加權，更重要的框有更大的權重
            # target_gt_idx記錄每個預測框對應的真實標籤索引

            # 關鍵點座標轉換
            # 將批次中的關鍵點移到GPU或CPU設備上
            # 將關鍵點的x座標(索引0)乘以圖像寬度(imgsz[1])
            # 將關鍵點的y座標(索引1)乘以圖像高度(imgsz[0])
            # 這樣做是將歸一化的關鍵點座標[0-1]轉換為實際像素座標
            keypoints = batch["keypoints"].to(self.device).float().clone()
            # print(f"keypoints shape: {keypoints.shape}", keypoints)
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            # print(f"keypoints shape: {keypoints.shape}", keypoints)

            if "teacher" in batch and batch["teacher"] is not None:
                keypoints2 = self.extract_keypoints(teacher_preds)
                # print(f"keypoints2 shape: {keypoints2.shape}", keypoints2)


            # 計算關鍵點損失
            # 計算關鍵點位置的損失(loss[1])和關鍵點可見性的損失(loss[2])
            # 參數包括：
            #   - fg_mask：前景掩碼，標記哪些預測是前景
            #   - target_gt_idx：每個預測對應的真實標籤索引
            #   - keypoints：轉換後的關鍵點座標
            #   - batch_idx：批次索引
            #   - stride_tensor：步長張量
            #   - target_bboxes：目標邊界框
            #   - pred_kpts：預測的關鍵點

            # # 顯示 target_gt_idx shape=[3, 8400]) 存起來，用時間做檔案名
            # if "teacher" in batch and batch["teacher"] is not None:
            #     now = datetime.now().strftime("%Y%m%d_%H%M%S")
            #     with open(f"target_gt_idx2_{now}.txt", "w") as f:
            #         for i in range(target_gt_idx2.shape[0]):
            #             for j in range(target_gt_idx2.shape[1]):
            #                 f.write(f"{target_gt_idx2[i, j]} ")
            #             f.write("\n")

            loss[1], loss[2], gt_kpts, gt_kpt_mask = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

            

            
            # print(f"gt_kpts: {describe_var(gt_kpts)}")
            # print(f"gt_kpt_mask: {describe_var(gt_kpt_mask)}")

            # # 將 gt_kpt_mask tensor([3, 8400, 17]) 寫到 .txt
            # with open("gt_kpt_mask.txt", "w") as f:
            #     for i in range(gt_kpt_mask.shape[0]):
            #         for j in range(gt_kpt_mask.shape[1]):
            #             for k in range(gt_kpt_mask.shape[2]):
            #                 f.write(f"{gt_kpt_mask[i, j, k]} ")
            #         f.write("\n")

            if "teacher" in batch and batch["teacher"] is not None:
                dpose, dkobj, gt_kpts2, gt_kpt_mask2 = self.calculate_keypoints_loss(
                    fg_mask2, target_gt_idx2, keypoints2, batch_idx2, stride_tensor, target_bboxes2, pred_kpts
                )
                # print(f"gt_kpts2: {describe_var(gt_kpts2)}")
                # print(f"gt_kpt_mask2: {describe_var(gt_kpt_mask2)}")

                # # 將 gt_kpt_mask2 tensor([3, 8400, 17]) 寫到 .txt
                # with open("gt_kpt_mask2.txt", "w") as f:
                #     for i in range(gt_kpt_mask2.shape[0]):
                #         for j in range(gt_kpt_mask2.shape[1]):
                #             for k in range(gt_kpt_mask2.shape[2]):
                #                 f.write(f"{gt_kpt_mask2[i, j, k]} ")
                #         f.write("\n")

            # gt_kpts, gt_kpt_mask = self.calculate_distill_keypoints_loss(
            #     teacher_fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            # )


            # if "teacher" in batch and batch["teacher"] is not None:
            #     # 可视化图像、边界框和关键点
            #     # print(f"gt_kpts: {describe_var(gt_kpts)}")
            #     target_keypoints = torch.cat([gt_kpts[..., :2] * stride_tensor.view(1, -1, 1, 1), gt_kpts[..., 2:]], dim=-1)
            #     # print(f"keypoints: {describe_var(keypoints)}")
            #     student_pred_kpts = torch.cat([pred_kpts[..., :2] * stride_tensor.view(1, -1, 1, 1), pred_kpts[..., 2:]], dim=-1)
            #     # print(f"student_pred_kpts: {describe_var(student_pred_kpts)}")
            #     teacher_pred_kpts = torch.cat([gt_kpts2[..., :2] * stride_tensor.view(1, -1, 1, 1), gt_kpts2[..., 2:]], dim=-1)
            #     # print(f"teacher_pred_kpts: {describe_var(teacher_pred_kpts)}")
            #     self.visualize_batch(
            #         images=batch["img"].detach(),
            #         target_bboxes=target_bboxes.detach() * stride_tensor,
            #         keypoints=target_keypoints,
            #         keypoint_mask=gt_kpt_mask,
            #         orig_img=batch.get("orig_img", None),
            #         student_pred_bboxes=pred_bboxes.detach() * stride_tensor,
            #         student_pred_kpts=student_pred_kpts,
            #         teacher_pred_bboxes=target_bboxes2.detach() * stride_tensor,
            #         teacher_pred_kpts=teacher_pred_kpts,
            #         teacher_keypoint_mask=gt_kpt_mask2,
            #         stride_tensor=stride_tensor.detach(),
            #         fg_mask=fg_mask.detach(),
            #         imgsz=imgsz.detach(),
            #         fg_mask2=fg_mask2.detach(),
            #     )

        if "teacher" in batch and batch["teacher"] is not None:
            loss[5] = dpose
            loss[6] = dkobj
        else:
            loss[5] = torch.zeros(1, device=self.device, requires_grad=True)
            loss[6] = torch.zeros(1, device=self.device, requires_grad=True)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.dpose  # dpose gain
        loss[6] *= self.hyp.dkobj  # dkobj gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def extract_keypoints(self,tensor_list):
        # Create a list to hold all extracted data
        all_keypoints = []
        
        # Process each tensor in the list
        for tensor in tensor_list:
            # Get the last 51 elements from each row and reshape to [n, 17, 3]
            keypoints = tensor[:, -51:].reshape(-1, 17, 3)
            all_keypoints.append(keypoints)
        
        # Concatenate all tensors along the first dimension
        return torch.cat(all_keypoints, dim=0)
    
    def create_batch_idx_tensor(self, tensor_list):
        batch_indices = []
        for i, tensor in enumerate(tensor_list):
            # For each tensor, add 'i' repeated 'tensor.shape[0]' times
            batch_indices.extend([i] * tensor.shape[0])
        
        # Convert to tensor and reshape to [N, 1]
        return torch.tensor(batch_indices, dtype=torch.float32).view(-1, 1)

    def preprocess2(self, tensors_list):
        # Find the batch size (number of tensors in the list)
        batch_size = len(tensors_list)
        
        # Find the maximum number of instances across all tensors
        max_instances = max(tensor.shape[0] for tensor in tensors_list)
        
        # Create empty output tensor with zeros [batch_size, max_instances, 5]
        result = torch.zeros(batch_size, max_instances, 5, 
                            dtype=tensors_list[0].dtype, 
                            device=tensors_list[0].device)
        
        # Fill in data for each batch
        for i, tensor in enumerate(tensors_list):
            # Get the number of instances in this tensor
            n_instances = tensor.shape[0]
            
            # Copy the first 5 columns for all instances in this tensor
            result[i, :n_instances] = torch.cat((torch.zeros(n_instances, 1, device=tensor.device), tensor[:, 0:4]), dim=1)
        
        return result

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

        # print("calculate_keypoints_loss called")

        # print(f"keypoints: {describe_var(keypoints)}")

        # 批次索引處理
        # 目的: 將批次索引展平為一維向量，並獲取批次大小
        # 說明: batch_idx 記錄每組關鍵點屬於哪個批次，這一步是為了後續處理做準備
        # print(f"batch_idx: {batch_idx}, shape: {batch_idx.shape}")
        batch_idx = batch_idx.flatten()
        # print(f"batch_idx2: {batch_idx}")
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        # 計算每個批次的最大關鍵點數量
        # 目的: 找出單個圖像中最多包含的關鍵點數量
        # 說明:
        # - torch.unique(batch_idx, return_counts=True) 返回兩個值：唯一的批次索引和每個批次包含的關鍵點數量
        # - [1].max() 獲取計數中的最大值，即最多的關鍵點數量
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        # 創建批次化關鍵點張量
        # 目的: 創建一個固定大小的張量來存儲所有批次的關鍵點
        # 說明:
        # 形狀為 [batch_size, max_kpts, N_kpts_per_object, kpts_dim]
        # 例如：對於 COCO 格式，N_kpts_per_object=17, kpts_dim=3 (x, y, visibility)
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )
        # print(f"batched_keypoints: {describe_var(batched_keypoints)}")

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        # 填充批次化關鍵點張量
        # 目的: 根據批次索引將關鍵點重新組織到批次中
        # 說明:
        #   - 這個循環處理每個批次
        #   - keypoints[batch_idx == i] 選擇屬於批次 i 的所有關鍵點
        #   - 將這些關鍵點放入 batched_keypoints 的對應位置
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # print(f"batched_keypoints2: {describe_var(batched_keypoints)}")

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        # 擴展 target_gt_idx 維度
        # 目的: 將 target_gt_idx 擴展為與 batched_keypoints 兼容的形狀
        # 說明:
        #   - 原始形狀: [BS, N_anchors]
        #   - 擴展後形狀: [BS, N_anchors, 1, 1]
        #   - 這樣做是為了後續的 gather 操作
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        # 使用索引選擇對應的關鍵點
        # 目的: 從重組後的關鍵點中選擇與每個前景錨點對應的真實關鍵點
        # 說明:
        #   - target_gt_idx_expanded.expand(...) 將索引擴展為與 batched_keypoints 相同的維度
        #   - gather(1, ...) 根據第1維（錨點維度）的索引選擇對應的關鍵點
        #   - 這一步確保每個前景錨點都與正確的真實關鍵點對應
        # # torch.Size([2, 12, 17, 3])
        # print(f"batched_keypoints.shape: {batched_keypoints.shape}", batched_keypoints)
        # # torch.Size([2, 8400, 1, 1])
        # print(f"target_gt_idx_expanded.shape: {target_gt_idx_expanded.shape}", target_gt_idx_expanded) 
        # # torch.Size([2, 8400, 1, 1])
        # print(f"keypoints.shape: {keypoints.shape}", keypoints)
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        # 座標縮放
        # 目的: 將關鍵點座標除以對應的步長，轉換到特徵圖尺度
        # 說明:
        #   - 只處理前兩個通道 (x, y)，不處理可見性通道
        #   - stride_tensor.view(1, -1, 1, 1) 重塑步長張量以便於廣播
        # print(f"selected_keypoints1: {describe_var(selected_keypoints)}")
        # print(f"stride_tensor.view(1, -1, 1, 1): {describe_var(stride_tensor.view(1, -1, 1, 1))}")
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        # print(f"selected_keypoints2: {describe_var(selected_keypoints)}")
        scaled_gt_keypoints = selected_keypoints.detach()
        # print(f"scaled_gt_keypoints: {describe_var(scaled_gt_keypoints.to(self.device))}")
        target_kpt_mask = scaled_gt_keypoints[..., 2] > 0.25 if scaled_gt_keypoints.shape[-1] == 3 else torch.full_like(scaled_gt_keypoints[..., 0], True)
        # print(f"target_kpt_mask: {describe_var(target_kpt_mask)}")
        # target_kpts = scaled_gt_keypoints[target_kpt_mask]
        # print(f"target_kpts: {describe_var(target_kpts)}")

        kpts_loss = 0
        kpts_obj_loss = 0

        # 計算損失
        # 目的: 計算關鍵點位置損失和關鍵點可見性損失
        if masks.any():
            # 擇前景掩碼為真的真實關鍵點
            gt_kpt = selected_keypoints[masks]
            # 計算面積
            # 將 xyxy 格式轉為 xywh 格式，取寬高進行相乘，得到面積
            # 面積用於損失計算中的標準化
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            # 選擇前景掩碼為真的預測關鍵點
            pred_kpt = pred_kpts[masks]
            # print("gt_kpt", describe_var(gt_kpt))
            # 創建關鍵點掩碼:
            # 如果關鍵點有可見性信息 (第3通道)，則只考慮可見的關鍵點 (可見性!=0)
            # 否則所有關鍵點都視為可見
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            # print("kpt_mask", describe_var(kpt_mask))
            # 位置損失計算:
            # 使用 OKS (Object Keypoint Similarity) 計算姿態損失
            # 考慮面積進行標準化，較大的物體允許有較大的絕對誤差
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss
            # 可見性損失計算:
            # 使用二元交叉熵損失計算關鍵點可見性預測的損失
            # 只在預測包含可見性通道時計算
            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        # kpts_loss: 關鍵點位置損失，評估預測關鍵點座標與真實關鍵點座標的差異
        # kpts_obj_loss: 關鍵點可見性損失，評估關鍵點可見性預測的準確度
        # 技術要點總結
        #  - 批次處理：函數能處理多張圖片的批次數據，通過批次索引追蹤每個關鍵點的來源
        #  - 索引映射：使用 target_gt_idx 將前景錨點映射到對應的真實關鍵點
        #  - 座標尺度轉換：通過除以步長將關鍵點座標從原始尺度轉換到特徵圖尺度
        #  - 自適應損失計算：考慮關鍵點可見性和物體面積，使損失計算更準確
        #  - 靈活性：支持帶可見性信息的關鍵點 (3通道) 和不帶可見性信息的關鍵點 (2通道)
        # 這個函數是 YOLOv8 姿態估計模型訓練的核心部分，通過精確的關鍵點損失計算，使模型能夠學習準確預測人體/物體的關鍵點位置。
        return kpts_loss, kpts_obj_loss, scaled_gt_keypoints, target_kpt_mask

    def visualize_batch(
        self, 
        images, 
        target_bboxes, 
        keypoints=None, 
        keypoint_mask=None,
        orig_img=None, 
        output_dir=Path('outputs/visualization'),
        student_pred_bboxes=None,
        student_pred_kpts=None,
        teacher_pred_bboxes=None,
        teacher_pred_kpts=None,
        stride_tensor=None,
        imgsz=None,
        fg_mask=None,
        fg_mask2=None,
        teacher_keypoint_mask=None,
    ):
        """
        可视化批次中的图像、边界框和关键点
        
        Args:
            images: 批次中的图像张量 [B, C, H, W]
            bboxes: 边界框坐标 [N, 4+] 格式为 [batch_idx, class, x, y, w, h]
            keypoints: 关键点坐标 [N, K, 3] 格式为 [x, y, visibility]
            orig_img: 原始图像列表（如果可用）
            output_dir: 输出目录
            student_pred_bboxes: 学生模型预测的边界框 [B, A, 4] 格式为 [x1, y1, x2, y2]
            student_pred_kpts: 学生模型预测的关键点 [B, A, K, 3] 格式为 [x, y, visibility]
            teacher_pred_bboxes: 教师模型预测的边界框 [B, A, 4] 格式为 [x1, y1, x2, y2]
            teacher_pred_kpts: 教师模型预测的关键点 [B, A, K, 3] 格式为 [x, y, visibility]
            stride_tensor: 步长张量，用于坐标转换
            imgsz: 图像大小，用于坐标转换
        """
        print(f"\n开始可视化批次，保存到: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        batch_size = images.shape[0]
        
        # 指定关键点连接关系（对于COCO 17个关键点）
        keypoint_connections = [
            (0, 1), (1, 3), (0, 2), (2, 4),  # 脸部和头部
            (5, 7), (7, 9), (6, 8), (8, 10),  # 胳膊
            (5, 6), (5, 11), (6, 12),  # 身体
            (11, 13), (13, 15), (12, 14), (14, 16)  # 腿部
        ]
        
        # 关键点颜色
        keypoint_colors = {
            'gt': 'green',         # 真实标签
            'student': 'blue',     # 学生模型预测
            'teacher': 'red'       # 教师模型预测
        }
        
        # 处理每张图像
        for idx in range(batch_size):
            print(f"处理图像: {idx}")
                
            # 获取图像
            if orig_img is not None and idx < len(orig_img):
                # 使用原始图像
                img = orig_img[idx].copy()
            else:
                # 转换处理后的图像
                img = images[idx].detach().permute(1, 2, 0).cpu().numpy()
                
                # 检查图像是否已归一化到[0-1]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 图像尺寸
            img_h, img_w = img.shape[:2]
            
            # 创建Matplotlib图形
            fig, ax = plt.subplots(1, figsize=(12, 12))
            ax.imshow(img)

            batch_fg_mask = fg_mask[idx]

            batch_target_bboxes = target_bboxes[idx]
            batch_fg_target_bboxes = batch_target_bboxes[batch_fg_mask]
            batch_target_keypoints = keypoints[idx]
            # print(f"batch_target_keypoints: {describe_var(batch_target_keypoints)}")
            # print(f"keypoint_mask: {describe_var(keypoint_mask)}")
            batch_fg_target_keypoints = batch_target_keypoints[batch_fg_mask]
            # print(f"batch_fg_target_keypoints: {describe_var(batch_fg_target_keypoints)}")

            # student
            batch_student_pred_bboxes = student_pred_bboxes[idx]
            batch_fg_student_pred_bboxes = batch_student_pred_bboxes[batch_fg_mask]
            batch_student_pred_kpts = student_pred_kpts[idx]
            batch_fg_student_pred_kpts = batch_student_pred_kpts[batch_fg_mask]
            
            # print(f"batch_fg_keypoints{idx}: {describe_var(batch_fg_keypoints)}", batch_fg_keypoints)
            
            # ------------------------------------------------------------

            batch_teacher_fg_mask = fg_mask2[idx]

            batch_teacher_pred_bboxes = teacher_pred_bboxes[idx]
            batch_fg_teacher_pred_bboxes = batch_teacher_pred_bboxes[batch_teacher_fg_mask]
            batch_teacher_pred_kpts = teacher_pred_kpts[idx]
            batch_fg_teacher_pred_kpts = batch_teacher_pred_kpts[batch_teacher_fg_mask]

            if batch_fg_target_bboxes is not None and len(batch_fg_target_bboxes) > 0:
                for i, bbox in enumerate(batch_fg_target_bboxes):
                    try:
                        # 获取预测框 (xyxy格式)
                        bbox_data = bbox.detach().cpu().numpy()
                        
                        # 只处理4维坐标 (x1, y1, x2, y2)
                        if len(bbox_data) >= 4:
                            x1, y1, x2, y2 = bbox_data[:4]
                            
                            # 计算宽高
                            w = x2 - x1
                            h = y2 - y1
                            
                            # 绘制矩形
                            rect = patches.Rectangle(
                                (x1, y1), w, h, 
                                linewidth=2, edgecolor='g', facecolor='none', linestyle='--'
                            )
                            ax.add_patch(rect)
                            ax.text(
                                x1, y1, f"GT {i}", 
                                color='white', fontsize=10,
                                bbox=dict(facecolor='green', alpha=0.5)
                            )
                            
                            print(f"  绘制目標预测框: 左上角=({x1:.1f}, {y1:.1f}), 宽高=({w:.1f}, {h:.1f})")
                    except Exception as e:
                        print(f"  绘制目標预测框时出错: {e}")

                for i, kpt in enumerate(batch_fg_target_keypoints):
                    try:
                        kpt_data = kpt.detach().cpu().numpy()
                        # x, y, v = kpt_data[:3]
                            
                        self._draw_keypoints(ax, kpt_data, keypoint_connections, 
                                            1, 1, scale=False, color=keypoint_colors['gt'])
                        print(f"  绘制目標预测框的关键点")
                    except Exception as e:
                        print(f"  绘制目標预测框的关键点时出错: {e}")

 


            # 绘制学生模型预测框
            if batch_fg_student_pred_bboxes is not None and len(batch_fg_student_pred_bboxes) > 0:
                for i, bbox in enumerate(batch_fg_student_pred_bboxes):
                    try:
                        # 获取预测框 (xyxy格式)
                        bbox_data = bbox.detach().cpu().numpy()
                        
                        # 只处理4维坐标 (x1, y1, x2, y2)
                        if len(bbox_data) >= 4:
                            x1, y1, x2, y2 = bbox_data[:4]
                            
                            # 计算宽高
                            w = x2 - x1
                            h = y2 - y1
                            
                            # 绘制矩形
                            rect = patches.Rectangle(
                                (x1, y1), w, h, 
                                linewidth=2, edgecolor='b', facecolor='none', linestyle='--'
                            )
                            ax.add_patch(rect)
                            ax.text(
                                x1, y1, f"Student {i}", 
                                color='white', fontsize=10,
                                bbox=dict(facecolor='blue', alpha=0.5)
                            )
                            
                            print(f"  绘制学生预测框: 左上角=({x1:.1f}, {y1:.1f}), 宽高=({w:.1f}, {h:.1f})")
                    except Exception as e:
                        print(f"  绘制学生预测框时出错: {e}")

                for i, kpt in enumerate(batch_fg_student_pred_kpts):
                    try:
                        kpt_data = kpt.detach().cpu().numpy()
                        # x, y, v = kpt_data[:3]
                            
                        self._draw_keypoints(ax, kpt_data, keypoint_connections, 
                                            1, 1, scale=False, color=keypoint_colors['student'])
                        print(f"  绘制学生预测框的关键点")
                    except Exception as e:
                        print(f"  绘制学生预测框的关键点时出错: {e}")



            # 绘制教師模型预测框
            if batch_fg_teacher_pred_bboxes is not None and len(batch_fg_teacher_pred_bboxes) > 0:
                for i, bbox in enumerate(batch_fg_teacher_pred_bboxes):
                    try:
                        # 获取预测框 (xyxy格式)
                        bbox_data = bbox.detach().cpu().numpy()
                        
                        # 只处理4维坐标 (x1, y1, x2, y2)
                        if len(bbox_data) >= 4:
                            x1, y1, x2, y2 = bbox_data[:4]
                            
                            # 计算宽高
                            w = x2 - x1
                            h = y2 - y1
                            
                            # 绘制矩形
                            rect = patches.Rectangle(
                                (x1, y1), w, h, 
                                linewidth=2, edgecolor='r', facecolor='none', linestyle='--'
                            )
                            ax.add_patch(rect)
                            ax.text(
                                x1, y1, f"Teacher {i}", 
                                color='white', fontsize=10,
                                bbox=dict(facecolor='red', alpha=0.5)
                            )
                            
                            print(f"  绘制教师预测框: 左上角=({x1:.1f}, {y1:.1f}), 宽高=({w:.1f}, {h:.1f})")
                    except Exception as e:
                        print(f"  绘制教师预测框时出错: {e}")

                for i, kpt in enumerate(batch_fg_teacher_pred_kpts):
                    try:
                        kpt_data = kpt.detach().cpu().numpy()
                        # x, y, v = kpt_data[:3]
                            
                        self._draw_keypoints(ax, kpt_data, keypoint_connections, 
                                            1, 1, scale=False, color=keypoint_colors['teacher'])
                        print(f"  绘制教師预测框的关键点")
                    except Exception as e:
                        print(f"  绘制教師预测框的关键点时出错: {e}")


            # 添加图例
            custom_lines = [
                patches.Patch(color='green', label='Ground Truth'),
                patches.Patch(color='blue', label='Student Prediction'),
                patches.Patch(color='red', label='Teacher Prediction')
            ]
            ax.legend(handles=custom_lines, loc='upper right')
            
            # 保存图像
            plt.axis('off')
            # plt.tight_layout()
            plt.savefig(output_dir / f'batch_img_{idx}.jpg', bbox_inches='tight', pad_inches=0.1, dpi=200)
            plt.close()
            
            print(f"已保存图像: {output_dir / f'batch_img_{idx}.jpg'}")

    def _draw_keypoints(self, ax, keypoints, connections, img_w, img_h, scale=True, color='g', linewidth=2, markersize=6):
        """
        绘制关键点和连接线
        
        Args:
            ax: matplotlib轴对象
            keypoints: 关键点数组，形状为(K, 3)，包含[x, y, visibility]
            connections: 关键点连接的列表，每项为(idx1, idx2)
            img_w, img_h: 图像尺寸
            scale: 是否缩放坐标到图像尺寸
            color: 绘制颜色
            linewidth: 线宽
            markersize: 点大小
        """
        if len(keypoints) == 0:
            return
            
        # 确保关键点是 NumPy 数组
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy()
            
        # 筛选可见的关键点 (可见性 > 0.5)
        visible = keypoints[:, 2] > 0.5 if keypoints.shape[1] >= 3 else np.ones(len(keypoints), dtype=bool)
        
        # 绘制关键点
        for i, kpt in enumerate(keypoints):
            if visible[i]:
                x, y = kpt[0], kpt[1]
                if scale:
                    x *= img_w
                    y *= img_h
                    
                ax.plot(x, y, 'o', color=color, markersize=markersize)
                ax.text(x+3, y+3, str(i), color=color, fontsize=8)
        
        # 绘制连接线
        for connection in connections:
            i, j = connection
            if i < len(keypoints) and j < len(keypoints) and visible[i] and visible[j]:
                x1, y1 = keypoints[i, 0], keypoints[i, 1]
                x2, y2 = keypoints[j, 0], keypoints[j, 1]
                if scale:
                    x1 *= img_w
                    y1 *= img_h
                    x2 *= img_w
                    y2 *= img_h
                    
                ax.plot([x1, x2], [y1, y2], '-', color=color, linewidth=linewidth, alpha=0.7)
