import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import LOGGER

class SpatialPoseDistillation(nn.Module):
    """空间感知的姿态特化蒸馏方法"""
    
    def __init__(self, channels_s, channels_t):
        super().__init__()
        
        # 确保所有模块在与模型相同的设备和数据类型上创建
        self.align = nn.ModuleList()
        for s_chan, t_chan in zip(channels_s, channels_t):
            self.align.append(
                nn.Sequential(
                    nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_chan)
                )
            )
        
        # 人体部位感知模块
        self.pose_attention = nn.ModuleList()
        for _ in range(len(channels_t)):
            self.pose_attention.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(1, 1, kernel_size=3, padding=1),
                    nn.Sigmoid()
                )
            )
            
    def forward(self, y_s, y_t):
        losses = []
        
        for i, (s, t) in enumerate(zip(y_s, y_t)):
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                continue
                
            # 获取输入特征的设备和数据类型
            device = s.device
            dtype = s.dtype
            
            # 确保模块的权重也使用相同的数据类型
            if next(self.align[i].parameters()).dtype != dtype:
                # 将模块转换为与输入相同的数据类型
                self.align[i] = self.align[i].to(dtype)
                if i < len(self.pose_attention):
                    self.pose_attention[i] = self.pose_attention[i].to(dtype)
                
            # 通道对齐，现在数据类型应该匹配
            try:
                s_aligned = self.align[i](s)
            except RuntimeError as e:
                # 如果仍然失败，尝试显式转换
                LOGGER.warning(f"类型转换失败，尝试显式转换: {e}")
                s = s.to(next(self.align[i].parameters()).dtype)
                s_aligned = self.align[i](s)
            
            # 确保教师特征也使用相同的数据类型
            t = t.to(dtype)
            
            # 空间对齐
            if s_aligned.size(2) != t.size(2) or s_aligned.size(3) != t.size(3):
                t = F.interpolate(t, size=(s_aligned.size(2), s_aligned.size(3)),
                                 mode='bilinear', align_corners=True)
            
            # 1. 一般特征图损失 - MSE
            general_loss = F.mse_loss(s_aligned, t)
            
            # 2. 高激活区域损失 - 可能是关键点区域
            # 提取可能包含关键点的高激活区域
            t_abs = torch.mean(torch.abs(t), dim=1, keepdim=True)
            t_mask = (t_abs > torch.mean(t_abs) * 1.5).float()
            
            # 对这些区域使用加权损失
            keypoint_region_loss = F.mse_loss(s_aligned * t_mask, t * t_mask) * 2.0
            
            # 3. 空间梯度损失 - 保持边缘和结构
            # 水平和垂直梯度
            s_dx = s_aligned[:, :, :, 1:] - s_aligned[:, :, :, :-1]
            t_dx = t[:, :, :, 1:] - t[:, :, :, :-1]
            s_dy = s_aligned[:, :, 1:, :] - s_aligned[:, :, :-1, :]
            t_dy = t[:, :, 1:, :] - t[:, :, :-1, :]
            
            # 梯度一致性损失
            gradient_loss = (F.mse_loss(s_dx, t_dx) + F.mse_loss(s_dy, t_dy)) * 0.5
            
            # 总损失
            total_loss = general_loss + keypoint_region_loss + gradient_loss
            losses.append(total_loss)
            
        return sum(losses)

class AdaptiveDistillationLoss(nn.Module):
    """自适应蒸馏损失，保护基本任务能力"""
    
    def __init__(self, channels_s, channels_t, original_performance=0.5):
        super().__init__()
        self.channels_s = channels_s
        self.channels_t = channels_t
        self.original_performance = original_performance
        self.current_performance = original_performance
        self.adaptivity = 0.5  # 自适应系数
        
        # 特征对齐模块
        self.align_layers = nn.ModuleList()
        for s_chan, t_chan in zip(channels_s, channels_t):
            self.align_layers.append(
                nn.Sequential(
                    nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_chan)
                )
            )
        
        LOGGER.info(f"初始化自适应蒸馏损失，基准性能设为 {original_performance:.4f}")
        
    def update_performance(self, current_perf):
        """更新当前性能，用于自适应调整"""
        self.current_performance = current_perf
        
    def forward(self, student_features, teacher_features):
        losses = []
        
        # 计算性能保护因子 - 性能下降严重时降低蒸馏强度
        if self.current_performance < 0.2 * self.original_performance:
            # 性能下降超过80%，强保护
            protection_factor = 0.2
            LOGGER.debug(f"性能严重下降至 {self.current_performance:.4f}，应用强保护因子 {protection_factor}")
        elif self.current_performance < 0.5 * self.original_performance:
            # 性能下降超过50%，中度保护
            protection_factor = 0.5
            LOGGER.debug(f"性能中度下降至 {self.current_performance:.4f}，应用中度保护因子 {protection_factor}")
        else:
            # 性能良好，正常蒸馏
            protection_factor = 1.0
        
        # 应用特征对齐和损失计算
        for i, (s, t) in enumerate(zip(student_features, teacher_features)):
            # 检查特征是否有效
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                continue
                
            try:
                # 对齐学生特征到教师特征空间
                s_aligned = self.align_layers[i](s)
                
                # 空间大小对齐
                if s_aligned.size(2) != t.size(2) or s_aligned.size(3) != t.size(3):
                    t = F.interpolate(t, size=(s_aligned.size(2), s_aligned.size(3)),
                                     mode='bilinear', align_corners=False)
                
                # 计算带保护的蒸馏损失
                distill_loss = F.mse_loss(s_aligned, t) * protection_factor
                losses.append(distill_loss)
            except Exception as e:
                LOGGER.error(f"处理特征时出错: {e}")
                continue
        
        if not losses:
            return torch.tensor(0.0, device=next(self.parameters()).device, requires_grad=True)
            
        return sum(losses)

class EnhancedDistillationLoss(nn.Module):
    """增强的蒸馏损失，专为姿态估计任务优化"""
    
    def __init__(self, channels_s, channels_t, distiller='enhancedfgd', loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        
        # 将所有模块移动到相同精度
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 转换为ModuleList并确保一致的数据类型
        self.align_module = nn.ModuleList()
        self.norm = nn.ModuleList()
        
        # 创建对齐模块
        for s_chan, t_chan in zip(channels_s, channels_t):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(t_chan, affine=False)
            ).to(device)
            self.align_module.append(align)
            
        # 创建归一化层
        for t_chan in channels_t:
            self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
            
        # 关键点位置注意力系数
        self.keypoint_attention = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to(device)
        
        LOGGER.info(f"初始化增强蒸馏损失函数: {distiller}")
            
    def forward(self, y_s, y_t):
        """计算特征层蒸馏损失"""
        min_len = min(len(y_s), len(y_t))
        y_s = y_s[:min_len]
        y_t = y_t[:min_len]

        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if idx >= len(self.align_module):
                break
            
            # 处理不同类型的特征
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                if isinstance(s, list) and len(s) > 0:
                    s = s[0]
                if isinstance(t, list) and len(t) > 0:
                    t = t[0]
            
            # 确保特征是张量
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                continue
                
            # 转换数据类型以匹配对齐模块
            s = s.type(next(self.align_module[idx].parameters()).dtype)
            t = t.type(next(self.align_module[idx].parameters()).dtype)

            try:
                # 对齐通道数
                if s.shape[1] != t.shape[1]:
                    s = self.align_module[idx](s)
                
                # 空间对齐
                if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                    t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
                stu_feats.append(s)
                tea_feats.append(t.detach())
            except Exception as e:
                LOGGER.error(f"处理特征时出错: {e}")
                continue

        # 安全检查
        if len(stu_feats) == 0 or len(tea_feats) == 0:
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
        
        return self._keypoint_aware_distillation(stu_feats, tea_feats)
    
    def _keypoint_aware_distillation(self, y_s, y_t):
        """关键点感知的蒸馏方法"""
        losses = []
        
        for s, t in zip(y_s, y_t):
            b, c, h, w = s.shape
            
            # 1. 生成特征激活图
            s_act = torch.mean(torch.abs(s), dim=1, keepdim=True)  # B,1,H,W
            t_act = torch.mean(torch.abs(t), dim=1, keepdim=True)  # B,1,H,W
            
            # 2. 识别教师模型中的高激活区域（可能是关键点位置）
            keypoint_mask = (t_act > t_act.mean() * 1.5).float()
            
            # 3. 使用关键点注意力模块增强掩码
            enhanced_mask = self.keypoint_attention(keypoint_mask)
            
            # 4. 加权蒸馏损失
            # 一般区域损失
            general_loss = F.mse_loss(s, t)
            
            # 关键点区域损失（加权）
            keypoint_loss = F.mse_loss(s * enhanced_mask, t * enhanced_mask) * 3.0
            
            # 空间注意力损失
            spatial_loss = F.mse_loss(s_act, t_act) * 1.5
            
            # 通道相关性损失
            s_flat = s.view(b, c, -1)
            t_flat = t.view(b, c, -1)
            s_corr = torch.bmm(s_flat, s_flat.transpose(1, 2)) / (h*w)
            t_corr = torch.bmm(t_flat, t_flat.transpose(1, 2)) / (h*w)
            channel_loss = F.mse_loss(s_corr, t_corr) * 0.5
            
            # 整合损失
            total_loss = general_loss + keypoint_loss + spatial_loss + channel_loss
            losses.append(total_loss)
            
        return sum(losses) * self.loss_weight

class CWDLoss(nn.Module):
    """Channel-wise Knowledge Distillation for Dense Prediction.
    https://arxiv.org/abs/2011.13256
    """
    def __init__(self, channels_s, channels_t, tau=1.0):
        super().__init__()
        self.tau = tau
        
    def forward(self, y_s, y_t):
        """計算CWD損失
        
        Args:
            y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
            y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            
        Returns:
            torch.Tensor: 所有階段的計算損失總和
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            N, C, H, W = s.shape
            # 在通道維度上歸一化
            softmax_pred_T = F.softmax(t.view(-1, W * H) / self.tau, dim=1)
            logsoftmax = torch.nn.LogSoftmax(dim=1)
            cost = torch.sum(
                softmax_pred_T * logsoftmax(t.view(-1, W * H) / self.tau) -
                softmax_pred_T * logsoftmax(s.view(-1, W * H) / self.tau)) * (self.tau ** 2)
            losses.append(cost / (C * N))
        loss = sum(losses)
        return loss

class MGDLoss(nn.Module):
    """Masked Generative Distillation"""
    def __init__(self, student_channels, teacher_channels, alpha_mgd=0.00002, lambda_mgd=0.65):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generation = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(t_chan, t_chan, kernel_size=3, padding=1)
            ).to(device) for s_chan, t_chan in zip(student_channels, teacher_channels)
        ])
        
    def forward(self, y_s, y_t, layer=None):
        """計算MGD損失
        
        Args:
            y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
            y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            layer (str, optional): 指定要蒸餾的層，如果是 "outlayer" 則使用最後一層
            
        Returns:
            torch.Tensor: 所有階段的計算損失總和
        """
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            if layer == "outlayer":
                idx = -1
            losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
        loss = sum(losses)
        return loss
    
    def get_dis_loss(self, preds_S, preds_T, idx):
        """計算單層的MGD損失
        
        Args:
            preds_S: 學生模型的預測
            preds_T: 教師模型的預測
            idx: 層索引
            
        Returns:
            torch.Tensor: 計算的損失值
        """
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device
        mat = torch.rand((N, 1, H, W)).to(device)
        mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)
        masked_fea = torch.mul(preds_S, mat)
        new_fea = self.generation[idx](masked_fea)
        dis_loss = loss_mse(new_fea, preds_T) / N
        return dis_loss

class ReviewKDLoss(nn.Module):
    """Review-KD: https://arxiv.org/abs/2104.09044"""
    def __init__(self, student_channels, teacher_channels, temperature=1.0):
        super(ReviewKDLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, y_s, y_t):
        """計算Review-KD損失
        
        Args:
            y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
            y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            
        Returns:
            torch.Tensor: 所有階段的計算損失總和
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            b, c, h, w = s.shape
            s = s.view(b, c, -1)
            t = t.view(b, c, -1)
            
            # 使用softmax和KL散度
            s = F.log_softmax(s / self.temperature, dim=2)
            t = F.softmax(t / self.temperature, dim=2)
            loss = F.kl_div(s, t, reduction='batchmean') * (self.temperature ** 2)
            losses.append(loss)
        loss = sum(losses)
        return loss

class FGDLoss(nn.Module):
    """增強版特徵引導蒸餾 - 專為姿態估計優化"""
    def __init__(self, student_channels, teacher_channels, spatial_weight=2.0, channel_weight=0.6):
        super(FGDLoss, self).__init__()
        self.spatial_weight = spatial_weight    # 更強的空間權重
        self.channel_weight = channel_weight    # 降低通道權重
        # 增加邊緣感知機制
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_filter.weight.data.copy_(torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3) / 8.0)
        self.edge_filter.requires_grad_(False)  # 凍結參數
        
    def forward(self, y_s, y_t):
        """計算增強版FGD損失，特別優化姿態估計的空間特徵保留
        
        Args:
            y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
            y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            
        Returns:
            torch.Tensor: 所有階段的計算損失總和
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            b, c, h, w = s.shape
            
            # 1. 基本特徵匹配 - 使用Huber損失減少異常值影響
            l2_loss = F.smooth_l1_loss(s, t)
            
            # 2. 增強版空間注意力損失
            s_spatial = torch.mean(s, dim=1, keepdim=True)  # [b, 1, h, w]
            t_spatial = torch.mean(t, dim=1, keepdim=True)  # [b, 1, h, w]
            
            # 提取空間特徵的邊緣信息
            s_edge = self.edge_filter(s_spatial)
            t_edge = self.edge_filter(t_spatial)
            
            # 空間注意力+邊緣感知的組合損失
            spatial_loss = (F.mse_loss(s_spatial, t_spatial) + 
                           F.mse_loss(s_edge, t_edge)) * self.spatial_weight
            
            # 3. 通道關係損失 - 專注於重要通道
            s_flat = s.view(b, c, -1)  # [b, c, h*w]
            t_flat = t.view(b, c, -1)  # [b, c, h*w]
            
            # 通道歸一化
            s_flat_norm = F.normalize(s_flat, dim=2)
            t_flat_norm = F.normalize(t_flat, dim=2)
            
            # 計算通道相關矩陣
            s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
            t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
            
            # 通道相關性損失
            channel_loss = F.mse_loss(s_corr, t_corr) * self.channel_weight
            
            # 4. 組合損失
            total_loss = l2_loss + spatial_loss + channel_loss
            losses.append(total_loss)
            
        loss = sum(losses)
        return loss

class PKDLoss(nn.Module):
    """Probabilistic Knowledge Distillation"""
    def __init__(self, student_channels, teacher_channels):
        super(PKDLoss, self).__init__()
        
    def forward(self, y_s, y_t):
        """計算PKD損失
        
        Args:
            y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
            y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            
        Returns:
            torch.Tensor: 所有階段的計算損失總和
        """
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            # 計算通道間相關性
            b, c, h, w = s.shape
            s_flat = s.view(b, c, -1)
            t_flat = t.view(b, c, -1)
            
            s_mean = s_flat.mean(dim=2, keepdim=True)
            t_mean = t_flat.mean(dim=2, keepdim=True)
            
            s_centered = s_flat - s_mean
            t_centered = t_flat - t_mean
            
            s_var = torch.mean(s_centered**2, dim=2) + 1e-6
            t_var = torch.mean(t_centered**2, dim=2) + 1e-6
            
            s_std = torch.sqrt(s_var).unsqueeze(2)
            t_std = torch.sqrt(t_var).unsqueeze(2)
            
            s_normalized = s_centered / s_std
            t_normalized = t_centered / t_std
            
            # 計算相關矩陣
            corr_s = torch.matmul(s_normalized, s_normalized.transpose(1, 2)) / h / w
            corr_t = torch.matmul(t_normalized, t_normalized.transpose(1, 2)) / h / w
            
            # PKD損失
            loss = F.mse_loss(corr_s, corr_t)
            losses.append(loss)
        loss = sum(losses)
        return loss

class EnhancedFGDLoss(nn.Module):
    """增強版特徵引導蒸餾 - 專為姿態估計優化，提供更好的關鍵點定位能力"""
    def __init__(self, student_channels, teacher_channels, spatial_weight=3.5, channel_weight=0.4):
        super(EnhancedFGDLoss, self).__init__()
        self.spatial_weight = spatial_weight    # 更強的空間權重
        self.channel_weight = channel_weight    # 降低通道權重
        
        # 增強版邊緣感知機制
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_filter.weight.data.copy_(torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3) / 8.0)
        self.edge_filter.requires_grad_(False)  # 凍結參數
        
        # 關鍵點注意力機制
        self.keypoint_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 用於通道調整的投影層
        self.align_layers = nn.ModuleList()
        for s_chan, t_chan in zip(student_channels, teacher_channels):
            if s_chan != t_chan:
                self.align_layers.append(nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False).to(
                    'cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                self.align_layers.append(nn.Identity())
                
        LOGGER.info(f"初始化增強版FGD損失: 空間權重={spatial_weight}, 通道權重={channel_weight}")
        
    def forward(self, y_s, y_t):
        """計算增強版FGD損失，特別優化姿態估計的空間特徵保留"""
        assert len(y_s) == len(y_t)
        losses = []
        
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # 確保索引在有效範圍內
            if idx >= len(self.align_layers):
                continue
                
            # 處理空層或None值
            if s is None or t is None:
                continue
            
            # 通道對齊
            try:
                if s.shape[1] != t.shape[1]:
                    # 如果通道不一致，使用投影層調整
                    s = self.align_layers[idx](s)
                    LOGGER.debug(f"調整特徵通道: {s.shape} -> {t.shape}")
            except Exception as e:
                LOGGER.warning(f"通道調整出錯，跳過此層: {e}")
                continue
                
            # 尺寸對齊    
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = torch.nn.functional.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            b, c, h, w = s.shape
            
            # 1. 自適應特徵匹配 - 安全處理
            try:
                # 對重要特徵賦予更高權重
                feature_importance = torch.sigmoid(torch.mean(t, dim=1, keepdim=True))
                l1_loss = torch.nn.functional.smooth_l1_loss(s, t)  # 直接計算損失，避免乘法導致的尺寸問題
            except Exception as e:
                LOGGER.warning(f"計算L1損失時出錯: {e}")
                l1_loss = torch.tensor(0.0, device=s.device)
            
            # 2. 空間注意力損失 - 安全處理
            try:
                s_spatial = torch.mean(s, dim=1, keepdim=True)  # [b, 1, h, w]
                t_spatial = torch.mean(t, dim=1, keepdim=True)  # [b, 1, h, w]
                
                # 提取空間特徵的邊緣信息
                s_edge = self.edge_filter(s_spatial)
                t_edge = self.edge_filter(t_spatial)
                
                # 關鍵點注意力增強
                keypoint_attn = self.keypoint_attention(t_spatial)
                
                # 空間注意力+邊緣感知+關鍵點注意力的組合損失
                spatial_loss = (torch.nn.functional.mse_loss(s_spatial, t_spatial) * 1.5 + 
                            torch.nn.functional.mse_loss(s_edge, t_edge) * 1.0 +
                            torch.nn.functional.mse_loss(s_spatial * keypoint_attn, t_spatial * keypoint_attn) * 1.5) * self.spatial_weight
            except Exception as e:
                LOGGER.warning(f"計算空間損失時出錯: {e}")
                spatial_loss = torch.tensor(0.0, device=s.device)
            
            # 3. 通道關係損失 - 安全處理
            try:
                s_flat = s.view(b, c, -1)  # [b, c, h*w]
                t_flat = t.view(b, c, -1)  # [b, c, h*w]
                
                # 通道歸一化
                s_flat_norm = torch.nn.functional.normalize(s_flat, dim=2)
                t_flat_norm = torch.nn.functional.normalize(t_flat, dim=2)
                
                # 計算通道相關矩陣
                s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
                t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
                
                # 通道相關性損失
                channel_loss = torch.nn.functional.mse_loss(s_corr, t_corr) * self.channel_weight
            except Exception as e:
                LOGGER.warning(f"計算通道損失時出錯: {e}")
                channel_loss = torch.tensor(0.0, device=s.device)
            
            # 4. 組合損失 - 確保所有損失都在相同設備上
            total_loss = l1_loss + spatial_loss + channel_loss
            losses.append(total_loss)
            
        if not losses:
            # 返回一個零張量，確保有梯度
            return torch.tensor(0.0, requires_grad=True, device=y_s[0].device if y_s and len(y_s) > 0 else 'cpu')
            
        loss = sum(losses)
        return loss

class FeatureLoss(nn.Module):
    """特征蒸馏损失，增加索引安全检查"""
    
    def __init__(self, channels_s, channels_t, distiller="fgd"):
        super().__init__()
        self.distiller = distiller
        
        # 确保通道列表长度一致
        if len(channels_s) != len(channels_t):
            LOGGER.error(f"通道列表长度不匹配: 学生={len(channels_s)}, 教师={len(channels_t)}")
            # 截断至较短的长度
            min_len = min(len(channels_s), len(channels_t))
            channels_s = channels_s[:min_len]
            channels_t = channels_t[:min_len]
            LOGGER.warning(f"截断通道列表至长度 {min_len}")
        
        # 创建对齐层
        self.align = nn.ModuleList()
        for s_chan, t_chan in zip(channels_s, channels_t):
            self.align.append(
                nn.Sequential(
                    nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False),
                    nn.BatchNorm2d(t_chan)
                )
            )
        
        LOGGER.info(f"创建了 {len(self.align)} 个特征对齐层")
        
    def forward(self, y_s, y_t):
        losses = []
        
        # 安全检查 - 确保特征列表长度一致 
        min_len = min(len(y_s), len(y_t), len(self.align))
        if min_len < len(y_s) or min_len < len(y_t):
            LOGGER.warning(f"特征列表长度不匹配: 学生={len(y_s)}, 教师={len(y_t)}, 对齐层={len(self.align)}")
            LOGGER.warning(f"将只使用前 {min_len} 个特征")
        
        # 只处理有效数量的特征
        for i in range(min_len):
            s, t = y_s[i], y_t[i]
            
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                continue
                
            # 获取设备和数据类型
            device = s.device
            dtype = s.dtype
            
            # 安全地处理对齐层
            try:
                # 确保对齐层使用正确的数据类型
                if i < len(self.align):  # 安全检查
                    if next(self.align[i].parameters()).dtype != dtype:
                        self.align[i] = self.align[i].to(dtype)
                    
                    # 通道对齐
                    s_aligned = self.align[i](s)
                else:
                    LOGGER.error(f"对齐层索引 {i} 超出范围(最大 {len(self.align)-1})")
                    continue
                
                # 空间对齐
                if s_aligned.size(2) != t.size(2) or s_aligned.size(3) != t.size(3):
                    t = F.interpolate(t, size=(s_aligned.size(2), s_aligned.size(3)),
                                     mode='bilinear', align_corners=True)
                
                # 计算损失
                if self.distiller == "fgd":
                    # FGD保留空间结构  
                    loss = F.mse_loss(s_aligned, t)
                    losses.append(loss)
                    
                elif self.distiller == "enhancedfgd":
                    # 增强版FGD，特别关注高激活区域
                    basic_loss = F.mse_loss(s_aligned, t)
                    
                    # 高激活区域（可能是关键点区域）
                    t_highlight = torch.mean(torch.abs(t), dim=1, keepdim=True)
                    threshold = torch.mean(t_highlight) * 1.5
                    t_mask = (t_highlight > threshold).float()
                    
                    # 关键区域损失
                    region_loss = F.mse_loss(s_aligned * t_mask, t * t_mask) * 2.0
                    
                    # 结合损失
                    total_loss = basic_loss + region_loss
                    losses.append(total_loss)
                
                else:  # 默认为标准CWD
                    # CWD通道级对齐
                    s_norm = self._channel_normalize(s_aligned)
                    t_norm = self._channel_normalize(t)
                    loss = torch.norm(s_norm - t_norm, p=2, dim=1).mean()
                    losses.append(loss)
                    
            except Exception as e:
                LOGGER.error(f"处理特征 {i} 时出错: {e}")
                continue
        
        if not losses:
            LOGGER.warning("没有计算任何蒸馏损失!")
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return sum(losses)
    
    def _channel_normalize(self, x):
        """通道级归一化"""
        channel_mean = x.mean(dim=1, keepdim=True)
        channel_std = x.std(dim=1, keepdim=True) + 1e-6
        return (x - channel_mean) / channel_std

class DistillationLoss:
    """增强版知识蒸馏损失，改进特征收集机制"""
    
    def __init__(self, models, modelt, distiller="fgd", layers=None, original_performance=None):
        self.models = models
        self.modelt = modelt
        self.distiller = distiller.lower()
        self.layers = layers if layers is not None else []
        self.remove_handle = []
        self.teacher_outputs = []
        self.student_outputs = []
        self.original_performance = original_performance
        
        # 检查是否启用混合精度训练
        self.using_amp = False
        if hasattr(models, 'args') and hasattr(models.args, 'amp'):
            self.using_amp = models.args.amp
            LOGGER.info(f"检测到混合精度训练设置: {self.using_amp}")
        
        # 更明确地记录找到的层
        self.found_student_layers = []
        self.found_teacher_layers = []
        
        # 寻找要蒸馏的层
        self._find_layers()
        
        if len(self.channels_s) == 0 or len(self.channels_t) == 0:
            LOGGER.error("未找到任何匹配的层进行蒸馏!")
            raise ValueError("无法找到适合蒸馏的层")
            
        LOGGER.info(f"成功配对 {len(self.teacher_module_pairs)} 对学生-教师特征层")
        LOGGER.info(f"学生层: {self.found_student_layers}")
        LOGGER.info(f"教师层: {self.found_teacher_layers}")
        LOGGER.info(f"学生通道: {self.channels_s}")
        LOGGER.info(f"教师通道: {self.channels_t}")
        
    def _find_layers(self):
        """寻找要蒸馏的层，增强错误处理和日志记录"""
        if not self.layers:
            LOGGER.warning("没有指定蒸馏层，尝试使用默认配置")
            self.layers = ["22"]  # 使用默认层
        
        # 列出模型模块以帮助调试
        LOGGER.debug("Available modules in student model:")
        student_modules = []
        for name, module in self.models.named_modules():
            if isinstance(name, str) and name.startswith("model."):
                student_modules.append(name)
        LOGGER.debug(f"Student modules: {student_modules[:10]}...")
        
        LOGGER.debug("Available modules in teacher model:")
        teacher_modules = []
        for name, module in self.modelt.named_modules():
            if isinstance(name, str) and name.startswith("model."):
                teacher_modules.append(name)
        LOGGER.debug(f"Teacher modules: {teacher_modules[:10]}...")
        
        # 查找学生模型中的层
        for layer_idx in self.layers:
            student_found = False
            teacher_found = False
            
            # 查找学生模型层
            for name, module in self.models.named_modules():
                if not name or not isinstance(name, str):
                    continue
                if f"model.{layer_idx}." in name and isinstance(module, nn.Conv2d):
                    self.student_module_pairs.append((name, module))
                    self.channels_s.append(module.out_channels)
                    student_found = True
                    self.found_student_layers.append(layer_idx)
                    LOGGER.info(f"学生模型找到层 {name} 用于蒸馏")
                    break
            
            # 如果学生层找到了，再查找相应的教师层
            if student_found:
                for name, module in self.modelt.named_modules():
                    if not name or not isinstance(name, str):
                        continue
                    if f"model.{layer_idx}." in name and isinstance(module, nn.Conv2d):
                        self.teacher_module_pairs.append((name, module))
                        self.channels_t.append(module.out_channels)
                        teacher_found = True
                        self.found_teacher_layers.append(layer_idx)
                        LOGGER.info(f"教师模型找到层 {name} 用于蒸馏")
                        break
            
            # 记录未找到的层
            if not student_found:
                LOGGER.warning(f"在学生模型中没有找到层 {layer_idx}")
            if not teacher_found and student_found:
                LOGGER.warning(f"在教师模型中没有找到对应的层 {layer_idx}")
                # 移除最后添加的学生层
                self.student_module_pairs.pop()
                self.channels_s.pop()
                self.found_student_layers.pop()
        
        # 确保列表长度一致
        min_len = min(len(self.student_module_pairs), len(self.teacher_module_pairs))
        if min_len < len(self.student_module_pairs):
            LOGGER.warning(f"截断学生模块列表从 {len(self.student_module_pairs)} 到 {min_len}")
            self.student_module_pairs = self.student_module_pairs[:min_len]
            self.channels_s = self.channels_s[:min_len]
            self.found_student_layers = self.found_student_layers[:min_len]
        
        if min_len < len(self.teacher_module_pairs):
            LOGGER.warning(f"截断教师模块列表从 {len(self.teacher_module_pairs)} 到 {min_len}")
            self.teacher_module_pairs = self.teacher_module_pairs[:min_len]
            self.channels_t = self.channels_t[:min_len]
            self.found_teacher_layers = self.found_teacher_layers[:min_len]
    
    def register_hook(self):
        """注册钩子函数，增强错误处理和日志记录"""
        # 清空之前的输出
        self.teacher_outputs = []
        self.student_outputs = []
        
        if not self.student_module_pairs or not self.teacher_module_pairs:
            LOGGER.error("无法注册钩子：模块对为空")
            return
        
        LOGGER.info(f"注册蒸馏钩子 - 学生层数: {len(self.student_module_pairs)}, 教师层数: {len(self.teacher_module_pairs)}")
        
        # 注册学生模型钩子
        for i, (name, module) in enumerate(self.student_module_pairs):
            self.remove_handle.append(
                module.register_forward_hook(
                    lambda m, inp, out, idx=i: self._student_hook(m, inp, out, idx)
                )
            )
            LOGGER.debug(f"为学生层 {name} 注册钩子(索引 {i})")
        
        # 注册教师模型钩子
        for i, (name, module) in enumerate(self.teacher_module_pairs):
            self.remove_handle.append(
                module.register_forward_hook(
                    lambda m, inp, out, idx=i: self._teacher_hook(m, inp, out, idx)
                )
            )
            LOGGER.debug(f"为教师层 {name} 注册钩子(索引 {i})")
    
    def _student_hook(self, module, input, output, idx):
        """学生模型钩子函数，增强安全处理"""
        try:
            # 填充列表直到idx位置
            while len(self.student_outputs) <= idx:
                self.student_outputs.append(None)
            
            # 设置输出
            self.student_outputs[idx] = output
            
        except Exception as e:
            LOGGER.error(f"学生钩子处理出错(idx={idx}): {e}")
    
    def _teacher_hook(self, module, input, output, idx):
        """教师模型钩子函数，增强安全处理"""
        try:
            # 填充列表直到idx位置
            while len(self.teacher_outputs) <= idx:
                self.teacher_outputs.append(None)
            
            # 设置输出
            self.teacher_outputs[idx] = output
            

        except Exception as e:
            LOGGER.error(f"教师钩子处理出错(idx={idx}): {e}")