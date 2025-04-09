import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import LOGGER

class CWDLoss(nn.Module):
    """Channel-wise Distillation for YOLO Pose with different channel dimensions.
    
    適用於教師和學生模型通道數不同的情況，專為姿勢識別特徵金字塔蒸餾優化。
    """
    def __init__(self, student_channels, teacher_channels, tau=4.0, 
                 channel_wise_adapt=True, normalize=True):
        super().__init__()
        self.tau = tau
        self.normalize = normalize
        self.channel_wise_adapt = channel_wise_adapt
        
        # 如果需要通道自適應，為不同層創建投影層
        if channel_wise_adapt and len(student_channels) > 0:
            self.adapters = nn.ModuleList()
            for s_ch, t_ch in zip(student_channels, teacher_channels):
                self.adapters.append(nn.Conv2d(s_ch, t_ch, kernel_size=1, bias=False))
        else:
            self.adapters = None
            
    def forward(self, student_features, teacher_features):
        """
        Args:
            student_features: 學生模型的特徵列表 [P3, P4, P5]
            teacher_features: 教師模型的特徵列表 [P3, P4, P5]
        """
        total_loss = 0.0
        assert len(student_features) == len(teacher_features), "特徵金字塔層數不匹配"
        
        for i, (s_feat, t_feat) in enumerate(zip(student_features, teacher_features)):
            # 可選：通道數自適應
            if self.adapters is not None:
                s_feat = self.adapters[i](s_feat)
            
            # 確保空間尺寸一致
            if s_feat.shape[2:] != t_feat.shape[2:]:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[2:], 
                                      mode='bilinear', align_corners=False)
            
            # 應用歸一化，提高穩定性
            if self.normalize:
                s_feat = F.normalize(s_feat, p=2, dim=1)
                t_feat = F.normalize(t_feat, p=2, dim=1)
            
            # 實現論文中的CWD方法
            loss = self._compute_cwd(s_feat, t_feat.detach())
            
            # 可以針對不同層設置不同權重
            layer_weight = 1.0  # 可以調整
            total_loss += loss * layer_weight
            
        return total_loss / len(student_features)
    
    def _compute_cwd(self, s_feat, t_feat):
        """計算單層的CWD損失"""
        N, C, H, W = s_feat.shape
        
        # 重塑為 [N, C, H*W]
        s_feat = s_feat.view(N, C, -1)
        t_feat = t_feat.view(N, C, -1)
        
        # 每個通道進行softmax，使其成為概率分布
        s_feat = F.softmax(s_feat / self.tau, dim=2)
        t_feat = F.softmax(t_feat / self.tau, dim=2)
        
        # KL散度計算
        loss = F.kl_div(
            torch.log(s_feat + 1e-7),
            t_feat,
            reduction='none'
        ).sum(2) * (self.tau ** 2)
        
        # 在批次和通道維度上求平均
        return loss.mean()

# class MGDLoss(nn.Module):
#     """Masked Generative Distillation"""
#     def __init__(self, student_channels, teacher_channels, alpha_mgd=0.00002, lambda_mgd=0.65):
#         super(MGDLoss, self).__init__()
#         self.alpha_mgd = alpha_mgd
#         self.lambda_mgd = lambda_mgd
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.generation = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(s_chan, t_chan, kernel_size=3, padding=1),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(t_chan, t_chan, kernel_size=3, padding=1)
#             ).to(device) for s_chan, t_chan in zip(student_channels, teacher_channels)
#         ])
        
#     def forward(self, y_s, y_t, layer=None):
#         """計算MGD損失
        
#         Args:
#             y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
#             y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
#             layer (str, optional): 指定要蒸餾的層，如果是 "outlayer" 則使用最後一層
            
#         Returns:
#             torch.Tensor: 所有階段的計算損失總和
#         """
#         losses = []
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             if s.size(2) != t.size(2) or s.size(3) != t.size(3):
#                 t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
#             if layer == "outlayer":
#                 idx = -1
#             losses.append(self.get_dis_loss(s, t, idx) * self.alpha_mgd)
#         loss = sum(losses)
#         return loss
    
#     def get_dis_loss(self, preds_S, preds_T, idx):
#         """計算單層的MGD損失
        
#         Args:
#             preds_S: 學生模型的預測
#             preds_T: 教師模型的預測
#             idx: 層索引
            
#         Returns:
#             torch.Tensor: 計算的損失值
#         """
#         loss_mse = nn.MSELoss(reduction='sum')
#         N, C, H, W = preds_T.shape
#         device = preds_S.device
#         mat = torch.rand((N, 1, H, W)).to(device)
#         mat = torch.where(mat > 1 - self.lambda_mgd, 0, 1).to(device)
#         masked_fea = torch.mul(preds_S, mat)
#         new_fea = self.generation[idx](masked_fea)
#         dis_loss = loss_mse(new_fea, preds_T) / N
#         return dis_loss

# class ReviewKDLoss(nn.Module):
#     """Review-KD: https://arxiv.org/abs/2104.09044"""
#     def __init__(self, student_channels, teacher_channels, temperature=1.0):
#         super(ReviewKDLoss, self).__init__()
#         self.temperature = temperature
        
#     def forward(self, y_s, y_t):
#         """計算Review-KD損失
        
#         Args:
#             y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
#             y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            
#         Returns:
#             torch.Tensor: 所有階段的計算損失總和
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             if s.size(2) != t.size(2) or s.size(3) != t.size(3):
#                 t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
#             b, c, h, w = s.shape
#             s = s.view(b, c, -1)
#             t = t.view(b, c, -1)
            
#             # 使用softmax和KL散度
#             s = F.log_softmax(s / self.temperature, dim=2)
#             t = F.softmax(t / self.temperature, dim=2)
#             loss = F.kl_div(s, t, reduction='batchmean') * (self.temperature ** 2)
#             losses.append(loss)
#         loss = sum(losses)
#         return loss

# class FGDLoss(nn.Module):
#     """增強版特徵引導蒸餾 - 專為姿態估計優化"""
#     def __init__(self, student_channels, teacher_channels, spatial_weight=2.0, channel_weight=0.6):
#         super(FGDLoss, self).__init__()
#         self.spatial_weight = spatial_weight    # 更強的空間權重
#         self.channel_weight = channel_weight    # 降低通道權重
#         # 增加邊緣感知機制
#         self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
#             'cuda' if torch.cuda.is_available() else 'cpu')
#         self.edge_filter.weight.data.copy_(torch.tensor([
#             [-1, -1, -1],
#             [-1,  8, -1],
#             [-1, -1, -1]
#         ]).view(1, 1, 3, 3) / 8.0)
#         self.edge_filter.requires_grad_(False)  # 凍結參數
        
#     def forward(self, y_s, y_t):
#         """計算增強版FGD損失，特別優化姿態估計的空間特徵保留
        
#         Args:
#             y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
#             y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            
#         Returns:
#             torch.Tensor: 所有階段的計算損失總和
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             if s.size(2) != t.size(2) or s.size(3) != t.size(3):
#                 t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
#             b, c, h, w = s.shape
            
#             # 1. 基本特徵匹配 - 使用Huber損失減少異常值影響
#             l2_loss = F.smooth_l1_loss(s, t)
            
#             # 2. 增強版空間注意力損失
#             s_spatial = torch.mean(s, dim=1, keepdim=True)  # [b, 1, h, w]
#             t_spatial = torch.mean(t, dim=1, keepdim=True)  # [b, 1, h, w]
            
#             # 提取空間特徵的邊緣信息
#             s_edge = self.edge_filter(s_spatial)
#             t_edge = self.edge_filter(t_spatial)
            
#             # 空間注意力+邊緣感知的組合損失
#             spatial_loss = (F.mse_loss(s_spatial, t_spatial) + 
#                            F.mse_loss(s_edge, t_edge)) * self.spatial_weight
            
#             # 3. 通道關係損失 - 專注於重要通道
#             s_flat = s.view(b, c, -1)  # [b, c, h*w]
#             t_flat = t.view(b, c, -1)  # [b, c, h*w]
            
#             # 通道歸一化
#             s_flat_norm = F.normalize(s_flat, dim=2)
#             t_flat_norm = F.normalize(t_flat, dim=2)
            
#             # 計算通道相關矩陣
#             s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
#             t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
            
#             # 通道相關性損失
#             channel_loss = F.mse_loss(s_corr, t_corr) * self.channel_weight
            
#             # 4. 組合損失
#             total_loss = l2_loss + spatial_loss + channel_loss
#             losses.append(total_loss)
            
#         loss = sum(losses)
#         return loss

# class PKDLoss(nn.Module):
#     """Probabilistic Knowledge Distillation"""
#     def __init__(self, student_channels, teacher_channels):
#         super(PKDLoss, self).__init__()
        
#     def forward(self, y_s, y_t):
#         """計算PKD損失
        
#         Args:
#             y_s (list): 學生模型的預測，形狀為 (N, C, H, W) 的張量列表
#             y_t (list): 教師模型的預測，形狀為 (N, C, H, W) 的張量列表
            
#         Returns:
#             torch.Tensor: 所有階段的計算損失總和
#         """
#         assert len(y_s) == len(y_t)
#         losses = []
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             if s.size(2) != t.size(2) or s.size(3) != t.size(3):
#                 t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
#             # 計算通道間相關性
#             b, c, h, w = s.shape
#             s_flat = s.view(b, c, -1)
#             t_flat = t.view(b, c, -1)
            
#             s_mean = s_flat.mean(dim=2, keepdim=True)
#             t_mean = t_flat.mean(dim=2, keepdim=True)
            
#             s_centered = s_flat - s_mean
#             t_centered = t_flat - t_mean
            
#             s_var = torch.mean(s_centered**2, dim=2) + 1e-6
#             t_var = torch.mean(t_centered**2, dim=2) + 1e-6
            
#             s_std = torch.sqrt(s_var).unsqueeze(2)
#             t_std = torch.sqrt(t_var).unsqueeze(2)
            
#             s_normalized = s_centered / s_std
#             t_normalized = t_centered / t_std
            
#             # 計算相關矩陣
#             corr_s = torch.matmul(s_normalized, s_normalized.transpose(1, 2)) / h / w
#             corr_t = torch.matmul(t_normalized, t_normalized.transpose(1, 2)) / h / w
            
#             # PKD損失
#             loss = F.mse_loss(corr_s, corr_t)
#             losses.append(loss)
#         loss = sum(losses)
#         return loss

# class EnhancedFGDLoss(nn.Module):
#     """增強版特徵引導蒸餾 - 專為姿態估計優化，提供更好的關鍵點定位能力"""
#     def __init__(self, student_channels, teacher_channels, spatial_weight=3.5, channel_weight=0.4):
#         super(EnhancedFGDLoss, self).__init__()
#         self.spatial_weight = spatial_weight    # 更強的空間權重
#         self.channel_weight = channel_weight    # 降低通道權重
        
#         # 增強版邊緣感知機制
#         self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
#             'cuda' if torch.cuda.is_available() else 'cpu')
#         self.edge_filter.weight.data.copy_(torch.tensor([
#             [-1, -1, -1],
#             [-1,  8, -1],
#             [-1, -1, -1]
#         ]).view(1, 1, 3, 3) / 8.0)
#         self.edge_filter.requires_grad_(False)  # 凍結參數
        
#         # 關鍵點注意力機制
#         self.keypoint_attention = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(8, 1, kernel_size=3, padding=1),
#             nn.Sigmoid()
#         ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # 用於通道調整的投影層
#         self.align_layers = nn.ModuleList()
#         for s_chan, t_chan in zip(student_channels, teacher_channels):
#             if s_chan != t_chan:
#                 self.align_layers.append(nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False).to(
#                     'cuda' if torch.cuda.is_available() else 'cpu'))
#             else:
#                 self.align_layers.append(nn.Identity())
                
#         LOGGER.info(f"初始化增強版FGD損失: 空間權重={spatial_weight}, 通道權重={channel_weight}")
        
#     def forward(self, y_s, y_t):
#         """計算增強版FGD損失，特別優化姿態估計的空間特徵保留"""
#         assert len(y_s) == len(y_t)
#         losses = []
        
#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             # 確保索引在有效範圍內
#             if idx >= len(self.align_layers):
#                 continue
                
#             # 處理空層或None值
#             if s is None or t is None:
#                 continue
            
#             # 通道對齊
#             try:
#                 if s.shape[1] != t.shape[1]:
#                     # 如果通道不一致，使用投影層調整
#                     s = self.align_layers[idx](s)
#                     LOGGER.debug(f"調整特徵通道: {s.shape} -> {t.shape}")
#             except Exception as e:
#                 LOGGER.warning(f"通道調整出錯，跳過此層: {e}")
#                 continue
                
#             # 尺寸對齊    
#             if s.size(2) != t.size(2) or s.size(3) != t.size(3):
#                 t = torch.nn.functional.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
#             b, c, h, w = s.shape
            
#             # 1. 自適應特徵匹配 - 安全處理
#             try:
#                 # 對重要特徵賦予更高權重
#                 feature_importance = torch.sigmoid(torch.mean(t, dim=1, keepdim=True))
#                 l1_loss = torch.nn.functional.smooth_l1_loss(s, t)  # 直接計算損失，避免乘法導致的尺寸問題
#             except Exception as e:
#                 LOGGER.warning(f"計算L1損失時出錯: {e}")
#                 l1_loss = torch.tensor(0.0, device=s.device)
            
#             # 2. 空間注意力損失 - 安全處理
#             try:
#                 s_spatial = torch.mean(s, dim=1, keepdim=True)  # [b, 1, h, w]
#                 t_spatial = torch.mean(t, dim=1, keepdim=True)  # [b, 1, h, w]
                
#                 # 提取空間特徵的邊緣信息
#                 s_edge = self.edge_filter(s_spatial)
#                 t_edge = self.edge_filter(t_spatial)
                
#                 # 關鍵點注意力增強
#                 keypoint_attn = self.keypoint_attention(t_spatial)
                
#                 # 空間注意力+邊緣感知+關鍵點注意力的組合損失
#                 spatial_loss = (torch.nn.functional.mse_loss(s_spatial, t_spatial) * 1.5 + 
#                             torch.nn.functional.mse_loss(s_edge, t_edge) * 1.0 +
#                             torch.nn.functional.mse_loss(s_spatial * keypoint_attn, t_spatial * keypoint_attn) * 1.5) * self.spatial_weight
#             except Exception as e:
#                 LOGGER.warning(f"計算空間損失時出錯: {e}")
#                 spatial_loss = torch.tensor(0.0, device=s.device)
            
#             # 3. 通道關係損失 - 安全處理
#             try:
#                 s_flat = s.view(b, c, -1)  # [b, c, h*w]
#                 t_flat = t.view(b, c, -1)  # [b, c, h*w]
                
#                 # 通道歸一化
#                 s_flat_norm = torch.nn.functional.normalize(s_flat, dim=2)
#                 t_flat_norm = torch.nn.functional.normalize(t_flat, dim=2)
                
#                 # 計算通道相關矩陣
#                 s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
#                 t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
                
#                 # 通道相關性損失
#                 channel_loss = torch.nn.functional.mse_loss(s_corr, t_corr) * self.channel_weight
#             except Exception as e:
#                 LOGGER.warning(f"計算通道損失時出錯: {e}")
#                 channel_loss = torch.tensor(0.0, device=s.device)
            
#             # 4. 組合損失 - 確保所有損失都在相同設備上
#             total_loss = l1_loss + spatial_loss + channel_loss
#             losses.append(total_loss)
            
#         if not losses:
#             # 返回一個零張量，確保有梯度
#             return torch.tensor(0.0, requires_grad=True, device=y_s[0].device if y_s and len(y_s) > 0 else 'cpu')
            
#         loss = sum(losses)
#         return loss

# class FeatureLoss(nn.Module):
#     """特徵層蒸餾損失，支持多種蒸餾方法，包括特徵對齊"""
#     def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
#         super(FeatureLoss, self).__init__()
#         self.loss_weight = loss_weight
#         self.distiller = distiller
        
#         # 將所有模塊移動到相同精度
#         device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
#         # 轉換為ModuleList並確保一致的數據類型
#         self.align_module = nn.ModuleList()
#         self.norm = nn.ModuleList()
#         self.norm1 = nn.ModuleList()
        
#         # 創建對齊模塊
#         for s_chan, t_chan in zip(channels_s, channels_t):
#             align = nn.Sequential(
#                 nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
#                 nn.BatchNorm2d(t_chan, affine=False)
#             ).to(device)
#             self.align_module.append(align)
            
#         # 創建歸一化層
#         for t_chan in channels_t:
#             self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
            
#         for s_chan in channels_s:
#             self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))
            
#         # 選擇蒸餾損失函數
#         if distiller == 'mgd':
#             self.feature_loss = MGDLoss(channels_s, channels_t)
#         elif distiller == 'cwd':
#             self.feature_loss = CWDLoss(channels_s, channels_t)
#         elif distiller == 'rev':
#             self.feature_loss = ReviewKDLoss(channels_s, channels_t)
#         elif distiller == 'fgd':
#             self.feature_loss = FGDLoss(channels_s, channels_t, spatial_weight=2.0, channel_weight=0.6)
#         elif distiller == 'enhancedfgd':
#             self.feature_loss = EnhancedFGDLoss(channels_s, channels_t, spatial_weight=3.5, channel_weight=0.4)
#             LOGGER.info("使用增強版FGD蒸餾方法")
#         elif distiller == 'pkd':
#             self.feature_loss = PKDLoss(channels_s, channels_t)
#         else:
#             raise NotImplementedError(f"Unknown distiller: {distiller}")
            
#     def forward(self, y_s, y_t):
#         """計算特徵層蒸餾損失
        
#         Args:
#             y_s (list): 學生模型的特徵
#             y_t (list): 教師模型的特徵
            
#         Returns:
#             torch.Tensor: 計算的損失值
#         """
#         # LOGGER.info(f"FeatureLoss.forward 被調用，特徵列表長度 - 學生: {len(y_s)}, 教師: {len(y_t)}")
        
#         min_len = min(len(y_s), len(y_t))
#         y_s = y_s[:min_len]
#         y_t = y_t[:min_len]

#         tea_feats = []
#         stu_feats = []

#         for idx, (s, t) in enumerate(zip(y_s, y_t)):
#             if idx >= len(self.align_module):
#                 LOGGER.warning(f"索引 {idx} 超出對齊模塊範圍 {len(self.align_module)}")
#                 break
            
#             # 處理不同類型的特徵
#             if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
#                 LOGGER.warning(f"特徵類型不是張量 - 學生: {type(s)}, 教師: {type(t)}")
#                 if isinstance(s, list) and len(s) > 0:
#                     s = s[0]
#                 if isinstance(t, list) and len(t) > 0:
#                     t = t[0]
            
#             # 確保特徵是張量
#             if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
#                 LOGGER.warning(f"轉換後特徵類型仍不是張量 - 學生: {type(s)}, 教師: {type(t)}")
#                 continue
                
#             # LOGGER.info(f"處理第 {idx+1} 對特徵 - 學生形狀: {s.shape}, 教師形狀: {t.shape}")
            
#             # 轉換數據類型以匹配對齊模塊
#             s = s.type(next(self.align_module[idx].parameters()).dtype)
#             t = t.type(next(self.align_module[idx].parameters()).dtype)

#             try:
#                 # 特別處理FGD方法，直接傳遞特徵而不進行標準化
#                 if self.distiller == "fgd":
#                     if s.shape[1] != t.shape[1]:  # 如果通道數不匹配
#                         s = self.align_module[idx](s)
#                     stu_feats.append(s)
#                     tea_feats.append(t.detach())
#                 elif self.distiller == "cwd":
#                     s = self.align_module[idx](s)
#                     # LOGGER.info(f"  對齊後學生特徵形狀: {s.shape}")
#                     stu_feats.append(s)
#                     tea_feats.append(t.detach())
#                 else:
#                     t = self.norm[idx](t)
#                     # LOGGER.info(f"  標準化後教師特徵形狀: {t.shape}")
#                     stu_feats.append(s)
#                     tea_feats.append(t.detach())
#             except Exception as e:
#                 LOGGER.error(f"處理特徵時出錯: {e}")
#                 import traceback
#                 LOGGER.error(traceback.format_exc())
#                 continue

#         # 安全檢查
#         if len(stu_feats) == 0 or len(tea_feats) == 0:
#             LOGGER.warning("沒有有效的特徵對進行蒸餾")
#             return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
        
#         try:
#             # LOGGER.info(f"調用 {self.feature_loss.__class__.__name__} 計算損失")
#             loss = self.feature_loss(stu_feats, tea_feats)
#             # LOGGER.info(f"計算的原始損失: {loss.item():.6f}, 加權後: {(self.loss_weight * loss).item():.6f}")
#             return self.loss_weight * loss
#         except Exception as e:
#             LOGGER.error(f"計算特徵損失時出錯: {e}")
#             import traceback
#             LOGGER.error(traceback.format_exc())
#             return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)

class DistillationLoss:
    """知識蒸餾損失實現，集成多種蒸餾方法，針對YOLO模型優化"""
    
    def __init__(self, models, modelt, distiller="cwd", layers=None):
        self.models = models
        self.modelt = modelt
        self.distiller = distiller.lower()
        self.layers = layers if layers is not None else []
        self.remove_handle = []
        self.teacher_outputs = []
        self.student_outputs = []
        
        # 初始化通道列表和模塊對
        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        
        # 尋找要蒸餾的層
        self._find_layers()
        
        if len(self.channels_s) == 0 or len(self.channels_t) == 0:
            LOGGER.error("未找到任何匹配的層進行蒸餾!")
            raise ValueError("無法找到適合蒸餾的層")
            
        LOGGER.info(f"成功配對 {len(self.teacher_module_pairs)} 對學生-教師特徵層")
        LOGGER.info(f"學生通道: {self.channels_s}")
        LOGGER.info(f"教師通道: {self.channels_t}")
        
        # # 創建蒸餾損失實例
        # self.distill_loss_fn = FeatureLoss(
        #     channels_s=self.channels_s, 
        #     channels_t=self.channels_t, 
        #     distiller=self.distiller
        # )
        
        LOGGER.info(f"使用 {self.distiller} 方法進行蒸餾")

    def _find_layers(self):
        """查找名為 cv2 且具有 conv 屬性的層"""
        self.channels_s = []
        self.channels_t = []
        self.teacher_module_pairs = []
        self.student_module_pairs = []
        
        # # 打印模型的一些層，以便於檢查結構
        # LOGGER.info("教師模型的一些關鍵層:")
        # for name, ml in self.modelt.named_modules():
        #     if "model" in name and ("16" in name or "19" in name or "22" in name):
        #         LOGGER.info(f"  - {name}: {ml.__class__.__name__}")
        #         if hasattr(ml, 'conv'):
        #             LOGGER.info(f"    有 conv 屬性，通道數: {ml.conv.out_channels}")
        
        # LOGGER.info("學生模型的一些關鍵層:")
        # for name, ml in self.models.named_modules():
        #     if "model" in name and ("16" in name or "19" in name or "22" in name):
        #         LOGGER.info(f"  - {name}: {ml.__class__.__name__}")
        #         if hasattr(ml, 'conv'):
        #             LOGGER.info(f"    有 conv 屬性，通道數: {ml.conv.out_channels}")
        
        # 首先查找教師模型中的目標層
        for name, ml in self.modelt.named_modules():
            if name is not None:
                name_parts = name.split(".")
                
                if name_parts[0] != "model":
                    continue
                if len(name_parts) >= 3:
                    if name_parts[1] in self.layers:
                        if "cv2" in name_parts[2]:
                            if hasattr(ml, 'conv'):
                                self.channels_t.append(ml.conv.out_channels)
                                self.teacher_module_pairs.append(ml)
                                LOGGER.info(f"找到教師模型層: {name}, 通道數: {ml.conv.out_channels}")
        
        # 然後查找學生模型中的目標層
        for name, ml in self.models.named_modules():
            if name is not None:
                name_parts = name.split(".")
                
                if name_parts[0] != "model":
                    continue
                if len(name_parts) >= 3:
                    if name_parts[1] in self.layers:
                        if "cv2" in name_parts[2]:
                            if hasattr(ml, 'conv'):
                                self.channels_s.append(ml.conv.out_channels)
                                self.student_module_pairs.append(ml)
                                LOGGER.info(f"找到學生模型層: {name}, 通道數: {ml.conv.out_channels}")
        
        # 確保通道數和模塊數量匹配
        nl = min(len(self.channels_s), len(self.channels_t))
          
        self.channels_s = self.channels_s[-nl:]
        self.channels_t = self.channels_t[-nl:]
        self.teacher_module_pairs = self.teacher_module_pairs[-nl:]
        self.student_module_pairs = self.student_module_pairs[-nl:]
        
        LOGGER.info(f"匹配了 {nl} 對層進行蒸餾")
        LOGGER.info(f"學生通道: {self.channels_s}")
        LOGGER.info(f"教師通道: {self.channels_t}")
        
        # 詳細列出所有匹配的模塊對
        for i, (t, s) in enumerate(zip(self.teacher_module_pairs, self.student_module_pairs)):
            LOGGER.info(f"模塊對 {i+1}:")
            LOGGER.info(f"  教師: {t.__class__.__name__}, 通道數: {self.channels_t[i]}")
            LOGGER.info(f"  學生: {s.__class__.__name__}, 通道數: {self.channels_s[i]}")
            
    def register_hook(self):
        """註冊用於捕獲特徵的鉤子"""
        # 移除可能存在的舊鉤子
        self.remove_handle_()
        
        self.teacher_outputs = []
        self.student_outputs = []

        # 為教師模型創建鉤子
        def make_teacher_hook(l):
            def forward_hook(m, input, output):
                LOGGER.debug(f"教師鉤子被觸發，輸出形狀: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                if isinstance(output, torch.Tensor):
                    l.append(output.detach().clone())  # 分離並複製教師輸出
                else:
                    l.append([o.detach().clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        # 為學生模型創建鉤子
        def make_student_hook(l):
            def forward_hook(m, input, output):
                LOGGER.debug(f"學生鉤子被觸發，輸出形狀: {output.shape if isinstance(output, torch.Tensor) else type(output)}")
                if isinstance(output, torch.Tensor):
                    out = output.clone()  # 複製以確保不修改原始輸出
                    l.append(out)
                else:
                    l.append([o.clone() if isinstance(o, torch.Tensor) else o for o in output])
            return forward_hook

        # 確保模塊對正確
        if len(self.teacher_module_pairs) == 0 or len(self.student_module_pairs) == 0:
            LOGGER.error("沒有找到匹配的模塊對進行蒸餾")
            return
            
        # 註冊鉤子到教師和學生模型的對應層
        for i, (ml, ori) in enumerate(zip(self.teacher_module_pairs, self.student_module_pairs)):
            self.remove_handle.append(ml.register_forward_hook(make_teacher_hook(self.teacher_outputs)))
            self.remove_handle.append(ori.register_forward_hook(make_student_hook(self.student_outputs)))

    def get_loss(self):
        """計算教師和學生模型之間的蒸餾損失，支持不同通道數的特徵圖"""        
        if not self.teacher_outputs or not self.student_outputs:
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)
        
        # 設定閾值常數
        DIFFERENCE_THRESHOLD = 1e-6  # 視為計算誤差的絕對差異閾值
        MAX_DIFF_THRESHOLD = 1e-5    # 整體最大差異閾值
        
        # 追蹤全局最大差異
        global_max_diff = 0.0
        
        # 特徵差異分析
        for i, (t, s) in enumerate(zip(self.teacher_outputs, self.student_outputs)):
            if isinstance(t, torch.Tensor) and isinstance(s, torch.Tensor):
                # 檢查形狀是否相同
                if t.shape == s.shape:
                    t_detached = t.detach()
                    diff = (s - t_detached).abs()
                    
                    # 基本統計
                    max_diff = diff.max().item()
                    global_max_diff = max(global_max_diff, max_diff)
                    mean_diff = diff.mean().item()
                    std_diff = diff.std().item()
                    
                    # 稀疏性分析
                    zero_elements = (diff < DIFFERENCE_THRESHOLD).sum().item() / diff.numel()
                    
                    # 分位數分析
                    percentiles = [50, 75, 90, 95, 99]
                    percentile_values = [torch.quantile(diff.float(), q/100).item() for q in percentiles]
                    
                    LOGGER.info(f"層 {i} 特徵差異統計:")
                    LOGGER.info(f"  形狀: 教師{t.shape}, 學生{s.shape}")
                    LOGGER.info(f"  最大差異: {max_diff:.8f}, 平均差異: {mean_diff:.8f}, 標準差: {std_diff:.8f}")
                    LOGGER.info(f"  接近零的元素比例: {zero_elements:.2%}")
                    LOGGER.info(f"  分位數分析: {', '.join([f'{p}%: {v:.8f}' for p, v in zip(percentiles, percentile_values)])}")
                else:
                    # 形狀不同時只記錄基本信息
                    LOGGER.info(f"層 {i} 特徵形狀不同:")
                    LOGGER.info(f"  形狀: 教師{t.shape}, 學生{s.shape}")
                    LOGGER.info(f"  將使用P5特化蒸餾方法")
                    
                # 檢查是否有NaN或Inf
                if torch.isnan(s).any() or torch.isinf(s).any() or torch.isnan(t).any() or torch.isinf(t).any():
                    LOGGER.warning(f"警告: 層 {i} 中發現NaN或Inf值!")
        
        # 僅對形狀相同的層檢查是否所有差異都可視為計算誤差
        # 對於形狀不同的層，我們需要用特殊方法蒸餾，不應該跳過
        same_shape_exists = any(t.shape == s.shape for t, s in zip(self.teacher_outputs, self.student_outputs) 
                            if isinstance(t, torch.Tensor) and isinstance(s, torch.Tensor))
        
        if same_shape_exists and global_max_diff < MAX_DIFF_THRESHOLD:
            LOGGER.info(f"所有相同形狀層的最大差異 ({global_max_diff:.8f}) 低於閾值 {MAX_DIFF_THRESHOLD}，將僅應用特化蒸餾")
        
        # 確保教師輸出已經分離
        teacher_outputs_detached = []
        for t in self.teacher_outputs:
            if isinstance(t, torch.Tensor):
                teacher_outputs_detached.append(t.detach())  # 確保分離教師輸出
            else:
                teacher_outputs_detached.append([o.detach() if isinstance(o, torch.Tensor) else o for o in t])
        
        # 累加損失，分別處理形狀相同和不同的層
        losses = []
        try:
            for i, (t, s) in enumerate(zip(teacher_outputs_detached, self.student_outputs)):
                if isinstance(t, torch.Tensor) and isinstance(s, torch.Tensor):
                    # 檢查是否是形狀不同的層
                    if t.shape != s.shape:
                        # 檢查是否是22.cv2.conv層(P5層)
                        is_p5_layer = False
                        
                        # 如果有記錄當前層名稱的屬性，則使用它判斷
                        if hasattr(self, 'current_layer_name'):
                            is_p5_layer = "model.22.cv2.conv" in self.current_layer_name
                        
                        # 如果沒有層名稱，則嘗試通過特徵形狀和索引猜測
                        # P5層通常是低分辨率高通道數的特徵圖
                        elif (t.shape[1] >= 512 and s.shape[1] >= 256 and t.shape[2] <= 20 and t.shape[3] <= 20):
                            is_p5_layer = True
                            LOGGER.info(f"通過特徵形狀猜測層 {i} 可能是P5層")
                        
                        if is_p5_layer:
                            LOGGER.info(f"對層 {i} 應用P5特化蒸餾方法")
                            loss = self.p5_feature_distillation(s, t)
                        else:
                            # 對於其他形狀不同的層，使用通用通道自適應方法
                            LOGGER.info(f"對層 {i} 應用通用通道自適應蒸餾方法")
                            loss = self.channel_adaptive_distillation(s, t)
                        
                        losses.append(loss)
                    else:
                        # 形狀相同的層，使用原有邏輯處理
                        # 計算差異
                        diff = (s - t).abs()
                        max_layer_diff = diff.max().item()
                        
                        # 如果層的最大差異低於閾值，則忽略該層的損失
                        if max_layer_diff < DIFFERENCE_THRESHOLD:
                            LOGGER.info(f"層 {i} 的最大差異 ({max_layer_diff:.8f}) 低於閾值，忽略該層的損失計算")
                            continue
                        
                        # 使用標準MSE損失
                        losses.append(F.mse_loss(s, t))
                
            # 如果有任何有效損失，則將它們相加
            if losses:
                loss = torch.sum(torch.stack(losses))
                LOGGER.info(f"計算出的總損失: {loss.item():.8f}")
            else:
                # 如果沒有有效損失，創建一個零損失但帶梯度
                loss = torch.zeros(1, device=next(self.models.parameters()).device, requires_grad=True)
                LOGGER.info("沒有計算任何有效損失，返回零損失")
        except Exception as e:
            LOGGER.error(f"計算蒸餾損失時出錯: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            # 返回零損失但帶梯度
            loss = torch.zeros(1, device=next(self.models.parameters()).device, requires_grad=True)
        
        # 清空教師和學生的輸出列表，避免二次使用
        self.teacher_outputs.clear()
        self.student_outputs.clear()
        
        return loss

    def p5_feature_distillation(self, student_feature, teacher_feature):
        """
        為姿勢估計優化的平衡FGD方法，降低高激活區域的權重
        """
        # 基本尺寸信息
        batch_size = student_feature.shape[0]
        student_channels = student_feature.shape[1]  # 256
        spatial_size = student_feature.shape[2:]  # (H, W)
        
        # 確保空間維度相同
        if student_feature.shape[2:] != teacher_feature.shape[2:]:
            teacher_feature = F.interpolate(
                teacher_feature, 
                size=student_feature.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # 1. 通道維度壓縮 - 只使用教師模型前256個通道
        t_selected = teacher_feature[:, :student_channels]  # [B, 256, H, W]
        
        # 2. 特徵通道注意力
        s_pool = F.adaptive_avg_pool2d(student_feature, 1).flatten(1)  # [B, 256]
        t_pool = F.adaptive_avg_pool2d(t_selected, 1).flatten(1)  # [B, 256]
        
        # 標準化通道注意力
        s_pool_norm = F.normalize(s_pool, p=2, dim=1)
        t_pool_norm = F.normalize(t_pool, p=2, dim=1)
        
        # 通道注意力損失 - 使用余弦相似度而不是MSE
        channel_loss = (1 - torch.mean(torch.sum(s_pool_norm * t_pool_norm, dim=1))) * 2.0
        
        # 3. 空間注意力
        s_spatial = torch.norm(student_feature, p=2, dim=1)  # [B, H, W]
        t_spatial = torch.norm(t_selected, p=2, dim=1)  # [B, H, W]
        
        # 標準化空間注意力
        s_spatial_norm = (s_spatial - s_spatial.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / \
                        (s_spatial.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] - 
                        s_spatial.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] + 1e-10)
        
        t_spatial_norm = (t_spatial - t_spatial.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]) / \
                        (t_spatial.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0] - 
                        t_spatial.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0] + 1e-10)
        
        # 空間注意力損失 - 使用L1損失而不是MSE，減少對異常值的敏感性
        spatial_loss = F.l1_loss(s_spatial_norm, t_spatial_norm) * 5.0
        
        # 4. 高激活區域特徵匹配 - 但降低其強度
        threshold = 0.5
        mask = (t_spatial_norm > threshold).float()
        
        # 確保至少有10%的區域被選中
        coverage = mask.mean()
        if coverage < 0.1:
            # 動態調整閾值
            k = int(0.1 * mask.shape[1] * mask.shape[2])
            for b in range(batch_size):
                values, _ = torch.topk(t_spatial_norm[b].reshape(-1), k)
                mask[b] = (t_spatial_norm[b] > values[-1]).float()
        
        # 在高激活區域進行溫和的特徵匹配
        mask_expanded = mask.unsqueeze(1)  # [B, 1, H, W]
        
        # 使用L1損失並降低強度
        masked_loss = F.l1_loss(
            student_feature * mask_expanded,
            t_selected * mask_expanded
        ) / (mask.mean() + 1e-10) * 2.0  # 大幅降低放大係數
        
        # 5. 溫和的全局特徵匹配
        global_loss = F.l1_loss(student_feature, t_selected) * 0.5
        
        # 6. 統計量匹配
        s_mean = student_feature.mean(dim=[2, 3])  # [B, 256]
        t_mean = t_selected.mean(dim=[2, 3])  # [B, 256]
        s_std = torch.sqrt(student_feature.var(dim=[2, 3]) + 1e-6)  # [B, 256]
        t_std = torch.sqrt(t_selected.var(dim=[2, 3]) + 1e-6)  # [B, 256]
        
        # 使用L1損失
        mean_loss = F.l1_loss(s_mean, t_mean)
        std_loss = F.l1_loss(s_std, t_std)
        stats_loss = (mean_loss + std_loss) * 1.0
        
        # 總損失計算 - 更平衡的權重分配
        total_loss = (
            0.20 * channel_loss +    # 通道注意力
            0.30 * spatial_loss +    # 空間注意力
            0.20 * masked_loss +     # 高激活區域匹配（降低權重）
            0.15 * global_loss +     # 全局特徵匹配
            0.15 * stats_loss        # 統計量匹配
        )
        
        # 不再額外放大損失
        # 如果pose_loss在增加，可能是我們的蒸餾太強了
        amplified_loss = total_loss
        
        # 日誌記錄
        LOGGER.info(f"P5層平衡FGD蒸餾損失詳情:")
        LOGGER.info(f"  通道注意力損失: {channel_loss.item():.6f} (權重: 0.20)")
        LOGGER.info(f"  空間注意力損失: {spatial_loss.item():.6f} (權重: 0.30)")
        LOGGER.info(f"  高激活區域損失: {masked_loss.item():.6f} (權重: 0.20)")
        LOGGER.info(f"  全局特徵損失: {global_loss.item():.6f} (權重: 0.15)")
        LOGGER.info(f"  統計量損失: {stats_loss.item():.6f} (權重: 0.15)")
        LOGGER.info(f"  總損失: {amplified_loss.item():.6f}")
        LOGGER.info(f"  高激活區域覆蓋率: {mask.mean().item()*100:.2f}%")
        
        return amplified_loss

    def channel_adaptive_distillation(self, student_feature, teacher_feature):
        """
        通用的通道自適應蒸餾方法，用於處理非P5層的不同通道數特徵圖
        
        Args:
            student_feature: 學生模型特徵圖
            teacher_feature: 教師模型特徵圖
        
        Returns:
            蒸餾損失
        """
        # 確保空間維度相同
        if student_feature.shape[2:] != teacher_feature.shape[2:]:
            teacher_feature = F.interpolate(
                teacher_feature, 
                size=student_feature.shape[2:],
                mode='bilinear', 
                align_corners=False
            )
        
        # 計算空間注意力圖
        student_spatial = torch.mean(student_feature, dim=1, keepdim=True)  # [B, 1, H, W]
        teacher_spatial = torch.mean(teacher_feature, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 空間注意力損失
        spatial_loss = F.mse_loss(student_spatial, teacher_spatial)
        
        # 計算通道注意力 (全局平均池化)
        s_channel = F.adaptive_avg_pool2d(student_feature, 1)  # [B, C_s, 1, 1]
        t_channel = F.adaptive_avg_pool2d(teacher_feature, 1)  # [B, C_t, 1, 1]
        
        # 標準化通道注意力
        s_channel = F.normalize(s_channel.squeeze(-1).squeeze(-1), p=2, dim=1)
        t_channel = F.normalize(t_channel.squeeze(-1).squeeze(-1), p=2, dim=1)
        
        # 取較小的通道數
        min_channels = min(student_feature.shape[1], teacher_feature.shape[1])
        channel_loss = F.mse_loss(s_channel[:, :min_channels], t_channel[:, :min_channels])
        
        # 組合損失
        total_loss = spatial_loss * 0.7 + channel_loss * 0.3
        
        LOGGER.info(f"通道自適應蒸餾損失詳情:")
        LOGGER.info(f"  空間注意力損失: {spatial_loss.item():.6f}")
        LOGGER.info(f"  通道注意力損失: {channel_loss.item():.6f}")
        LOGGER.info(f"  總損失: {total_loss.item():.6f}")
        
        return total_loss


    # def get_loss(self):
    #     """計算教師和學生模型之間的蒸餾損失，直接使用CWD方法"""        
    #     if not self.teacher_outputs or not self.student_outputs:
    #         return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)
        
    #     # 準備有效的特徵張量列表
    #     valid_teacher_features = []
    #     valid_student_features = []
        
    #     for i, (t, s) in enumerate(zip(self.teacher_outputs, self.student_outputs)):
    #         if isinstance(t, torch.Tensor) and isinstance(s, torch.Tensor):
    #             # 基本檢查，確保特徵是有效的
    #             if torch.isnan(s).any() or torch.isinf(s).any() or torch.isnan(t).any() or torch.isinf(t).any():
    #                 LOGGER.warning(f"警告: 層 {i} 中發現NaN或Inf值!")
    #                 continue
                    
    #             # 將有效特徵添加到列表中
    #             valid_teacher_features.append(t.detach())  # 確保分離教師輸出
    #             valid_student_features.append(s)  # 保持學生特徵的梯度
        
    #     # 如果沒有有效特徵對，返回零損失
    #     if not valid_teacher_features or not valid_student_features:
    #         LOGGER.warning("沒有找到有效的特徵對進行蒸餾")
    #         return torch.zeros(1, device=next(self.models.parameters()).device, requires_grad=True)
        
    #     try:
    #         # 取得特徵通道數
    #         student_channels = [f.shape[1] for f in valid_student_features]
    #         teacher_channels = [f.shape[1] for f in valid_teacher_features]
            
    #         # 檢查是否需要初始化或更新CWD損失函數
    #         if not hasattr(self, 'cwd_loss') or self.cwd_loss is None:
    #             # 首次初始化CWD損失
    #             self.cwd_loss = CWDLoss(
    #                 student_channels=student_channels,
    #                 teacher_channels=teacher_channels,
    #                 tau=4.0,
    #                 normalize=True
    #             ).to(next(self.models.parameters()).device)
    #             LOGGER.info(f"已初始化CWD損失函數，學生通道: {student_channels}, 教師通道: {teacher_channels}")
    #         elif (len(student_channels) != len(self.cwd_loss.adapters) or 
    #             any(s.shape[1] != adapter.weight.shape[0] for s, adapter in zip(valid_student_features, self.cwd_loss.adapters))):
    #             # 通道數變化，需要重新初始化
    #             LOGGER.info(f"特徵通道數已變化，重新初始化CWD損失函數")
    #             LOGGER.info(f"新的學生通道: {student_channels}, 教師通道: {teacher_channels}")
    #             self.cwd_loss = CWDLoss(
    #                 student_channels=student_channels,
    #                 teacher_channels=teacher_channels,
    #                 tau=4.0,
    #                 normalize=True
    #             ).to(next(self.models.parameters()).device)
            
    #         # 計算損失
    #         loss = self.cwd_loss(valid_student_features, valid_teacher_features)
            
    #         # 記錄損失值
    #         LOGGER.debug(f"CWD蒸餾損失: {loss.item():.6f}")
    #     except Exception as e:
    #         LOGGER.error(f"計算CWD蒸餾損失時出錯: {e}")
    #         import traceback
    #         LOGGER.error(traceback.format_exc())
    #         # 返回零損失但帶梯度
    #         loss = torch.zeros(1, device=next(self.models.parameters()).device, requires_grad=True)
        
    #     # 清空教師和學生的輸出列表，避免二次使用
    #     self.teacher_outputs.clear()
    #     self.student_outputs.clear()
        
    #     return loss

    def remove_handle_(self):
        """安全移除已註冊的鉤子"""
        for rm in self.remove_handle:
            try:
                rm.remove()
            except Exception as e:
                LOGGER.warning(f"移除鉤子時出錯: {e}")
        self.remove_handle.clear()
        
        # 清空收集的特徵列表，避免保留對特徵的不必要引用
        if hasattr(self, 'teacher_outputs'):
            self.teacher_outputs.clear()
        if hasattr(self, 'student_outputs'):
            self.student_outputs.clear()
        
        LOGGER.debug("已移除蒸餾鉤子")
