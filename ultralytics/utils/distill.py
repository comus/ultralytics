import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.utils import LOGGER

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
    """特徵層蒸餾損失，支持多種蒸餾方法，包括特徵對齊"""
    def __init__(self, channels_s, channels_t, distiller='mgd', loss_weight=1.0):
        super(FeatureLoss, self).__init__()
        self.loss_weight = loss_weight
        self.distiller = distiller
        
        # 將所有模塊移動到相同精度
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 轉換為ModuleList並確保一致的數據類型
        self.align_module = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        
        # 創建對齊模塊
        for s_chan, t_chan in zip(channels_s, channels_t):
            align = nn.Sequential(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(t_chan, affine=False)
            ).to(device)
            self.align_module.append(align)
            
        # 創建歸一化層
        for t_chan in channels_t:
            self.norm.append(nn.BatchNorm2d(t_chan, affine=False).to(device))
            
        for s_chan in channels_s:
            self.norm1.append(nn.BatchNorm2d(s_chan, affine=False).to(device))
            
        # 選擇蒸餾損失函數
        if distiller == 'mgd':
            self.feature_loss = MGDLoss(channels_s, channels_t)
        elif distiller == 'cwd':
            self.feature_loss = CWDLoss(channels_s, channels_t)
        elif distiller == 'rev':
            self.feature_loss = ReviewKDLoss(channels_s, channels_t)
        elif distiller == 'fgd':
            self.feature_loss = FGDLoss(channels_s, channels_t, spatial_weight=2.0, channel_weight=0.6)
        elif distiller == 'enhancedfgd':
            self.feature_loss = EnhancedFGDLoss(channels_s, channels_t, spatial_weight=3.5, channel_weight=0.4)
            LOGGER.info("使用增強版FGD蒸餾方法")
        elif distiller == 'pkd':
            self.feature_loss = PKDLoss(channels_s, channels_t)
        else:
            raise NotImplementedError(f"Unknown distiller: {distiller}")
            
    def forward(self, y_s, y_t):
        """計算特徵層蒸餾損失
        
        Args:
            y_s (list): 學生模型的特徵
            y_t (list): 教師模型的特徵
            
        Returns:
            torch.Tensor: 計算的損失值
        """
        # LOGGER.info(f"FeatureLoss.forward 被調用，特徵列表長度 - 學生: {len(y_s)}, 教師: {len(y_t)}")
        
        min_len = min(len(y_s), len(y_t))
        y_s = y_s[:min_len]
        y_t = y_t[:min_len]

        tea_feats = []
        stu_feats = []

        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if idx >= len(self.align_module):
                LOGGER.warning(f"索引 {idx} 超出對齊模塊範圍 {len(self.align_module)}")
                break
            
            # 處理不同類型的特徵
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                LOGGER.warning(f"特徵類型不是張量 - 學生: {type(s)}, 教師: {type(t)}")
                if isinstance(s, list) and len(s) > 0:
                    s = s[0]
                if isinstance(t, list) and len(t) > 0:
                    t = t[0]
            
            # 確保特徵是張量
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                LOGGER.warning(f"轉換後特徵類型仍不是張量 - 學生: {type(s)}, 教師: {type(t)}")
                continue
                
            # LOGGER.info(f"處理第 {idx+1} 對特徵 - 學生形狀: {s.shape}, 教師形狀: {t.shape}")
            
            # 轉換數據類型以匹配對齊模塊
            s = s.type(next(self.align_module[idx].parameters()).dtype)
            t = t.type(next(self.align_module[idx].parameters()).dtype)

            try:
                # 特別處理FGD方法，直接傳遞特徵而不進行標準化
                if self.distiller == "fgd":
                    if s.shape[1] != t.shape[1]:  # 如果通道數不匹配
                        s = self.align_module[idx](s)
                    stu_feats.append(s)
                    tea_feats.append(t.detach())
                elif self.distiller == "cwd":
                    s = self.align_module[idx](s)
                    # LOGGER.info(f"  對齊後學生特徵形狀: {s.shape}")
                    stu_feats.append(s)
                    tea_feats.append(t.detach())
                else:
                    t = self.norm[idx](t)
                    # LOGGER.info(f"  標準化後教師特徵形狀: {t.shape}")
                    stu_feats.append(s)
                    tea_feats.append(t.detach())
            except Exception as e:
                LOGGER.error(f"處理特徵時出錯: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())
                continue

        # 安全檢查
        if len(stu_feats) == 0 or len(tea_feats) == 0:
            LOGGER.warning("沒有有效的特徵對進行蒸餾")
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)
        
        try:
            # LOGGER.info(f"調用 {self.feature_loss.__class__.__name__} 計算損失")
            loss = self.feature_loss(stu_feats, tea_feats)
            # LOGGER.info(f"計算的原始損失: {loss.item():.6f}, 加權後: {(self.loss_weight * loss).item():.6f}")
            return self.loss_weight * loss
        except Exception as e:
            LOGGER.error(f"計算特徵損失時出錯: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            return torch.tensor(0.0, requires_grad=True, device=next(self.parameters()).device)

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

        # 進行模型預熱
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            try:
                dummy_input = torch.randn(1, 3, 640, 640)
                _ = self.models(dummy_input.to(device))
                _ = self.modelt(dummy_input.to(device))
                LOGGER.info("模型預熱成功")
            except Exception as e:
                LOGGER.warning(f"模型預熱失敗: {e}")
        
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
        
        # 創建蒸餾損失實例
        self.distill_loss_fn = FeatureLoss(
            channels_s=self.channels_s, 
            channels_t=self.channels_t, 
            distiller=self.distiller
        )
        
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
        """計算教師和學生模型之間的蒸餾損失"""            
        if not self.teacher_outputs or not self.student_outputs:
            LOGGER.warning(f"沒有收集到特徵 - 教師: {len(self.teacher_outputs) if hasattr(self, 'teacher_outputs') else 0}, 學生: {len(self.student_outputs) if hasattr(self, 'student_outputs') else 0}")
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)
        
        if len(self.teacher_outputs) != len(self.student_outputs):
            LOGGER.warning(f"輸出不匹配 - 教師: {len(self.teacher_outputs)}, 學生: {len(self.student_outputs)}")
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)
        
        try:
            teacher_outputs = [t.detach() for t in self.teacher_outputs]
            quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=teacher_outputs)
            
            # 檢查損失值是否合理
            if torch.isnan(quant_loss) or torch.isinf(quant_loss):
                LOGGER.warning(f"蒸餾損失值無效: {quant_loss}")
                quant_loss = torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)
            elif quant_loss == 0:
                LOGGER.warning("蒸餾損失為零，可能計算出錯")
            else:
                pass
            
            # 清除收集的特徵，準備下一個批次
            self.teacher_outputs.clear()
            self.student_outputs.clear()
            
            return quant_loss
        except Exception as e:
            LOGGER.error(f"計算蒸餾損失時出錯: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())  # 打印完整的錯誤堆疊
            return torch.tensor(0.0, requires_grad=True, device=next(self.models.parameters()).device)

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
