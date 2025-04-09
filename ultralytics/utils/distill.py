# distill.py - 修复版本

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
        """計算CWD損失"""
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

class FGDLoss(nn.Module):
    """增强版特征引导蒸馏 - 专为姿态估计优化"""
    def __init__(self, student_channels, teacher_channels, spatial_weight=2.0, channel_weight=0.6):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight
        
        # 增加边缘感知机制
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.edge_filter.weight.data.copy_(torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3) / 8.0)
        self.edge_filter.requires_grad_(False)
        
        # 添加通道适配层
        self.align_channels = nn.ModuleList()
        for s_chan, t_chan in zip(student_channels, teacher_channels):
            if s_chan != t_chan:
                self.align_channels.append(
                    nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False)
                )
            else:
                self.align_channels.append(nn.Identity())
                
        LOGGER.info(f"初始化FGD损失: 空间权重={spatial_weight}, 通道权重={channel_weight}")
        LOGGER.info(f"创建了 {len(self.align_channels)} 个通道适配层，用于处理通道不匹配")
        
    def forward(self, y_s, y_t):
        """计算增强版FGD损失，特别优化姿态估计的空间特征保留"""
        losses = []
        min_len = min(len(y_s), len(y_t), len(self.align_channels))
        
        for idx in range(min_len):
            s, t = y_s[idx], y_t[idx]
            
            # 跳过非张量特征
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                continue
                
            # 确保align_channels在同一设备上
            self.align_channels[idx] = self.align_channels[idx].to(s.device)
            self.edge_filter = self.edge_filter.to(s.device)
                
            # 1. 通道对齐
            s = self.align_channels[idx](s)
            LOGGER.debug(f"特征形状 - 学生: {s.shape}, 教师: {t.shape}")
                
            # 2. 空间尺寸对齐
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = F.interpolate(t, size=(s.size(2), s.size(3)), 
                                 mode='bilinear', align_corners=False)
                
            b, c, h, w = s.shape
            
            # 3. 基本特征匹配
            l2_loss = F.smooth_l1_loss(s, t)
            
            # 4. 空间注意力损失
            s_spatial = torch.mean(s, dim=1, keepdim=True)
            t_spatial = torch.mean(t, dim=1, keepdim=True)
            
            s_edge = self.edge_filter(s_spatial)
            t_edge = self.edge_filter(t_spatial)
            
            spatial_loss = (F.mse_loss(s_spatial, t_spatial) + 
                           F.mse_loss(s_edge, t_edge)) * self.spatial_weight
            
            # 5. 通道关系损失
            s_flat = s.view(b, c, -1)
            t_flat = t.view(b, c, -1)
            
            s_flat_norm = F.normalize(s_flat, dim=2)
            t_flat_norm = F.normalize(t_flat, dim=2)
            
            s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)
            t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)
            
            channel_loss = F.mse_loss(s_corr, t_corr) * self.channel_weight
            
            # 6. 总损失
            total_loss = l2_loss + spatial_loss + channel_loss
            losses.append(total_loss)
            
        if not losses:
            return torch.tensor(0.0, device=y_s[0].device if len(y_s) > 0 and isinstance(y_s[0], torch.Tensor) else 'cpu', requires_grad=True)
            
        return sum(losses)

class ReviewKDLoss(nn.Module):
    """Review-KD: https://arxiv.org/abs/2104.09044"""
    def __init__(self, student_channels, teacher_channels, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, y_s, y_t):
        """计算Review-KD损失"""
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

class EnhancedFGDLoss(nn.Module):
    """增强版特征引导蒸馏 - 专为姿态估计优化，提供更好的关键点定位能力"""
    def __init__(self, student_channels, teacher_channels, spatial_weight=3.5, channel_weight=0.4):
        super().__init__()
        self.spatial_weight = spatial_weight
        self.channel_weight = channel_weight
        
        # 增强版边缘感知机制
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.edge_filter.weight.data.copy_(torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3) / 8.0)
        self.edge_filter.requires_grad_(False)
        
        # 关键点注意力机制
        self.keypoint_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 用于通道调整的投影层
        self.align_layers = nn.ModuleList()
        for s_chan, t_chan in zip(student_channels, teacher_channels):
            self.align_layers.append(
                nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False)
            )
        
        LOGGER.info(f"初始化增强版FGD损失，学生通道: {student_channels}, 教师通道: {teacher_channels}")
        
    def forward(self, y_s, y_t):
        """计算增强版FGD损失，特别优化姿态估计的空间特征保留"""
        losses = []
        
        min_len = min(len(y_s), len(y_t), len(self.align_layers))
        LOGGER.debug(f"处理 {min_len} 对特征")
        
        for idx in range(min_len):
            s, t = y_s[idx], y_t[idx]
            
            # 处理空层或None值
            if not isinstance(s, torch.Tensor) or not isinstance(t, torch.Tensor):
                continue
            
            # 将所有模块移动到当前数据所在的设备
            device = s.device
            self.align_layers[idx] = self.align_layers[idx].to(device)
            self.edge_filter = self.edge_filter.to(device)
            self.keypoint_attention = self.keypoint_attention.to(device)
            
            # 通道对齐 - 强制使用对齐层
            s_aligned = self.align_layers[idx](s)
            LOGGER.debug(f"特征通道对齐: {s.shape[1]} -> {s_aligned.shape[1]} (目标: {t.shape[1]})")
                
            # 尺寸对齐    
            if s_aligned.size(2) != t.size(2) or s_aligned.size(3) != t.size(3):
                t = F.interpolate(t, size=(s_aligned.size(2), s_aligned.size(3)), 
                                 mode='bilinear', align_corners=False)
                
            b, c, h, w = s_aligned.shape
            
            try:
                # 1. L1损失
                l1_loss = F.smooth_l1_loss(s_aligned, t)
                
                # 2. 空间注意力损失
                s_spatial = torch.mean(s_aligned, dim=1, keepdim=True)
                t_spatial = torch.mean(t, dim=1, keepdim=True)
                
                s_edge = self.edge_filter(s_spatial)
                t_edge = self.edge_filter(t_spatial)
                
                keypoint_attn = self.keypoint_attention(t_spatial)
                
                spatial_loss = (
                    F.mse_loss(s_spatial, t_spatial) * 1.5 + 
                    F.mse_loss(s_edge, t_edge) * 1.0 +
                    F.mse_loss(s_spatial * keypoint_attn, t_spatial * keypoint_attn) * 1.5
                ) * self.spatial_weight
                
                # 3. 通道相关性损失
                s_flat = s_aligned.view(b, c, -1)
                t_flat = t.view(b, c, -1)
                
                s_flat_norm = F.normalize(s_flat, dim=2)
                t_flat_norm = F.normalize(t_flat, dim=2)
                
                s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)
                t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)
                
                channel_loss = F.mse_loss(s_corr, t_corr) * self.channel_weight
                
                # 4. 组合损失
                total_loss = l1_loss + spatial_loss + channel_loss
                losses.append(total_loss)
                
            except Exception as e:
                LOGGER.error(f"计算损失时出错: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())
                continue
            
        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=y_s[0].device if len(y_s) > 0 and isinstance(y_s[0], torch.Tensor) else 'cuda')
            
        return sum(losses)

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
        
        # 初始化属性 - 修复缺失的属性
        self.student_module_pairs = []
        self.teacher_module_pairs = []
        self.channels_s = []
        self.channels_t = []
        
        # 记录找到的层
        self.found_student_layers = []
        self.found_teacher_layers = []
        
        # 安全地获取设备
        try:
            self.device = next(self.models.parameters()).device
        except:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查是否启用混合精度训练
        self.using_amp = False
        if hasattr(models, 'args') and hasattr(models.args, 'amp'):
            self.using_amp = models.args.amp
            LOGGER.info(f"检测到混合精度训练设置: {self.using_amp}")
        
        # 寻找要蒸馏的层
        try:
            self._find_layers()
        except Exception as e:
            LOGGER.error(f"查找蒸馏层时出错: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            raise
        
        if len(self.channels_s) == 0 or len(self.channels_t) == 0:
            LOGGER.error("未找到任何匹配的层进行蒸馏!")
            raise ValueError("无法找到适合蒸馏的层")
            
        LOGGER.info(f"成功配对 {len(self.teacher_module_pairs)} 对学生-教师特征层")
        LOGGER.info(f"学生层: {self.found_student_layers}")
        LOGGER.info(f"教师层: {self.found_teacher_layers}")
        LOGGER.info(f"学生通道: {self.channels_s}")
        LOGGER.info(f"教师通道: {self.channels_t}")
        
        # 创建蒸馏损失实例
        self._init_distill_loss()

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
    
    def _init_distill_loss(self):
        """初始化蒸馏损失函数"""
        # 获取模型的设备
        device = next(self.models.parameters()).device
        
        # 根据蒸馏方法选择对应的损失函数
        if self.distiller == "cwd":
            LOGGER.info("使用Channel-wise Knowledge Distillation(CWD)进行蒸馏")
            self.distill_loss_fn = CWDLoss(
                channels_s=self.channels_s,
                channels_t=self.channels_t,
                tau=1.0
            ).to(device)
        elif self.distiller == "reviewkd":
            LOGGER.info("使用Review KD进行蒸馏")
            self.distill_loss_fn = ReviewKDLoss(
                student_channels=self.channels_s,
                teacher_channels=self.channels_t,
                temperature=1.0
            ).to(device)
        elif self.distiller == "enhancedfgd":
            LOGGER.info("使用Enhanced FGD进行蒸馏 - 姿态优化版")
            self.distill_loss_fn = EnhancedFGDLoss(
                student_channels=self.channels_s,
                teacher_channels=self.channels_t,
                spatial_weight=3.5,
                channel_weight=0.4
            ).to(device)
        else:  # 默认使用FGD
            LOGGER.info("使用Feature Guided Distillation(FGD)进行蒸馏")
            self.distill_loss_fn = FGDLoss(
                student_channels=self.channels_s,
                teacher_channels=self.channels_t,
                spatial_weight=2.0,
                channel_weight=0.6
            ).to(device)
    
    def register_hook(self):
        """注册钩子函数，增强错误处理和日志记录"""
        # 清空之前的输出和钩子
        self.remove_handle_()
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
    
    def remove_handle_(self):
        """移除所有钩子"""
        for handle in self.remove_handle:
            handle.remove()
        self.remove_handle = []
        LOGGER.debug("移除所有蒸馏钩子")
    
    def get_loss(self):
        """计算蒸馏损失，处理数据类型不匹配问题"""
        if not self.teacher_outputs or not self.student_outputs:
            LOGGER.debug("没有收集到特征输出，返回零损失")
            # 使用更安全的方式获取设备
            device = next(self.models.parameters()).device if hasattr(self.models, 'parameters') else 'cuda'
            return torch.tensor(0.0).to(device)
        
        # 解除教师特征的梯度
        teacher_outputs = [t.detach() if isinstance(t, torch.Tensor) else t for t in self.teacher_outputs]
        
        try:
            # 尝试正常计算损失
            quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=teacher_outputs)
            return quant_loss
        except Exception as e:
            LOGGER.error(f"计算蒸馏损失时出现未知错误: {e}")
            import traceback
            LOGGER.error(traceback.format_exc())
            
            # 使用更安全的方式获取设备
            device = (self.student_outputs[0].device 
                    if len(self.student_outputs) > 0 and isinstance(self.student_outputs[0], torch.Tensor)
                    else next(self.models.parameters()).device if hasattr(self.models, 'parameters')
                    else 'cuda')
            
            return torch.tensor(0.0).to(device)