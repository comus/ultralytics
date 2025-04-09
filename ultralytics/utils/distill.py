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
        self.spatial_weight = spatial_weight    # 更强的空间权重
        self.channel_weight = channel_weight    # 降低通道权重
        # 增加边缘感知机制
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_filter.weight.data.copy_(torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3) / 8.0)
        self.edge_filter.requires_grad_(False)  # 冻结参数
        
    def forward(self, y_s, y_t):
        """计算增强版FGD损失，特别优化姿态估计的空间特征保留"""
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = F.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            b, c, h, w = s.shape
            
            # 1. 基本特征匹配 - 使用Huber损失减少异常值影响
            l2_loss = F.smooth_l1_loss(s, t)
            
            # 2. 增强版空间注意力损失
            s_spatial = torch.mean(s, dim=1, keepdim=True)  # [b, 1, h, w]
            t_spatial = torch.mean(t, dim=1, keepdim=True)  # [b, 1, h, w]
            
            # 提取空间特征的边缘信息
            s_edge = self.edge_filter(s_spatial)
            t_edge = self.edge_filter(t_spatial)
            
            # 空间注意力+边缘感知的组合损失
            spatial_loss = (F.mse_loss(s_spatial, t_spatial) + 
                           F.mse_loss(s_edge, t_edge)) * self.spatial_weight
            
            # 3. 通道关系损失 - 专注于重要通道
            s_flat = s.view(b, c, -1)  # [b, c, h*w]
            t_flat = t.view(b, c, -1)  # [b, c, h*w]
            
            # 通道归一化
            s_flat_norm = F.normalize(s_flat, dim=2)
            t_flat_norm = F.normalize(t_flat, dim=2)
            
            # 计算通道相关矩阵
            s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
            t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
            
            # 通道相关性损失
            channel_loss = F.mse_loss(s_corr, t_corr) * self.channel_weight
            
            # 4. 组合损失
            total_loss = l2_loss + spatial_loss + channel_loss
            losses.append(total_loss)
            
        loss = sum(losses)
        return loss

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
        self.spatial_weight = spatial_weight    # 更强的空间权重
        self.channel_weight = channel_weight    # 降低通道权重
        
        # 增强版边缘感知机制
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_filter.weight.data.copy_(torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3) / 8.0)
        self.edge_filter.requires_grad_(False)  # 冻结参数
        
        # 关键点注意力机制
        self.keypoint_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 用于通道调整的投影层
        self.align_layers = nn.ModuleList()
        for s_chan, t_chan in zip(student_channels, teacher_channels):
            if s_chan != t_chan:
                self.align_layers.append(nn.Conv2d(s_chan, t_chan, kernel_size=1, bias=False).to(
                    'cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                self.align_layers.append(nn.Identity())
                
        LOGGER.info(f"初始化增强版FGD损失: 空间权重={spatial_weight}, 通道权重={channel_weight}")
        
    def forward(self, y_s, y_t):
        """计算增强版FGD损失，特别优化姿态估计的空间特征保留"""
        assert len(y_s) == len(y_t)
        losses = []
        
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            # 确保索引在有效范围内
            if idx >= len(self.align_layers):
                continue
                
            # 处理空层或None值
            if s is None or t is None:
                continue
            
            # 通道对齐
            try:
                if s.shape[1] != t.shape[1]:
                    # 如果通道不一致，使用投影层调整
                    s = self.align_layers[idx](s)
                    LOGGER.debug(f"调整特征通道: {s.shape} -> {t.shape}")
            except Exception as e:
                LOGGER.warning(f"通道调整出错，跳过此层: {e}")
                continue
                
            # 尺寸对齐    
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = torch.nn.functional.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            b, c, h, w = s.shape
            
            # 1. 自适应特征匹配 - 安全处理
            try:
                # 对重要特征赋予更高权重
                feature_importance = torch.sigmoid(torch.mean(t, dim=1, keepdim=True))
                l1_loss = torch.nn.functional.smooth_l1_loss(s, t)  # 直接计算损失，避免乘法导致的尺寸问题
            except Exception as e:
                LOGGER.warning(f"计算L1损失时出错: {e}")
                l1_loss = torch.tensor(0.0, device=s.device)
            
            # 2. 空间注意力损失 - 安全处理
            try:
                s_spatial = torch.mean(s, dim=1, keepdim=True)  # [b, 1, h, w]
                t_spatial = torch.mean(t, dim=1, keepdim=True)  # [b, 1, h, w]
                
                # 提取空间特征的边缘信息
                s_edge = self.edge_filter(s_spatial)
                t_edge = self.edge_filter(t_spatial)
                
                # 关键点注意力增强
                keypoint_attn = self.keypoint_attention(t_spatial)
                
                # 空间注意力+边缘感知+关键点注意力的组合损失
                spatial_loss = (torch.nn.functional.mse_loss(s_spatial, t_spatial) * 1.5 + 
                            torch.nn.functional.mse_loss(s_edge, t_edge) * 1.0 +
                            torch.nn.functional.mse_loss(s_spatial * keypoint_attn, t_spatial * keypoint_attn) * 1.5) * self.spatial_weight
            except Exception as e:
                LOGGER.warning(f"计算空间损失时出错: {e}")
                spatial_loss = torch.tensor(0.0, device=s.device)
            
            # 3. 通道关系损失 - 安全处理
            try:
                s_flat = s.view(b, c, -1)  # [b, c, h*w]
                t_flat = t.view(b, c, -1)  # [b, c, h*w]
                
                # 通道归一化
                s_flat_norm = torch.nn.functional.normalize(s_flat, dim=2)
                t_flat_norm = torch.nn.functional.normalize(t_flat, dim=2)
                
                # 计算通道相关矩阵
                s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
                t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
                
                # 通道相关性损失
                channel_loss = torch.nn.functional.mse_loss(s_corr, t_corr) * self.channel_weight
            except Exception as e:
                LOGGER.warning(f"计算通道损失时出错: {e}")
                channel_loss = torch.tensor(0.0, device=s.device)
            
            # 4. 组合损失 - 确保所有损失都在相同设备上
            total_loss = l1_loss + spatial_loss + channel_loss
            losses.append(total_loss)
            
        if not losses:
            # 返回一个零张量，确保有梯度
            return torch.tensor(0.0, requires_grad=True, device=y_s[0].device if y_s and len(y_s) > 0 else 'cpu')
            
        loss = sum(losses)
        return loss

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
        
        # 检查是否启用混合精度训练
        self.using_amp = False
        if hasattr(models, 'args') and hasattr(models.args, 'amp'):
            self.using_amp = models.args.amp
            LOGGER.info(f"检测到混合精度训练设置: {self.using_amp}")
        
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
            return torch.tensor(0.0).to(self.models.device)
        
        # 检查特征列表长度
        if len(self.teacher_outputs) != len(self.student_outputs):
            LOGGER.warning(f"特征输出长度不匹配: 学生={len(self.student_outputs)}, 教师={len(self.teacher_outputs)}")
        
        # 解除教师特征的梯度
        teacher_outputs = [t.detach() if isinstance(t, torch.Tensor) else t for t in self.teacher_outputs]
        
        try:
            # 尝试正常计算损失
            quant_loss = self.distill_loss_fn(y_s=self.student_outputs, y_t=teacher_outputs)
            return quant_loss
        except RuntimeError as e:
            # 如果遇到数据类型错误，尝试转换数据类型
            if "Input type" in str(e) and "weight type" in str(e):
                LOGGER.warning(f"蒸馏损失计算出现数据类型不匹配: {e}")
                
                # 获取模型权重的数据类型
                model_dtype = next(self.distill_loss_fn.parameters()).dtype
                LOGGER.info(f"尝试将特征转换为模型数据类型: {model_dtype}")
                
                # 转换特征
                converted_student = [s.to(model_dtype) if isinstance(s, torch.Tensor) else s 
                                    for s in self.student_outputs]
                converted_teacher = [t.to(model_dtype) if isinstance(t, torch.Tensor) else t 
                                   for t in teacher_outputs]
                
                # 再次尝试计算损失
                try:
                    quant_loss = self.distill_loss_fn(y_s=converted_student, y_t=converted_teacher)
                    LOGGER.info("数据类型转换后成功计算损失")
                    return quant_loss
                except Exception as e2:
                    LOGGER.error(f"尝试转换数据类型后仍然失败: {e2}")
                    # 返回零损失避免训练中断
                    return torch.tensor(0.0).to(self.models.device)
            else:
                # 如果是其他错误，记录并返回零损失
                LOGGER.error(f"计算蒸馏损失时出现未知错误: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())
                return torch.tensor(0.0).to(self.models.device)