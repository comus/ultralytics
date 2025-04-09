# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import torch.nn as nn
import torch
import torch.optim as optim

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks
from ultralytics.utils.distill import DistillationLoss
from ultralytics.utils.plotting import plot_images, plot_results


class PoseTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training YOLO pose estimation models.

    This trainer specializes in handling pose estimation tasks, managing model training, validation, and visualization
    of pose keypoints alongside bounding boxes.

    Attributes:
        args (dict): Configuration arguments for training.
        model (PoseModel): The pose estimation model being trained.
        data (dict): Dataset configuration including keypoint shape information.
        loss_names (Tuple[str]): Names of the loss components used in training.

    Methods:
        get_model: Retrieves a pose estimation model with specified configuration.
        set_model_attributes: Sets keypoints shape attribute on the model.
        get_validator: Creates a validator instance for model evaluation.
        plot_training_samples: Visualizes training samples with keypoints.
        plot_metrics: Generates and saves training/validation metric plots.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseTrainer
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml", epochs=3)
        >>> trainer = PoseTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize a PoseTrainer object for training YOLO pose estimation models.

        This initializes a trainer specialized for pose estimation tasks, setting the task to 'pose' and
        handling specific configurations needed for keypoint detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (list, optional): List of callback functions to be executed during training.

        Notes:
            This trainer will automatically set the task to 'pose' regardless of what is provided in overrides.
            A warning is issued when using Apple MPS device due to known bugs with pose models.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseTrainer
            >>> args = dict(model="yolov8n-pose.pt", data="coco8-pose.yaml", epochs=3)
            >>> trainer = PoseTrainer(overrides=args)
            >>> trainer.train()
        """
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose"

        # 設置教師模型和蒸餾方法
        self.teacher = overrides.get("teacher", None)
        self.distill_loss_instance = None
        
        # 如果有教師模型和蒸餾方法
        if self.teacher is not None:
            # 創建自定義回調列表
            if _callbacks is None:
                _callbacks = callbacks.get_default_callbacks()

            _callbacks["on_train_start"].append(self.distill_on_train_start)
            _callbacks["on_train_epoch_start"].append(self.distill_on_epoch_start)
            _callbacks["on_train_epoch_end"].append(self.distill_on_epoch_end)
            _callbacks["on_val_start"].append(self.distill_on_val_start)
            _callbacks["on_val_end"].append(self.distill_on_val_end)
            _callbacks["on_train_end"].append(self.distill_on_train_end)
            _callbacks["teardown"].append(self.distill_teardown)
            
            LOGGER.info("已啟用蒸餾訓練策略")
        
        # 調用父類的初始化方法
        super().__init__(cfg, overrides, _callbacks)

        # 初始化教師模型（如果存在）
        if self.teacher is not None:
            # 凍結教師模型參數
            for k, v in self.teacher.named_parameters():
                v.requires_grad = False
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            LOGGER.info(f"初始化教師模型已完成，設為評估模式")

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )


    def distill_on_train_start(self, trainer):
        pass

    def distill_on_epoch_start(self, trainer):
        if self.epoch == 0:
            LOGGER.info("阶段1: 稳定特征预热")
            
            # 保持原始模型任务训练能力
            self.model.args.box = 0.5    # 保留边界框损失
            self.model.args.pose = 0.7   # 保留姿态损失 
            self.model.args.kobj = 0.5   # 保留关键点置信度损失
            self.model.args.cls = 0.5    # 保留分类损失
            self.model.args.dfl = 0.5    # 保留分布焦点损失
            self.model.args.distill = 0.1 # 轻微蒸馏
            
            # 使用单层温和蒸馏
            distillation_loss = "spatial_pose"
            distillation_layers = ["22"]
            
            # 初始化蒸馏损失实例
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=distillation_loss,
                layers=distillation_layers,
            )
            
            # 解冻检测头和部分蒸馏层 - 关键步骤
            self._selective_unfreeze(
                unfreeze_layers=["22", "23", "24", "25"],  # 解冻检测头和输出层
                partial_layers=["19"],   # 部分解冻其他层
                keep_ratio=0.1           # 部分解冻比例
            )
            
        elif self.epoch == 3:
            LOGGER.info("阶段2: 特征辅助蒸馏")
            
            # 逐步平衡原始任务和蒸馏
            self.model.args.box = 0.4
            self.model.args.pose = 0.5
            self.model.args.kobj = 0.4
            self.model.args.cls = 0.4
            self.model.args.dfl = 0.4
            self.model.args.distill = 0.3  # 增加蒸馏权重
            
            # 多层蒸馏
            distillation_loss = "spatial_pose"  # 继续使用温和蒸馏
            distillation_layers = ["19", "22"]
            
            self.distill_loss_instance.remove_handle_()
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=distillation_loss,
                layers=distillation_layers,
            )
            
            # 逐步解冻更多层
            self._selective_unfreeze(
                unfreeze_layers=["19", "22", "23", "24", "25"],
                partial_layers=["15", "17", "18", "20", "21"],
                keep_ratio=0.2
            )
            
        elif self.epoch == 8:
            LOGGER.info("阶段3: 平衡蒸馏")
            
            # 平衡原始任务和蒸馏
            self.model.args.box = 0.3
            self.model.args.pose = 0.4
            self.model.args.kobj = 0.3
            self.model.args.cls = 0.3
            self.model.args.dfl = 0.3
            self.model.args.distill = 0.5  # 进一步增加蒸馏权重
            
            # 使用增强蒸馏
            distillation_loss = "spatial_pose"
            distillation_layers = ["19", "22"]
            
            self.distill_loss_instance.remove_handle_()
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=distillation_loss,
                layers=distillation_layers,
            )
            
            # 解冻所有相关层
            self._selective_unfreeze(
                unfreeze_layers=["15", "17", "18", "19", "20", "21", "22", "23", "24", "25"],
                partial_layers=[],
                keep_ratio=0
            )
            
        elif self.epoch == 12:
            LOGGER.info("阶段4: 重点蒸馏")
            
            # 降低任务损失权重，提高蒸馏权重
            self.model.args.box = 0.2
            self.model.args.pose = 0.3
            self.model.args.kobj = 0.2
            self.model.args.cls = 0.2
            self.model.args.dfl = 0.2
            self.model.args.distill = 0.8  # 提高蒸馏权重
            
            # 降低学习率，确保稳定
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                LOGGER.info(f"降低学习率到 {param_group['lr']:.6f}")
                
        # 注册钩子
        self.distill_loss_instance.register_hook()

    def _selective_unfreeze(self, unfreeze_layers=None, partial_layers=None, keep_ratio=0.0):
        """选择性解冻函数
        
        Args:
            unfreeze_layers: 完全解冻的层
            partial_layers: 部分解冻的层
            keep_ratio: 任务损失保留比例
        """
        if unfreeze_layers is None:
            unfreeze_layers = []
        if partial_layers is None:
            partial_layers = []
        
        # 首先冻结所有层
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        
        # 解冻检测头和输出层 - 关键步骤，确保任务损失能够正常工作
        unfrozen_layers = 0
        unfrozen_bn_layers = 0
        
        # 完全解冻选定层
        for name, module in self.model.named_modules():
            if not name or not isinstance(name, str):
                continue
                
            # 检查是否是需要解冻的层
            is_target = False
            for layer in unfreeze_layers:
                if f"model.{layer}." in name:
                    is_target = True
                    break
            
            if is_target:
                # 解冻该模块的所有参数
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    unfrozen_layers += 1
                
                # 设置为训练模式
                module.train()
                LOGGER.info(f"完全解冻层: {name}")
        
        # 部分解冻选定层（只解冻部分卷积层）
        partial_count = 0
        for layer in partial_layers:
            layer_modules = []
            # 收集该层的所有模块
            for name, module in self.model.named_modules():
                if f"model.{layer}." in name and isinstance(module, nn.Conv2d):
                    layer_modules.append((name, module))
            
            # 随机选择一部分模块解冻
            if layer_modules:
                sample_size = max(1, int(len(layer_modules) * 0.3))  # 至少解冻一个模块
                selected_modules = random.sample(layer_modules, sample_size)
                
                for name, module in selected_modules:
                    for param_name, param in module.named_parameters():
                        param.requires_grad = True
                        partial_count += 1
                    module.train()
                    LOGGER.info(f"部分解冻层: {name}")
        
        # 处理BN层 - 将所有相关BN层设为训练模式
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # 检查是否与解冻层相关
                is_related = False
                for layer in unfreeze_layers + partial_layers:
                    if f"model.{layer}." in name:
                        is_related = True
                        break
                
                if is_related:
                    module.train()
                    module.track_running_stats = True
                    for param_name, param in module.named_parameters():
                        param.requires_grad = True
                    unfrozen_bn_layers += 1
        
        # 使用小的BN动量
        def set_low_bn_momentum(m):
            if isinstance(m, nn.BatchNorm2d) and m.training:
                m.momentum = 0.01
        
        self.model.apply(set_low_bn_momentum)
        
        # 统计和报告
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        LOGGER.info(f"解冻 {unfrozen_layers} 个常规层和 {unfrozen_bn_layers} 个BN层")
        LOGGER.info(f"部分解冻 {partial_count} 个卷积层")
        LOGGER.info(f"可训练参数占比: {trainable_params/total_params:.2%}")
        LOGGER.info(f"任务损失保留比例: box={self.model.args.box}, pose={self.model.args.pose}")

    def _unfreeze_all_layers(self):
        """解冻所有层，用于最终阶段"""
        unfrozen_count = 0
        
        # 解冻所有参数
        for name, param in self.model.named_parameters():
            param.requires_grad = True
            unfrozen_count += 1
        
        # 设置所有模块为训练模式
        self.model.train()
        
        # 启用所有BN层的统计更新
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()
                module.track_running_stats = True
                # 使用较小的动量增强稳定性
                module.momentum = 0.01
        
        LOGGER.info(f"全模型优化: 解冻所有 {unfrozen_count} 个参数")    

    def distill_on_epoch_end(self, trainer):
        self.distill_loss_instance.remove_handle_()
        LOGGER.debug(f"Removed distillation hooks at epoch {self.epoch} end")
        LOGGER.info(f"self.lr: {self.lr}")

    def distill_on_val_start(self, validator):
        self.distill_loss_instance.remove_handle_()
        LOGGER.debug("Ensuring distillation hooks are removed for validation")

    def _get_performance_metric(self, validator):
        """安全地获取性能指标"""
        try:
            # 记录验证器和度量对象信息
            LOGGER.debug(f"验证器类型: {type(validator)}")
            
            # 尝试多种方式获取性能指标
            if hasattr(validator, 'metrics'):
                LOGGER.debug(f"度量对象类型: {type(validator.metrics)}")
                
                # 方法1: 从pose对象获取
                if hasattr(validator.metrics, 'pose'):
                    if hasattr(validator.metrics.pose, 'mp'):
                        mp_value = validator.metrics.pose.mp
                        if hasattr(mp_value, 'mean'):
                            return float(mp_value.mean())
                        elif isinstance(mp_value, (float, int)):
                            return float(mp_value)
                
                # 方法2: 从fitness属性获取
                if hasattr(validator.metrics, 'fitness'):
                    return float(validator.metrics.fitness)
                
                # 方法3: 尝试从results_dict获取
                if hasattr(validator.metrics, 'results_dict'):
                    results = validator.metrics.results_dict
                    if 'metrics/mAP50-95(P)' in results:
                        return results['metrics/mAP50-95(P)']
            
            # 如果找不到任何指标，记录警告
            LOGGER.warning("无法找到有效的性能指标")
            return 0.01
            
        except Exception as e:
            LOGGER.error(f"获取性能指标时出错: {e}")
            return 0.01

    def distill_on_val_end(self, validator):
        """验证结束回调，用于监控性能并调整蒸馏强度"""
        try:
            # 获取当前姿态性能
            current_performance = self._get_performance_metric(validator)
            LOGGER.info(f"当前性能指标: {current_performance:.4f}")
            
            # 监控性能变化
            if hasattr(self, 'best_performance'):
                if current_performance < 0.3 * self.best_performance:
                    # 性能下降超过70%，紧急干预
                    LOGGER.warning(f"性能严重下降! 从 {self.best_performance:.4f} 到 {current_performance:.4f}")
                    LOGGER.warning("紧急调整训练参数...")
                    
                    # 增强任务损失权重
                    self.model.args.pose = min(self.model.args.pose * 2.0, 1.0)  # 加倍姿态损失，最大为1.0
                    self.model.args.box = min(self.model.args.box * 1.5, 1.0)    # 增加边界框损失，最大为1.0
                    
                    # 降低蒸馏权重
                    self.model.args.distill = max(self.model.args.distill * 0.3, 0.05)  # 降低蒸馏权重，最小为0.05
                    
                    # 降低学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.5
                        
                    LOGGER.warning(f"调整后参数: pose={self.model.args.pose:.2f}, "
                                f"distill={self.model.args.distill:.2f}, "
                                f"lr={param_group['lr']:.6f}")
            else:
                # 首次验证，记录基准性能
                self.best_performance = max(current_performance, 0.01)  # 确保至少有0.01作为基准
                LOGGER.info(f"设置基准性能: {self.best_performance:.4f}")
        
        except Exception as e:
            LOGGER.error(f"在验证结束回调中出错: {e}")


    def distill_on_train_end(self, trainer):
        self.distill_loss_instance.remove_handle_()
        LOGGER.info("Cleaned up distillation resources at training end")

    def distill_teardown(self, trainer):
        self.distill_loss_instance.remove_handle_()
        self.distill_loss_instance = None
        LOGGER.debug("Final cleanup of distillation resources during teardown")

    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        Get pose estimation model with specified configuration and weights.

        Args:
            cfg (str | Path | dict | None): Model configuration file path or dictionary.
            weights (str | Path | None): Path to the model weights file.
            verbose (bool): Whether to display model information.

        Returns:
            (PoseModel): Initialized pose estimation model.
        """
        model = PoseModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose)
        if weights:
            model.load(weights)

        return model
    
    def preprocess_batch(self, batch):
        """預處理批次數據並添加蒸餾相關信息"""
        batch = super().preprocess_batch(batch)
        
        # 添加蒸餾相關信息，僅在啟用蒸餾時添加
        if hasattr(self, 'distill_loss_instance') and self.distill_loss_instance is not None:
            batch["distill_instance"] = self.distill_loss_instance
            LOGGER.debug(f"Added distillation instance to batch for epoch {self.epoch}")
        
        return batch

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "d_loss"
        return yolo.pose.PoseValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """
        Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints.

        Args:
            batch (dict): Dictionary containing batch data with the following keys:
                - img (torch.Tensor): Batch of images
                - keypoints (torch.Tensor): Keypoints coordinates for pose estimation
                - cls (torch.Tensor): Class labels
                - bboxes (torch.Tensor): Bounding box coordinates
                - im_file (list): List of image file paths
                - batch_idx (torch.Tensor): Batch indices for each instance
            ni (int): Current training iteration number used for filename

        The function saves the plotted batch as an image in the trainer's save directory with the filename
        'train_batch{ni}.jpg', where ni is the iteration number.
        """
        images = batch["img"]
        kpts = batch["keypoints"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]
        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png
