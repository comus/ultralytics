# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import random
from copy import copy

import torch.nn as nn
import torch

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
        self.distillation_loss = overrides.get("distillation_loss", None)
        self.distillation_layers = overrides.get("distillation_layers", None)
        self.distill_loss_instance = None
        self.pure_distill = overrides.get("pure_distill", False)

        # 蒸餾方法映射字典，方便將簡短名稱映射到完整實現
        self.distillation_methods = {
            "cwd": "cwd",  # Channel-wise Distillation
            "mgd": "mgd",  # Masked Generative Distillation
            "rev": "rev",  # Review KD
            "fgd": "fgd",  # Feature Guided Distillation
            "pkd": "pkd",  # Probabilistic Knowledge Distillation
            "kd": "cwd",   # 默認使用 CWD
            "review": "rev",
            "feature": "fgd",
            "enhancedfgd": "enhancedfgd"  # 增強版特徵引導蒸餾
        }
        
        # 標準化蒸餾方法名稱
        if self.distillation_loss:
            self.distillation_loss = self.distillation_methods.get(
                self.distillation_loss.lower(), self.distillation_loss
            )
        
        # 如果有教师模型和蒸馏方法，添加对应的回调
        if self.teacher is not None and self.distillation_loss is not None:
            # 创建自定义回调列表
            if _callbacks is None:
                _callbacks = callbacks.get_default_callbacks()
                
            # 添加蒸馏相关回调到各事件中
            _callbacks["on_train_start"].append(self.distill_on_train_start)
            _callbacks["on_train_epoch_start"].append(self.distill_on_epoch_start)
            _callbacks["on_train_epoch_end"].append(self.distill_on_epoch_end)
            _callbacks["on_val_start"].append(self.distill_on_val_start)
            _callbacks["on_val_end"].append(self.distill_on_val_end)
            _callbacks["on_train_end"].append(self.distill_on_train_end)
            _callbacks["teardown"].append(self.distill_teardown)
            
            # 添加极限学习率调整和高级数据增强回调
            _callbacks["on_train_epoch_start"].append(self.extreme_adaptive_lr_callback)
            _callbacks["on_train_epoch_start"].append(self.advanced_augmentation_callback)
            _callbacks["on_train_epoch_start"].append(self.optimizer_config_callback)
            
            # 添加训练进度监控回调
            _callbacks["on_fit_epoch_end"].append(self.training_progress_callback)
        
        # 調用父類的初始化方法
        super().__init__(cfg, overrides, _callbacks)

        # 初始化教師模型（如果存在）
        if self.teacher is not None:
            # 凍結教師模型參數
            for k, v in self.teacher.named_parameters():
                v.requires_grad = False
            self.teacher = self.teacher.to(self.device)
            self.teacher.eval()
            LOGGER.info(f"Using {self.distillation_loss} distillation with teacher model")

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def extreme_adaptive_lr_callback(self, trainer):
        """極限版自適應學習率調整策略，用於顯著突破性能瓶頸"""
        # 在訓練的不同階段實施更激進策略
        LOGGER.info(f"執行學習率調整 - 當前 epoch: {trainer.epoch}")
        
        if trainer.epoch == 0:
            # 初始階段使用較高學習率
            target_lr = 0.00007
            LOGGER.info(f"Initial phase: setting higher learning rate to {target_lr}")
        elif trainer.epoch == 3:
            # 第一次大幅提高學習率突破平台
            target_lr = 0.0002
            LOGGER.info(f"First major learning rate boost to {target_lr}")
        elif trainer.epoch == 5:
            # 第二次提高學習率突破下一個平台
            target_lr = 0.00025
            LOGGER.info(f"Second learning rate boost to {target_lr}")
        elif trainer.epoch == 8:
            # 恢復到中等學習率進行優化
            target_lr = 0.00005
            LOGGER.info(f"Restoring to medium learning rate: {target_lr}")
        elif trainer.epoch == 12:
            # 適度降低學習率
            target_lr = 0.00002
            LOGGER.info(f"Moderate tuning phase: reduced learning rate to {target_lr}")
        elif trainer.epoch == 15:
            # 降低學習率進行精細優化階段
            target_lr = 0.000005
            LOGGER.info(f"Fine-tuning phase: low learning rate of {target_lr}")
        elif trainer.epoch == 20:
            # 極限精細優化階段
            target_lr = 0.000001
            LOGGER.info(f"Ultra-fine tuning phase: minimal learning rate of {target_lr}")
        else:
            # 其他epoch不更改學習率，保持原始設置
            return
            
        # 顯示當前學習率
        for i, g in enumerate(trainer.optimizer.param_groups):
            g['lr'] = target_lr
            LOGGER.info(f"Epoch {trainer.epoch}: Group {i} learning rate = {g['lr']:.8f}")
            
    def advanced_augmentation_callback(self, trainer):
        """根據訓練階段高度靈活地調整增強策略"""
        if trainer.epoch == 0:
            # 初始階段：使用適中增強
            LOGGER.info("Initial phase: moderate augmentation")
            if hasattr(trainer.train_loader.dataset, 'hsv_values'):
                trainer.train_loader.dataset.hsv_values = [0.15, 0.15, 0.15]  # 適中HSV變化
            if hasattr(trainer.train_loader.dataset, 'mosaic'):
                trainer.train_loader.dataset.mosaic = False   # 初始就禁用馬賽克
            if hasattr(trainer.train_loader.dataset, 'mixup'):
                trainer.train_loader.dataset.mixup = False    # 禁用mixup
                
        elif trainer.epoch == 3:
            # 大幅提高學習率的同時增加增強強度
            LOGGER.info("Boosting phase: stronger augmentation")
            if hasattr(trainer.train_loader.dataset, 'hsv_values'):
                trainer.train_loader.dataset.hsv_values = [0.25, 0.25, 0.2]  # 較強HSV變化
            if hasattr(trainer.train_loader.dataset, 'translate'):
                trainer.train_loader.dataset.translate = 0.15  # 輕微平移
            if hasattr(trainer.train_loader.dataset, 'scale'):
                trainer.train_loader.dataset.scale = 0.2  # 增加縮放
            
        elif trainer.epoch == 8:
            # 學習率下降時減少增強強度
            LOGGER.info("Mid phase: moderate augmentation")
            if hasattr(trainer.train_loader.dataset, 'hsv_values'):
                trainer.train_loader.dataset.hsv_values = [0.15, 0.15, 0.1]  # 適中HSV變化
            if hasattr(trainer.train_loader.dataset, 'translate'):
                trainer.train_loader.dataset.translate = 0.1  # 輕微平移
            if hasattr(trainer.train_loader.dataset, 'scale'):
                trainer.train_loader.dataset.scale = 0.1  # 減少縮放
                
        elif trainer.epoch == 15:
            # 後期：減少增強專注精細優化
            LOGGER.info("Late phase: minimal augmentation for fine-tuning")
            if hasattr(trainer.train_loader.dataset, 'hsv_values'):
                trainer.train_loader.dataset.hsv_values = [0.05, 0.05, 0.05]  # 輕微HSV
            if hasattr(trainer.train_loader.dataset, 'translate'):
                trainer.train_loader.dataset.translate = 0.0  # 禁用平移
            if hasattr(trainer.train_loader.dataset, 'scale'):
                trainer.train_loader.dataset.scale = 0.0  # 禁用縮放
            if hasattr(trainer.train_loader.dataset, 'fliplr'):
                trainer.train_loader.dataset.fliplr = 0.3     # 只保留少量水平翻轉
            
        elif trainer.epoch == 20:
            # 最終階段：完全禁用所有增強以實現極致精度
            LOGGER.info("Final phase: zero augmentation for ultimate precision")
            if hasattr(trainer.train_loader.dataset, 'hsv_values'):
                trainer.train_loader.dataset.hsv_values = [0.0, 0.0, 0.0]  # 禁用HSV
            if hasattr(trainer.train_loader.dataset, 'mosaic'):
                trainer.train_loader.dataset.mosaic = False   # 確保禁用馬賽克
            if hasattr(trainer.train_loader.dataset, 'mixup'):
                trainer.train_loader.dataset.mixup = False    # 確保禁用mixup
            if hasattr(trainer.train_loader.dataset, 'fliplr'):
                trainer.train_loader.dataset.fliplr = 0.0     # 禁用水平翻轉
                
    def training_progress_callback(self, trainer):
        """增強版訓練進度監控，設定更高的目標"""
        if not hasattr(trainer, 'metrics') or trainer.metrics is None:
            return
        
        metrics = trainer.metrics
        if "pose_map50-95" in metrics:
            current_map = metrics.get("pose_map50-95", 0)
            LOGGER.info(f"Epoch {trainer.epoch}: Pose mAP50-95 = {current_map:.6f}")
            
            # 突破官方記錄提示，設定更高目標
            if current_map > 0.505:
                LOGGER.info(f"🚀 突破官方記錄！當前mAP: {current_map:.6f} > 0.505")
                
            if current_map > 0.510:
                LOGGER.info(f"🔥 明顯超越官方記錄！當前mAP: {current_map:.6f} > 0.510")
                
            if current_map > 0.515:
                LOGGER.info(f"💯 大幅超越官方記錄！當前mAP: {current_map:.6f} > 0.515")
                
            if current_map > 0.520:
                LOGGER.info(f"🏆 極限突破！當前mAP: {current_map:.6f} > 0.520")
                
    def optimizer_config_callback(self, trainer):
        """優化優化器參數以實現更好的收斂和突破"""
        # 只在第一個epoch設置
        if trainer.epoch == 0:
            for g in trainer.optimizer.param_groups:
                # 對於AdamW優化器，設置beta值
                if 'betas' in g:
                    g['betas'] = (0.937, 0.999)  # 優化的beta值

    def distill_on_train_start(self, trainer):
        """訓練開始時初始化蒸餾損失實例和凍結非目標層"""
        if self.teacher is not None and self.distillation_loss is not None:
            # 初始化蒸餾損失實例，支持增強版FGD
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=self.distillation_loss,
                layers=self.distillation_layers
            )

            # 在純蒸餾模式下進行優化參數選擇
            if self.pure_distill:
                # 獲取需要蒸餾的層ID
                target_layers = self.distillation_layers
                
                # 預設凍結所有層
                for name, param in self.model.named_parameters():
                    param.requires_grad = False
                
                # 只解凍目標層的cv2.conv參數
                unfrozen_count = 0
                unfrozen_names = []
                for name, param in self.model.named_parameters():
                    if "model." in name and any(f".{layer}." in name for layer in target_layers):
                        if ".cv2.conv" in name:  # 精確匹配cv2的卷積層參數
                            param.requires_grad = True
                            unfrozen_count += 1
                            unfrozen_names.append(name)
                
                # 計算可訓練參數比例
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                
                LOGGER.info(f"純蒸餾模式：只優化層 {target_layers} 中的 cv2.conv 參數")
                LOGGER.info(f"解凍了 {unfrozen_count} 個參數組，可訓練參數比例: {trainable_params/total_params:.2%}")

                # 記錄凍結狀態的更詳細資訊
                LOGGER.info("--------- 參數凍結狀態總結 ---------")
                LOGGER.info("以下參數將被訓練 (requires_grad=True):")
                for name in unfrozen_names:
                    LOGGER.info(f"  - {name}")
                
                # 凍結所有BN層並記錄
                bn_layer_names = []
                for name, m in self.model.named_modules():
                    if isinstance(m, torch.nn.BatchNorm2d):
                        m.eval()  # 設置為評估模式
                        m.track_running_stats = False  # 停止更新統計量
                        bn_layer_names.append(name)
                
                LOGGER.info(f"\n已凍結 {len(bn_layer_names)} 個 BN 層，這些層不會更新統計量:")
                # 顯示部分BN層名稱作為示例
                for i, name in enumerate(bn_layer_names):
                    if i < 10 or i >= len(bn_layer_names) - 5:  # 顯示前10個和最後5個
                        LOGGER.info(f"  - {name}")
                    elif i == 10:
                        LOGGER.info(f"  ... (省略 {len(bn_layer_names) - 15} 個 BN 層) ...")
                
                LOGGER.info("純蒸餾模式: 所有BN層已凍結，不再更新統計量")
                LOGGER.info("------------------------------------")

    def distill_on_epoch_start(self, trainer):
        """每個 epoch 開始時註冊鉤子"""
        if hasattr(self, 'distill_loss_instance') and self.distill_loss_instance is not None:
            self.distill_loss_instance.register_hook()
            # 添加這行來測試鉤子是否正確註冊
            for i, h in enumerate(self.distill_loss_instance.remove_handle):
                LOGGER.info(f"鉤子 {i+1} 已註冊: {h}")

    def distill_on_epoch_end(self, trainer):
        """每個 epoch 結束時取消鉤子"""
        if hasattr(self, 'distill_loss_instance') and self.distill_loss_instance is not None:
            self.distill_loss_instance.remove_handle_()
            LOGGER.debug(f"Removed distillation hooks at epoch {self.epoch} end")

    def distill_on_val_start(self, validator):
        """驗證開始時確保鉤子被移除"""
        if hasattr(self, 'distill_loss_instance') and self.distill_loss_instance is not None:
            # 確保在驗證過程中沒有鉤子
            self.distill_loss_instance.remove_handle_()
            LOGGER.debug("Ensuring distillation hooks are removed for validation")

    def distill_on_val_end(self, validator):
        """驗證結束後不需要做任何事，因為每個 epoch 開始時會重新註冊鉤子"""
        pass

    def distill_on_train_end(self, trainer):
        """訓練結束時清理資源"""
        if hasattr(self, 'distill_loss_instance') and self.distill_loss_instance is not None:
            self.distill_loss_instance.remove_handle_()
            LOGGER.info("Cleaned up distillation resources at training end")

    def distill_teardown(self, trainer):
        """最終清理，確保鉤子被移除"""
        if hasattr(self, 'distill_loss_instance') and self.distill_loss_instance is not None:
            try:
                self.distill_loss_instance.remove_handle_()
            except Exception as e:
                LOGGER.warning(f"Error removing hooks during teardown: {e}")
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

        # 添加純蒸餾標誌
        if hasattr(self, 'pure_distill') and self.pure_distill:
            batch["pure_distill"] = True
        
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
