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
            LOGGER.info("階段1: 純蒸餾")

            distillation_loss = "cwd"
            distillation_layers = ["22"]
            self.model.args.distill = 1.0

            # 初始化蒸餾損失實例
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=distillation_loss,
                layers=distillation_layers,
            )

        # 在訓練開始前檢查教師模型
        LOGGER.info("=" * 50)
        LOGGER.info("教師模型狀態檢查:")
        LOGGER.info(f"教師模型評估模式: {'開啟' if not self.teacher.training else '關閉'}")
        
        teacher_frozen_params = 0
        teacher_trainable_params = 0
        teacher_total_params = 0
        teacher_frozen_layers = []
        teacher_trainable_layers = []
        
        for name, param in self.teacher.named_parameters():
            teacher_total_params += param.numel()
            if param.requires_grad:
                teacher_trainable_params += param.numel()
                teacher_trainable_layers.append(name)
            else:
                teacher_frozen_params += param.numel()
                teacher_frozen_layers.append(name)
                
        LOGGER.info(f"教師模型參數統計:")
        LOGGER.info(f"總參數數量: {teacher_total_params:,}")
        LOGGER.info(f"凍結參數數量: {teacher_frozen_params:,} ({teacher_frozen_params/teacher_total_params:.2%})")
        LOGGER.info(f"可訓練參數數量: {teacher_trainable_params:,} ({teacher_trainable_params/teacher_total_params:.2%})")
        LOGGER.info(f"凍結層數: {len(teacher_frozen_layers)}")
        LOGGER.info(f"可訓練層數: {len(teacher_trainable_layers)}")
        
        # 如果有可訓練層，這是個警告
        if teacher_trainable_params > 0:
            LOGGER.warning(f"警告: 教師模型有 {len(teacher_trainable_layers)} 個可訓練層!")
            if len(teacher_trainable_layers) > 0:
                LOGGER.warning(f"可訓練層示例: {teacher_trainable_layers[:5]}")
        
        # 檢查教師模型的BN層狀態
        teacher_bn_train_count = 0
        teacher_bn_eval_count = 0
        
        for name, module in self.teacher.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if module.training:
                    teacher_bn_train_count += 1
                else:
                    teacher_bn_eval_count += 1
        
        LOGGER.info(f"教師模型BN層統計: 訓練模式 {teacher_bn_train_count} 個, 評估模式 {teacher_bn_eval_count} 個")
        if teacher_bn_train_count > 0:
            LOGGER.warning(f"警告: 教師模型有 {teacher_bn_train_count} 個BN層處於訓練模式!")
        LOGGER.info("=" * 50)

        # 檢查學生模型參數訓練狀態
        LOGGER.info("學生模型狀態檢查:")
        LOGGER.info(f"學生模型評估模式: {'關閉' if self.model.training else '開啟'}")
        
        total_params = 0
        trainable_params = 0
        frozen_layers = []
        trainable_layers = []

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                trainable_layers.append(name)
            else:
                frozen_layers.append(name)

        LOGGER.info(f"學生模型參數統計:")
        LOGGER.info(f"總參數數量: {total_params:,}")
        LOGGER.info(f"可訓練參數數量: {trainable_params:,} ({trainable_params/total_params:.2%})")
        LOGGER.info(f"凍結參數數量: {total_params-trainable_params:,} ({(total_params-trainable_params)/total_params:.2%})")
        LOGGER.info(f"可訓練層數: {len(trainable_layers)}")
        LOGGER.info(f"凍結層數: {len(frozen_layers)}")

        # 檢查BN層的訓練狀態
        bn_train_count = 0
        bn_eval_count = 0
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                if module.training:
                    bn_train_count += 1
                else:
                    bn_eval_count += 1
        
        LOGGER.info(f"BN層統計: 訓練模式 {bn_train_count} 個, 評估模式 {bn_eval_count} 個")
        
        # 可以選擇性地列印部分訓練層和凍結層的名稱（避免輸出過多）
        if len(trainable_layers) > 0:
            LOGGER.info(f"訓練層示例 (前5個): {trainable_layers[:5]}")
        if len(frozen_layers) > 0:
            LOGGER.info(f"凍結層示例 (前5個): {frozen_layers[:5]}")
        LOGGER.info("=" * 50)

        # 檢查模型中的隨機性來源
        LOGGER.info("檢查模型中的隨機性來源:")
        
        # 檢查教師模型
        teacher_dropout_layers = []
        for name, module in self.teacher.named_modules():
            if isinstance(module, nn.Dropout):
                teacher_dropout_layers.append(name)
        
        if teacher_dropout_layers:
            LOGGER.info(f"教師模型包含 {len(teacher_dropout_layers)} 個Dropout層:")
            for name in teacher_dropout_layers[:5]:  # 只顯示前5個
                LOGGER.info(f"  - {name}")
        else:
            LOGGER.info("教師模型不包含任何Dropout層")
        
        # 檢查學生模型的隨機性來源
        student_dropout_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Dropout):
                student_dropout_layers.append(name)
        
        if student_dropout_layers:
            LOGGER.info(f"學生模型包含 {len(student_dropout_layers)} 個Dropout層:")
            for name in student_dropout_layers[:5]:  # 只顯示前5個
                LOGGER.info(f"  - {name}")
        else:
            LOGGER.info("學生模型不包含任何Dropout層")

        # 註冊鉤子
        self.distill_loss_instance.register_hook()
        for i, h in enumerate(self.distill_loss_instance.remove_handle):
            LOGGER.info(f"鉤子 {i+1} 已註冊: {h}")
            

    def distill_on_epoch_end(self, trainer):
        self.distill_loss_instance.remove_handle_()
        LOGGER.debug(f"Removed distillation hooks at epoch {self.epoch} end")
        LOGGER.info(f"self.lr: {self.lr}")

    def distill_on_val_start(self, validator):
        self.distill_loss_instance.remove_handle_()
        LOGGER.debug("Ensuring distillation hooks are removed for validation")

    def distill_on_val_end(self, validator):
        pass

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
