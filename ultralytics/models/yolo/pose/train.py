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
            self.model.args.distill = 0.0

            # 初始化蒸餾損失實例
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=distillation_loss,
                layers=distillation_layers,
            )

            # # 輸出模型所有層的名字
            # for name, param in self.model.named_parameters():
            #     LOGGER.info(f"模型層名: {name}")
            
            # 預設凍結所有層
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            # 凍結所有BN層並記錄
            # bn_layer_names = []
            # for name, m in self.model.named_modules():
            #     if isinstance(m, torch.nn.BatchNorm2d):
            #         m.eval()  # 設置為評估模式
            #         m.track_running_stats = False  # 停止更新統計量
            #         bn_layer_names.append(name)
            
            # LOGGER.info(f"已凍結 {len(bn_layer_names)} 個 BN 層，這些層不會更新統計量")

        if self.epoch == 1:
            LOGGER.info("階段2: 解凍蒸餾層及相關BN層")

            distillation_loss = "enhancedfgd"
            distillation_layers = ["22"]
            self.model.args.distill = 0.0  # 從較小權重開始

            self.distill_loss_instance.remove_handle_()
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=distillation_loss,
                layers=distillation_layers,
            )

            # 預設凍結所有層
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            # 跟踪已解凍的模塊
            unfrozen_modules = []
            bn_modules = []
            
            # 1. 首先找到所有需要解凍的卷積模塊
            for name, module in self.model.named_modules():
                if name is not None:
                    name_parts = name.split(".")
                    
                    if name_parts[0] != "model":
                        continue
                    if len(name_parts) >= 3:
                        if name_parts[1] in distillation_layers:
                            if "cv2" in name_parts[2]:
                                unfrozen_modules.append(name)
                                
                                # 解凍該模塊的所有參數
                                for param_name, param in module.named_parameters():
                                    param.requires_grad = True
                                
                                # 設置為訓練模式
                                module.train()
                                LOGGER.info(f"解凍卷積模塊: {name}")
            
            # 2. 識別並解凍相關的BN層
            # 遍歷所有BN層
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # 檢查此BN層是否與解凍的卷積層相關聯
                    is_related = False
                    for unfrozen in unfrozen_modules:
                        # 例：如果卷積層是"model.22.cv2"，相關的BN層可能是"model.22.bn2"
                        prefix = unfrozen.rsplit(".", 1)[0]  # 獲取最後一個'.'之前的部分
                        if name.startswith(prefix):
                            is_related = True
                            break
                    
                    # 對於block結構，可能需要檢查更複雜的關係
                    # 例：如果cv2在C3模塊中，則需要解凍其後的BN層
                    # 這需要根據具體模型結構調整
                    for layer_num in distillation_layers:
                        if f"model.{layer_num}." in name:
                            # 檢查是否為cv2後的BN層或相關層
                            if ".cv2.bn" in name or ".bn2" in name:
                                is_related = True
                                break
                    
                    if is_related:
                        # 解凍BN層
                        bn_modules.append(name)
                        
                        # 設置BN層為訓練模式
                        module.train()
                        
                        # 解凍BN層的所有參數
                        for param_name, param in module.named_parameters():
                            param.requires_grad = True
                        
                        # 確保運行統計數據會被更新
                        module.track_running_stats = True
                        
                        LOGGER.info(f"解凍BN層: {name}")
            
            # 3. 確保BN層在評估時使用訓練期間收集的統計數據
            def set_bn_train(m):
                if isinstance(m, nn.BatchNorm2d):
                    if m.training:
                        # 僅對訓練模式的BN層執行
                        # 設置動量值較小可以更快地適應新統計數據
                        m.momentum = 0.01  # 降低動量使統計量更穩定
            
            # 應用到整個模型
            self.model.apply(set_bn_train)
            
            LOGGER.info(f"總共解凍 {len(unfrozen_modules)} 個卷積模塊和 {len(bn_modules)} 個BN層")
            
            # 4. 降低學習率以穩定訓練
            for param_group in self.optimizer.param_groups:
                original_lr = param_group['lr']
                param_group['lr'] = original_lr * 0.2  # 降低到原來的20%
                LOGGER.info(f"學習率從 {original_lr:.6f} 降低到 {param_group['lr']:.6f}")

            # for name, param in self.model.named_parameters():
            #     if "model." in name and any(f".{layer}." in name for layer in distillation_layers):
            #         # 記錄下蒸餾層的前綴，例如 "model.1" 或 "model.2"
            #         for layer in distillation_layers:
            #             if f".{layer}." in name:
            #                 layer_prefix = name.split(f".{layer}.")[0] + f".{layer}"
            #                 if layer_prefix not in distill_layer_prefixes:
            #                     distill_layer_prefixes.append(layer_prefix)
                            
            #         if ".cv2.conv" in name:  # 精確匹配cv2的卷積層參數
            #             param.requires_grad = True
            #             unfrozen_count += 1
            #             unfrozen_names.append(name)

            # # 處理BN層：蒸餾層的BN保持訓練模式，其他BN設為評估模式
            # frozen_bn_count = 0
            # active_bn_count = 0
            # for name, m in self.model.named_modules():
            #     if isinstance(m, torch.nn.BatchNorm2d):
            #         # 檢查此BN層是否屬於蒸餾層
            #         is_in_distill_layer = any(prefix in name for prefix in distill_layer_prefixes)
                    
            #         if is_in_distill_layer:
            #             # 蒸餾層的BN保持訓練模式
            #             m.train()
            #             m.track_running_stats = True
            #             active_bn_count += 1
            #         else:
            #             # 非蒸餾層的BN設為評估模式
            #             m.eval()
            #             m.track_running_stats = False
            #             frozen_bn_count += 1

            # # 凍結所有BN層並記錄
            # bn_layer_names = []
            # for name, m in self.model.named_modules():
            #     if isinstance(m, torch.nn.BatchNorm2d):
            #         m.eval()  # 設置為評估模式
            #         m.track_running_stats = False  # 停止更新統計量
            #         bn_layer_names.append(name)
            
            # LOGGER.info(f"已凍結 {len(bn_layer_names)} 個 BN 層，這些層不會更新統計量")

            # # 計算可訓練參數比例
            # total_params = sum(p.numel() for p in self.model.parameters())
            # trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # LOGGER.info(f"純蒸餾模式：只優化層 {distillation_layers} 中的 cv2.conv 參數")
            # LOGGER.info(f"解凍了 {unfrozen_count} 個參數組，可訓練參數比例: {trainable_params/total_params:.2%}")
            # # LOGGER.info(f"保持 {active_bn_count} 個蒸餾層的BN處於訓練模式，凍結了 {frozen_bn_count} 個非蒸餾層的BN")

            # # 記錄凍結狀態的更詳細資訊
            # LOGGER.info("--------- 參數凍結狀態總結 ---------")
            # LOGGER.info("以下參數將被訓練 (requires_grad=True):")
            # for name in unfrozen_names:
            #     LOGGER.info(f"  - {name}")
            # LOGGER.info("以下層的BN保持訓練模式:")
            # for prefix in distill_layer_prefixes:
            #     LOGGER.info(f"  - {prefix}")


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
