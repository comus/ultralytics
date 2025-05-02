# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy
import os

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks
from ultralytics.utils.plotting import plot_images, plot_results
import torch
from ultralytics import YOLO
import torch.distributed as dist


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

        self.teacher_path = overrides.get("teacher", None)  # Store the teacher path instead of model
        self.teacher = None  # Initialized to None, will load the model later
        self.distill = overrides.get("distill", 1.0)
        self.freezeAllBN = overrides.get("freezeAllBN", False)
        self.target_layers = overrides.get("target_layers", [])
        
        # For collecting features from layers
        self.teacher_features = {}
        self.student_features = {}
        self.teacher_hooks = []
        self.student_hooks = []

        # Initialize parent class first to set up device and other attributes
        super().__init__(cfg, overrides, _callbacks)

        # Now we can initialize the teacher model since self.device is available
        if self.teacher_path is not None:
            self.init_teacher_model()
            
            if _callbacks is None:
                _callbacks = callbacks.get_default_callbacks()

            _callbacks["on_train_start"].append(self.on_train_start)
            _callbacks["on_train_epoch_start"].append(self.on_epoch_start)
            _callbacks["on_train_epoch_end"].append(self.on_epoch_end)
            _callbacks["on_val_start"].append(self.on_val_start)
            _callbacks["on_val_end"].append(self.on_val_end)
            _callbacks["on_train_end"].append(self.on_train_end)
            _callbacks["teardown"].append(self.teardown)
            _callbacks["on_batch_end"].append(self.on_batch_end)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def init_teacher_model(self):
        """Initialize the teacher model on the current device."""
        if self.teacher_path is not None and self.teacher is None:
            # Get rank for distributed training
            rank = dist.get_rank() if dist.is_initialized() else 0
            gpu_id = self.device.index if hasattr(self.device, 'index') else 0
            
            # Log detailed information about which GPU is loading the teacher
            log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
            LOGGER.info(f"{log_prefix}Loading teacher model from {self.teacher_path}")
            
            # Load teacher model
            try:
                self.teacher = YOLO(self.teacher_path).model.to(self.device)
                
                # Record memory usage before and after loading teacher model
                if hasattr(torch.cuda, 'memory_allocated'):
                    mem_before = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
                
                # Freeze teacher parameters
                for k, v in self.teacher.named_parameters():
                    v.requires_grad = False

                # Set teacher model to eval mode
                self.teacher.eval()

                # Freeze BN layers
                for m in self.teacher.modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                        m.eval()  # Only set BN layers to eval mode
                        for param in m.parameters():
                            param.requires_grad = False
                
                if hasattr(torch.cuda, 'memory_allocated'):
                    mem_after = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
                    mem_used = mem_after - mem_before
                    LOGGER.info(f"{log_prefix}Teacher model loaded successfully. Memory used: {mem_used:.2f} MB")
                else:
                    LOGGER.info(f"{log_prefix}Teacher model loaded successfully.")
                
                # Log teacher model structure details
                teacher_params = sum(p.numel() for p in self.teacher.parameters())
                LOGGER.info(f"{log_prefix}Teacher model has {teacher_params:,} parameters")
                
                # Log process ID for debugging
                LOGGER.info(f"{log_prefix}Process ID: {os.getpid()}")
                
            except Exception as e:
                LOGGER.error(f"{log_prefix}Error loading teacher model: {str(e)}")
                raise
            
    def register_teacher_hooks(self):
        """Register hooks on the teacher model to capture intermediate features."""
        if self.teacher is not None:
            # Clear any existing hooks
            for hook in self.teacher_hooks:
                hook.remove()
            self.teacher_hooks = []
            
            # Get rank for distributed training
            rank = dist.get_rank() if dist.is_initialized() else 0
            gpu_id = self.device.index if hasattr(self.device, 'index') else 0
            log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
            
            LOGGER.info(f"{log_prefix}Registering hooks for teacher model:")
            
            # 建立模塊名稱到模塊的映射
            module_dict = {}
            for name, module in self.teacher.named_modules():
                module_dict[name] = module
                
            # 處理每個目標層
            for target in self.target_layers:
                if isinstance(target, int):
                    # 如果是整數索引，直接獲取對應層
                    layer = self.teacher.model[target]
                    layer_full_name = f"model.{target}"
                    layer_idx = target  # 用於hook的layer_idx
                else:
                    # 如果是字符串路徑，從module_dict中查找
                    if target in module_dict:
                        layer = module_dict[target]
                        layer_full_name = target
                        # 對於字符串路徑，我們使用一個唯一標識作為layer_idx
                        layer_idx = target
                    else:
                        LOGGER.warning(f"{log_prefix}在教師模型中未找到指定層: {target}")
                        continue
                
                # 獲取層的類型
                layer_type = layer.__class__.__name__
                
                # 構建詳細的層信息
                layer_info = f"層名稱: {layer_full_name}, 類型: {layer_type}"
                
                # 直接檢查該層是否有conv屬性
                if hasattr(layer, 'conv'):
                    layer_info += f", 通道數: {layer.conv.out_channels}"
                else:
                    # 動態查找所有子模塊中的conv
                    conv_modules = []
                    # 獲取層的所有模塊
                    for name, module in layer.named_modules():
                        if hasattr(module, 'conv') and name != '':  # 排除模塊本身
                            if layer_full_name == "model":  # 處理特殊情況
                                full_path = f"{layer_full_name}.{name}.conv"
                            else:
                                full_path = f"{layer_full_name}.{name}.conv" if name else f"{layer_full_name}.conv"
                            conv_info = f"{full_path}: {module.conv.out_channels}通道"
                            conv_modules.append(conv_info)
                    
                    if conv_modules:
                        layer_info += f"\n  子模塊包含:"
                        # 顯示所有conv模塊，但限制數量避免輸出過多
                        max_show = min(len(conv_modules), 5)
                        for j in range(max_show):
                            layer_info += f"\n    - {conv_modules[j]}"
                        if len(conv_modules) > max_show:
                            layer_info += f"\n    ... 等{len(conv_modules)}個conv模塊"
                
                LOGGER.info(f"{log_prefix}{layer_info}")
                
                # 註冊勾子，使用自定義的_save_feature方法並傳遞layer_idx
                self.teacher_hooks.append(layer.register_forward_hook(
                    lambda module, input, output, idx=layer_idx: self._save_teacher_feature(idx, output)))
            
            LOGGER.info(f"{log_prefix}教師模型勾子註冊完成，共 {len(self.teacher_hooks)} 個勾子")
            
    def register_student_hooks(self):
        """Register hooks on the student model to capture intermediate features."""
        # Clear any existing hooks
        for hook in self.student_hooks:
            hook.remove()
        self.student_hooks = []
        
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Registering hooks for student model:")
        
        # 建立模塊名稱到模塊的映射
        module_dict = {}
        for name, module in self.model.named_modules():
            module_dict[name] = module
            
        # 處理每個目標層
        for target in self.target_layers:
            if isinstance(target, int):
                # 如果是整數索引，直接獲取對應層
                layer = self.model.model[target]
                layer_full_name = f"model.{target}"
                layer_idx = target  # 用於hook的layer_idx
            else:
                # 如果是字符串路徑，從module_dict中查找
                if target in module_dict:
                    layer = module_dict[target]
                    layer_full_name = target
                    # 對於字符串路徑，我們使用一個唯一標識作為layer_idx
                    layer_idx = target
                else:
                    LOGGER.warning(f"{log_prefix}在學生模型中未找到指定層: {target}")
                    continue
            
            # 獲取層的類型
            layer_type = layer.__class__.__name__
            
            # 構建詳細的層信息
            layer_info = f"層名稱: {layer_full_name}, 類型: {layer_type}"
            
            # 直接檢查該層是否有conv屬性
            if hasattr(layer, 'conv'):
                layer_info += f", 通道數: {layer.conv.out_channels}"
            else:
                # 動態查找所有子模塊中的conv
                conv_modules = []
                # 獲取層的所有模塊
                for name, module in layer.named_modules():
                    if hasattr(module, 'conv') and name != '':  # 排除模塊本身
                        if layer_full_name == "model":  # 處理特殊情況
                            full_path = f"{layer_full_name}.{name}.conv"
                        else:
                            full_path = f"{layer_full_name}.{name}.conv" if name else f"{layer_full_name}.conv"
                        conv_info = f"{full_path}: {module.conv.out_channels}通道"
                        conv_modules.append(conv_info)
                
                if conv_modules:
                    layer_info += f"\n  子模塊包含:"
                    # 顯示所有conv模塊，但限制數量避免輸出過多
                    max_show = min(len(conv_modules), 5)
                    for j in range(max_show):
                        layer_info += f"\n    - {conv_modules[j]}"
                    if len(conv_modules) > max_show:
                        layer_info += f"\n    ... 等{len(conv_modules)}個conv模塊"
            
            LOGGER.info(f"{log_prefix}{layer_info}")
            
            # 註冊勾子
            self.student_hooks.append(layer.register_forward_hook(
                lambda module, input, output, idx=layer_idx: self._save_student_feature(idx, output)))
                
        LOGGER.info(f"{log_prefix}學生模型勾子註冊完成，共 {len(self.student_hooks)} 個勾子")

    def _save_teacher_feature(self, layer_idx, feature):
        """Save features from the teacher model."""
        self.teacher_features[layer_idx] = feature
        
    def _save_student_feature(self, layer_idx, feature):
        """Save features from the student model."""
        self.student_features[layer_idx] = feature

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        # 凍結BN層，讓它們的統計數據(running_mean, running_var)不會更新
        if self.freezeAllBN:
            for m in self.model.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.eval()  # 只有BN層設為評估模式
                    for param in m.parameters():
                        param.requires_grad = False

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        # Add teacher to batch if it exists
        if self.teacher is not None:
            batch["teacher"] = self.teacher
                
            # Store features in the batch
            batch["teacher_features"] = self.teacher_features
            batch["student_features"] = self.student_features

        return batch

    def on_train_start(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Starting training...")
        
        if self.teacher is not None:
            # 打印教師模型和學生模型的結構
            LOGGER.info(f"{log_prefix}" + "=" * 80)
            LOGGER.info(f"{log_prefix}教師模型結構:")
            for name, module in self.teacher.named_modules():
                if name.startswith("model.") and len(name.split(".")) <= 3:
                    module_type = module.__class__.__name__
                    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    has_conv = hasattr(module, 'conv')
                    channels_info = f", 通道數: {module.conv.out_channels}" if has_conv else ""
                    LOGGER.info(f"{log_prefix}  - {name}: {module_type} (參數量: {num_params}){channels_info}")
            
            LOGGER.info(f"\n{log_prefix}學生模型結構:")
            for name, module in self.model.named_modules():
                if name.startswith("model.") and len(name.split(".")) <= 3:
                    module_type = module.__class__.__name__
                    num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    has_conv = hasattr(module, 'conv')
                    channels_info = f", 通道數: {module.conv.out_channels}" if has_conv else ""
                    LOGGER.info(f"{log_prefix}  - {name}: {module_type} (參數量: {num_params}){channels_info}")
            LOGGER.info(f"{log_prefix}" + "=" * 80)

            # Register hooks for the teacher model
            self.register_teacher_hooks()
            # Register hooks for the student model
            self.register_student_hooks()

    def on_epoch_start(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Starting epoch {trainer.epoch}/{trainer.epochs}")
        
        self.model.epoch = trainer.epoch
        self.model.epochs = trainer.epochs
        self.model.is_first_batch_in_epoch = True

    def on_epoch_end(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Finished epoch {trainer.epoch}/{trainer.epochs}")

    def on_val_start(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Starting validation...")

    def on_val_end(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Validation completed")
    
    def on_train_end(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Training completed, cleaning up...")
        
        # Remove hooks when training ends
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()

        # Clear the stored features
        self.teacher_features = {}
        self.student_features = {}
        
        LOGGER.info(f"{log_prefix}Cleanup completed")
    
    def teardown(self, trainer):
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        LOGGER.info(f"{log_prefix}Teardown in progress...")
        
        # Make sure all hooks are removed
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()
            
        LOGGER.info(f"{log_prefix}Teardown completed")
    
    def on_batch_end(self, trainer):
        self.model.is_first_batch_in_epoch = False
        pass

    def set_target_layers(self, new_target_layers):
        """
        設置新的目標層並重新註冊勾子。
        
        Args:
            new_target_layers (list): 包含層索引或層名稱的列表，例如 [0, 6, 13] 或 ["model.0.conv", "model.6.cv1"]
        """
        # Get rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0
        gpu_id = self.device.index if hasattr(self.device, 'index') else 0
        log_prefix = f"[Rank {rank}, GPU {gpu_id}] "
        
        # 更新目標層
        self.target_layers = new_target_layers
        LOGGER.info(f"{log_prefix}更新目標層為: {self.target_layers}")
        
        # 重新註冊勾子
        self.register_teacher_hooks()
        self.register_student_hooks()
        
        return self.target_layers

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
        model = PoseModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "d_loss"
        return yolo.pose.PoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
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
