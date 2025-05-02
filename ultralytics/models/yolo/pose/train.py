# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks
from ultralytics.utils.plotting import plot_images, plot_results
import torch
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

        # 保存teacher路徑，而不是直接加載模型
        self.teacher_path = overrides.get("teacher", None)
        self.distill = overrides.get("distill", 1.0)
        self.freezeAllBN = overrides.get("freezeAllBN", False)
        self.target_layers = overrides.get("target_layers", [])
        
        # For collecting features from layers
        self.teacher_features = {}
        self.student_features = {}
        self.teacher_hooks = []
        self.student_hooks = []

        # 先初始化基類，設置好設備環境
        super().__init__(cfg, overrides, _callbacks)

        # 在基類初始化後，確定正確的設備後加載teacher模型
        if self.teacher_path is not None:
            # 檢查是否在DDP環境中
            is_ddp = isinstance(self.model, torch.nn.parallel.DistributedDataParallel)
            using_ddp = is_ddp or torch.cuda.device_count() > 1
            
            # 獲取DDP中的本地rank（多GPU訓練中每個進程的設備ID）
            local_rank = getattr(self, 'rank', 0) % torch.cuda.device_count()
            
            # 記錄當前設備，方便調試
            current_device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
            LOGGER.info(f"在設備 {current_device} 上載入教師模型: {self.teacher_path} (DDP模式: {using_ddp})")
            
            try:
                # 在當前進程對應的GPU上加載teacher模型
                if torch.cuda.is_available():
                    with torch.cuda.device(local_rank):
                        self.teacher = YOLO(self.teacher_path).model
                        self.teacher = self.teacher.to(torch.device(current_device))
                else:
                    self.teacher = YOLO(self.teacher_path).model
                
                # 凍結教師模型參數
                for k, v in self.teacher.named_parameters():
                    v.requires_grad = False

                # 設置教師模型為評估模式，確保BN層不更新
                self.teacher.eval()

                # 凍結BN層，讓它們的統計數據(running_mean, running_var)不會更新
                for m in self.teacher.modules():
                    if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                        m.eval()  # 只有BN層設為評估模式
                        for param in m.parameters():
                            param.requires_grad = False

                # 在每個進程上打印資訊，確認教師模型已在所有設備上正確載入
                if using_ddp:
                    # 在DDP中，每個進程獨立載入模型
                    world_size = dist.get_world_size() if dist.is_initialized() else 1
                    current_rank = dist.get_rank() if dist.is_initialized() else 0
                    
                    # 等待所有進程到達此點
                    if dist.is_initialized():
                        dist.barrier()
                    
                    # 分別打印每個進程的信息
                    LOGGER.info(f"進程 {current_rank}/{world_size-1} 在設備 {current_device} 上成功載入教師模型")
                else:
                    LOGGER.info(f"教師模型成功載入到設備 {current_device}")
            except Exception as e:
                LOGGER.error(f"教師模型載入失敗: {e}")
                self.teacher = None

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
            
    def register_teacher_hooks(self):
        """Register hooks on the teacher model to capture intermediate features."""
        if self.teacher is not None:
            # Clear any existing hooks
            for hook in self.teacher_hooks:
                hook.remove()
            self.teacher_hooks = []
            
            LOGGER.info(f"為教師模型註冊勾子，目標層: {self.target_layers}")
            
            # 建立模塊名稱到模塊的映射
            module_dict = {}
            for name, module in self.teacher.named_modules():
                module_dict[name] = module
                
            # 處理每個目標層
            for target in self.target_layers:
                if isinstance(target, int):
                    # 如果是整數索引，直接獲取對應層
                    try:
                        # 檢查索引是否存在於teacher模型中
                        # 注意：在DDP中，model屬性可能是_orig_mod
                        if hasattr(self.teacher, 'model'):
                            layer = self.teacher.model[target]
                            layer_full_name = f"model.{target}"
                        elif len(list(self.teacher.modules())) > target:
                            # 如果沒有model屬性，嘗試直接訪問modules
                            modules = list(self.teacher.modules())
                            layer = modules[target + 1]  # +1跳過模型本身
                            layer_full_name = f"module_{target}"
                        else:
                            LOGGER.warning(f"在教師模型中未找到索引層 {target}")
                            continue
                        layer_idx = target  # 用於hook的layer_idx
                    except (IndexError, AttributeError) as e:
                        LOGGER.warning(f"在教師模型中訪問索引層 {target} 時出錯: {e}")
                        continue
                else:
                    # 如果是字符串路徑，從module_dict中查找
                    if target in module_dict:
                        layer = module_dict[target]
                        layer_full_name = target
                        # 對於字符串路徑，我們使用一個唯一標識作為layer_idx
                        layer_idx = target
                    else:
                        LOGGER.warning(f"在教師模型中未找到指定層: {target}")
                        continue
                
                # 獲取層的類型
                layer_type = layer.__class__.__name__
                
                # 構建詳細的層信息
                layer_info = f"層名稱: {layer_full_name}, 類型: {layer_type}"
                
                LOGGER.info(f"註冊教師模型勾子: {layer_info}")
                
                # 使用捕獲idx的方式註冊鉤子，避免閉包問題
                def get_hook(idx):
                    def hook(module, input, output):
                        self._save_teacher_feature(idx, output)
                    return hook
                
                # 註冊勾子
                hook = layer.register_forward_hook(get_hook(layer_idx))
                self.teacher_hooks.append(hook)
                LOGGER.info(f"成功註冊教師模型勾子: {layer_full_name} (ID: {layer_idx})")
            
    def register_student_hooks(self):
        """Register hooks on the student model to capture intermediate features."""
        # Clear any existing hooks
        for hook in self.student_hooks:
            hook.remove()
        self.student_hooks = []
        
        # 確保我們有模型實例
        if hasattr(self, 'model'):
            model = self.model
            if hasattr(model, 'module'):  # 處理DDP包裝的模型
                model = model.module
        else:
            LOGGER.warning("找不到學生模型實例，無法註冊勾子")
            return
            
        LOGGER.info(f"為學生模型註冊勾子，目標層: {self.target_layers}")
        
        # 建立模塊名稱到模塊的映射
        module_dict = {}
        for name, module in model.named_modules():
            module_dict[name] = module
            
        # 處理每個目標層
        for target in self.target_layers:
            if isinstance(target, int):
                # 如果是整數索引，直接獲取對應層
                try:
                    # 檢查索引是否存在於學生模型中
                    if hasattr(model, 'model'):
                        layer = model.model[target]
                        layer_full_name = f"model.{target}"
                    elif len(list(model.modules())) > target:
                        # 如果沒有model屬性，嘗試直接訪問modules
                        modules = list(model.modules())
                        layer = modules[target + 1]  # +1跳過模型本身
                        layer_full_name = f"module_{target}"
                    else:
                        LOGGER.warning(f"在學生模型中未找到索引層 {target}")
                        continue
                    layer_idx = target  # 用於hook的layer_idx
                except (IndexError, AttributeError) as e:
                    LOGGER.warning(f"在學生模型中訪問索引層 {target} 時出錯: {e}")
                    continue
            else:
                # 如果是字符串路徑，從module_dict中查找
                if target in module_dict:
                    layer = module_dict[target]
                    layer_full_name = target
                    # 對於字符串路徑，我們使用一個唯一標識作為layer_idx
                    layer_idx = target
                else:
                    LOGGER.warning(f"在學生模型中未找到指定層: {target}")
                    continue
            
            # 獲取層的類型
            layer_type = layer.__class__.__name__
            
            # 構建詳細的層信息
            layer_info = f"層名稱: {layer_full_name}, 類型: {layer_type}"
            
            LOGGER.info(f"註冊學生模型勾子: {layer_info}")
            
            # 使用捕獲idx的方式註冊鉤子，避免閉包問題
            def get_hook(idx):
                def hook(module, input, output):
                    self._save_student_feature(idx, output)
                return hook
            
            # 註冊勾子
            hook = layer.register_forward_hook(get_hook(layer_idx))
            self.student_hooks.append(hook)
            LOGGER.info(f"成功註冊學生模型勾子: {layer_full_name} (ID: {layer_idx})")

    def _save_teacher_feature(self, layer_idx, feature):
        """Save features from the teacher model."""
        try:
            # 在存儲特徵之前進行克隆，避免梯度信息影響和後續計算干擾
            if isinstance(feature, torch.Tensor):
                self.teacher_features[layer_idx] = feature.clone().detach()
                # 每100批次或第一批次記錄一次特徵信息，避免日誌過多
                if getattr(self, 'batch_count', 0) % 100 == 0 or not hasattr(self, 'batch_count'):
                    shape_info = f"形狀: {feature.shape}, 類型: {feature.dtype}, 設備: {feature.device}"
                    LOGGER.debug(f"保存教師模型第 {layer_idx} 層特徵: {shape_info}")
            else:
                # 處理非張量輸出的情況
                LOGGER.debug(f"教師模型第 {layer_idx} 層輸出不是張量，而是 {type(feature)}")
                # 如果是元組，列表或字典，嘗試保存第一個張量
                if isinstance(feature, (tuple, list)) and len(feature) > 0:
                    self.teacher_features[layer_idx] = feature[0].clone().detach() if isinstance(feature[0], torch.Tensor) else None
                    LOGGER.debug(f"已保存教師模型第 {layer_idx} 層的第一個元素")
                elif isinstance(feature, dict) and len(feature) > 0:
                    first_key = next(iter(feature))
                    self.teacher_features[layer_idx] = feature[first_key].clone().detach() if isinstance(feature[first_key], torch.Tensor) else None
                    LOGGER.debug(f"已保存教師模型第 {layer_idx} 層的鍵 '{first_key}' 對應的特徵")
                else:
                    self.teacher_features[layer_idx] = None
                    LOGGER.warning(f"無法保存教師模型第 {layer_idx} 層的特徵")
        except Exception as e:
            LOGGER.error(f"保存教師模型第 {layer_idx} 層特徵時發生錯誤: {e}")
            self.teacher_features[layer_idx] = None
            
    def _save_student_feature(self, layer_idx, feature):
        """Save features from the student model."""
        try:
            # 儲存當前批次計數
            if not hasattr(self, 'batch_count'):
                self.batch_count = 0
            else:
                self.batch_count += 1
                
            # 在存儲特徵之前進行克隆，避免梯度信息影響和後續計算干擾
            if isinstance(feature, torch.Tensor):
                self.student_features[layer_idx] = feature.clone()
                # 每100批次或第一批次記錄一次特徵信息，避免日誌過多
                if self.batch_count % 100 == 0 or self.batch_count == 0:
                    shape_info = f"形狀: {feature.shape}, 類型: {feature.dtype}, 設備: {feature.device}"
                    LOGGER.debug(f"保存學生模型第 {layer_idx} 層特徵: {shape_info}")
            else:
                # 處理非張量輸出的情況
                LOGGER.debug(f"學生模型第 {layer_idx} 層輸出不是張量，而是 {type(feature)}")
                # 如果是元組，列表或字典，嘗試保存第一個張量
                if isinstance(feature, (tuple, list)) and len(feature) > 0:
                    self.student_features[layer_idx] = feature[0].clone() if isinstance(feature[0], torch.Tensor) else None
                    LOGGER.debug(f"已保存學生模型第 {layer_idx} 層的第一個元素")
                elif isinstance(feature, dict) and len(feature) > 0:
                    first_key = next(iter(feature))
                    self.student_features[layer_idx] = feature[first_key].clone() if isinstance(feature[first_key], torch.Tensor) else None
                    LOGGER.debug(f"已保存學生模型第 {layer_idx} 層的鍵 '{first_key}' 對應的特徵")
                else:
                    self.student_features[layer_idx] = None
                    LOGGER.warning(f"無法保存學生模型第 {layer_idx} 層的特徵")
        except Exception as e:
            LOGGER.error(f"保存學生模型第 {layer_idx} 層特徵時發生錯誤: {e}")
            self.student_features[layer_idx] = None

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
        """
        預處理批次數據，確保在多GPU環境中正確處理教師模型和特徵
        
        Args:
            batch (dict): 批次數據字典
            
        Returns:
            dict: 預處理後的批次數據
        """
        batch = super().preprocess_batch(batch)

        # 添加教師模型資訊
        if hasattr(self, 'teacher') and self.teacher is not None:
            # 在DDP環境下，確保教師模型與當前批次在同一設備上
            target_device = batch["img"].device
            # 獲取教師模型的設備（通過模型的參數獲取）
            teacher_device = next(self.teacher.parameters()).device if list(self.teacher.parameters()) else target_device
            
            if teacher_device != target_device:
                # 只在需要移動時輸出日誌，避免過多輸出
                if not dist.is_initialized() or dist.get_rank() == 0:
                    LOGGER.debug(f"將教師模型從 {teacher_device} 移動到 {target_device}")
                self.teacher = self.teacher.to(target_device)
            
            # 將教師模型加入批次
            batch["teacher"] = self.teacher
            
            # 添加特徵字典 - 這些字典已經在__init__中初始化
            batch["teacher_features"] = self.teacher_features
            batch["student_features"] = self.student_features
        elif self.teacher_path is not None:
            # 教師模型路徑存在但模型未載入，可能是出錯了
            LOGGER.warning("教師模型路徑存在但模型未成功載入，蒸餾訓練可能無法進行")

        return batch

    def on_train_start(self, trainer):
        # 打印教師模型和學生模型的結構
        LOGGER.info("=" * 80)
        
        if hasattr(self, 'teacher') and self.teacher is not None:
            LOGGER.info(f"教師模型結構 (設備: {next(self.teacher.parameters()).device}):")
            
            # 檢查模型結構並顯示關鍵層
            if hasattr(self.teacher, 'model'):
                for i, m in enumerate(self.teacher.model):
                    module_type = m.__class__.__name__
                    num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                    LOGGER.info(f"  - model.{i}: {module_type} (參數量: {num_params})")
            else:
                LOGGER.info("  教師模型沒有標準的model屬性")
                for i, (name, m) in enumerate(list(self.teacher.named_children())[:10]):  # 只顯示前10個子模塊
                    module_type = m.__class__.__name__
                    num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                    LOGGER.info(f"  - {name}: {module_type} (參數量: {num_params})")
                if len(list(self.teacher.named_children())) > 10:
                    LOGGER.info(f"  ... 等 {len(list(self.teacher.named_children()))-10} 個模塊")
        else:
            LOGGER.warning("教師模型未載入，蒸餾可能無法正常進行")
        
        # 處理學生模型可能被DDP包裝的情況
        student_model = self.model
        if hasattr(student_model, 'module'):
            student_model = student_model.module
        
        LOGGER.info("\n學生模型結構:")
        if hasattr(student_model, 'model'):
            for i, m in enumerate(student_model.model):
                module_type = m.__class__.__name__
                num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                LOGGER.info(f"  - model.{i}: {module_type} (參數量: {num_params})")
        else:
            LOGGER.info("  學生模型沒有標準的model屬性")
            for i, (name, m) in enumerate(list(student_model.named_children())[:10]):  # 只顯示前10個子模塊
                module_type = m.__class__.__name__
                num_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                LOGGER.info(f"  - {name}: {module_type} (參數量: {num_params})")
            if len(list(student_model.named_children())) > 10:
                LOGGER.info(f"  ... 等 {len(list(student_model.named_children()))-10} 個模塊")
                
        LOGGER.info("=" * 80)
        LOGGER.info(f"目標蒸餾層: {self.target_layers}")
        LOGGER.info("=" * 80)

        # Register hooks for the teacher model
        self.register_teacher_hooks()
        # Register hooks for the student model
        self.register_student_hooks()

    def on_epoch_start(self, trainer):
        self.model.epoch = trainer.epoch
        self.model.epochs = trainer.epochs
        self.model.is_first_batch_in_epoch = True

    def on_epoch_end(self, trainer):
        pass

    def on_val_start(self, trainer):
        pass

    def on_val_end(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        # Remove hooks when training ends
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()

        # Clear the stored features
        self.teacher_features = {}
        self.student_features = {}
    
    def teardown(self, trainer):
        # Make sure all hooks are removed
        for hook in self.teacher_hooks:
            hook.remove()
        for hook in self.student_hooks:
            hook.remove()
    
    def on_batch_end(self, trainer):
        self.model.is_first_batch_in_epoch = False
        pass

    def set_target_layers(self, new_target_layers):
        """
        設置新的目標層並重新註冊勾子。
        
        Args:
            new_target_layers (list): 包含層索引或層名稱的列表，例如 [0, 6, 13] 或 ["model.0.conv", "model.6.cv1"]
        """
        # 更新目標層
        self.target_layers = new_target_layers
        LOGGER.info(f"更新目標層為: {self.target_layers}")
        
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
