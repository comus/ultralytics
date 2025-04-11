# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, callbacks
from ultralytics.utils.plotting import plot_images, plot_results
import torch


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

        self.teacher = overrides.get("teacher", None)

        super().__init__(cfg, overrides, _callbacks)

        if self.teacher is not None:
            # 凍結教師模型參數
            for k, v in self.teacher.named_parameters():
                v.requires_grad = False
            
            # 設置教師模型為訓練模式，但凍結BN層統計數據
            self.teacher = self.teacher.to(self.device)
            # self.teacher.train()  # 設為訓練模式而非評估模式
            self.teacher.eval()
            
            # 凍結BN層，讓它們的統計數據(running_mean, running_var)不會更新
            for m in self.teacher.modules():
                if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                    m.eval()  # 只有BN層設為評估模式
                    
            LOGGER.info(f"初始化教師模型已完成，設為訓練模式但凍結BN層")

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
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        # 測試完要刪除以下代碼

        # # 凍結學生模型參數
        # for k, v in self.model.named_parameters():
        #     v.requires_grad = False

        # 凍結BN層，讓它們的統計數據(running_mean, running_var)不會更新
        for m in self.model.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                m.eval()  # 只有BN層設為評估模式

        # 列出有哪些層是訓練模式
        print("============= 學生模型層 =============")
        for n, m in self.model.named_modules():
            if m.training:
                print(f"!!!!!!!!!!!_model_train, m.training: {n}")
            else:
                print(f"!!!!!!!!!!!_model_train, m.eval: {n}")
        
        # 列出參數的 requires_grad 狀態
        print("============= 學生模型參數 =============")
        for n, p in self.model.named_parameters():
            print(f"參數: {n}, requires_grad: {p.requires_grad}")
            
        if self.teacher is not None:
            print("============= 教師模型層 =============")
            for n, m in self.teacher.named_modules():
                if m.training:
                    print(f"!!!!!!!!!!!_model_train, m.training: {n}")
                else:
                    print(f"!!!!!!!!!!!_model_train, m.eval: {n}")
                    
            # 列出教師模型參數的 requires_grad 狀態
            print("============= 教師模型參數 =============")
            for n, p in self.teacher.named_parameters():
                print(f"參數: {n}, requires_grad: {p.requires_grad}")

    def preprocess_batch(self, batch):
        batch = super().preprocess_batch(batch)

        # add teacher
        if self.teacher is not None:
            batch["teacher"] = self.teacher

        return batch

    def on_train_start(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_val_start(self, trainer):
        pass

    def on_val_end(self, trainer):
        pass
    
    def on_train_end(self, trainer):
        pass

    def teardown(self, trainer):
        pass
    
    def on_batch_end(self, trainer):
        pass

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

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = "box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss", "dpose_loss", "dkobj_loss"
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
