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


    def compare_bn_statistics(self, teacher_model, student_model, sample_layers=10):
        """比較教師和學生模型的BN層統計數據"""
        teacher_bns = [m for m in teacher_model.modules() if isinstance(m, nn.BatchNorm2d)]
        student_bns = [m for m in student_model.modules() if isinstance(m, nn.BatchNorm2d)]
        
        # 記錄兩個模型的BN層數量
        LOGGER.info(f"BN層數量: 教師({len(teacher_bns)}), 學生({len(student_bns)})")
        
        # 創建兩個列表存儲BN層的大小信息
        teacher_bn_sizes = [bn.num_features for bn in teacher_bns]
        student_bn_sizes = [bn.num_features for bn in student_bns]
        
        # 只比較大小相同的BN層
        comparable_bns = []
        for i, (t_bn, s_bn) in enumerate(zip(teacher_bns, student_bns)):
            if t_bn.num_features == s_bn.num_features:
                comparable_bns.append((i, t_bn, s_bn))
        
        LOGGER.info(f"可比較的BN層數量: {len(comparable_bns)}/{min(len(teacher_bns), len(student_bns))}")
        
        if not comparable_bns:
            LOGGER.warning("沒有可比較的BN層，跳過統計數據比較")
            return {
                'avg_mean_diff': 0,
                'avg_var_diff': 0,
                'avg_mean_rel_diff': 0,
                'avg_var_rel_diff': 0,
                'max_mean_diff': 0,
                'max_var_diff': 0,
                'max_mean_rel_diff': 0, 
                'max_var_rel_diff': 0,
                'mean_diffs': [],
                'var_diffs': []
            }
        
        # 計算所有可比較BN層統計數據的差異
        mean_diffs = []
        var_diffs = []
        mean_rel_diffs = []  # 相對差異
        var_rel_diffs = []   # 相對差異
        
        for idx, t_bn, s_bn in comparable_bns:
            # 計算絕對差異
            mean_diff = (t_bn.running_mean - s_bn.running_mean).abs().mean().item()
            var_diff = (t_bn.running_var - s_bn.running_var).abs().mean().item()
            
            # 計算相對差異 (避免除以零)
            t_mean_abs = t_bn.running_mean.abs().mean().item()
            t_var_abs = t_bn.running_var.abs().mean().item()
            
            mean_rel_diff = mean_diff / (t_mean_abs + 1e-6)
            var_rel_diff = var_diff / (t_var_abs + 1e-6)
            
            mean_diffs.append(mean_diff)
            var_diffs.append(var_diff)
            mean_rel_diffs.append(mean_rel_diff)
            var_rel_diffs.append(var_rel_diff)
        
        # 計算總體統計數據
        avg_mean_diff = sum(mean_diffs) / len(mean_diffs) if mean_diffs else 0
        avg_var_diff = sum(var_diffs) / len(var_diffs) if var_diffs else 0
        avg_mean_rel_diff = sum(mean_rel_diffs) / len(mean_rel_diffs) if mean_rel_diffs else 0
        avg_var_rel_diff = sum(var_rel_diffs) / len(var_rel_diffs) if var_rel_diffs else 0
        
        max_mean_diff = max(mean_diffs) if mean_diffs else 0
        max_var_diff = max(var_diffs) if var_diffs else 0
        max_mean_rel_diff = max(mean_rel_diffs) if mean_rel_diffs else 0
        max_var_rel_diff = max(var_rel_diffs) if var_rel_diffs else 0
        
        # 打印總體統計數據
        LOGGER.info("=" * 50)
        LOGGER.info("BatchNorm層統計數據比較:")
        LOGGER.info(f"總計比較了 {len(comparable_bns)} 個BN層")
        LOGGER.info(f"平均絕對差異 - 均值: {avg_mean_diff:.6f}, 方差: {avg_var_diff:.6f}")
        LOGGER.info(f"平均相對差異 - 均值: {avg_mean_rel_diff:.2%}, 方差: {avg_var_rel_diff:.2%}")
        LOGGER.info(f"最大絕對差異 - 均值: {max_mean_diff:.6f}, 方差: {max_var_diff:.6f}")
        LOGGER.info(f"最大相對差異 - 均值: {max_mean_rel_diff:.2%}, 方差: {max_var_rel_diff:.2%}")
        
        # 打印部分層的詳細統計數據
        LOGGER.info("-" * 50)
        LOGGER.info("部分BN層的詳細統計數據:")
        
        # 選擇一些有代表性的層(前幾層、中間幾層、後幾層)
        sample_indices = []
        if len(comparable_bns) <= sample_layers:
            sample_indices = list(range(len(comparable_bns)))
        else:
            # 選擇前幾層、中間幾層和後幾層
            front = sample_layers // 3
            middle = sample_layers // 3
            back = sample_layers - front - middle
            
            sample_indices = list(range(front))
            sample_indices += list(range(len(comparable_bns)//2 - middle//2, len(comparable_bns)//2 + middle//2 + middle%2))
            sample_indices += list(range(len(comparable_bns) - back, len(comparable_bns)))
        
        for i in sample_indices:
            if i < len(comparable_bns):
                idx, t_bn, s_bn = comparable_bns[i]
                
                # 計算基本統計數據
                t_mean = t_bn.running_mean.mean().item()
                t_var = t_bn.running_var.mean().item()
                s_mean = s_bn.running_mean.mean().item()
                s_var = s_bn.running_var.mean().item()
                
                mean_diff = mean_diffs[i]
                var_diff = var_diffs[i]
                mean_rel_diff = mean_rel_diffs[i]
                var_rel_diff = var_rel_diffs[i]
                
                LOGGER.info(f"BN層 {idx} (特徵數: {t_bn.num_features}):")
                LOGGER.info(f"  教師 - 均值: {t_mean:.6f}, 方差: {t_var:.6f}")
                LOGGER.info(f"  學生 - 均值: {s_mean:.6f}, 方差: {s_var:.6f}")
                LOGGER.info(f"  絕對差異 - 均值: {mean_diff:.6f}, 方差: {var_diff:.6f}")
                LOGGER.info(f"  相對差異 - 均值: {mean_rel_diff:.2%}, 方差: {var_rel_diff:.2%}")
        
        LOGGER.info("=" * 50)
        
        # 返回差異指標，可用於進一步分析
        return {
            'avg_mean_diff': avg_mean_diff,
            'avg_var_diff': avg_var_diff,
            'avg_mean_rel_diff': avg_mean_rel_diff,
            'avg_var_rel_diff': avg_var_rel_diff,
            'max_mean_diff': max_mean_diff,
            'max_var_diff': max_var_diff,
            'max_mean_rel_diff': max_mean_rel_diff,
            'max_var_rel_diff': max_var_rel_diff,
            'mean_diffs': mean_diffs,
            'var_diffs': var_diffs
        }
    
    def verify_initial_model_state(self):
        """在訓練開始前驗證教師和學生模型的初始狀態"""
        LOGGER.info("=" * 50)
        LOGGER.info("驗證教師和學生模型的初始狀態")
        
        # 1. 比較模型參數
        teacher_params = {name: param.clone() for name, param in self.teacher.named_parameters()}
        student_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # 找出共同的參數名稱
        common_params = set(teacher_params.keys()).intersection(set(student_params.keys()))
        
        # 記錄參數總數和可比較數量
        total_t_params = len(teacher_params)
        total_s_params = len(student_params)
        common_param_count = len(common_params)
        
        LOGGER.info(f"參數統計: 教師({total_t_params}), 學生({total_s_params}), 共同參數({common_param_count})")
        
        # 只比較大小相同的參數
        comparable_params = []
        for name in common_params:
            t_param = teacher_params[name]
            s_param = student_params[name]
            
            if t_param.size() == s_param.size():
                comparable_params.append(name)
        
        LOGGER.info(f"可比較的參數數量: {len(comparable_params)}/{common_param_count}")
        
        param_diffs = []
        for name in comparable_params:
            diff = (teacher_params[name] - student_params[name]).abs().max().item()
            param_diffs.append((name, diff))
        
        if param_diffs:
            max_param_diff = max(param_diffs, key=lambda x: x[1])
            LOGGER.info(f"參數最大差異: {max_param_diff[1]:.10f} (在層 {max_param_diff[0]})")
        else:
            LOGGER.info("沒有可比較的參數")
        
        # 2. 比較模型前向傳播 (使用相同輸入)
        # 準備固定的測試輸入
        test_input = torch.rand(1, 3, 640, 640, device=self.device)
        
        # 保存教師和學生的當前模式
        teacher_training = self.teacher.training
        student_training = self.model.training
        
        # 設置兩個模型為評估模式
        self.teacher.eval()
        self.model.eval()
        
        # 收集中間特徵
        teacher_features = []
        student_features = []
        teacher_hooks = []
        student_hooks = []
        
        # 定義特徵收集函數
        def hook_fn(features_list):
            return lambda module, input, output: features_list.append(output.detach())
        
        # 嘗試為特定層註冊hooks
        teacher_hook_layers = []
        student_hook_layers = []
        hook_layer_patterns = ["22.cv2"]  # 使用模式匹配
        
        # 嘗試查找教師模型中的對應層
        for name, module in self.teacher.named_modules():
            for pattern in hook_layer_patterns:
                if pattern in name:
                    teacher_hook_layers.append((name, module))
                    LOGGER.info(f"為教師模型註冊hook: {name}, 輸出形狀: {getattr(module, 'out_channels', 'unknown')}")
                    teacher_hooks.append(module.register_forward_hook(hook_fn(teacher_features)))
        
        # 嘗試查找學生模型中的對應層
        for name, module in self.model.named_modules():
            for pattern in hook_layer_patterns:
                if pattern in name:
                    student_hook_layers.append((name, module))
                    LOGGER.info(f"為學生模型註冊hook: {name}, 輸出形狀: {getattr(module, 'out_channels', 'unknown')}")
                    student_hooks.append(module.register_forward_hook(hook_fn(student_features)))
        
        # 如果找不到特定層，記錄所有可用層以供參考
        if not teacher_hook_layers or not student_hook_layers:
            LOGGER.warning(f"未找到匹配的層: {hook_layer_patterns}")
            LOGGER.info("教師模型可用層:")
            for name, _ in self.teacher.named_modules():
                if any(name.endswith(f".cv{i}") for i in range(1, 4)):
                    LOGGER.info(f"  - {name}")
            
            LOGGER.info("學生模型可用層:")
            for name, _ in self.model.named_modules():
                if any(name.endswith(f".cv{i}") for i in range(1, 4)):
                    LOGGER.info(f"  - {name}")
        
        # 執行前向傳播
        with torch.no_grad():
            try:
                teacher_output = self.teacher(test_input)
                student_output = self.model(test_input)
                forward_success = True
            except Exception as e:
                LOGGER.error(f"前向傳播失敗: {str(e)}")
                forward_success = False
        
        # 移除hooks
        for hook in teacher_hooks + student_hooks:
            hook.remove()
        
        if forward_success:
            # 比較最終輸出 (只在輸出是Tensor或具有相同結構時比較)
            if (isinstance(teacher_output, torch.Tensor) and isinstance(student_output, torch.Tensor) and 
                teacher_output.size() == student_output.size()):
                output_diff = (teacher_output - student_output).abs().max().item()
                LOGGER.info(f"最終輸出最大差異: {output_diff:.10f}")
            else:
                LOGGER.info("無法比較最終輸出 (不同的輸出類型或形狀)")
                if isinstance(teacher_output, torch.Tensor):
                    LOGGER.info(f"教師輸出形狀: {teacher_output.shape}")
                if isinstance(student_output, torch.Tensor):
                    LOGGER.info(f"學生輸出形狀: {student_output.shape}")
            
            # 比較中間特徵 (只比較大小相同的特徵)
            comparable_features = []
            for i, (t_feat, s_feat) in enumerate(zip(teacher_features, student_features)):
                if isinstance(t_feat, torch.Tensor) and isinstance(s_feat, torch.Tensor) and t_feat.size() == s_feat.size():
                    comparable_features.append((i, t_feat, s_feat))
            
            LOGGER.info(f"可比較的特徵數量: {len(comparable_features)}/{len(teacher_features)}")
            
            for i, t_feat, s_feat in comparable_features:
                feat_diff = (t_feat - s_feat).abs()
                max_diff = feat_diff.max().item()
                mean_diff = feat_diff.mean().item()
                std_diff = feat_diff.std().item()
                
                LOGGER.info(f"層 {i} 特徵差異統計 (形狀: {t_feat.shape}):")
                LOGGER.info(f"  最大差異: {max_diff:.10f}, 平均差異: {mean_diff:.10f}, 標準差: {std_diff:.10f}")
        
        # 恢復原始模式
        self.teacher.train(teacher_training)
        self.model.train(student_training)
        
        # 3. 特殊檢查 - 模型內部狀態
        LOGGER.info("-" * 30)
        LOGGER.info("檢查模型內部狀態:")
        
        # 檢查模型是否有相同的buffer (如BN層的running_mean/var)
        teacher_buffers = {name: buf.clone() for name, buf in self.teacher.named_buffers()}
        student_buffers = {name: buf.clone() for name, buf in self.model.named_buffers()}
        
        # 找出共同的緩衝區名稱
        common_buffers = set(teacher_buffers.keys()).intersection(set(student_buffers.keys()))
        
        # 只比較大小相同的緩衝區
        comparable_buffers = []
        for name in common_buffers:
            if teacher_buffers[name].size() == student_buffers[name].size():
                comparable_buffers.append(name)
        
        LOGGER.info(f"可比較的buffers數量: {len(comparable_buffers)}/{len(common_buffers)}")
        
        buffer_diffs = []
        for name in comparable_buffers:
            diff = (teacher_buffers[name] - student_buffers[name]).abs().max().item()
            buffer_diffs.append((name, diff))
        
        if buffer_diffs:
            max_buffer_diff = max(buffer_diffs, key=lambda x: x[1])
            LOGGER.info(f"Buffer最大差異: {max_buffer_diff[1]:.10f} (在 {max_buffer_diff[0]})")
        else:
            LOGGER.info("沒有可比較的buffer")
        
        LOGGER.info("=" * 50)

    def freeze_bn_statistics(self, model):
        """凍結模型中所有BN層的統計量更新"""
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False  # 停止更新running_mean和running_var
                m.running_mean.requires_grad = False
                m.running_var.requires_grad = False
        
        LOGGER.info("已凍結模型的BN統計量更新")


    def distill_on_train_start(self, trainer):
        # 設置確定性計算環境
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def distill_on_epoch_start(self, trainer):
        # 在比較模型狀態後添加
        LOGGER.info("比較教師和學生模型的BN層統計數據...")
        bn_diff_stats = self.compare_bn_statistics(self.teacher, self.model)
        
        # 根據差異大小發出警告
        if bn_diff_stats['avg_mean_rel_diff'] > 0.5 or bn_diff_stats['avg_var_rel_diff'] > 0.5:
            LOGGER.warning("警告：BN層統計數據差異顯著，可能影響蒸餾效果!")

        self.verify_initial_model_state()

        if self.epoch == 0:
            LOGGER.info("階段1: 純蒸餾")

            distillation_loss = "cwd"
            distillation_layers = ["22"]
            self.model.args.distill = 0.1

            # 初始化蒸餾損失實例
            self.distill_loss_instance = DistillationLoss(
                models=self.model,
                modelt=self.teacher,
                distiller=distillation_loss,
                layers=distillation_layers,
            )

            # 選擇: 凍結BN統計量更新
            self.freeze_bn_statistics(self.model)

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
