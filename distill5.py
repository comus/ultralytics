# distill5.py - 超級終極突破策略 - 極限版
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import torch.nn as nn
import torch

# 使用最佳FGD模型作為起點
student_model = YOLO("distill_projects/yolo11n_fgd_breakthrough/weights/best.pt")
teacher_model = YOLO("yolo11x-pose.pt")  # 最強教師模型

# 自定義更強大的FGD蒸餾損失 - 直接插入到工作流程中
class EnhancedFGDLoss(nn.Module):
    """極限增強版特徵引導蒸餾 - 專為突破性能瓶頸優化"""
    def __init__(self, student_channels, teacher_channels, spatial_weight=3.0, channel_weight=0.4):
        super(EnhancedFGDLoss, self).__init__()
        self.spatial_weight = spatial_weight    # 更強的空間權重
        self.channel_weight = channel_weight    # 降低通道權重
        
        # 增強版邊緣感知機制
        self.edge_filter = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False).to(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.edge_filter.weight.data.copy_(torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ]).view(1, 1, 3, 3) / 8.0)
        self.edge_filter.requires_grad_(False)  # 凍結參數
        
        # 關鍵點注意力機制
        self.keypoint_attention = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        ).to('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, y_s, y_t):
        """計算極限增強版FGD損失，特別優化姿態估計的空間特徵保留"""
        assert len(y_s) == len(y_t)
        losses = []
        for idx, (s, t) in enumerate(zip(y_s, y_t)):
            if s.size(2) != t.size(2) or s.size(3) != t.size(3):
                t = torch.nn.functional.interpolate(t, size=(s.size(2), s.size(3)), mode='bilinear', align_corners=False)
                
            b, c, h, w = s.shape
            
            # 1. 自適應特徵匹配 - 使用Huber損失減少異常值影響
            # 對重要特徵賦予更高權重
            feature_importance = torch.sigmoid(torch.sum(t, dim=1, keepdim=True))
            l1_loss = torch.nn.functional.smooth_l1_loss(s * feature_importance, t * feature_importance)
            
            # 2. 超級增強版空間注意力損失
            s_spatial = torch.mean(s, dim=1, keepdim=True)  # [b, 1, h, w]
            t_spatial = torch.mean(t, dim=1, keepdim=True)  # [b, 1, h, w]
            
            # 提取空間特徵的邊緣信息
            s_edge = self.edge_filter(s_spatial)
            t_edge = self.edge_filter(t_spatial)
            
            # 關鍵點注意力增強
            keypoint_attn = self.keypoint_attention(t_spatial)
            
            # 空間注意力+邊緣感知+關鍵點注意力的組合損失
            spatial_loss = (torch.nn.functional.mse_loss(s_spatial, t_spatial) * 1.5 + 
                         torch.nn.functional.mse_loss(s_edge, t_edge) * 1.0 +
                         torch.nn.functional.mse_loss(s_spatial * keypoint_attn, t_spatial * keypoint_attn) * 1.5) * self.spatial_weight
            
            # 3. 優化的通道關係損失 - 專注於重要通道
            s_flat = s.view(b, c, -1)  # [b, c, h*w]
            t_flat = t.view(b, c, -1)  # [b, c, h*w]
            
            # 通道歸一化
            s_flat_norm = torch.nn.functional.normalize(s_flat, dim=2)
            t_flat_norm = torch.nn.functional.normalize(t_flat, dim=2)
            
            # 計算通道相關矩陣
            s_corr = torch.bmm(s_flat_norm, s_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
            t_corr = torch.bmm(t_flat_norm, t_flat_norm.transpose(1, 2)) / (h*w)  # [b, c, c]
            
            # 通道相關性損失
            channel_loss = torch.nn.functional.mse_loss(s_corr, t_corr) * self.channel_weight
            
            # 4. 組合損失
            total_loss = l1_loss + spatial_loss + channel_loss
            losses.append(total_loss)
            
        loss = sum(losses)
        return loss

# 初始化增強版FGD損失函數
try:
    LOGGER.info("初始化增強版FGD損失函數...")
    enhanced_fgd = EnhancedFGDLoss(
        student_channels=[256, 512, 512], 
        teacher_channels=[256, 512, 512]
    )
    # 將自定義損失函數注入utils.distill模塊
    import ultralytics.utils.distill
    ultralytics.utils.distill.FGDLoss = EnhancedFGDLoss
    LOGGER.info("成功注入增強版FGD損失函數！")
except Exception as e:
    LOGGER.warning(f"注入增強FGD損失函數失敗: {e}")

# 極限版自適應學習率調整回調
def extreme_adaptive_lr_callback(trainer):
    """極限版自適應學習率調整策略，用於顯著突破性能瓶頸"""
    # 在訓練的不同階段實施更激進策略
    if trainer.epoch == 0:
        # 初始階段使用較高學習率
        LOGGER.info("Initial phase: using higher learning rate")
        for g in trainer.optimizer.param_groups:
            g['lr'] = 0.00005  # 開始階段更高
    
    elif trainer.epoch == 3:
        # 第一次大幅提高學習率突破平台
        LOGGER.info("First major learning rate boost to escape initial plateau")
        for g in trainer.optimizer.param_groups:
            g['lr'] = 0.0001  # 第一次大幅提高
            
    elif trainer.epoch == 5:
        # 第二次提高學習率突破下一個平台
        LOGGER.info("Second learning rate boost for breakthrough")
        for g in trainer.optimizer.param_groups:
            g['lr'] = 0.00012  # 第二次大幅提高
            
    elif trainer.epoch == 8:
        # 恢復到中等學習率進行優化
        LOGGER.info("Restoring to medium learning rate for optimization")
        for g in trainer.optimizer.param_groups:
            g['lr'] = 0.00002  # 中等學習率精細優化
    
    elif trainer.epoch == 15:
        # 降低學習率進行精細優化階段
        LOGGER.info("Fine-tuning phase: low learning rate")
        for g in trainer.optimizer.param_groups:
            g['lr'] = 0.000005  # 精細優化學習率
            
    elif trainer.epoch == 20:
        # 極限精細優化階段
        LOGGER.info("Ultra-fine tuning phase: minimal learning rate")
        for g in trainer.optimizer.param_groups:
            g['lr'] = 0.000001  # 極限精細優化學習率
            
    # 顯示當前學習率
    for i, g in enumerate(trainer.optimizer.param_groups):
        LOGGER.info(f"Epoch {trainer.epoch}: Group {i} learning rate = {g['lr']:.8f}")

# 將極限版回調添加到模型中
student_model.add_callback("on_train_epoch_start", extreme_adaptive_lr_callback)

# 增強的監控訓練進度回調 - 維持不變，目標設定已經足夠積極
def training_progress_callback(trainer):
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

# 添加監控回調
student_model.add_callback("on_fit_epoch_end", training_progress_callback)

# 高級選擇性增強回調 - 根據訓練階段靈活調整數據增強策略
def advanced_augmentation_callback(trainer):
    """根據訓練階段高度靈活地調整增強策略"""
    if trainer.epoch == 0:
        # 初始階段：使用輕微增強
        LOGGER.info("Initial phase: very light augmentation")
        trainer.train_loader.dataset.hsv_values = [0.05, 0.05, 0.05]  # 極輕微HSV變化
        if hasattr(trainer.train_loader.dataset, 'mosaic'):
            trainer.train_loader.dataset.mosaic = False   # 初始就禁用馬賽克
        if hasattr(trainer.train_loader.dataset, 'mixup'):
            trainer.train_loader.dataset.mixup = False    # 禁用mixup
            
    elif trainer.epoch == 5:
        # 中期階段：增加變化以幫助泛化
        LOGGER.info("Mid phase: moderate augmentation for diversity")
        trainer.train_loader.dataset.hsv_values = [0.15, 0.15, 0.1]  # 適中HSV變化
        if hasattr(trainer.train_loader.dataset, 'translate'):
            trainer.train_loader.dataset.translate = 0.1  # 輕微平移
        
    elif trainer.epoch == 10:
        # 中後期：減少增強以提高精度
        LOGGER.info("Late mid phase: reducing augmentation for precision")
        trainer.train_loader.dataset.hsv_values = [0.1, 0.1, 0.05]  # 降低HSV變化
        if hasattr(trainer.train_loader.dataset, 'translate'):
            trainer.train_loader.dataset.translate = 0.0  # 禁用平移
            
    elif trainer.epoch == 15:
        # 後期：禁用大部分增強專注精細優化
        LOGGER.info("Late phase: minimal augmentation for fine-tuning")
        trainer.train_loader.dataset.hsv_values = [0.0, 0.0, 0.0]  # 禁用HSV
        if hasattr(trainer.train_loader.dataset, 'mosaic'):
            trainer.train_loader.dataset.mosaic = False   # 確保禁用馬賽克
        if hasattr(trainer.train_loader.dataset, 'mixup'):
            trainer.train_loader.dataset.mixup = False    # 確保禁用mixup
        if hasattr(trainer.train_loader.dataset, 'fliplr'):
            trainer.train_loader.dataset.fliplr = 0.3     # 只保留少量水平翻轉
        
    elif trainer.epoch == 20:
        # 最終階段：完全禁用所有增強以實現極致精度
        LOGGER.info("Final phase: zero augmentation for ultimate precision")
        trainer.train_loader.dataset.hsv_values = [0.0, 0.0, 0.0]  # 禁用HSV
        if hasattr(trainer.train_loader.dataset, 'mosaic'):
            trainer.train_loader.dataset.mosaic = False   # 確保禁用馬賽克
        if hasattr(trainer.train_loader.dataset, 'mixup'):
            trainer.train_loader.dataset.mixup = False    # 確保禁用mixup
        if hasattr(trainer.train_loader.dataset, 'fliplr'):
            trainer.train_loader.dataset.fliplr = 0.0     # 禁用水平翻轉
        
# 添加高級選擇性增強回調
student_model.add_callback("on_train_epoch_start", advanced_augmentation_callback)

# 自定義優化器設置回調 - 控制動量和beta值
def optimizer_config_callback(trainer):
    """優化優化器參數以實現更好的收斂和突破"""
    # 只在第一個epoch設置
    if trainer.epoch == 0:
        for g in trainer.optimizer.param_groups:
            # 對於AdamW優化器，設置beta值
            if 'betas' in g:
                g['betas'] = (0.937, 0.999)  # 優化的beta值
                LOGGER.info(f"設置優化的beta值: {g['betas']}")
            
            # 設置動量值（如果適用）
            if 'momentum' in g:
                g['momentum'] = 0.95  # 優化的動量值
                LOGGER.info(f"設置優化的動量值: {g['momentum']}")

# 添加優化器設置回調
student_model.add_callback("on_train_epoch_start", optimizer_config_callback)

# 超極限終極訓練設置
student_model.train(
    data="coco-pose.yaml",
    teacher=teacher_model.model,
    distillation_loss="fgd",           # 使用我們增強版的FGD
    distillation_layers=["4", "6", "8", "11", "13", "16", "19", "22"],  # 更全面的層選擇
    epochs=30,                         # 大幅延長訓練時間
    batch=16,                          # 較小批次提高精度
    workers=8,
    lr0=0.00005,                       # 更高初始學習率
    lrf=0.00001,                       # 更激進的衰減
    optimizer="AdamW",                 # 更先進的優化器
    weight_decay=0.0001,               # 適度增加正則化
    cos_lr=False,                      # 禁用餘弦調度，使用自定義調度
    patience=35,                       # 增加耐心值
    device=0,
    project="distill_projects",
    name="yolo11n_extreme_breakthrough",
    val=True,
    save_period=1,
    plots=True,
    distill=0.45,                      # 進一步增強蒸餾權重
    pose=45.0,                         # 進一步增強姿態權重
    warmup_epochs=0.0,                 # 無需熱身
    close_mosaic=0,                    # 從頭關閉mosaic
    amp=False,                         # 確保全精度
    overlap_mask=True,                 # 改善遮擋情況 (修正overlap→overlap_mask)
    
    # 進階數據處理設置
    hsv_h=0.05,                        # 初始極輕微色調增強
    hsv_s=0.05,                        # 初始極輕微飽和度增強
    hsv_v=0.05,                        # 初始極輕微亮度增強
    degrees=0.0,                       # 禁用旋轉增強
    translate=0.0,                     # 初始禁用平移增強
    scale=0.0,                         # 禁用縮放增強
    fliplr=0.5,                        # 啟用水平翻轉以增加多樣性
    mosaic=0.0,                        # 禁用馬賽克增強
    
    # 特殊關鍵點設置 (移除無效的kpt_shape參數)
    single_cls=False,                  # 維持多類別區分
    nbs=8,                             # 標稱批次大小
    rect=False,                        # 非矩形訓練
    save_json=True,                    # 保存評估JSON
    half=False,                        # 確保全精度訓練
    augment=False,                     # 初始禁用強增強
    fraction=1.0,                      # 使用全部數據
    cache="disk",                      # 使用磁盤緩存加速
    
    # 額外優化參數
    dropout=0.0,                       # 禁用dropout以保留全部特徵
    verbose=True,                      # 詳細日誌
    seed=42,                           # 更優的隨機種子
) 