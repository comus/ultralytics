from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils.dev import describe_var
from ultralytics.utils.ops import nms_rotated, xywh2xyxy
import time
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_indices_on_feature_maps(
    student_preds, student_indices, student_confidences=None, student_results=None, 
    student_pred_scores=None, student_fg_mask=None, student_target_gt_idx=None, 
    student_target_scores=None, student_high_conf_not_assigned=None,
    teacher_preds=None, teacher_indices=None, teacher_confidences=None, teacher_results=None,
    teacher_pred_scores=None, teacher_fg_mask=None, teacher_target_gt_idx=None,
    teacher_target_scores=None, teacher_high_conf_not_assigned=None
):
    """
    在特徵圖上可視化預測索引的分佈，同時支持學生模型和教師模型的對比
    
    Args:
        student_preds (tuple): 學生模型的原始預測輸出，包含特徵圖信息
        student_indices (list of tensors): 學生模型每張圖片的預測索引列表
        student_confidences (list of tensors, optional): 學生模型每個索引的置信度
        student_results (list of Results, optional): 學生模型後處理後的結果，包含關鍵點信息
        student_pred_scores (tensor, optional): 學生模型預測分數，形狀為 [batch_size, 8400, 1]
        student_fg_mask (tensor, optional): 學生模型前景掩碼，形狀為 [batch_size, 8400]
        student_target_gt_idx (tensor, optional): 學生模型目標GT索引，形狀為 [batch_size, 8400]
        student_target_scores (tensor, optional): 學生模型目標分數，形狀為 [batch_size, 8400, 1]
        student_high_conf_not_assigned (tensor, optional): 學生模型高置信度但未被分配的掩碼，形狀為 [batch_size, 8400]
        
        teacher_preds (tuple, optional): 教師模型的原始預測輸出，包含特徵圖信息
        teacher_indices (list of tensors, optional): 教師模型每張圖片的預測索引列表
        teacher_confidences (list of tensors, optional): 教師模型每個索引的置信度
        teacher_results (list of Results, optional): 教師模型後處理後的結果，包含關鍵點信息
        teacher_pred_scores (tensor, optional): 教師模型預測分數，形狀為 [batch_size, 8400, 1]
        teacher_fg_mask (tensor, optional): 教師模型前景掩碼，形狀為 [batch_size, 8400]
        teacher_target_gt_idx (tensor, optional): 教師模型目標GT索引，形狀為 [batch_size, 8400]
        teacher_target_scores (tensor, optional): 教師模型目標分數，形狀為 [batch_size, 8400, 1]
        teacher_high_conf_not_assigned (tensor, optional): 教師模型高置信度但未被分配的掩碼，形狀為 [batch_size, 8400]
    """
    # 检查是否同时提供了学生和教师模型数据
    has_teacher = teacher_preds is not None and teacher_indices is not None
    
    # 從pred中獲取特徵圖
    student_feature_maps = student_preds[1][0]  # pred[1]是(x, kpt)，pred[1][0]是x，即特徵圖列表
    
    # 如果有教师模型，获取教师模型的特征图
    teacher_feature_maps = None
    if has_teacher:
        teacher_feature_maps = teacher_preds[1][0]
    
    # 檢查輸入格式
    if not isinstance(student_indices, list) or not student_indices or not hasattr(student_indices[0], 'cpu'):
        print("錯誤：學生模型索引必須是包含tensor的列表")
        return
    
    if not isinstance(student_feature_maps, list) or not student_feature_maps:
        print("錯誤：學生模型特徵圖必須是列表格式")
        return
    
    if has_teacher:
        if not isinstance(teacher_indices, list) or not teacher_indices or not hasattr(teacher_indices[0], 'cpu'):
            print("錯誤：教師模型索引必須是包含tensor的列表")
            return
        
        if not isinstance(teacher_feature_maps, list) or not teacher_feature_maps:
            print("錯誤：教師模型特徵圖必須是列表格式")
            return
    
    # 設置matplotlib為非互動模式
    import matplotlib
    matplotlib.use('Agg')
    
    # 獲取批次大小
    batch_size = len(student_indices)
    print(f"處理 {batch_size} 張圖片...")
    
    # 檢查學生模型每個特徵圖的batch size是否匹配
    for fm in student_feature_maps:
        if not isinstance(fm, torch.Tensor) or fm.shape[0] != batch_size:
            print(f"錯誤：學生模型特徵圖形狀不匹配。特徵圖shape: {fm.shape}, 索引列表長度: {batch_size}")
            return
    
    # 如果有教师模型，检查教师模型特征图的batch size是否匹配
    if has_teacher:
        for fm in teacher_feature_maps:
            if not isinstance(fm, torch.Tensor) or fm.shape[0] != batch_size:
                print(f"錯誤：教師模型特徵圖形狀不匹配。特徵圖shape: {fm.shape}, 索引列表長度: {batch_size}")
                return
    
    # 處理每張圖片
    for i in range(batch_size):
        # 準備學生模型當前圖片的數據
        student_current_indices = student_indices[i].cpu().numpy().tolist()
        student_current_confidences = None if student_confidences is None else student_confidences[i].cpu().numpy().tolist()
        student_current_feature_maps = [fm[i:i+1] for fm in student_feature_maps]
        
        student_current_pred_scores = None if student_pred_scores is None else student_pred_scores[i].cpu()
        student_current_fg_mask = None if student_fg_mask is None else student_fg_mask[i].cpu()
        student_current_target_gt_idx = None if student_target_gt_idx is None else student_target_gt_idx[i].cpu()
        student_current_target_scores = None if student_target_scores is None else student_target_scores[i].cpu()
        student_current_high_conf_not_assigned = None if student_high_conf_not_assigned is None else student_high_conf_not_assigned[i].cpu()
        student_current_result = None if student_results is None else student_results[i]
        
        # 如果有教师模型，准备教师模型当前图片的数据
        teacher_current_indices = None
        teacher_current_confidences = None
        teacher_current_feature_maps = None
        teacher_current_pred_scores = None
        teacher_current_fg_mask = None
        teacher_current_target_gt_idx = None
        teacher_current_target_scores = None
        teacher_current_high_conf_not_assigned = None
        teacher_current_result = None
        
        if has_teacher:
            teacher_current_indices = teacher_indices[i].cpu().numpy().tolist()
            teacher_current_confidences = None if teacher_confidences is None else teacher_confidences[i].cpu().numpy().tolist()
            teacher_current_feature_maps = [fm[i:i+1] for fm in teacher_feature_maps]
            
            teacher_current_pred_scores = None if teacher_pred_scores is None else teacher_pred_scores[i].cpu()
            teacher_current_fg_mask = None if teacher_fg_mask is None else teacher_fg_mask[i].cpu()
            teacher_current_target_gt_idx = None if teacher_target_gt_idx is None else teacher_target_gt_idx[i].cpu()
            teacher_current_target_scores = None if teacher_target_scores is None else teacher_target_scores[i].cpu()
            teacher_current_high_conf_not_assigned = None if teacher_high_conf_not_assigned is None else teacher_high_conf_not_assigned[i].cpu()
            teacher_current_result = None if teacher_results is None else teacher_results[i]
        
        print(f"\n生成圖片 {i+1} 的特徵圖索引分佈...")
        
        # 繪製當前圖片的特徵圖可視化（学生和教师模型对比）
        _visualize_comparison(
            student_indices=student_current_indices, 
            student_feature_maps=student_current_feature_maps, 
            student_confidences=student_current_confidences, 
            student_result=student_current_result, 
            student_pred_scores=student_current_pred_scores, 
            student_fg_mask=student_current_fg_mask, 
            student_target_gt_idx=student_current_target_gt_idx, 
            student_target_scores=student_current_target_scores, 
            student_high_conf_not_assigned=student_current_high_conf_not_assigned,
            
            teacher_indices=teacher_current_indices,
            teacher_feature_maps=teacher_current_feature_maps,
            teacher_confidences=teacher_current_confidences,
            teacher_result=teacher_current_result,
            teacher_pred_scores=teacher_current_pred_scores,
            teacher_fg_mask=teacher_current_fg_mask,
            teacher_target_gt_idx=teacher_current_target_gt_idx,
            teacher_target_scores=teacher_current_target_scores,
            teacher_high_conf_not_assigned=teacher_current_high_conf_not_assigned
        )
        
        # 為每張圖片保存單獨的文件
        import os
        if os.path.exists('feature_map_indices.png'):
            os.rename('feature_map_indices.png', f'feature_map_indices_img{i+1}.png')
            print(f"已保存為: feature_map_indices_img{i+1}.png")

def _visualize_comparison(
    student_indices, student_feature_maps, student_confidences=None, student_result=None, 
    student_pred_scores=None, student_fg_mask=None, student_target_gt_idx=None, 
    student_target_scores=None, student_high_conf_not_assigned=None,
    teacher_indices=None, teacher_feature_maps=None, teacher_confidences=None, teacher_result=None,
    teacher_pred_scores=None, teacher_fg_mask=None, teacher_target_gt_idx=None,
    teacher_target_scores=None, teacher_high_conf_not_assigned=None
):
    """
    可視化單張圖片的特徵圖索引分佈，同時比較學生模型和教師模型
    
    Args:
        student_* : 學生模型的各種參數
        teacher_* : 教師模型的各種參數，全部為可選
    """
    if not student_indices:
        print("沒有學生模型索引可供可視化")
        return
    
    # 检查是否同时提供了学生和教师模型数据
    has_teacher = teacher_indices is not None and teacher_feature_maps is not None
    
    # 1. 準備特徵圖信息
    # ------------------
    student_feature_maps_info = _extract_feature_maps_info(student_feature_maps)
    teacher_feature_maps_info = None
    if has_teacher:
        teacher_feature_maps_info = _extract_feature_maps_info(teacher_feature_maps)
    
    if not student_feature_maps_info:
        print("沒有有效的學生模型特徵圖")
        return
    
    if has_teacher and not teacher_feature_maps_info:
        print("沒有有效的教師模型特徵圖")
        return
    
    n_maps = len(student_feature_maps_info)
    print(f"學生模型特徵圖預測總數: {sum(fm['predictions'] for fm in student_feature_maps_info)}")
    if has_teacher:
        print(f"教師模型特徵圖預測總數: {sum(fm['predictions'] for fm in teacher_feature_maps_info)}")
    
    # 2. 設置matplotlib
    # ------------------
    # 使用更合适的布局
    rows = 4  # 4行布局
    cols = 7  # 每行7列，适应更多图表
    
    # 创建一个更大的图表用于比较
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 5*rows), gridspec_kw={'wspace': 0.3, 'hspace': 0.4})
    axes = axes.flatten()
    
    plt.rcParams['axes.formatter.useoffset'] = False
    
    # 布局规划：
    # 第1行(0-6): 学生模型特征图置信度(0,1,2) + 学生模型预测分数(3,4,5) + 学生图像预览(6)
    # 第2行(7-13): 教师模型特征图置信度(7,8,9) + 教师模型预测分数(10,11,12) + 教师图像预览(13)
    # 第3行(14-20): 学生模型目标分数和GT索引(14,15,16) + 学生模型高置信度未分配(17,18,19) + 空白(20)
    # 第4行(21-27): 教师模型目标分数和GT索引(21,22,23) + 教师模型高置信度未分配(24,25,26) + 空白(27)
    
    # 第1行：学生模型特征图置信度 + 预测分数 + 图像预览
    # 绘制学生模型的特征图置信度
    for i, fm_info in enumerate(student_feature_maps_info):
        if i >= 3:  # 只处理前三个特征图
            break
            
        _draw_feature_map_confidences(
            ax=axes[i], 
            fm_info=fm_info, 
            indices=student_indices, 
            confidences=student_confidences,
            title_prefix="(student)",
            print_info=True
        )
    
    # 绘制学生模型的预测分数热力图
    if student_pred_scores is not None:
        for i, fm_info in enumerate(student_feature_maps_info):
            if i >= 3:  # 只处理前三个特征图
                break
                
            _draw_pred_scores(
                ax=axes[3+i], 
                fm_info=fm_info, 
                pred_scores=student_pred_scores,
                title_prefix="(student)"
            )
    
    # 绘制学生模型的目标图像和关键点
    if student_result is not None:
        _draw_original_image(
            ax=axes[6], 
            result=student_result, 
            title_prefix="(student)"
        )
    
    # 第2行：教师模型特征图置信度 + 预测分数 + 图像预览
    if has_teacher:
        # 绘制教师模型的特征图置信度
        for i, fm_info in enumerate(teacher_feature_maps_info):
            if i >= 3:  # 只处理前三个特征图
                break
                
            _draw_feature_map_confidences(
                ax=axes[7+i], 
                fm_info=fm_info, 
                indices=teacher_indices, 
                confidences=teacher_confidences,
                title_prefix="(teacher)",
                print_info=True
            )
        
        # 绘制教师模型的预测分数热力图
        if teacher_pred_scores is not None:
            for i, fm_info in enumerate(teacher_feature_maps_info):
                if i >= 3:  # 只处理前三个特征图
                    break
                    
                _draw_pred_scores(
                    ax=axes[10+i], 
                    fm_info=fm_info, 
                    pred_scores=teacher_pred_scores,
                    title_prefix="(teacher)"
                )
        
        # 绘制教师模型的目标图像和关键点
        if teacher_result is not None:
            _draw_original_image(
                ax=axes[13], 
                result=teacher_result, 
                title_prefix="(teacher)"
            )
    
    # 第3行：学生模型目标分数和GT索引 + 高置信度未分配掩码
    # 绘制学生模型的目标分数和GT索引
    if student_target_scores is not None and student_fg_mask is not None and student_target_gt_idx is not None:
        for i, fm_info in enumerate(student_feature_maps_info):
            if i >= 3:  # 只处理前三个特征图
                break
                
            _draw_target_scores(
                ax=axes[14+i],
                fm_info=fm_info,
                target_scores=student_target_scores,
                target_gt_idx=student_target_gt_idx,
                fg_mask=student_fg_mask,
                title_prefix="(student)"
            )
    
    # 绘制学生模型的高置信度未分配掩码
    if student_high_conf_not_assigned is not None and student_pred_scores is not None:
        for i, fm_info in enumerate(student_feature_maps_info):
            if i >= 3:  # 只处理前三个特征图
                break
                
            _draw_high_conf_not_assigned(
                ax=axes[17+i],
                fm_info=fm_info,
                high_conf_not_assigned=student_high_conf_not_assigned,
                pred_scores=student_pred_scores,
                title_prefix="(student)"
            )
    
    # 第4行：教师模型目标分数和GT索引 + 高置信度未分配掩码
    if has_teacher:
        # 绘制教师模型的目标分数和GT索引
        if teacher_target_scores is not None and teacher_fg_mask is not None and teacher_target_gt_idx is not None:
            for i, fm_info in enumerate(teacher_feature_maps_info):
                if i >= 3:  # 只处理前三个特征图
                    break
                    
                _draw_target_scores(
                    ax=axes[21+i],
                    fm_info=fm_info,
                    target_scores=teacher_target_scores,
                    target_gt_idx=teacher_target_gt_idx,
                    fg_mask=teacher_fg_mask,
                    title_prefix="(teacher)"
                )
        
        # 绘制教师模型的高置信度未分配掩码
        if teacher_high_conf_not_assigned is not None and teacher_pred_scores is not None:
            for i, fm_info in enumerate(teacher_feature_maps_info):
                if i >= 3:  # 只处理前三个特征图
                    break
                    
                _draw_high_conf_not_assigned(
                    ax=axes[24+i],
                    fm_info=fm_info,
                    high_conf_not_assigned=teacher_high_conf_not_assigned,
                    pred_scores=teacher_pred_scores,
                    title_prefix="(teacher)"
                )
    
    # 删除没有使用的子图
    for i, ax in enumerate(axes):
        if i in [20, 27]:  # 第3、4行的最后一列保持空白
            if ax is not None and hasattr(ax, 'set_visible'):
                ax.set_visible(False)
    
    # 保存图像
    plt.savefig('feature_map_indices.png', dpi=150, bbox_inches='tight')

def _extract_feature_maps_info(feature_maps):
    """提取特征图的关键信息"""
    feature_maps_info = []
    start_idx = 0
    
    for i, fm in enumerate(feature_maps):
        if not isinstance(fm, torch.Tensor) or fm.dim() != 4:
            continue
            
        h, w = fm.shape[2], fm.shape[3]
        grid_cells = h * w
        
        feature_maps_info.append({
            "name": f"Feature Map {i+1}",
            "size": (h, w),
            "predictions": grid_cells,
            "start_idx": start_idx,
            "end_idx": start_idx + grid_cells - 1
        })
        
        start_idx += grid_cells
    
    return feature_maps_info

def _draw_feature_map_confidences(ax, fm_info, indices, confidences=None, title_prefix="", print_info=False):
    """绘制特征图的置信度图"""
    h, w = fm_info["size"]
    img = np.zeros((h, w, 3))  # 创建RGB图像
    
    # 设置标题
    ax.set_title(f"{title_prefix} {fm_info['name']}\nConfidence ({h}×{w})", 
                fontsize=10, fontweight='bold')
    
    # 设置颜色映射
    if confidences is not None and len(confidences) > 0:
        # 使用置信度创建热力图颜色映射
        confidences = np.array(confidences)
        confidence_map = dict(zip(indices, confidences))
        cmap = plt.cm.viridis  # 从深蓝到黄色的颜色方案
    else:
        confidence_map = None
        # 每个索引一个唯一的颜色
        unique_colors = plt.cm.rainbow(np.linspace(0, 1, max(1, len(indices))))
        color_map = {idx: unique_colors[i % len(unique_colors)][:3] for i, idx in enumerate(indices)}
    
    # 查找并绘制属于此特征图的索引
    map_indices = []
    for idx in indices:
        if not (fm_info["start_idx"] <= idx <= fm_info["end_idx"]):
            continue
            
        # 计算网格位置
        rel_idx = idx - fm_info["start_idx"]
        y, x = divmod(rel_idx, w)  # 使用divmod简化计算
        
        # 检查坐标是否有效
        if not (0 <= y < h and 0 <= x < w):
            continue
        
        # 获取颜色和保存数据
        if confidence_map is not None and idx in confidence_map:
            conf = confidence_map[idx]
            map_indices.append((idx, x, y, conf))
            
            # 根据置信度设置颜色，使用固定范围0-1
            norm_conf = conf  # 直接使用置信度值，无需再次标准化
            color = cmap(norm_conf)[:3]
        else:
            map_indices.append((idx, x, y))
            color = color_map.get(idx, (1.0, 0.0, 0.0))
        
        # 在图像上标记
        img[y, x] = color
        
        # 添加矩形标记
        rect = Rectangle((x-0.5, y-0.5), 1, 1, linewidth=0.8, 
                       edgecolor='white', facecolor='none')
        ax.add_patch(rect)
    
    # 显示图像
    im = ax.imshow(img)
    
    # 为每个特征图添加侧边颜色条（不带标签）
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    if confidence_map is not None:
        plt.colorbar(im, cax=cax)
    else:
        # 如果没有置信度，使用虚拟颜色条
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.rainbow, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, cax=cax)
    
    # 设置网格和刻度
    _setup_grid(ax, h, w)
    
    # 输出找到的索引
    if print_info:
        print(f"{title_prefix} {fm_info['name']} ({h}×{w}): 找到 {len(map_indices)} 个索引")
        _print_indices_info(map_indices)

def _draw_original_image(ax, result, title_prefix=""):
    """绘制原始图像和关键点"""
    # 设置标题
    ax.set_title(f"{title_prefix} Original Image\nwith Boxes & Keypoints", fontsize=10, fontweight='bold')
    
    # 获取原始图像
    orig_img = result.orig_img
    
    # 绘制原始图像（BGR转RGB）
    ax.imshow(orig_img[:, :, ::-1])
    
    # 绘制边界框
    if hasattr(result, 'boxes') and result.boxes is not None:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            # 创建矩形
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                            linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            
            # 可选：显示置信度
            conf_text = f"{conf:.2f}"
            ax.text(x1, y1-5, conf_text, color='lime', 
                  fontsize=8, backgroundcolor='black')
    
    # 绘制关键点
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        # 直接在 ax 上绘制关键点
        draw_keypoints_on_axes(ax, result.keypoints.data, min_confidence=0.5)
    
    # 关闭不必要的刻度
    ax.set_xticks([])
    ax.set_yticks([])

def _draw_pred_scores(ax, fm_info, pred_scores, title_prefix=""):
    """绘制预测分数热力图"""
    h, w = fm_info["size"]
    
    # 从预测分数中提取该特征图的分数
    start_idx = fm_info["start_idx"]
    end_idx = fm_info["end_idx"] + 1
    scores_slice = pred_scores[start_idx:end_idx].reshape(h, w)
    
    # 设置标题
    ax.set_title(f"{title_prefix} {fm_info['name']}\nPred Scores ({h}×{w})", fontsize=10, fontweight='bold')
    
    # 显示热力图
    heat_cmap = plt.cm.hot  # 热力图颜色方案
    im = ax.imshow(scores_slice, cmap=heat_cmap, vmin=0, vmax=1)
    
    # 设置网格和刻度
    _setup_grid(ax, h, w)
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def _setup_grid(ax, h, w):
    """設置軸的網格和刻度"""
    # 設置背景和基本屬性
    ax.set_facecolor('black')
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='x', which='major', labelbottom=True)
    ax.tick_params(axis='y', which='major', labelleft=True)
    
    # 根據特徵圖大小設置網格
    if min(h, w) <= 20:  # 小尺寸特徵圖
        ax.set_xticks(np.arange(0, w, 1))
        ax.set_yticks(np.arange(0, h, 1))
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        
        # 網格線
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.5, alpha=0.9)
        ax.grid(which='major', visible=False)
        
        # 強調每5個格子
        for val in range(-5, max(h, w), 5):
            if -0.5 <= val < w-0.5:
                ax.axvline(x=val-0.5, color='w', linewidth=1.0, alpha=0.8)
            if -0.5 <= val < h-0.5:
                ax.axhline(y=val-0.5, color='w', linewidth=1.0, alpha=0.8)
    
    elif min(h, w) <= 40:  # 中等尺寸特徵圖
        ax.set_xticks(np.arange(0, w, 5))
        ax.set_yticks(np.arange(0, h, 5))
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        
        # 網格線
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.3, alpha=0.7)
        ax.grid(which='major', visible=False)
        
        # 強調每10個格子
        for val in range(0, max(h, w), 10):
            if -0.5 <= val-0.5 < w-0.5:
                ax.axvline(x=val-0.5, color='w', linewidth=0.7, alpha=0.8)
            if -0.5 <= val-0.5 < h-0.5:
                ax.axhline(y=val-0.5, color='w', linewidth=0.7, alpha=0.8)
    
    else:  # 大尺寸特徵圖
        ax.set_xticks(np.arange(0, w, 10))
        ax.set_yticks(np.arange(0, h, 10))
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        
        # 網格線
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.2, alpha=0.5)
        ax.grid(which='major', visible=False)
        
        # 強調每20個格子
        for val in range(0, max(h, w), 20):
            if -0.5 <= val-0.5 < w-0.5:
                ax.axvline(x=val-0.5, color='w', linewidth=0.5, alpha=0.8)
            if -0.5 <= val-0.5 < h-0.5:
                ax.axhline(y=val-0.5, color='w', linewidth=0.5, alpha=0.8)
    
    # 設置範圍
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)  # y軸倒置

def _print_indices_info(map_indices):
    """打印索引信息，限制輸出數量"""
    for j, map_idx in enumerate(map_indices):
        if j >= 20:
            print(f"  ... 還有 {len(map_indices) - 20} 個索引未顯示 ...")
            break
            
        if len(map_idx) > 3:  # 有置信度值
            idx, x, y, conf = map_idx
            print(f"  索引 {idx}: 位置 ({x},{y}), 置信度: {conf:.4f}")
        else:
            idx, x, y = map_idx
            print(f"  索引 {idx}: 位置 ({x},{y})")

def draw_keypoints_on_axes(ax, keypoints, min_confidence=0.5, thickness=2, circle_radius=5):
    """
    直接在指定的matplotlib軸上繪製關鍵點
    
    Args:
        ax (matplotlib.axes.Axes): 要繪製關鍵點的軸
        keypoints (numpy.ndarray | torch.Tensor): 關鍵點數據
        min_confidence (float): 最小置信度閾值
        thickness (int): 線條粗細
        circle_radius (int): 關鍵點圓圈半徑
    """
    # 轉換為numpy數組（如果不是）
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.cpu().numpy()
    
    # 關鍵點連接順序，用於繪製骨架線條
    skeleton = [  # 關鍵點之間的連接關係
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
        [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # 定義顏色
    limb_colors = ['#FF3399', '#990066', '#990033', 
                  '#CC0033', '#FF0033', '#FF3333', 
                  '#FF6633', '#FF9933', '#FF9966',
                  '#FFCC66', '#FFFF33', '#CCFF33', 
                  '#99FF33', '#66FF33', '#33FF33', 
                  '#33FF66', '#33FF99', '#33FFCC']
    
    kpt_colors = ['#FF0000', '#FF5500', '#FFAA00', '#FFFF00', 
                 '#AAFF00', '#55FF00', '#00FF00', '#00FF55', 
                 '#00FFAA', '#00FFFF', '#00AAFF', '#0055FF',
                 '#0000FF', '#5500FF', '#AA00FF', '#FF00FF',
                 '#FF00AA']
    
    # 繪製骨架
    for person_kpts in keypoints:
        # 畫骨架線
        for i, (kpt_idx1, kpt_idx2) in enumerate(skeleton):
            kpt1 = person_kpts[kpt_idx1 - 1]
            kpt2 = person_kpts[kpt_idx2 - 1]
            
            # 檢查關鍵點置信度（如果有的話）
            if len(kpt1) > 2 and len(kpt2) > 2:
                conf1 = kpt1[2] if len(kpt1) > 2 else 1.0
                conf2 = kpt2[2] if len(kpt2) > 2 else 1.0
                if conf1 > min_confidence and conf2 > min_confidence:
                    color = limb_colors[i % len(limb_colors)]
                    ax.plot([kpt1[0], kpt2[0]], [kpt1[1], kpt2[1]], 
                           color=color, linewidth=thickness)
        
        # 畫關鍵點
        for i, kpt in enumerate(person_kpts):
            if len(kpt) > 2:
                conf = kpt[2]
                if conf > min_confidence:  # 只繪製高置信度的關鍵點
                    color = kpt_colors[i % len(kpt_colors)]
                    ax.scatter(kpt[0], kpt[1], s=circle_radius**2, 
                              color=color, zorder=2)

def _draw_high_conf_not_assigned(ax, fm_info, high_conf_not_assigned, pred_scores, title_prefix=""):
    """绘制高置信度未分配掩码热力图"""
    h, w = fm_info["size"]
    
    # 从数据中提取该特征图的信息
    start_idx = fm_info["start_idx"]
    end_idx = fm_info["end_idx"] + 1
    
    # 提取当前特征图的高置信度未分配掩码
    high_conf_not_assigned_slice = high_conf_not_assigned[start_idx:end_idx].reshape(h, w)
    
    # 提取预测分数
    pred_scores_slice = pred_scores[start_idx:end_idx].reshape(h, w)
    
    # 设置标题
    ax.set_title(f"{title_prefix} {fm_info['name']}\nHigh Conf Not Assigned ({h}×{w})", fontsize=10, fontweight='bold')
    
    # 创建高置信度未分配掩码的带有分数值的热力图
    masked_high_conf_scores = np.zeros((h, w))
    masked_high_conf_scores[:] = np.nan  # 设置非高置信度未分配区域为NaN
    
    # 只显示高置信度未分配的区域，其值为原始预测分数
    masked_high_conf_scores[high_conf_not_assigned_slice] = pred_scores_slice[high_conf_not_assigned_slice]
    
    # 显示热力图
    high_conf_cmap = plt.cm.cool  # 使用cool颜色方案以区分其他图表
    im = ax.imshow(masked_high_conf_scores, cmap=high_conf_cmap, vmin=0, vmax=1)
    
    # 设置网格和刻度
    _setup_grid(ax, h, w)
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    # 计算并显示高置信度未分配预测的数量
    high_conf_count = high_conf_not_assigned_slice.sum().item()
    print(f"{title_prefix} {fm_info['name']}: {high_conf_count} 个高置信度未分配预测")

def _draw_target_scores(ax, fm_info, target_scores, target_gt_idx, fg_mask, title_prefix=""):
    """绘制目标分数和GT索引热力图"""
    h, w = fm_info["size"]
    
    # 从目标数据中提取该特征图的数据
    start_idx = fm_info["start_idx"]
    end_idx = fm_info["end_idx"] + 1
    target_scores_slice = target_scores[start_idx:end_idx].reshape(h, w)
    target_gt_idx_slice = target_gt_idx[start_idx:end_idx].reshape(h, w)
    fg_mask_slice = fg_mask[start_idx:end_idx].reshape(h, w)
    
    # 设置标题
    ax.set_title(f"{title_prefix} {fm_info['name']}\nTarget Scores & GT Idx ({h}×{w})", fontsize=10, fontweight='bold')
    
    # 创建热力图
    masked_target_scores = np.zeros((h, w))
    masked_target_scores[:] = np.nan  # 设置非前景区域为NaN，使其透明
    masked_target_scores[fg_mask_slice] = target_scores_slice[fg_mask_slice]
    
    # 使用plasma颜色方案
    target_cmap = plt.cm.plasma
    im = ax.imshow(masked_target_scores, cmap=target_cmap, vmin=0, vmax=1)
    
    # 在前景区域添加GT索引标签
    for y in range(h):
        for x in range(w):
            if fg_mask_slice[y, x]:
                gt_idx = int(target_gt_idx_slice[y, x])
                score = float(masked_target_scores[y, x])
                # 根据背景颜色选择文字颜色
                color = 'white' if score < 0.7 else 'black'
                ax.text(x, y, f"{gt_idx}", ha='center', va='center', 
                       color=color, fontsize=7, fontweight='bold')
    
    # 设置网格和刻度
    _setup_grid(ax, h, w)
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

