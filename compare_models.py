import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from ultralytics import YOLO
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torchvision.transforms as transforms
import random
import argparse

# 設置隨機種子，確保結果可重現
RANDOM_SEED = 42  # 可以更改為任何整數
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)  # 如果使用多個GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print(f"Random seed set to: {RANDOM_SEED}")

# 設置路徑
VAL_IMAGES_PATH = "/Users/region/yolo11-pose-distiller/datasets/coco-pose/images/val2017"
YOLO_MODEL_PATH = "yolo11n-pose.pt"  # 請確保路徑正確
GDE_MODEL_PATH = "super_phase5_1_v2/weights/best.pt"  # 請確保路徑正確
OUTPUT_DIR = "model_comparison_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
    def register_hook(self, layer_name, layer):
        def hook(module, input, output):
            # 確保輸出是可處理的類型
            if isinstance(output, torch.Tensor):
                self.features[layer_name] = output.detach().cpu()
            elif isinstance(output, tuple) and all(isinstance(o, torch.Tensor) for o in output):
                self.features[layer_name] = output[0].detach().cpu()
            
        handle = layer.register_forward_hook(hook)
        self.hooks.append(handle)
        
    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def clear_features(self):
        self.features = {}

def get_layer_by_id(model, layer_id):
    """通過索引獲取模型的層"""
    try:
        return model.model.model[layer_id]
    except (IndexError, AttributeError):
        return None

def list_model_layers(model):
    """列出模型的所有層結構"""
    layers = []
    try:
        for i, m in enumerate(model.model.model):
            layers.append((i, type(m).__name__, m))
    except AttributeError:
        print("無法訪問模型層")
    return layers

def load_models():
    print("Loading models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    gde_model = YOLO(GDE_MODEL_PATH)
    
    print("YOLO model layers:")
    yolo_layers = list_model_layers(yolo_model)
    for i, type_name, _ in yolo_layers[:10]:  # 只顯示前10層
        print(f"Layer {i}: {type_name}")
    print("...")
    
    print("\nGDE model layers:")
    gde_layers = list_model_layers(gde_model)
    for i, type_name, _ in gde_layers[:10]:  # 只顯示前10層
        print(f"Layer {i}: {type_name}")
    print("...")
    
    return yolo_model, gde_model, yolo_layers, gde_layers

def identify_comparable_layers(yolo_layers, gde_layers):
    """識別兩個模型中可比較的層"""
    paired_layers = []
    
    # 按類型分組配對相似的層
    yolo_by_type = {}
    gde_by_type = {}
    
    # 排除一些通用層 (如 Concat, Upsample 等)
    excluded_types = {"Concat", "nn.Upsample", "Upsample", "nn.modules.upsampling.Upsample"}
    
    for i, type_name, layer in yolo_layers:
        if type_name not in excluded_types:
            if type_name not in yolo_by_type:
                yolo_by_type[type_name] = []
            yolo_by_type[type_name].append((i, layer))
    
    for i, type_name, layer in gde_layers:
        if type_name not in excluded_types:
            # 特殊處理: 配對 C3k2 與 C3k2_Ghost, C3k2_DFFM
            mapped_type = type_name
            if type_name in ["C3k2_Ghost", "C3k2_DFFM"]:
                mapped_type = "C3k2"  # 映射到相應的YOLO層類型
                
            if mapped_type not in gde_by_type:
                gde_by_type[mapped_type] = []
            gde_by_type[mapped_type].append((i, layer))
    
    # 尋找共同的層類型
    common_types = set(yolo_by_type.keys()) & set(gde_by_type.keys())
    
    print(f"Found common layer types: {common_types}")
    
    # 為每種類型配對層
    for type_name in common_types:
        yolo_type_layers = yolo_by_type[type_name]
        gde_type_layers = gde_by_type[type_name]
        
        min_count = min(len(yolo_type_layers), len(gde_type_layers))
        
        # 配對相同位置的層
        for i in range(min_count):
            yolo_idx, yolo_layer = yolo_type_layers[i]
            gde_idx, gde_layer = gde_type_layers[i]
            paired_layers.append((yolo_idx, gde_idx, yolo_layer, gde_layer, type_name))
    
    # 特殊處理: 配對 GDEPose / Pose 層
    if "Pose" in yolo_by_type and "GDEPose" in gde_by_type:
        y_idx, y_layer = yolo_by_type["Pose"][0]
        g_idx, g_layer = gde_by_type["GDEPose"][0]
        paired_layers.append((y_idx, g_idx, y_layer, g_layer, "Pose/GDEPose"))
    
    # 根據層在模型中的位置排序
    paired_layers.sort(key=lambda x: x[0])
    
    return paired_layers

def process_images(yolo_model, gde_model, paired_layers, num_images=50):
    """處理圖像並提取特徵"""
    # 獲取圖像列表
    image_files = os.listdir(VAL_IMAGES_PATH)
    image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if num_images > 0 and num_images < len(image_files):
        image_files = image_files[:num_images]
    
    # 設置特徵提取器
    yolo_extractor = FeatureExtractor(yolo_model)
    gde_extractor = FeatureExtractor(gde_model)
    
    # 為每一對層註冊鉤子
    for yolo_idx, gde_idx, yolo_layer, gde_layer, layer_type in paired_layers:
        yolo_extractor.register_hook(f"{layer_type}_{yolo_idx}", yolo_layer)
        gde_extractor.register_hook(f"{layer_type}_{gde_idx}", gde_layer)
    
    all_similarities = {f"{layer_type}_{yolo_idx}_{gde_idx}": [] for yolo_idx, gde_idx, _, _, layer_type in paired_layers}
    
    # 處理每張圖像
    for img_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(VAL_IMAGES_PATH, img_file)
        
        # 清除之前的特徵
        yolo_extractor.clear_features()
        gde_extractor.clear_features()
        
        # 使用兩個模型處理圖像
        yolo_results = yolo_model(img_path, verbose=False)
        gde_results = gde_model(img_path, verbose=False)
        
        # 計算每一對層的相似度
        for yolo_idx, gde_idx, _, _, layer_type in paired_layers:
            yolo_key = f"{layer_type}_{yolo_idx}"
            gde_key = f"{layer_type}_{gde_idx}"
            
            if yolo_key in yolo_extractor.features and gde_key in gde_extractor.features:
                yolo_feat = yolo_extractor.features[yolo_key]
                gde_feat = gde_extractor.features[gde_key]
                
                # 確保特徵維度相符，如果不相符則嘗試適配
                try:
                    if yolo_feat.shape != gde_feat.shape:
                        # 如果只是批次大小不同，可以選第一個元素
                        if len(yolo_feat.shape) > 1 and len(gde_feat.shape) > 1 and yolo_feat.shape[1:] == gde_feat.shape[1:]:
                            yolo_feat = yolo_feat[0:1]
                            gde_feat = gde_feat[0:1]
                            
                    # 將特徵轉為扁平的numpy數組
                    yf_flat = yolo_feat.view(yolo_feat.size(0), -1).numpy()
                    gf_flat = gde_feat.view(gde_feat.size(0), -1).numpy()
                    
                    # 計算相關性
                    corr_matrix = np.corrcoef(yf_flat.flatten(), gf_flat.flatten())
                    pearson_corr = corr_matrix[0, 1]
                    
                    # 計算餘弦相似度
                    cos_sim = np.dot(yf_flat.flatten(), gf_flat.flatten()) / (
                        np.linalg.norm(yf_flat.flatten()) * np.linalg.norm(gf_flat.flatten()) + 1e-8
                    )
                    
                    # 計算MSE
                    mse = np.mean((yf_flat.flatten() - gf_flat.flatten()) ** 2)
                    
                    all_similarities[f"{layer_type}_{yolo_idx}_{gde_idx}"].append({
                        'image': img_file,
                        'pearson_corr': pearson_corr,
                        'cosine_sim': cos_sim,
                        'mse': mse
                    })
                except Exception as e:
                    print(f"Error processing features for layer {yolo_key}/{gde_key}: {e}")
    
    # 清除鉤子
    yolo_extractor.clear_hooks()
    gde_extractor.clear_hooks()
    
    return all_similarities, yolo_extractor, gde_extractor

def analyze_similarities(all_similarities, paired_layers):
    """分析特徵相似度並產生統計數據"""
    results = {}
    
    for yolo_idx, gde_idx, _, _, layer_type in paired_layers:
        key = f"{layer_type}_{yolo_idx}_{gde_idx}"
        
        if key in all_similarities and all_similarities[key]:
            stats = all_similarities[key]
            
            # 計算平均和標準差
            avg_pearson = np.mean([s['pearson_corr'] for s in stats if not np.isnan(s['pearson_corr'])])
            std_pearson = np.std([s['pearson_corr'] for s in stats if not np.isnan(s['pearson_corr'])])
            
            avg_cosine = np.mean([s['cosine_sim'] for s in stats if not np.isnan(s['cosine_sim'])])
            std_cosine = np.std([s['cosine_sim'] for s in stats if not np.isnan(s['cosine_sim'])])
            
            avg_mse = np.mean([s['mse'] for s in stats if not np.isnan(s['mse'])])
            std_mse = np.std([s['mse'] for s in stats if not np.isnan(s['mse'])])
            
            results[key] = {
                'yolo_idx': yolo_idx,
                'gde_idx': gde_idx,
                'layer_type': layer_type,
                'avg_pearson': avg_pearson,
                'std_pearson': std_pearson,
                'avg_cosine': avg_cosine,
                'std_cosine': std_cosine,
                'avg_mse': avg_mse,
                'std_mse': std_mse,
                'samples': len(stats)
            }
    
    return results

def visualize_results(analysis_results, paired_layers):
    """可視化分析結果"""
    if not analysis_results:
        print("沒有可視化的結果")
        return pd.DataFrame()
    
    # 準備數據
    layer_keys = list(analysis_results.keys())
    layer_names = [f"{results['layer_type']}\nYOLO_{results['yolo_idx']}/GDE_{results['gde_idx']}" 
                  for key, results in analysis_results.items()]
    pearson_values = [results['avg_pearson'] for key, results in analysis_results.items()]
    cosine_values = [results['avg_cosine'] for key, results in analysis_results.items()]
    mse_values = [results['avg_mse'] for key, results in analysis_results.items()]
    
    # 1. 相似度條形圖
    plt.figure(figsize=(18, 10))
    
    # Pearson相關係數
    plt.subplot(3, 1, 1)
    bars = plt.bar(range(len(layer_names)), pearson_values, color='skyblue')
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=0.7, color='g', linestyle='-', alpha=0.3)
    plt.ylabel('Pearson Correlation')
    plt.title('Average Pearson Correlation Between YOLO and GDE Layers')
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    
    # 為每個柱子添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 餘弦相似度
    plt.subplot(3, 1, 2)
    bars = plt.bar(range(len(layer_names)), cosine_values, color='lightgreen')
    plt.axhline(y=0.5, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=0.7, color='g', linestyle='-', alpha=0.3)
    plt.ylabel('Cosine Similarity')
    plt.title('Average Cosine Similarity Between YOLO and GDE Layers')
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # MSE
    plt.subplot(3, 1, 3)
    bars = plt.bar(range(len(layer_names)), mse_values, color='salmon')
    plt.ylabel('Mean Squared Error')
    plt.title('Average MSE Between YOLO and GDE Layers')
    plt.xticks(range(len(layer_names)), layer_names, rotation=90)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "layer_similarity_metrics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 網絡深度相關性圖
    plt.figure(figsize=(15, 8))
    
    # 按層索引排序
    sorted_indices = [i for i in range(len(layer_keys))]
    sorted_indices.sort(key=lambda i: analysis_results[layer_keys[i]]['yolo_idx'])
    
    sorted_pearson = [pearson_values[i] for i in sorted_indices]
    sorted_cosine = [cosine_values[i] for i in sorted_indices]
    sorted_layer_names = [layer_names[i] for i in sorted_indices]
    
    plt.plot(range(len(sorted_layer_names)), sorted_pearson, 'o-', label='Pearson Correlation', color='blue')
    plt.plot(range(len(sorted_layer_names)), sorted_cosine, 's-', label='Cosine Similarity', color='green')
    plt.xticks(range(len(sorted_layer_names)), sorted_layer_names, rotation=90)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3, label='Threshold 0.5')
    plt.axhline(y=0.7, color='purple', linestyle='--', alpha=0.3, label='Threshold 0.7')
    plt.title('Layer Similarity Through Network Depth')
    plt.ylabel('Similarity Score')
    plt.xlabel('Model Layers')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "depth_similarity.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 熱力圖 - 如果層數足夠多
    if len(layer_keys) > 5:
        similarity_matrix = np.zeros((len(layer_keys), 3))
        for i, key in enumerate(layer_keys):
            similarity_matrix[i, 0] = analysis_results[key]['avg_pearson']
            similarity_matrix[i, 1] = analysis_results[key]['avg_cosine']
            similarity_matrix[i, 2] = analysis_results[key]['avg_mse']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                    annot=True, 
                    fmt=".3f", 
                    xticklabels=['Pearson', 'Cosine', 'MSE'],
                    yticklabels=layer_names,
                    cmap='viridis')
        plt.title('Layer Similarity Metrics Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "similarity_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. 保存數值結果到CSV
    results_df = pd.DataFrame([
        {
            'YOLO Index': results['yolo_idx'],
            'GDE Index': results['gde_idx'],
            'Layer Type': results['layer_type'],
            'Avg Pearson': results['avg_pearson'],
            'Std Pearson': results['std_pearson'],
            'Avg Cosine': results['avg_cosine'],
            'Std Cosine': results['std_cosine'],
            'Avg MSE': results['avg_mse'],
            'Std MSE': results['std_mse'],
            'Samples': results['samples']
        }
        for key, results in analysis_results.items()
    ])
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, "layer_similarity_stats.csv"), index=False)
    
    return results_df

def visualize_feature_maps(yolo_model, gde_model, paired_layers, yolo_extractor, gde_extractor, num_images=3):
    """可視化並比較兩個模型的特徵圖"""
    # 獲取圖像列表並排序，確保順序一致
    image_files = os.listdir(VAL_IMAGES_PATH)
    image_files = [f for f in image_files if f.endswith(('.jpg', '.jpeg', '.png'))]
    image_files.sort()  # 排序以確保順序一致
    
    # 使用固定的隨機種子選擇圖像
    # np.random.seed(RANDOM_SEED)  # 種子已在腳本開始時設置
    if len(image_files) > num_images:
        selected_indices = np.random.choice(len(image_files), num_images, replace=False)
        selected_images = [image_files[i] for i in selected_indices]
        print(f"Selected images for visualization: {selected_images}")
    else:
        selected_images = image_files
        print(f"Using all {len(selected_images)} available images for visualization")
    
    # 使用所有配對層進行可視化
    visualize_layers = paired_layers
    
    # 重新註冊鉤子
    yolo_extractor.clear_hooks()
    gde_extractor.clear_hooks()
    
    for yolo_idx, gde_idx, yolo_layer, gde_layer, layer_type in visualize_layers:
        yolo_extractor.register_hook(f"{layer_type}_{yolo_idx}", yolo_layer)
        gde_extractor.register_hook(f"{layer_type}_{gde_idx}", gde_layer)
    
    for img_file in selected_images:
        img_path = os.path.join(VAL_IMAGES_PATH, img_file)
        
        # 清除之前的特徵
        yolo_extractor.clear_features()
        gde_extractor.clear_features()
        
        # 處理圖像
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 運行模型
        yolo_model(img_path, verbose=False)
        gde_model(img_path, verbose=False)
        
        # 為每個層創建特徵圖可視化
        for yolo_idx, gde_idx, _, _, layer_type in visualize_layers:
            yolo_key = f"{layer_type}_{yolo_idx}"
            gde_key = f"{layer_type}_{gde_idx}"
            
            if yolo_key in yolo_extractor.features and gde_key in gde_extractor.features:
                yolo_feat = yolo_extractor.features[yolo_key]
                gde_feat = gde_extractor.features[gde_key]
                
                # 確保有特徵可視化
                if yolo_feat is None or gde_feat is None:
                    continue
                    
                # 將特徵轉換為可視化格式
                def prepare_feature_for_viz(feature):
                    # 如果特徵是4D張量 [batch, channels, height, width]
                    if len(feature.shape) == 4:
                        # Batch內平均
                        if feature.shape[0] > 1:
                            feature = feature.mean(0, keepdim=True)
                            
                        # 計算通道均值得到空間特徵圖
                        feature_map = feature[0].mean(0).numpy()
                        
                        # 將特徵圖標準化到 [0,1] 範圍
                        if feature_map.max() > feature_map.min():
                            feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())
                        feature_map = np.uint8(feature_map * 255)
                        
                        # 調整大小以便於顯示
                        feature_map = cv2.resize(feature_map, (224, 224))
                        return feature_map
                    else:
                        # 如果不是4D張量，則尋找其他維度
                        print(f"不支援的特徵形狀: {feature.shape}")
                        return None
                
                try:
                    yolo_viz = prepare_feature_for_viz(yolo_feat)
                    gde_viz = prepare_feature_for_viz(gde_feat)
                    
                    if yolo_viz is not None and gde_viz is not None:
                        # 創建彩色熱力圖
                        yolo_heatmap = cv2.applyColorMap(yolo_viz, cv2.COLORMAP_JET)
                        gde_heatmap = cv2.applyColorMap(gde_viz, cv2.COLORMAP_JET)
                        
                        # 將原始圖像調整為相同大小
                        resized_img = cv2.resize(original_img, (224, 224))
                        
                        # 計算差異圖
                        diff_map = cv2.absdiff(yolo_viz, gde_viz)
                        diff_heatmap = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
                        
                        # 計算相似度指標
                        yolo_flat = yolo_viz.flatten().astype(float) / 255.0
                        gde_flat = gde_viz.flatten().astype(float) / 255.0
                        
                        cos_sim = np.dot(yolo_flat, gde_flat) / (
                            np.linalg.norm(yolo_flat) * np.linalg.norm(gde_flat) + 1e-8
                        )
                        
                        corr = np.corrcoef(yolo_flat, gde_flat)[0, 1]
                        
                        # 創建組合圖像
                        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                        axes[0, 0].imshow(resized_img)
                        axes[0, 0].set_title("Original Image")
                        axes[0, 0].axis('off')
                        
                        axes[0, 1].imshow(cv2.cvtColor(yolo_heatmap, cv2.COLOR_BGR2RGB))
                        axes[0, 1].set_title(f"YOLO Feature (Layer {yolo_idx})")
                        axes[0, 1].axis('off')
                        
                        axes[1, 0].imshow(cv2.cvtColor(gde_heatmap, cv2.COLOR_BGR2RGB))
                        axes[1, 0].set_title(f"GDE Feature (Layer {gde_idx})")
                        axes[1, 0].axis('off')
                        
                        axes[1, 1].imshow(cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB))
                        axes[1, 1].set_title(f"Difference Map\nCosine: {cos_sim:.3f}, Corr: {corr:.3f}")
                        axes[1, 1].axis('off')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_file)[0]}_{layer_type}_{yolo_idx}_{gde_idx}_featuremaps.png"), dpi=300, bbox_inches='tight')
                        plt.close()
                except Exception as e:
                    print(f"Error visualizing features for {yolo_key}/{gde_key}: {e}")
    
    # 清除鉤子
    yolo_extractor.clear_hooks()
    gde_extractor.clear_hooks()

def main(num_images=50, num_viz_images=3):
    # 載入模型並獲取層列表
    yolo_model, gde_model, yolo_layers, gde_layers = load_models()
    
    # 識別可比較的層
    paired_layers = identify_comparable_layers(yolo_layers, gde_layers)
    print(f"識別出 {len(paired_layers)} 對可比較的層")
    for yolo_idx, gde_idx, _, _, layer_type in paired_layers[:10]:  # 只顯示前10對
        print(f"YOLO Layer {yolo_idx} <-> GDE Layer {gde_idx} (Type: {layer_type})")
    if len(paired_layers) > 10:
        print(f"... 及其他 {len(paired_layers) - 10} 對層")
    
    # 處理圖像並提取特徵
    print(f"處理圖像並提取特徵 (使用 {num_images} 張圖像)...")
    all_similarities, yolo_extractor, gde_extractor = process_images(yolo_model, gde_model, paired_layers, num_images=num_images)
    
    # 分析特徵相似度
    print("分析特徵相似度...")
    analysis_results = analyze_similarities(all_similarities, paired_layers)
    
    # 可視化結果
    print("生成可視化結果...")
    results_df = visualize_results(analysis_results, paired_layers)
    print(results_df)
    
    # 可視化特徵圖
    print(f"生成特徵圖可視化 (使用 {num_viz_images} 張圖像)...")
    visualize_feature_maps(yolo_model, gde_model, paired_layers, yolo_extractor, gde_extractor, num_images=num_viz_images)
    
    print(f"分析完成。結果保存在 {OUTPUT_DIR} 目錄")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="比較YOLO和GDE模型的特徵")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="隨機種子")
    parser.add_argument("--num-images", type=int, default=50, help="用於分析的圖像數量")
    parser.add_argument("--num-viz-images", type=int, default=3, help="用於可視化的圖像數量")
    args = parser.parse_args()
    
    # 重新設置隨機種子（如果與默認值不同）
    if args.seed != RANDOM_SEED:
        print(f"更新隨機種子: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
    
    main(num_images=args.num_images, num_viz_images=args.num_viz_images)