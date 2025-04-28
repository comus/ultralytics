import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import imagehash
from tqdm import tqdm

def get_image_hash(image_path):
    """Calculate perceptual hash of image"""
    try:
        img = Image.open(image_path)
        # 使用感知哈希，比較不受圖像變形的影響
        hash_value = imagehash.phash(img)
        return hash_value
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def calculate_similarity(hash1, hash2):
    """Calculate similarity between two hashes (0 to 100)"""
    if hash1 is None or hash2 is None:
        return 0
    
    # 計算漢明距離（相異位元的數量）
    hamming_distance = hash1 - hash2
    # 轉換為相似度百分比 (64位哈希值)
    max_bits = 64
    similarity = 100 - (hamming_distance / max_bits * 100)
    return similarity

def create_comparison_image(train_path, val_path, output_path, similarity):
    """創建兩張圖片的對比圖"""
    try:
        # 讀取兩張圖片
        train_img = Image.open(train_path)
        val_img = Image.open(val_path)
        
        # 調整大小以匹配
        max_width = 600  # 每張圖片的最大寬度
        train_width, train_height = train_img.size
        val_width, val_height = val_img.size
        
        # 計算縮放比例
        train_ratio = min(max_width / train_width, 400 / train_height)
        val_ratio = min(max_width / val_width, 400 / val_height)
        
        # 調整大小
        new_train_width = int(train_width * train_ratio)
        new_train_height = int(train_height * train_ratio)
        new_val_width = int(val_width * val_ratio)
        new_val_height = int(val_height * val_ratio)
        
        train_img = train_img.resize((new_train_width, new_train_height), Image.Resampling.LANCZOS)
        val_img = val_img.resize((new_val_width, new_val_height), Image.Resampling.LANCZOS)
        
        # 創建新圖片（水平放置兩張圖片）
        total_width = new_train_width + new_val_width
        max_height = max(new_train_height, new_val_height) + 40  # 增加標題的空間
        
        comparison_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        
        # 將兩張圖片貼上
        comparison_img.paste(train_img, (0, 40))
        comparison_img.paste(val_img, (new_train_width, 40))
        
        # 添加標題
        draw = ImageDraw.Draw(comparison_img)
        try:
            # 嘗試加載字體，如果失敗則使用默認
            font = ImageFont.truetype("Arial", 16)
        except IOError:
            font = ImageFont.load_default()
        
        draw.text((10, 10), f"Train: {Path(train_path).name}", (0, 0, 0), font=font)
        draw.text((new_train_width + 10, 10), f"Val: {Path(val_path).name}", (0, 0, 0), font=font)
        draw.text((total_width // 2 - 50, 10), f"相似度: {similarity:.2f}%", (255, 0, 0), font=font)
        
        # 保存圖片
        comparison_img.save(output_path)
        return True
    except Exception as e:
        print(f"創建對比圖時出錯: {e}")
        return False

def main():
    # 定義路徑
    train_dir = Path('datasets/yoga82/images/train')
    val_dir = Path('datasets/yoga82/images/val')
    output_dir = Path('similar_images_comparison')
    
    # 創建輸出目錄
    output_dir.mkdir(exist_ok=True)
    
    # 相似度閾值 (%)
    similarity_threshold = 80
    
    # 常見的圖像擴展名
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # 獲取訓練目錄中的所有圖像文件
    train_files = []
    for ext in image_extensions:
        train_files.extend(list(train_dir.glob(f'*{ext}')))
        train_files.extend(list(train_dir.glob(f'*{ext.upper()}')))
    
    print(f"找到 {len(train_files)} 個訓練圖像文件")
    
    # 獲取驗證目錄中的所有圖像文件
    val_files = []
    for ext in image_extensions:
        val_files.extend(list(val_dir.glob(f'*{ext}')))
        val_files.extend(list(val_dir.glob(f'*{ext.upper()}')))
    
    print(f"找到 {len(val_files)} 個驗證圖像文件")
    
    # 為驗證文件創建字典以便快速查找
    val_files_dict = {file_path.name: file_path for file_path in val_files}
    
    # 為驗證文件計算哈希值
    print("正在計算驗證圖像的哈希值...")
    val_hashes = {}
    for file_path in tqdm(val_files):
        img_hash = get_image_hash(file_path)
        if img_hash is not None:
            val_hashes[file_path.name] = img_hash
    
    # 檢查相似圖像
    print("檢查相似圖像...")
    similar_images = []
    
    for train_file in tqdm(train_files):
        train_hash = get_image_hash(train_file)
        if train_hash is None:
            continue
            
        for val_name, val_hash in val_hashes.items():
            similarity = calculate_similarity(train_hash, val_hash)
            if similarity >= similarity_threshold:
                similar_images.append((train_file, val_name, similarity))
    
    # 按相似度排序結果（從高到低）
    similar_images.sort(key=lambda x: x[2], reverse=True)
    
    # 創建對比圖
    print("正在創建對比圖...")
    created_count = 0
    
    for train_file, val_name, similarity in tqdm(similar_images):
        val_file = val_files_dict[val_name]
        output_filename = f"similar_{created_count}_{train_file.stem}_{val_name}_{similarity:.2f}.jpg"
        output_path = output_dir / output_filename
        
        if create_comparison_image(train_file, val_file, output_path, similarity):
            created_count += 1
    
    # 打印結果
    if similar_images:
        print(f"\n找到 {len(similar_images)} 個相似度達到 {similarity_threshold}% 或更高的圖像對")
        print(f"創建了 {created_count} 張對比圖，保存在 {output_dir} 目錄中")
    else:
        print(f"\n沒有找到相似度達到 {similarity_threshold}% 或更高的圖像對。")

if __name__ == "__main__":
    main() 