```
git clone https://github.com/comus/ultralytics.git
cd ultralytics
conda create -n yolo python=3.11
conda activate yolo
pip install ultralytics psutil requests seaborn pandas numpy scikit-learn pycocotools mlx onnxruntime tqdm pyyaml opencv-python matplotlib
pip install torch torchvision
```

運行 predict (我自己弄的多模型比較)

查看 predict.py, 修改圖片路徑

```
python predict.py
```

官方的 predict

查看 predict_single.py
- 修改圖片路徑
- 修改模型路徑

```
python predict_single.py
```

驗證

查看 val.py
- 修改模型路徑

```
python val.py
```

模型

- 模型檔放在 models 資料夾
- models/official 是官方的模型
- models/yolo11n-pose 是我訓練的模型
  - 用官方的 yolo11n-pose 沒有預訓練權重的架構，重新訓練，數據集是 coco-pose
  - models/yolo11n-pose/train
    - 第一次訓練，只訓練了 30 個 epoch
    - 訓練參數看 train_yolo11n_pose.py
    - weights
      - 訓練好的模型權重
    - val.txt
      - 驗證模型的數據
    - results.csv
      - 訓練過程的結果
  - models/yolo11n-pose/train_stage2
    - 第二次訓練，訓練了 70 個 epoch
    - 訓練參數看 train_yolo11n_pose_stage2.py
    - weights
      - 訓練好的模型權重
    - val.txt
      - 驗證模型的數據 (我沒弄，自己運行 val.py 驗證)
    - results.csv
      - 訓練過程的結果
- models/yolo11n-pose-distill1 是我訓練的模型
  - 用官方 yolo11n-pose  的預訓練權重，重新訓練，數據集是 coco-pose，用了第一個版本的蒸餾損失函數
    - 損失函數請找 pose_loss.py
  - models/yolo11n-pose-distill1/train
    - 第一次訓練，只訓練了 15 個 epoch
    - 訓練參數看 train_yolo11n_pose_distill1.py
    - weights
      - 訓練好的模型權重
    - val.txt
      - 驗證模型的數據
    - results.csv
      - 訓練過程的結果
  - models/yolo11n-pose-distill1/train_stage2
    - 第二次訓練，訓練了 25 個 epoch
    - 訓練參數看 train_yolo11n_pose_distill1_stage2.py
    - weights
      - 訓練好的模型權重
    - val.txt
      - 驗證模型的數據 (我沒弄，自己運行 val.py 驗證)
    - results.csv
      - 訓練過程的結果
- models/yolo11n-pose-distill2 是我訓練的模型
  - 用官方 yolo11n-pose 的預訓練權重，重新訓練，數據集是 coco-pose，用了第二個版本的蒸餾損失函數
    - 損失函數請找 pose_loss.py
  - models/yolo11n-pose-distill2/train
    - 第一次訓練，只訓練了 15 個 epoch
    - 訓練參數看 train_yolo11n_pose_distill2.py
    - weights
      - 訓練好的模型權重
    - val.txt
      - 驗證模型的數據 (我沒弄，自己運行 val.py 驗證)
    - results.csv
      - 訓練過程的結果
  - models/yolo11n-pose-distill2/train_stage2
    - 第二次訓練，訓練了 25 個 epoch
    - 訓練參數看 train_yolo11n_pose_distill2_stage2.py
    - weights
      - 訓練好的模型權重
    - val.txt
      - 驗證模型的數據
    - results.csv
      - 訓練過程的結果

訓練詳情

- log_yolo11n_pose.txt
  - 第一次訓練詳情
- log_yolo11n_pose_stage2.txt
  - 第二次訓練詳情
- log_distill1.txt
  - 第一次訓練詳情
- log_distill1_stage2.txt
  - 第二次訓練詳情
- log_distill2.txt
  - 第一次訓練詳情
- log_distill2_stage2.txt
  - 第二次訓練詳情

---

第三次訓練，進行中，完成會更新在這裡




