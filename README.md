# 本身 yolo11n pose 模型的精度就已經好好

```
yolo val pose data=coco-pose.yaml device=0 model=yolo11n-pose.p
```

pycocotools 評估 yolo11n-pose.pt 的精度是 (pose) mAP50=0.806, mAP50-95=0.500

和官方的說法差不多

https://docs.ultralytics.com/tasks/pose/#models

<details>
<summary>官方模型驗證</summary>

```
Evaluating pycocotools mAP using /root/autodl-tmp/ultralytics/runs/pose/val5/predictions.json and /root/autodl-tmp/datasets/coco-pose/annotations/person_keypoints_val2017.json...
loading annotations into memory...
Done (t=0.16s)
creating index...
index created!
Loading and preparing results...
DONE (t=1.58s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6.43s).
Accumulating evaluation results...
DONE (t=0.52s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.719
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.588
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.657
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.210
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.749
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.865
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=6.03s).
Accumulating evaluation results...
DONE (t=0.15s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.806
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.534
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.436
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.859
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.622
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.689
Results saved to /root/autodl-tmp/ultralytics/runs/pose/val5
```

</details>

經過無數次嘗試，蒸餾大模型到小模型，最後的結果是 yolo11n-pose 的精度沒有任何改變，或至還跌了少少

因此結論是 yolo11 官方的模型已經是最優解

這是我最最最後嘗試改善 yolo11n-pose 的過程 (有關蒸餾損失函數也是在裡面)

https://poe.com/s/1jVchmCmAfe9msarZt8l

https://poe.com/s/W5uKVMNvZuVsBhVtFO2U

# 另外的方向: 自行建立另一小模型，再蒸餾

主打輕量級模型，我訓練速度快，訓練時間短。

低精度的模型也有場景的

https://poe.com/s/OWI97xqrv3108DApozqJ

# 方向一，先訓練另一小模型，再蒸餾大模型到小模型

我先依照不同模型架構，訓練出兩個模型

建立模型 yaml 的方法我也是問 AI 的

https://poe.com/s/gMxZfYKFOLfz7iQEr7vo

## train_lite.py

  這是我叫 AI 替我想的模型架構

  <details>
  <summary>只有 1.2 GFLOPs</summary>

  ```
                   from  n    params  module                                       arguments
  0                  -1  1       232  ultralytics.nn.modules.conv.Conv             [3, 8, 3, 2]
  1                  -1  1      1184  ultralytics.nn.modules.conv.Conv             [8, 16, 3, 2]
  2                  -1  1      1304  ultralytics.nn.modules.block.C3k2            [16, 16, 1, False, 0.5]
  3                  -1  1       928  ultralytics.nn.modules.block.SCDown          [16, 32, 3, 2]
  4                  -1  1      6314  ultralytics.nn.modules.block.C3k2            [32, 32, 1, False, 0.6]
  5                  -1  1      2880  ultralytics.nn.modules.block.SCDown          [32, 64, 3, 2]
  6                  -1  1     25550  ultralytics.nn.modules.block.C3k2            [64, 64, 1, False, 0.6]
  7                  -1  1      9856  ultralytics.nn.modules.block.SCDown          [64, 128, 3, 2]
  8                  -1  1     78528  ultralytics.nn.modules.block.C3k2            [128, 128, 1, False, 0.5]
  9                  -1  1     41344  ultralytics.nn.modules.block.SPPF            [128, 128, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1     12416  ultralytics.nn.modules.conv.Conv             [192, 64, 1, 1]
 13                  -1  1     25550  ultralytics.nn.modules.block.C3k2            [64, 64, 1, False, 0.6]
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 16                  -1  1      3136  ultralytics.nn.modules.conv.Conv             [96, 32, 1, 1]
 17                  -1  1      6314  ultralytics.nn.modules.block.C3k2            [32, 32, 1, False, 0.6]
 18                  -1  1      1440  ultralytics.nn.modules.block.SCDown          [32, 32, 3, 2]
 19            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 20                  -1  1      6272  ultralytics.nn.modules.conv.Conv             [96, 64, 1, 1]
 21                  -1  1     25550  ultralytics.nn.modules.block.C3k2            [64, 64, 1, False, 0.6]
 22                  -1  1      4928  ultralytics.nn.modules.block.SCDown          [64, 64, 3, 2]
 23             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 24                  -1  1     24832  ultralytics.nn.modules.conv.Conv             [192, 128, 1, 1]
 25                  -1  1     78528  ultralytics.nn.modules.block.C3k2            [128, 128, 1, False, 0.5]
 26        [17, 21, 25]  1    448734  ultralytics.nn.modules.head.Pose             [1, [17, 3], [32, 64, 128]]
lite summary: 171 layers, 805,820 parameters, 805,804 gradients, 3.0 GFLOPs
  ```
  </details>

- 模型架構：自己看 lite.yaml
- 訓練參數：看 train_lite.py
  - 參數說明看官方的文檔 https://docs.ultralytics.com/usage/cfg/#train-settings

## train_base.py

  這是我叫 AI 替我想的模型架構

  <details>
  <summary>只有 2.9 GFLOPs</summary>

  ```
                   from  n    params  module                                       arguments
  0                  -1  1       232  ultralytics.nn.modules.conv.Conv             [3, 8, 3, 2]
  1                  -1  1      1184  ultralytics.nn.modules.conv.Conv             [8, 16, 3, 2]
  2                  -1  1      1720  ultralytics.nn.modules.block.C3k2            [16, 32, 1, False, 0.25]
  3                  -1  1      9280  ultralytics.nn.modules.conv.Conv             [32, 32, 3, 2]
  4                  -1  1      6640  ultralytics.nn.modules.block.C3k2            [32, 64, 1, False, 0.25]
  5                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
  6                  -1  1     22016  ultralytics.nn.modules.block.C3k2            [64, 64, 1, True]
  7                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
  8                  -1  1     22016  ultralytics.nn.modules.block.C3k2            [64, 64, 1, True]
  9                  -1  1     10432  ultralytics.nn.modules.block.SPPF            [64, 64, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1     23904  ultralytics.nn.modules.block.C3k2            [128, 64, 1, False]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1      8112  ultralytics.nn.modules.block.C3k2            [128, 32, 1, False]
 16                  -1  1      9280  ultralytics.nn.modules.conv.Conv             [32, 32, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1     21856  ultralytics.nn.modules.block.C3k2            [96, 64, 1, False]
 19                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1     26112  ultralytics.nn.modules.block.C3k2            [128, 64, 1, True]
 22        [15, 18, 21]  1    379742  ultralytics.nn.modules.head.Pose             [1, [17, 3], [32, 64, 64]]
base summary: 178 layers, 653,502 parameters, 653,486 gradients, 2.9 GFLOPs
  ```
  </details>

- 模型架構：自己看 base.yaml
- 訓練參數：看 train_base.py
  - 參數說明看官方的文檔 https://docs.ultralytics.com/usage/cfg/#train-settings
## 如何用我的程式碼

```
git clone https://github.com/comus/ultralytics
git checkout base

cd ultralytics
conda create -n yolo python=3.11
conda activate yolo
pip install ultralytics psutil requests seaborn pandas numpy scikit-learn pycocotools mlx onnxruntime tqdm pyyaml opencv-python matplotlib
pip install torch torchvision

# 之後運行 python 檔案, predict.py 或者訓練的程式或者驗證的程式
python predict.py
```

我其實沒有改很多東西，以下是我從官方修改了什麼

https://github.com/comus/ultralytics/pull/6/files


---

<details>
<summary>====== 以下未開始做 ======</summary>

# 方向二，在訓練小模型時，使用蒸餾

這兩個模型訓練過程，我會使用蒸餾損失函數，將大模型蒸餾到小模型

## train_lite_distill.py

- 模型架構：自己看 lite.yaml
- 訓練參數：看 train_lite_distill.py

## train_base_distill.py

- 模型架構：自己看 base.yaml
- 訓練參數：看 train_base_distill.py


# 方向三，先純蒸餾新模型，再訓練小模型

</details>










