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

- 模型架構：自己看 lite.yaml
- 訓練參數：看 train_lite.py

## train_base.py

- 模型架構：自己看 base.yaml
- 訓練參數：看 train_base.py

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










