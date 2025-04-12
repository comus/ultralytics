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

最大的原因是：官方的模型已經是最佳解，再怎麼蒸餾，精度都不會提升，甚至會下降。

我先依照不同模型架構，訓練出兩個模型

建立模型 yaml 的方法我也是問 AI 的

https://poe.com/s/gMxZfYKFOLfz7iQEr7vo





## 訓練層

將新創建的模型變成訓練模式，即是所有層都可以接受梯度變化，包括所有 BN 層，沒有凍結任何層。




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

第一次訓練，經過 60 epochs 後，pose mAP50-95=0.209

<details>
<summary>訓練過程</summary>

```
(yolo) [root@autodl-container-35c04cbe2d-b163da95 ~/autodl-tmp/ultralytics] [TMX:final2]$ python train_lite.py
WARNING ⚠️ no model scale passed. Assuming scale='lite'.
New https://pypi.org/project/ultralytics/8.3.107 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.105 🚀 Python-3.11.11 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 4090, 24111MiB)
engine/trainer: task=pose, mode=train, model=lite.yaml, data=coco-pose.yaml, epochs=60, time=None, patience=30, batch=64, imgsz=640, save=True, save_period=1, cache=disk, device=0, workers=16, project=yolo11-pose-lite, name=train, exist_ok=True, pretrained=True, optimizer=AdamW, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=True, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, teacher=None, distill=1.0, freezeAllBN=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.5, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=yolo11-pose-lite/train
Overriding model.yaml nc=80 with nc=1
WARNING ⚠️ no model scale passed. Assuming scale='lite'.

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

Freezing layer 'model.26.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /root/autodl-tmp/datasets/coco-pose/labels/train2017.cache... 56599 images, 0 backgrounds, 0 corrupt: 100%|██████████| 56599/56599 [00:00<?, ?it/s]
train: Caching images (43.7GB Disk): 100%|██████████| 56599/56599 [00:01<00:00, 44138.55it/s]
val: Scanning /root/autodl-tmp/datasets/coco-pose/labels/val2017.cache... 2346 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2346/2346 [00:00<?, ?it/s]
val: Caching images (1.8GB Disk): 100%|██████████| 2346/2346 [00:00<00:00, 25989.10it/s]
Plotting labels to yolo11-pose-lite/train/labels.jpg...
optimizer: AdamW(lr=0.01, momentum=0.937) with parameter groups 74 weight(decay=0.0), 84 weight(decay=0.0005), 83 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 16 dataloader workers
Logging results to yolo11-pose-lite/train
Starting training for 60 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       1/60      21.2G      2.374      8.712     0.8732      2.589      2.407          0        168        768: 100%|██████████| 885/885 [02:46<00:00,  5.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.44it/s]
                   all       2346       6352      0.438      0.297      0.301      0.123      0.175     0.0647     0.0363    0.00694

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       2/60      14.5G      1.869      7.671     0.7529      2.002      1.919          0        112        704: 100%|██████████| 885/885 [02:41<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.54it/s]
                   all       2346       6352      0.515      0.439      0.434        0.2      0.229      0.127     0.0681     0.0139

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       3/60      14.6G      1.738      7.165     0.7113      1.825       1.78          0        116        512: 100%|██████████| 885/885 [02:41<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.58it/s]
                   all       2346       6352      0.528      0.392      0.419      0.202        0.3      0.154      0.104     0.0267

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       4/60      14.6G      1.643      6.781     0.6813      1.697      1.693          0        161        544: 100%|██████████| 885/885 [02:45<00:00,  5.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.666      0.557      0.628      0.346      0.483      0.304      0.261      0.072

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       5/60      14.6G      1.585      6.541     0.6609      1.601      1.627          0         98        576: 100%|██████████| 885/885 [02:40<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.49it/s]
                   all       2346       6352      0.668      0.581      0.653      0.365        0.5      0.326      0.288     0.0853

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       6/60      14.6G      1.544      6.355     0.6469      1.551      1.596          0         84        896: 100%|██████████| 885/885 [02:40<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.55it/s]
                   all       2346       6352      0.723      0.594      0.685      0.403      0.551      0.368      0.329     0.0995

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       7/60      14.6G      1.514      6.244     0.6353      1.504      1.567          0        127        608: 100%|██████████| 885/885 [02:41<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.724      0.608      0.696      0.419      0.574      0.394      0.369      0.115

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       8/60      14.6G      1.483      6.102     0.6257      1.469       1.55          0        138        576: 100%|██████████| 885/885 [02:44<00:00,  5.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.742      0.621      0.718       0.44      0.591      0.424      0.398      0.131

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       9/60      14.6G      1.466      6.053     0.6205      1.438      1.533          0        111        640: 100%|██████████| 885/885 [02:39<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.762      0.649      0.738      0.456      0.636      0.443      0.437      0.148

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      10/60      14.6G      1.452      5.985     0.6161      1.421      1.518          0         80        480: 100%|██████████| 885/885 [02:33<00:00,  5.76it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.57it/s]
                   all       2346       6352      0.763      0.663      0.753      0.466       0.64      0.459      0.446      0.156

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      11/60      14.6G      1.437      5.927     0.6096      1.403      1.507          0         94        320: 100%|██████████| 885/885 [02:42<00:00,  5.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.51it/s]
                   all       2346       6352      0.765      0.657      0.754      0.472      0.644      0.467      0.455      0.164

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      12/60      14.6G       1.43      5.893     0.6084      1.385      1.497          0        133        928: 100%|██████████| 885/885 [02:39<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.769      0.657      0.759       0.48      0.656      0.477      0.469      0.167

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      13/60      14.6G      1.417      5.825     0.6025      1.371       1.49          0        102        608: 100%|██████████| 885/885 [02:41<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.50it/s]
                   all       2346       6352      0.775      0.663      0.764      0.485      0.669      0.476      0.475      0.175

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      14/60      14.6G      1.406      5.785     0.6009      1.357      1.482          0        120        704: 100%|██████████| 885/885 [02:41<00:00,  5.46it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.783      0.668      0.772      0.493      0.661      0.488      0.478      0.177

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      15/60      14.6G      1.399      5.734     0.5969      1.341      1.474          0         93        896: 100%|██████████| 885/885 [02:41<00:00,  5.46it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.53it/s]
                   all       2346       6352      0.779      0.675      0.774      0.496      0.666      0.492       0.49      0.182

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      16/60      14.6G      1.383      5.672     0.5909       1.33      1.471          0         99        960: 100%|██████████| 885/885 [02:46<00:00,  5.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.791      0.676      0.778      0.499      0.676      0.491      0.489      0.185

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      17/60      14.6G       1.38      5.661     0.5903      1.323       1.46          0        154        608: 100%|██████████| 885/885 [02:42<00:00,  5.46it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.52it/s]
                   all       2346       6352      0.788      0.677      0.778      0.502      0.671      0.499      0.495      0.187

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      18/60      14.6G      1.378      5.646     0.5895      1.315      1.457          0        127        864: 100%|██████████| 885/885 [02:42<00:00,  5.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.789      0.674      0.779      0.504      0.679      0.497      0.497      0.189

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      19/60      14.6G      1.372      5.632      0.589      1.304      1.449          0        108        896: 100%|██████████| 885/885 [02:39<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.787      0.678      0.781      0.506      0.687      0.497      0.501       0.19

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      20/60      14.6G      1.361      5.562     0.5843      1.296      1.445          0        119        896: 100%|██████████| 885/885 [02:42<00:00,  5.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.785      0.677      0.781      0.508      0.693      0.498      0.504      0.193

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      21/60      14.6G      1.363      5.605     0.5851      1.293      1.442          0        127        352: 100%|██████████| 885/885 [02:41<00:00,  5.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352       0.79      0.679      0.783      0.509      0.695        0.5      0.507      0.195

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      22/60      14.6G      1.358      5.569     0.5825      1.287      1.436          0        115        960: 100%|██████████| 885/885 [02:39<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.786      0.679      0.783       0.51      0.692      0.503      0.509      0.196

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      23/60      14.6G      1.349      5.533       0.58      1.278      1.431          0        129        512: 100%|██████████| 885/885 [02:40<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.789      0.676      0.784      0.511      0.693      0.506      0.512      0.198

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      24/60      14.6G       1.34      5.468     0.5766      1.264      1.429          0        151        928: 100%|██████████| 885/885 [02:42<00:00,  5.46it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.791      0.676      0.785      0.512      0.696      0.508      0.514      0.199

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      25/60      14.6G      1.334      5.463     0.5752      1.261      1.424          0        107        704: 100%|██████████| 885/885 [02:41<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.44it/s]
                   all       2346       6352      0.787       0.68      0.785      0.513      0.703      0.507      0.515        0.2

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      26/60      14.6G      1.331      5.422     0.5728      1.252       1.42          0        113        576: 100%|██████████| 885/885 [02:43<00:00,  5.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.787       0.68      0.785      0.514      0.702      0.508      0.515      0.201

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      27/60      14.6G      1.328      5.411     0.5726      1.251      1.422          0        122        864: 100%|██████████| 885/885 [02:43<00:00,  5.40it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.795      0.678      0.786      0.515      0.704      0.507      0.517      0.202

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      28/60      14.6G      1.317      5.364     0.5693      1.244      1.416          0        121        832: 100%|██████████| 885/885 [02:44<00:00,  5.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.792      0.679      0.786      0.515      0.707      0.506      0.518      0.204

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      29/60      14.6G      1.317      5.374     0.5706       1.24      1.414          0        136        736: 100%|██████████| 885/885 [02:44<00:00,  5.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.787      0.682      0.787      0.516      0.706      0.507      0.522      0.205

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      30/60      14.6G      1.317       5.36     0.5678      1.235       1.41          0        127        896: 100%|██████████| 885/885 [02:38<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.52it/s]
                   all       2346       6352      0.793      0.677      0.788      0.517      0.706      0.507      0.523      0.206

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      31/60      14.6G      1.312      5.343     0.5673      1.226      1.406          0        133        576: 100%|██████████| 885/885 [02:40<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.793      0.678      0.787      0.517      0.702       0.51      0.524      0.208

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      32/60      14.6G      1.312      5.332     0.5664      1.225      1.401          0        154        416: 100%|██████████| 885/885 [02:39<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.787      0.683      0.787      0.518      0.702      0.514      0.527      0.209

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      33/60      14.6G      1.307      5.308     0.5666      1.219      1.404          0        127        736: 100%|██████████| 885/885 [02:43<00:00,  5.42it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.789      0.684      0.789      0.519      0.705      0.517      0.529       0.21

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      34/60      14.6G      1.302      5.285     0.5626      1.213      1.397          0        116        832: 100%|██████████| 885/885 [02:39<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.788      0.685       0.79       0.52      0.702      0.518      0.529      0.212

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      35/60      14.6G        1.3      5.274     0.5633      1.213      1.397          0        136        480: 100%|██████████| 885/885 [02:40<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.792      0.685      0.791      0.522      0.701      0.517      0.529      0.212

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      36/60      14.6G      1.299      5.275      0.562      1.206       1.39          0         99        800: 100%|██████████| 885/885 [02:38<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.52it/s]
                   all       2346       6352      0.792      0.686      0.792      0.523      0.701      0.518      0.529      0.214

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      37/60      14.6G      1.292      5.247     0.5613      1.204      1.389          0        119        736: 100%|██████████| 885/885 [02:40<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.794      0.685      0.793      0.525      0.698      0.521      0.532      0.216

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      38/60      14.6G      1.287      5.215     0.5597      1.197      1.389          0        128        448: 100%|██████████| 885/885 [02:42<00:00,  5.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.793      0.687      0.794      0.526      0.698      0.524      0.535      0.217

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      39/60      14.6G      1.279      5.185     0.5585      1.191      1.383          0        103        960: 100%|██████████| 885/885 [02:41<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.792      0.688      0.795      0.527      0.697      0.526      0.537      0.218

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      40/60      14.6G      1.278       5.18     0.5562      1.189      1.382          0        116        352: 100%|██████████| 885/885 [02:42<00:00,  5.46it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.793      0.688      0.795      0.528      0.699      0.527      0.539      0.219

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      41/60      14.6G      1.275      5.155     0.5551      1.184      1.384          0        101        608: 100%|██████████| 885/885 [02:43<00:00,  5.40it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.58it/s]
                   all       2346       6352      0.794      0.691      0.796      0.529      0.703      0.528      0.539       0.22

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      42/60      14.6G      1.267      5.126     0.5518      1.182       1.38          0        169        576: 100%|██████████| 885/885 [02:44<00:00,  5.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.789      0.694      0.797      0.529      0.705      0.529      0.542      0.221

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      43/60      14.6G       1.27      5.146     0.5546       1.18      1.376          0        147        640: 100%|██████████| 885/885 [02:40<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.788      0.695      0.798       0.53      0.709      0.529      0.542      0.222

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      44/60      14.6G      1.267      5.118     0.5523      1.175      1.379          0        105        736: 100%|██████████| 885/885 [02:44<00:00,  5.39it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.788      0.698      0.799      0.531      0.709      0.531      0.542      0.223

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      45/60      14.6G      1.271      5.135     0.5517      1.172      1.377          0        112        416: 100%|██████████| 885/885 [02:41<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.52it/s]
                   all       2346       6352      0.789      0.697        0.8      0.532      0.713      0.531      0.543      0.223

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      46/60      14.6G      1.268       5.14     0.5536      1.174      1.373          0        122        800: 100%|██████████| 885/885 [02:39<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.58it/s]
                   all       2346       6352      0.792      0.694        0.8      0.533      0.716      0.533      0.545      0.224

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      47/60      14.6G      1.263      5.104      0.552      1.168      1.371          0        155        320: 100%|██████████| 885/885 [02:40<00:00,  5.51it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.795      0.694      0.801      0.534      0.718      0.533      0.547      0.225

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      48/60      14.6G      1.262      5.103     0.5518      1.168      1.371          0        133        832: 100%|██████████| 885/885 [02:42<00:00,  5.45it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.794      0.696      0.802      0.534      0.723      0.533      0.549      0.226

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      49/60      14.6G      1.261      5.075     0.5492      1.169      1.373          0         99        352: 100%|██████████| 885/885 [02:43<00:00,  5.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.795      0.697      0.802      0.535      0.721      0.531      0.548      0.226

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      50/60      14.6G      1.262      5.098     0.5503      1.163      1.367          0        120        448: 100%|██████████| 885/885 [02:40<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.794      0.697      0.803      0.536      0.722      0.534       0.55      0.227
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      51/60      14.6G      1.223      4.561     0.5445      1.067      1.341          0         50        896: 100%|██████████| 885/885 [02:38<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.50it/s]
                   all       2346       6352        0.8      0.696      0.804      0.536      0.723      0.532      0.551      0.228

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      52/60      14.6G       1.21        4.5     0.5391      1.053      1.338          0         87        320: 100%|██████████| 885/885 [02:31<00:00,  5.82it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.799      0.698      0.804      0.537      0.725      0.536      0.554       0.23

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      53/60      14.6G      1.208        4.5     0.5392      1.049      1.332          0         80        800: 100%|██████████| 885/885 [02:35<00:00,  5.70it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.802      0.698      0.805      0.538      0.725      0.537      0.555       0.23

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      54/60      14.6G      1.206      4.494     0.5377      1.048      1.335          0         60        352: 100%|██████████| 885/885 [02:38<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.69it/s]
                   all       2346       6352      0.802      0.701      0.806      0.539      0.725       0.54      0.557      0.232

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      55/60      14.6G      1.204      4.478     0.5372      1.041      1.332          0         63        352: 100%|██████████| 885/885 [02:37<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.70it/s]
                   all       2346       6352      0.801      0.705      0.807       0.54      0.726       0.54       0.56      0.233

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      56/60      14.6G      1.202      4.473     0.5364      1.039      1.331          0         64        480: 100%|██████████| 885/885 [02:37<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.67it/s]
                   all       2346       6352        0.8      0.706      0.807      0.541      0.728      0.536      0.559      0.234

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      57/60      14.6G      1.199      4.444     0.5353      1.038      1.334          0         89        576: 100%|██████████| 885/885 [02:42<00:00,  5.46it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.69it/s]
                   all       2346       6352      0.801      0.706      0.808      0.541      0.724      0.539       0.56      0.235

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      58/60      14.6G        1.2      4.468     0.5359      1.036      1.329          0         45        864: 100%|██████████| 885/885 [02:39<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:08<00:00,  2.35it/s]
                   all       2346       6352      0.803      0.705      0.808      0.542      0.725       0.54      0.562      0.235

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      59/60      14.6G      1.203      4.466      0.536       1.04      1.333          0         71        672: 100%|██████████| 885/885 [02:46<00:00,  5.32it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.71it/s]
                   all       2346       6352      0.799      0.709      0.809      0.543      0.729      0.541      0.564      0.236

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      60/60      14.6G      1.199      4.456     0.5359      1.035       1.33          0         63        896: 100%|██████████| 885/885 [02:41<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.55it/s]
                   all       2346       6352      0.801      0.708       0.81      0.543       0.73      0.542      0.565      0.237

60 epochs completed in 2.818 hours.
Optimizer stripped from yolo11-pose-lite/train/weights/last.pt, 1.9MB
Optimizer stripped from yolo11-pose-lite/train/weights/best.pt, 1.9MB

Validating yolo11-pose-lite/train/weights/best.pt...
Ultralytics 8.3.105 🚀 Python-3.11.11 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 4090, 24111MiB)
lite summary (fused): 97 layers, 801,939 parameters, 0 gradients, 2.9 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:13<00:00,  1.42it/s]
                   all       2346       6352      0.801      0.707      0.809      0.543      0.728      0.542      0.565      0.237
Speed: 0.1ms preprocess, 0.2ms inference, 0.0ms loss, 0.5ms postprocess per image
Saving yolo11-pose-lite/train/predictions.json...

Evaluating pycocotools mAP using yolo11-pose-lite/train/predictions.json and /root/autodl-tmp/datasets/coco-pose/annotations/person_keypoints_val2017.json...
loading annotations into memory...
Done (t=0.15s)
creating index...
index created!
Loading and preparing results...
DONE (t=2.36s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=12.81s).
Accumulating evaluation results...
DONE (t=0.86s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.640
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.445
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.500
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.177
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.234
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.654
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.775
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=7.05s).
Accumulating evaluation results...
DONE (t=0.18s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.231
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.552
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.154
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.207
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.296
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.321
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.655
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.399
Results saved to yolo11-pose-lite/train
```
</details>





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

第一次訓練，經過 60 epochs 後，pose mAP50-95=0.209

<details>
<summary>訓練過程</summary>

```
(yolo) [root@autodl-container-8ad2498c83-81a97822 ~/autodl-tmp/ultralytics] [TMX:final2]$ python train_base.py
WARNING ⚠️ no model scale passed. Assuming scale='base'.
New https://pypi.org/project/ultralytics/8.3.107 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.105 🚀 Python-3.11.11 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 4090, 24111MiB)
engine/trainer: task=pose, mode=train, model=base.yaml, data=coco-pose.yaml, epochs=60, time=None, patience=30, batch=64, imgsz=640, save=True, save_period=1, cache=disk, device=0, workers=16, project=yolo11-pose-base, name=train, exist_ok=True, pretrained=True, optimizer=AdamW, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=True, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=True, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=True, opset=None, workspace=None, nms=False, teacher=None, distill=1.0, freezeAllBN=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.5, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, copy_paste_mode=flip, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=yolo11-pose-base/train
Overriding model.yaml nc=80 with nc=1
WARNING ⚠️ no model scale passed. Assuming scale='base'.

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

Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Scanning /root/autodl-tmp/datasets/coco-pose/labels/train2017.cache... 56599 images, 0 backgrounds, 0 corrupt: 100%|██████████| 56599/56599 [00:00<?, ?it/s]
train: Caching images (43.7GB Disk): 100%|██████████| 56599/56599 [00:01<00:00, 35533.39it/s]
val: Scanning /root/autodl-tmp/datasets/coco-pose/labels/val2017.cache... 2346 images, 0 backgrounds, 0 corrupt: 100%|██████████| 2346/2346 [00:00<?, ?it/s]
val: Caching images (1.8GB Disk): 100%|██████████| 2346/2346 [00:00<00:00, 26319.51it/s]
Plotting labels to yolo11-pose-base/train/labels.jpg...
optimizer: AdamW(lr=0.01, momentum=0.937) with parameter groups 80 weight(decay=0.0), 90 weight(decay=0.0005), 89 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 16 dataloader workers
Logging results to yolo11-pose-base/train
Starting training for 60 epochs...

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       1/60      19.9G      2.355      8.682     0.8712      2.549      2.422          0        168        768: 100%|██████████| 885/885 [02:43<00:00,  5.41it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:08<00:00,  2.28it/s]
                   all       2346       6352      0.483      0.386      0.399      0.173      0.173      0.099      0.045     0.0077

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       2/60        20G      1.841      7.659     0.7501       1.99      1.918          0        112        704: 100%|██████████| 885/885 [02:38<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.55it/s]
                   all       2346       6352      0.565      0.435      0.473      0.219      0.316      0.153      0.101     0.0226

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       3/60        20G      1.717      7.183       0.71      1.824      1.782          0        116        512: 100%|██████████| 885/885 [02:33<00:00,  5.76it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.56it/s]
                   all       2346       6352      0.632      0.528      0.584      0.295      0.378      0.256      0.186     0.0459

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       4/60        20G      1.631       6.81     0.6811      1.708      1.698          0        161        544: 100%|██████████| 885/885 [02:41<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.55it/s]
                   all       2346       6352      0.697      0.546      0.622       0.33      0.455      0.303      0.248      0.067

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       5/60        20G       1.58      6.579     0.6626      1.621      1.638          0         98        576: 100%|██████████| 885/885 [02:37<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.54it/s]
                   all       2346       6352      0.705      0.573      0.663      0.375      0.516       0.34      0.293     0.0833

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       6/60        20G      1.546      6.395     0.6497      1.571       1.61          0         84        896: 100%|██████████| 885/885 [02:37<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.705      0.589      0.675       0.39      0.535      0.364      0.325     0.0979

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       7/60        20G      1.514      6.287     0.6383      1.525      1.581          0        127        608: 100%|██████████| 885/885 [02:38<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.57it/s]
                   all       2346       6352      0.745      0.605      0.706      0.423      0.579      0.395      0.373      0.117

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       8/60        20G      1.488      6.151     0.6297      1.494      1.567          0        138        576: 100%|██████████| 885/885 [02:40<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.748      0.622      0.721      0.435      0.596      0.416      0.393      0.129

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
       9/60        20G      1.474      6.111     0.6246      1.466      1.549          0        111        640: 100%|██████████| 885/885 [02:36<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.54it/s]
                   all       2346       6352      0.743      0.645      0.732      0.444      0.623      0.438      0.427      0.141

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      10/60        20G      1.461      6.047     0.6208      1.446      1.536          0         80        480: 100%|██████████| 885/885 [02:39<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.53it/s]
                   all       2346       6352      0.758      0.645       0.74      0.457      0.619      0.434       0.42      0.142

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      11/60        20G      1.448      5.986     0.6153      1.427      1.525          0         94        320: 100%|██████████| 885/885 [02:38<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.56it/s]
                   all       2346       6352      0.763      0.651      0.747      0.465      0.639      0.447      0.441      0.153

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      12/60        20G       1.44      5.952     0.6129      1.413      1.515          0        133        928: 100%|██████████| 885/885 [02:37<00:00,  5.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.56it/s]
                   all       2346       6352      0.773      0.644      0.752      0.472      0.638      0.461      0.454      0.156

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      13/60        20G      1.428      5.894     0.6088      1.396      1.508          0        102        608: 100%|██████████| 885/885 [02:38<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.53it/s]
                   all       2346       6352      0.777      0.657       0.76      0.478      0.644      0.466      0.461      0.162

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      14/60        20G      1.421      5.863     0.6074      1.389      1.503          0        120        704: 100%|██████████| 885/885 [02:38<00:00,  5.59it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.776      0.661      0.762      0.482      0.655      0.467      0.465      0.166

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      15/60        20G      1.413      5.816     0.6035      1.374      1.496          0         93        896: 100%|██████████| 885/885 [02:39<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.45it/s]
                   all       2346       6352      0.778      0.664      0.766      0.487      0.666      0.471      0.474      0.168

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      16/60        20G      1.397      5.753     0.5981      1.361      1.494          0         99        960: 100%|██████████| 885/885 [02:41<00:00,  5.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.781      0.667      0.769      0.491       0.66      0.476      0.476      0.171

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      17/60        20G      1.393      5.741     0.5961      1.354      1.484          0        154        608: 100%|██████████| 885/885 [02:39<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.788      0.672      0.771      0.493      0.638       0.49      0.477      0.173

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      18/60        20G      1.393      5.732     0.5971      1.347      1.479          0        127        864: 100%|██████████| 885/885 [02:40<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.784      0.675      0.773      0.495      0.638      0.491      0.479      0.174

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      19/60        20G      1.385      5.726     0.5958      1.337      1.471          0        108        896: 100%|██████████| 885/885 [02:37<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.788      0.675      0.775      0.498      0.639      0.496      0.483      0.177

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      20/60        20G      1.379      5.657      0.591      1.332      1.471          0        119        896: 100%|██████████| 885/885 [02:39<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.789      0.678      0.776      0.499      0.639      0.496      0.485      0.179

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      21/60        20G      1.379        5.7     0.5922      1.324      1.465          0        127        352: 100%|██████████| 885/885 [02:38<00:00,  5.60it/s]q
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.57it/s]
                   all       2346       6352      0.793      0.677      0.778        0.5      0.641      0.499      0.488       0.18

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      22/60        20G      1.376      5.665     0.5902      1.323      1.462          0        115        960: 100%|██████████| 885/885 [02:37<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.57it/s]
                   all       2346       6352      0.791       0.68      0.779      0.502      0.653      0.497      0.492      0.183

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      23/60        20G      1.364      5.632     0.5875       1.31      1.453          0        129        512: 100%|██████████| 885/885 [02:37<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.788      0.677      0.778      0.503      0.656      0.498      0.495      0.184

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      24/60        20G      1.355      5.565     0.5842        1.3      1.449          0        151        928: 100%|██████████| 885/885 [02:38<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.786      0.678       0.78      0.504      0.662      0.497      0.496      0.185

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      25/60        20G      1.351      5.566     0.5832      1.296      1.446          0        107        704: 100%|██████████| 885/885 [02:38<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.785      0.679       0.78      0.504      0.665      0.499        0.5      0.187

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      26/60        20G      1.347       5.53     0.5813      1.287       1.44          0        113        576: 100%|██████████| 885/885 [02:37<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.786      0.679      0.781      0.505       0.67        0.5      0.503      0.188

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      27/60        20G      1.346       5.52     0.5809      1.289      1.444          0        122        864: 100%|██████████| 885/885 [02:38<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.791      0.676      0.782      0.507      0.667      0.504      0.504      0.189

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      28/60        20G      1.334      5.476      0.578      1.277      1.436          0        121        832: 100%|██████████| 885/885 [02:40<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.51it/s]
                   all       2346       6352      0.792      0.674      0.782      0.508      0.667      0.505      0.507       0.19

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      29/60        20G      1.335      5.487     0.5792      1.274      1.434          0        136        736: 100%|██████████| 885/885 [02:41<00:00,  5.49it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.66it/s]
                   all       2346       6352      0.793      0.675      0.783      0.509      0.669      0.505      0.507      0.191

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      30/60        20G      1.334      5.471     0.5763      1.271       1.43          0        127        896: 100%|██████████| 885/885 [02:35<00:00,  5.70it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.53it/s]
                   all       2346       6352      0.797      0.672      0.783       0.51      0.673      0.505      0.509      0.193

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      31/60        20G      1.328      5.455     0.5763       1.26      1.427          0        133        576: 100%|██████████| 885/885 [02:39<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.803      0.669      0.784      0.511      0.676      0.506      0.509      0.194

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      32/60        20G       1.33      5.444     0.5757       1.26      1.419          0        154        416: 100%|██████████| 885/885 [02:37<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.56it/s]
                   all       2346       6352      0.803       0.67      0.785      0.511      0.679      0.507      0.511      0.195

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      33/60        20G      1.322      5.421     0.5754      1.257      1.425          0        127        736: 100%|██████████| 885/885 [02:40<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.56it/s]
                   all       2346       6352      0.803      0.672      0.786      0.512      0.679      0.506      0.511      0.196

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      34/60        20G      1.321      5.402     0.5717      1.249      1.419          0        116        832: 100%|██████████| 885/885 [02:36<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.62it/s]
                   all       2346       6352      0.806       0.67      0.786      0.513      0.689      0.505      0.514      0.197

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      35/60        20G      1.319      5.392     0.5722      1.249      1.418          0        136        480: 100%|██████████| 885/885 [02:37<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352        0.8      0.677      0.787      0.515      0.692      0.505      0.517      0.199

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      36/60        20G      1.317      5.395      0.571      1.241       1.41          0         99        800: 100%|██████████| 885/885 [02:35<00:00,  5.69it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.799      0.677      0.788      0.516       0.69      0.509      0.517        0.2

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      37/60        20G      1.311      5.369     0.5709      1.239      1.412          0        119        736: 100%|██████████| 885/885 [02:37<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.65it/s]
                   all       2346       6352      0.803      0.676      0.789      0.516      0.695      0.509      0.522      0.201

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      38/60        20G      1.306      5.338     0.5689      1.231      1.409          0        128        448: 100%|██████████| 885/885 [02:38<00:00,  5.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.801      0.679      0.789      0.517      0.693      0.507      0.519      0.202

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      39/60        20G      1.298       5.31     0.5682      1.225      1.404          0        103        960: 100%|██████████| 885/885 [02:37<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.803       0.68      0.791      0.519      0.689      0.513      0.523      0.203

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      40/60        20G      1.299      5.306     0.5657      1.224      1.404          0        116        352: 100%|██████████| 885/885 [02:37<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.799      0.685      0.791      0.519      0.692      0.513      0.525      0.203

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      41/60        20G      1.295      5.286     0.5653      1.222      1.405          0        101        608: 100%|██████████| 885/885 [02:40<00:00,  5.52it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.807       0.68      0.792       0.52      0.691      0.513      0.525      0.204

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      42/60        20G      1.287      5.256     0.5619      1.218      1.403          0        169        576: 100%|██████████| 885/885 [02:40<00:00,  5.50it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.801      0.686      0.793      0.521      0.692      0.514      0.526      0.205

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      43/60        20G      1.291      5.272     0.5643      1.215      1.398          0        147        640: 100%|██████████| 885/885 [02:38<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.58it/s]
                   all       2346       6352      0.803      0.684      0.794      0.522      0.687      0.516      0.527      0.206

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      44/60        20G      1.285      5.253     0.5624      1.214      1.399          0        105        736: 100%|██████████| 885/885 [02:41<00:00,  5.48it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352        0.8      0.689      0.795      0.523      0.688      0.517      0.528      0.207

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      45/60        20G       1.29      5.267      0.562      1.209      1.397          0        112        416: 100%|██████████| 885/885 [02:39<00:00,  5.55it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.58it/s]
                   all       2346       6352      0.801      0.688      0.796      0.524      0.691      0.517      0.528      0.208

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      46/60        20G      1.288      5.269     0.5636      1.211      1.394          0        122        800: 100%|██████████| 885/885 [02:36<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.796      0.692      0.796      0.525      0.689      0.517      0.528      0.208

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      47/60        20G      1.283      5.239     0.5621      1.205      1.393          0        155        320: 100%|██████████| 885/885 [02:37<00:00,  5.60it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.795      0.693      0.797      0.526      0.689      0.519      0.529      0.209

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      48/60        20G      1.284       5.24     0.5616      1.206      1.395          0        133        832: 100%|██████████| 885/885 [02:39<00:00,  5.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.795      0.693      0.797      0.527      0.691       0.52       0.53      0.209

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      49/60        20G      1.281      5.216     0.5593      1.205      1.394          0         99        352: 100%|██████████| 885/885 [02:40<00:00,  5.53it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.796      0.693      0.798      0.527      0.694      0.517       0.53       0.21

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      50/60        20G      1.282      5.231     0.5603        1.2       1.39          0        120        448: 100%|██████████| 885/885 [02:37<00:00,  5.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.54it/s]
                   all       2346       6352      0.797      0.693      0.798      0.528      0.688      0.519       0.53       0.21
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      51/60        20G      1.248       4.72      0.555      1.107      1.369          0         50        896: 100%|██████████| 885/885 [02:37<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.57it/s]
                   all       2346       6352      0.802      0.693      0.799      0.529      0.694      0.518      0.532      0.211

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      52/60        20G      1.237      4.656     0.5504      1.094      1.367          0         87        320: 100%|██████████| 885/885 [02:37<00:00,  5.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352        0.8      0.696        0.8      0.529      0.694       0.52      0.533      0.212

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      53/60        20G      1.233      4.654     0.5499      1.087      1.361          0         80        800: 100%|██████████| 885/885 [02:33<00:00,  5.77it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.797      0.698        0.8       0.53      0.695       0.52      0.534      0.212

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      54/60        20G      1.233      4.656     0.5494      1.087      1.365          0         60        352: 100%|██████████| 885/885 [02:37<00:00,  5.61it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.60it/s]
                   all       2346       6352      0.799      0.697      0.801      0.531      0.693      0.522      0.535      0.213

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      55/60        20G       1.23      4.635     0.5486      1.082      1.361          0         63        352: 100%|██████████| 885/885 [02:36<00:00,  5.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.801      0.696      0.801      0.531      0.697      0.524      0.537      0.213

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      56/60        20G      1.226      4.633     0.5478      1.079      1.358          0         64        480: 100%|██████████| 885/885 [02:31<00:00,  5.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.59it/s]
                   all       2346       6352      0.803      0.695      0.802      0.532      0.701      0.524      0.539      0.214

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      57/60        20G      1.225       4.61     0.5464      1.079      1.363          0         89        576: 100%|██████████| 885/885 [02:39<00:00,  5.54it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.65it/s]
                   all       2346       6352      0.802      0.697      0.802      0.532      0.698      0.525      0.539      0.215

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      58/60        20G      1.226      4.628      0.547      1.075      1.358          0         45        864: 100%|██████████| 885/885 [02:35<00:00,  5.69it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.63it/s]
                   all       2346       6352      0.801      0.699      0.803      0.533      0.698      0.526      0.539      0.215

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      59/60        20G      1.227      4.632     0.5469       1.08      1.361          0         71        672: 100%|██████████| 885/885 [02:38<00:00,  5.58it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.61it/s]
                   all       2346       6352      0.805      0.695      0.803      0.533      0.698      0.526      0.538      0.216

      Epoch    GPU_mem   box_loss  pose_loss  kobj_loss   cls_loss   dfl_loss     d_loss  Instances       Size
      60/60        20G      1.224      4.615     0.5472      1.075      1.358          0         63        896: 100%|██████████| 885/885 [02:36<00:00,  5.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:07<00:00,  2.64it/s]
                   all       2346       6352      0.805      0.695      0.803      0.534      0.695      0.527      0.538      0.216

60 epochs completed in 2.768 hours.
Optimizer stripped from yolo11-pose-base/train/weights/last.pt, 1.6MB
Optimizer stripped from yolo11-pose-base/train/weights/best.pt, 1.6MB

Validating yolo11-pose-base/train/weights/best.pt...
Ultralytics 8.3.105 🚀 Python-3.11.11 torch-2.6.0+cu124 CUDA:0 (NVIDIA GeForce RTX 4090, 24111MiB)
base summary (fused): 98 layers, 650,600 parameters, 0 gradients, 2.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95): 100%|██████████| 19/19 [00:12<00:00,  1.58it/s]
                   all       2346       6352      0.806      0.695      0.803      0.534      0.694      0.527      0.538      0.216
Speed: 0.1ms preprocess, 0.2ms inference, 0.0ms loss, 0.4ms postprocess per image
Saving yolo11-pose-base/train/predictions.json...

Evaluating pycocotools mAP using yolo11-pose-base/train/predictions.json and /root/autodl-tmp/datasets/coco-pose/annotations/person_keypoints_val2017.json...
loading annotations into memory...
Done (t=0.16s)
creating index...
index created!
Loading and preparing results...
DONE (t=2.31s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=10.34s).
Accumulating evaluation results...
DONE (t=0.81s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.406
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.631
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.438
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.176
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.551
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769
Running per image evaluation...
Evaluate annotation type *keypoints*
DONE (t=6.64s).
Accumulating evaluation results...
DONE (t=0.16s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.209
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.532
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.125
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.641
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.247
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.363
Results saved to yolo11-pose-base/train
```
</details>

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


## 若你想自己訓練自己新的模型架構

1. 在 ultralytics 資料夾下, 建立 yaml 檔，例如我的 lite.yaml

- 然後複製問 AI 參考他建立 yaml 檔
- 或者用官方的 https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/11/yolo11-pose.yaml
- 重點是要他參考一下官方模型的 head 層，最好類似。

2. 在 ultralytics 資料夾下, 建立 train.py 檔

- 叫 AI 幫你建立，參考 `train_lite.py` 或者 `train_base.py`
- 叫 AI 替你改訓練參數

3. 然後運行 `train.py`

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










