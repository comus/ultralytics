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
from visualize import visualize_indices_on_feature_maps

def pre_transform(im, stride, imgsz):
    """
    Pre-transform input image before inference.

    Args:
        im (List[np.ndarray]): Images of shape (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

    Returns:
        (List[np.ndarray]): A list of transformed images.
    """
    # same_shapes = len({x.shape for x in im}) == 1

    letterbox = LetterBox(
        imgsz,  # Use model's configuration size instead of hardcoded value
        auto=False,
        stride=stride,  # Use model's stride
    )
    return [letterbox(image=x) for x in im]

def preprocess(im, stride, imgsz):
    """
    Prepares input image before inference.

    Args:
        im (torch.Tensor | List(np.ndarray)): Images of shape (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
    """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(pre_transform(im, stride, imgsz))
        im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)

    im = im.to("cpu")
    im = im.float()  # uint8 to fp16/32
    if not_tensor:
        im /= 255  # 0 - 255 to 0.0 - 1.0
    return im

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.
        end2end (bool): If the model doesn't require NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """
    import torchvision  # scope for faster 'import ultralytics'

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output

def postprocess(preds, img, orig_imgs, **kwargs):
    """
    Post-process predictions and return a list of Results objects.

    This method applies non-maximum suppression to raw model predictions and prepares them for visualization and
    further analysis.

    Args:
        preds (torch.Tensor): Raw predictions from the model.
        img (torch.Tensor): Processed input image tensor in model input format.
        orig_imgs (torch.Tensor | list): Original input images before preprocessing.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        (list): List of Results objects containing the post-processed predictions.

    Examples:
        >>> predictor = DetectionPredictor(overrides=dict(model="yolov8n.pt"))
        >>> results = predictor.predict("path/to/image.jpg")
        >>> processed_results = predictor.postprocess(preds, img, orig_imgs)
    """
    # 應用NMS
    preds = non_max_suppression(
        preds,
        0.25,
        1.0,
        None,
        False,
        max_det=300,
        nc=1,
        end2end=False,
        rotated=False,
    )

    if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

    return construct_results(preds, img, orig_imgs, **kwargs)

def construct_results(preds, img, orig_imgs):
    """
    Construct a list of Results objects from model predictions.

    Args:
        preds (List[torch.Tensor]): List of predicted bounding boxes and scores for each image.
        img (torch.Tensor): Batch of preprocessed images used for inference.
        orig_imgs (List[np.ndarray]): List of original images before preprocessing.

    Returns:
        (List[Results]): List of Results objects containing detection information for each image.
    """
    return [
        construct_result(pred, img, orig_img)
        for pred, orig_img in zip(preds, orig_imgs)
    ]

def construct_result0(pred, img, orig_img, img_path):
    """
    Construct a single Results object from one image prediction.

    Args:
        pred (torch.Tensor): Predicted boxes and scores with shape (N, 6) where N is the number of detections.
        img (torch.Tensor): Preprocessed image tensor used for inference.
        orig_img (np.ndarray): Original image before preprocessing.
        img_path (str): Path to the original image file.

    Returns:
        (Results): Results object containing the original image, image path, class names, and scaled bounding boxes.
    """
    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

    # 得到轉換後的 57 個點

    # 在 construct_result0 函數中，使用 scale_boxes 函數將模型輸出的座標轉換為原始圖像的座標
    # scale_boxes 函數的工作原理是：
    #  - 計算模型輸入圖像與原始圖像之間的縮放比例 (gain)
    #  - 計算填充值 (padding)
    #  - 移除填充：boxes[..., 0] -= pad[0]（移除 x 軸填充）
    #  - 進行縮放：boxes[..., :4] /= gain（將座標除以縮放比例）
    #  - 最後確保座標不超出圖像邊界：clip_boxes(boxes, img0_shape)
    # 這個過程將 1.9143e+02 轉換為 1.7643e+02，代表將模型處理後的座標轉回原始圖像上的實際位置。
    return Results(orig_img, path=img_path, names={0: 'person'}, boxes=pred[:, :6])

def construct_result(pred, img, orig_img, img_path = None):
    """
    Construct the result object from the prediction, including keypoints.

    This method extends the parent class implementation by extracting keypoint data from predictions
    and adding them to the result object.

    Args:
        pred (torch.Tensor): The predicted bounding boxes, scores, and keypoints with shape (N, 6+K*D) where N is
            the number of detections, K is the number of keypoints, and D is the keypoint dimension.
        img (torch.Tensor): The processed input image tensor with shape (B, C, H, W).
        orig_img (np.ndarray): The original unprocessed image as a numpy array.
        img_path (str): The path to the original image file.

    Returns:
        (Results): The result object containing the original image, image path, class names, bounding boxes, and keypoints.
    """
    result = construct_result0(pred, img, orig_img, img_path)
    # Extract keypoints from prediction and reshape according to model's keypoint shape
    pred_kpts = pred[:, 6:].view(len(pred), 17, 3) if len(pred) else pred[:, 6:]
    # Scale keypoints coordinates to match the original image dimensions
    pred_kpts = ops.scale_coords(img.shape[2:], pred_kpts, orig_img.shape)
    result.update(keypoints=pred_kpts)

    # 最後使用 scale_coords 函數將關鍵點座標從模型輸入尺寸縮放至原始圖像尺寸
    # scale_coords 函數的原理與 scale_boxes 類似，但專門用於處理關鍵點座標：
    #  - 計算縮放比例和填充值
    #  - 移除填充：coords[..., 0] -= pad[0]（x座標減去x軸填充）
    #  - 進行縮放：coords[..., 0] /= gain（x座標除以縮放比例）
    #  - 確保座標不超出圖像邊界：clip_coords(coords, img0_shape)
    return result

def draw_pose_skeleton(image, keypoints, min_confidence=0.5, thickness=2, circle_radius=5):
    """
    在圖像上繪製人體姿勢骨架
    
    Args:
        image (numpy.ndarray): 輸入圖像
        keypoints (numpy.ndarray): 關鍵點數據，形狀為 (num_persons, num_keypoints, 3)，
                                  其中最後一維是 [x, y, confidence]
        min_confidence (float): 最小置信度閾值，只繪製高於此閾值的關鍵點
        thickness (int): 線條粗細
        circle_radius (int): 關鍵點圓圈半徑
        
    Returns:
        numpy.ndarray: 繪製了骨架的圖像
    """
    if image is None or keypoints is None:
        return image
    
    # 複製圖像以避免修改原始圖像
    img_with_kpts = image.copy() if image is not None else None
    
    # 關鍵點連接順序，用於繪製骨架線條
    skeleton = [  # 關鍵點之間的連接關係
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
        [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # 定義顏色
    limb_colors = [(255, 51, 153), (153, 0, 102), (153, 0, 51), 
                   (204, 0, 51), (255, 0, 51), (255, 51, 51), 
                   (255, 102, 51), (255, 153, 51), (255, 153, 102),
                   (255, 204, 102), (255, 255, 51), (204, 255, 51), 
                   (153, 255, 51), (102, 255, 51), (51, 255, 51), 
                   (51, 255, 102), (51, 255, 153), (51, 255, 204)]
    
    kpt_colors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), 
                 (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85), 
                 (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                 (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
                 (255, 0, 170)]
    
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
                    pt1 = (int(kpt1[0]), int(kpt1[1]))
                    pt2 = (int(kpt2[0]), int(kpt2[1]))
                    cv2.line(img_with_kpts, pt1, pt2, color, thickness)
        
        # 畫關鍵點
        for i, kpt in enumerate(person_kpts):
            if len(kpt) > 2:
                conf = kpt[2]
                if conf > min_confidence:  # 只繪製高置信度的關鍵點
                    color = kpt_colors[i % len(kpt_colors)]
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(img_with_kpts, (x, y), circle_radius, color, -1)
                    
    return img_with_kpts

def transform_bboxes_xywh(bboxes, batch_idx=None, original_shape=None, imgsz=None, stride=32):
    """
    使用与preprocess函数相同的LetterBox逻辑转换边界框，支持批量处理
    
    Args:
        bboxes (torch.Tensor): 邊界框 tensor，形狀為 [N, 4]，格式為 xywh
        batch_idx (torch.Tensor, optional): 每個邊界框對應的批次索引，形狀為 [N, 1]
        original_shape (list): 每個圖像的原始形狀列表，[(h1,w1), (h2,w2), ...]
        imgsz (int or tuple): 目标尺寸，与preprocess中的imgsz相同
        stride (int): 步长，与preprocess中的stride相同
        
    Returns:
        torch.Tensor: 转换后的边界框，格式为 [N, 4]
    """
    # print("bboxes", describe_var(bboxes))
    # print("batch_idx", describe_var(batch_idx))
    # print("original_shape", describe_var(original_shape))
    # print("imgsz", describe_var(imgsz))
    # print("stride", describe_var(stride))

    import torch
    
    # 确保输入是tensor
    if not isinstance(bboxes, torch.Tensor):
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
    
    # 确保imgsz是元组
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    
    # 处理使用batch_idx的情况（新添加）
    if batch_idx is not None:
        if not isinstance(batch_idx, torch.Tensor):
            batch_idx = torch.tensor(batch_idx, dtype=torch.float32)
        
        # 获取唯一的batch索引
        unique_batches = batch_idx.unique()
        num_batches = len(unique_batches)
        
        # 确保original_shape长度匹配batch数量
        if len(original_shape) != num_batches:
            raise ValueError(f"原始形狀列表長度({len(original_shape)})與批次數量({num_batches})不匹配")
        
        # 为每个边界框应用对应的变换
        transformed_bboxes = torch.zeros_like(bboxes)
        for i, batch_id in enumerate(unique_batches):
            # 找出当前batch的所有box的索引
            mask = (batch_idx.view(-1) == batch_id)
            batch_boxes = bboxes[mask]
            
            print("mask shape:", mask.shape)
            print("batch_boxes shape:", batch_boxes.shape)
            print("bboxes shape:", bboxes.shape)
            
            h, w = original_shape[i]
            
            # 计算缩放比例
            r = min(imgsz[0] / h, imgsz[1] / w)
            
            # 计算padding
            new_unpad = int(round(w * r)), int(round(h * r))
            dw, dh = imgsz[1] - new_unpad[0], imgsz[0] - new_unpad[1]
            dw /= 2  # 居中padding
            dh /= 2
            
            # 计算具体padding
            top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
            
            # 检查是否归一化
            is_normalized = torch.max(batch_boxes) <= 1.0
            
            # 如果是归一化坐标，先转为绝对坐标
            if is_normalized:
                bbox_abs = batch_boxes.clone()
                bbox_abs[:, 0] *= w
                bbox_abs[:, 1] *= h
                bbox_abs[:, 2] *= w
                bbox_abs[:, 3] *= h
            else:
                bbox_abs = batch_boxes.clone()
            
            # 应用缩放和padding
            bbox_abs[:, 0] = bbox_abs[:, 0] * r + left  # x
            bbox_abs[:, 1] = bbox_abs[:, 1] * r + top   # y
            bbox_abs[:, 2] = bbox_abs[:, 2] * r         # width
            bbox_abs[:, 3] = bbox_abs[:, 3] * r         # height
            
            # 重新归一化
            if is_normalized:
                bbox_norm = bbox_abs.clone()
                bbox_norm[:, 0] /= imgsz[1]
                bbox_norm[:, 1] /= imgsz[0]
                bbox_norm[:, 2] /= imgsz[1]
                bbox_norm[:, 3] /= imgsz[0]
                transformed_bboxes[mask] = bbox_norm
            else:
                transformed_bboxes[mask] = bbox_abs
        
        return transformed_bboxes
    
    # 以下是原有逻辑（处理没有batch_idx的情况）
    # 处理单图片和批次图片的情况
    is_batch = isinstance(original_shape, (list, tuple)) and isinstance(original_shape[0], (list, tuple))
    
    if not is_batch:
        # 单图片情况
        original_shapes = [original_shape]
        if not isinstance(bboxes[0], (list, tuple, np.ndarray, torch.Tensor)):
            # 如果是单个边界框 [x,y,w,h]
            bboxes_list = [bboxes]
        else:
            # 如果是多个边界框 [[x,y,w,h], ...]
            bboxes_list = bboxes
    else:
        # 批次情况
        original_shapes = original_shape
        if isinstance(bboxes[0], (list, tuple, np.ndarray, torch.Tensor)):
            # 如果是边界框列表 [[x,y,w,h], ...]
            bboxes_list = bboxes
        else:
            # 单个边界框展开成列表
            bboxes_list = [bboxes]
    
    # 确保长度匹配
    if len(bboxes_list) != len(original_shapes):
        # 如果只有一个边界框但有多个图片，复制边界框
        if len(bboxes_list) == 1 and len(original_shapes) > 1:
            bboxes_list = bboxes_list * len(original_shapes)
        else:
            raise ValueError(f"边界框数量({len(bboxes_list)})与图片数量({len(original_shapes)})不匹配")
    
    # 转换所有边界框
    transformed_bboxes = []
    for i, (bbox, shape) in enumerate(zip(bboxes_list, original_shapes)):
        h, w = shape
        
        # 计算缩放比例
        r = min(imgsz[0] / h, imgsz[1] / w)
        
        # 计算padding
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = imgsz[1] - new_unpad[0], imgsz[0] - new_unpad[1]
        dw /= 2  # 居中padding
        dh /= 2
        
        # 计算具体padding
        top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
        
        # 转为numpy数组
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        bbox = np.array(bbox).reshape(-1, 4)
        
        # 检查是否归一化
        is_normalized = np.max(bbox) <= 1.0
        
        # 如果是归一化坐标，先转为绝对坐标
        if is_normalized:
            bbox_abs = bbox.copy()
            bbox_abs[:, 0] *= w
            bbox_abs[:, 1] *= h
            bbox_abs[:, 2] *= w
            bbox_abs[:, 3] *= h
        else:
            bbox_abs = bbox.copy()
        
        # 应用缩放和padding
        bbox_abs[:, 0] = bbox_abs[:, 0] * r + left  # x
        bbox_abs[:, 1] = bbox_abs[:, 1] * r + top   # y
        bbox_abs[:, 2] = bbox_abs[:, 2] * r         # width
        bbox_abs[:, 3] = bbox_abs[:, 3] * r         # height
        
        # 重新归一化
        if is_normalized:
            bbox_norm = bbox_abs.copy()
            bbox_norm[:, 0] /= imgsz[1]
            bbox_norm[:, 1] /= imgsz[0]
            bbox_norm[:, 2] /= imgsz[1]
            bbox_norm[:, 3] /= imgsz[0]
            transformed_bboxes.append(bbox_norm)
        else:
            transformed_bboxes.append(bbox_abs)
    
    # 组合结果并转换为tensor
    result = np.vstack([b.reshape(-1, 4) for b in transformed_bboxes])
    return torch.tensor(result, dtype=torch.float32)