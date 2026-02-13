import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[4], boxB[4])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[3] - boxA[1]) * (boxA[4] - boxA[2])
    areaB = (boxB[3] - boxB[1]) * (boxB[4] - boxB[2])

    return inter / float(areaA + areaB - inter)


def compute_metrics(gt_boxes, pred_boxes, iou_thresh=0.5):
    """
    gt_boxes   : [(class, x1, y1, x2, y2), ...]
    pred_boxes : [(class, x1, y1, x2, y2), ...]
    """

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0, 0.0, 0.0

    matched_gt = set()
    tp = 0

    for p in pred_boxes:
        for i, g in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            if p[0] == g[0] and iou(p, g) >= iou_thresh:
                tp += 1
                matched_gt.add(i)
                break

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ap = precision  # single-image AP approximation

    return precision, recall, ap
