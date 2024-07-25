import numpy as np

def generate_boxes(box_size):
    return [ [i+1, i+1, i+1, i+1] for i in range(box_size) ]

def non_max_suppresions(boxes, scores, iou_threshold):
    """
    Perform non-maximum suppression.

    Parameters:
    boxes (numpy.ndarray): Array of bounding boxes, each defined by [x1, y1, x2, y2].
    scores (numpy.ndarray): Array of confidence scores for each bounding box.
    iou_threshold (float): Overlap threshold for suppression.

    Returns:
    numpy.ndarray: Indices of bounding boxes to keep.
    """
    if len(boxes) == 0: return np.array([])

    X_o = boxes[:, 0]
    Y_o = boxes[:, 1]
    X_1 = boxes[:, 2]
    Y_1 = boxes[:, 3]

    areas = (X_1 - X_o + 1) * (Y_1 - Y_o + 1)

    sorted_score_indices = scores.argsort()[::-1]
    keep = np.array([], dtype=int)

    while sorted_score_indices.size > 0:
        i = sorted_score_indices[0]
        keep = np.append(keep, i)

        X_min = np.maximum(X_o[i], X_o[sorted_score_indices[1:]])
        Y_min = np.maximum(Y_o[i], Y_o[sorted_score_indices[1:]])
        X_max = np.minimum(X_1[i], X_1[sorted_score_indices[1:]])
        Y_max = np.minimum(Y_1[i], Y_1[sorted_score_indices[1:]])

        w = np.maximum(0, X_max - X_min + 1)
        h = np.maximum(0, Y_max - Y_min + 1)

        inter_area = w * h
        iou = inter_area / (areas[i] + areas[sorted_score_indices[1:]] - inter_area)

        keep_ious = np.where(iou <= iou_threshold)[0]
        sorted_score_indices = sorted_score_indices[keep_ious + 1]
    
    return keep



def main():
    # Base IOU
    boxes = [
        [12, 84, 140, 212],
        [24, 84, 152, 212],
        [36, 84, 164, 212],
        [12, 96, 140, 224],
        [24, 96, 152, 224],
    ]
    np_boxes = np.array(boxes)
    Xo_min, Yo_min, Xo_max, Yo_max = np_boxes[0, :]
    target_boxes = np_boxes[1:, :]

    np_X_min = target_boxes[:, 0]
    np_Y_min = target_boxes[:, 1]
    np_X_max = target_boxes[:, 2]
    np_Y_max = target_boxes[:, 3]

    area_box_o = (Xo_max - Xo_min + 1) * (Yo_max - Yo_min + 1) # min 1 pixel
    areas_box_1 = (np_X_max - np_X_min + 1) * (np_Y_max - np_Y_min + 1)

    np_X_1 = np.maximum(Xo_min, np_X_min)
    np_Y_1 = np.maximum(Yo_min, np_Y_min)
    np_X_2 = np.minimum(Xo_max, np_X_max)
    np_Y_2 = np.minimum(Yo_max, np_Y_max)

    w = np.maximum(0, np_X_2 - np_X_1 + 1)
    h = np.maximum(0, np_Y_2 - np_Y_1 + 1)

    inter_area = w * h

    iou = inter_area / (areas_box_1 + area_box_o - inter_area)

    # Improve IOU
    # Non maximum Suppresion
    # Test case 1
    boxes = np.array([
        [12, 84, 140, 212],
        [24, 84, 152, 212],
        [36, 84, 164, 212],
        [12, 96, 140, 224],
        [24, 96, 152, 224],
        [24, 108, 152, 236]
    ])
    scores = np.array([0.5, 0.3, 0.7, 0.4, 0.6, 0.2])
    iou_threshold = 0.3

    kept_indices = non_max_suppresions(boxes, scores, iou_threshold)
    kept_boxes = boxes[kept_indices] #[boxes[i] for i in kept_indices]

    print(kept_indices)
    print(kept_boxes)

    # Test case 2
    boxes = np.array([
        [100, 100, 210, 210],
        [105, 105, 215, 215],
        [150, 150, 250, 250]
    ])

    scores = np.array([0.9, 0.8, 0.7])
    iou_threshold = 0.5

    kept_indices = non_max_suppresions(boxes, scores, iou_threshold)
    kept_boxes = boxes[kept_indices]

    print(kept_indices)
    print(kept_boxes)

if __name__ == "__main__":
    main()
