# There can be different metrics for association:
# - IoU (this takes care of occlusion by another car, I think)
# - Distance between bounding boxes (can also do Cosine distance)
# - Feature of bounding box

def corners_to_center(box):
    """
    Convert bounding box from corner format (x1, y1, x2, y2) to center format (x, y, w, h).
    """
    x = int((box[0] + box[2]) / 2)
    y = int((box[1] + box[3]) / 2)
    w = int(box[2] - box[0])
    h = int(box[3] - box[1])
    return [x, y, w, h]


def center_to_corners(box):
    """
    Convert bounding box from center format (x, y, w, h) to corner format (x1, y1, x2, y2).
    """
    x1 = int(box[0] - (box[2] / 2))
    y1 = int(box[1] - (box[3] / 2))
    x2 = int(box[0] + (box[2] / 2))
    y2 = int(box[1] + (box[3] / 2))
    return [x1, y1, x2, y2]


def xywh_to_corners(box):
    """
    Convert bounding box from XYWH format (x, y, w, h) to corner format (x1, y1, x2, y2).
    """
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def calculate_iou(box_a, box_b, xywh_rep=False):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box_a: Bounding box A in the format (x1, y1, x2, y2) or other formats.
    - box_b: Bounding box B in the format (x1, y1, x2, y2) or other formats.
    - xywh_rep: Boolean indicating if the input boxes are in XYWH format.

    Returns:
    - IoU: Intersection over Union between the two bounding boxes.
    """
    if not xywh_rep:  # Convert to corner representation
        box_a = center_to_corners(box_a)
        box_b = center_to_corners(box_b)
    else:
        box_a = xywh_to_corners(box_a)
        box_b = xywh_to_corners(box_b)

    # Calculate the coordinates of the intersection rectangle
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    # Calculate the area of the intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both bounding boxes
    area_box_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_box_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Calculate the union area
    union_area = area_box_a + area_box_b - intersection_area

    # Calculate the Intersection over Union (IoU)
    iou = intersection_area / union_area

    return iou
