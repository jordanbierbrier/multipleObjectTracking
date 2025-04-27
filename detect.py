import cv2
import numpy as np


def detect_cars_single_image(frame):
    """
    Detect cars in a single image frame using the YOLOv3 object detection model.
    
    Args:
        frame (numpy.ndarray): The input image frame in the form of a NumPy array 
                               (height x width x channels).

    Returns:
        tuple: A tuple containing:
            - updated_boxes (list of list of int): A list of bounding boxes for detected cars, 
              where each bounding box is represented as [x, y, width, height].
            - updated_confidences (list of float): A list of confidence scores corresponding 
              to the detected cars.
    """
    # Load YOLOv4
    net = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')

    layer_names = net.getUnconnectedOutLayersNames()

    # Get image dimensions
    height, width, _ = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layer_names)

    # Initialize lists for detected objects
    car_class = 2  # Class ID for 'car' in COCO dataset
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == car_class:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-maximum suppression to remove duplicate and low-confidence detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    updated_boxes = []
    updated_confidences = []

    # Collect final bounding boxes and confidences
    for i in indices.flatten():
        updated_boxes.append(boxes[i])
        updated_confidences.append(confidences[i])

    return updated_boxes, updated_confidences
