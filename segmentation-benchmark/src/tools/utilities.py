import os
import json

import cv2
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import time

class BaseDetector:
    def __init__(self, name):
        self.name = name

    def run(self, image_path):
        pass  # Implement bounding box detection logic here

def generate_report(data):
    # return markdown table 
    pass

def load_dataset(coco_path):
    with open(coco_path, 'r') as file:
        data = json.load(file)
    return data


def annotate_image(img_path, pred_boxes, color="red", output_path=None):
    # Load the image
    img = cv2.imread(img_path)

    # Define color mappings
    color_map = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255)
    }

    # Get the BGR color
    box_color = color_map.get(color.lower(), (0, 0, 255))  # Default to red

    # Draw bounding boxes
    for (x, y, w, h) in pred_boxes:
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(img, top_left, bottom_right, box_color, thickness=2)

    if output_path:
        # Save the annotated image
        cv2.imwrite(output_path, img)
    else:
        # Display the image
        cv2.imshow("Annotated Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def compute_iou(boxA, boxB):
    """Compute the Intersection over Union (IoU) between two bounding boxes in (x, y, w, h) format."""

    # Convert (x, y, w, h) -> (x_min, y_min, x_max, y_max)
    xA_min, yA_min, wA, hA = boxA
    xB_min, yB_min, wB, hB = boxB

    xA_max, yA_max = xA_min + wA, yA_min + hA
    xB_max, yB_max = xB_min + wB, yB_min + hB

    # Compute intersection
    xI_min = max(xA_min, xB_min)
    yI_min = max(yA_min, yB_min)
    xI_max = min(xA_max, xB_max)
    yI_max = min(yA_max, yB_max)

    inter_width = max(0, xI_max - xI_min)
    inter_height = max(0, yI_max - yI_min)
    interArea = inter_width * inter_height

    # Compute areas of both boxes
    areaA = wA * hA
    areaB = wB * hB

    # Compute IoU
    unionArea = areaA + areaB - interArea
    iou = interArea / unionArea if unionArea > 0 else 0

    return iou

def save_benchmark(name, result):
    metrics_file_path = "./segmentation-benchmark/output/metrics.json"
    metrics_dir = os.path.dirname(metrics_file_path)

    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    try:
        with open(metrics_file_path, 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = {}

    metrics[name] = result

    with open(metrics_file_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        print("Benchmarks saved to segmentation-benchmark/output folder")


def run_benchmarks(detector, dataset):
    """Benchmark object detector using COCO-formatted annotations."""
    result = {
        "avg_processing_time": 0,
        "avg_iou": 0,
        "avg_precision": 0,
        "avg_recall": 0,
        "avg_f1_score": 0,
        "images": {}
    }

    # results = {
    #     "images": [],
    #     "precision": [],
    #     "recall": [],
    #      "f1_score": [],
    #     "processing_time": [],
    #     "iou_scores": []
    # }

    # Create a mapping from image_id to its bounding boxes
    image_id_to_boxes = {}
    for ann in dataset["annotations"]:
        image_id = ann["image_id"]
        bbox = ann["bbox"]  # COCO format: [x, y, width, height]
        if image_id not in image_id_to_boxes:
            image_id_to_boxes[image_id] = []
        image_id_to_boxes[image_id].append(bbox)

    # Process each image
    for img_data in dataset["images"]:
        img_id = img_data["id"]
        result["images"][img_id] = {}
        print("Processing " + str(img_id))
        img_path = "./segmentation-benchmark/dataset/images/" + img_data["file_name"]
        print("image: " + str(img_id))

        if img_id not in image_id_to_boxes:
            continue  # Skip images without annotations

        gt_boxes = image_id_to_boxes[img_id]

        # Run detector
        start_time = time.time()
        pred_boxes = detector.predict(img_path)  # List of (x, y, w, h)
        processing_time = time.time() - start_time
        result["images"][img_id]["processing_time"] = processing_time

        # Compute IoU and classification metrics
        true_labels = [1] * len(gt_boxes)  # All ground truth boxes are positive
        pred_labels = []
        ious = []

        output_directory = f"./segmentation-benchmark/output/{detector.name}/"
        os.makedirs(output_directory, exist_ok=True)

        output_file = output_directory + str(img_id) + '.png'

        # save annotated image returned from detector.predict
        annotate_image(img_path, gt_boxes, "green", output_file)
        annotate_image(output_file, pred_boxes, "red", output_file)

        for gt_box in gt_boxes:
            max_iou = max((compute_iou(gt_box, pred_box) for pred_box in pred_boxes), default=0)
            ious.append(max_iou)
            pred_labels.append(1 if max_iou > 0.5 else 0)
        result["images"][img_id]["iou_scores"] = ious
        result["images"][img_id]["iou_avg"] = np.average(ious)
        result["images"][img_id]["precision"] = precision_score(true_labels, pred_labels, zero_division=1)
        result["images"][img_id]["recall"] = recall_score(true_labels, pred_labels, zero_division=1)
        result["images"][img_id]["f1_score"] = f1_score(true_labels, pred_labels, zero_division=1)


    # Compute average metrics for the tool
    result["avg_precision"] = np.average([img["precision"] for img in result["images"].values()])
    result["avg_recall"] = np.average([img["recall"] for img in result["images"].values()])
    result["avg_f1_score"] = np.average([img["f1_score"] for img in result["images"].values()])
    result["avg_processing_time"] = np.average([img["processing_time"] for img in result["images"].values()])
    result["avg_iou"] = np.average([img["iou_avg"] for img in result["images"].values()])

    for image in result["images"]:
        result["images"][image]["iou_scores"] = []

    return result


