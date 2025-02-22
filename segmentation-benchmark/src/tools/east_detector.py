from utilities import BaseDetector, run_benchmarks, save_benchmark, load_dataset
import cv2
import numpy as np

class EASTDetector(BaseDetector):
    name = "east"
    threshold = 0.5  # Confidence threshold

    def __init__(self, name, east_model_path="segmentation-benchmark/dataset/frozen_east_text_detection.pb"):
        self.name = name
        self.net = cv2.dnn.readNet(east_model_path)

    def predict(self, image_path):
        image = cv2.imread(image_path)
        orig_height, orig_width = image.shape[:2]

        # Resize for EAST model
        new_width, new_height = 320, 320
        image_resized = cv2.resize(image, (new_width, new_height))
        ratio_width = orig_width / new_width
        ratio_height = orig_height / new_height

        blob = cv2.dnn.blobFromImage(image_resized, 1.0, (new_width, new_height),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        scores, geometry = self.net.forward(["feature_fusion/Conv_7/Sigmoid",
                                             "feature_fusion/concat_3"])

        # Decode results
        bounding_boxes = []
        num_rows, num_cols = scores.shape[2:4]
        for y in range(num_rows):
            for x in range(num_cols):
                score = scores[0, 0, y, x]
                if score < self.threshold:
                    continue

                # Get bounding box
                offset_x, offset_y = x * 4.0, y * 4.0
                angle = geometry[0, 4, y, x]
                cos, sin = np.cos(angle), np.sin(angle)
                h, w = geometry[0, 0, y, x], geometry[0, 1, y, x]
                end_x = int(offset_x + (cos * w) + (sin * h))
                end_y = int(offset_y - (sin * w) + (cos * h))
                start_x, start_y = int(end_x - w), int(end_y - h)

                # Scale back to original image
                bounding_boxes.append((int(start_x * ratio_width), int(start_y * ratio_height),
                                       int(w * ratio_width), int(h * ratio_height)))

        return bounding_boxes

detector = EASTDetector("EAST")

dataset = load_dataset("./segmentation-benchmark/dataset/coco.json")
result = run_benchmarks(detector, dataset)

save_benchmark(detector.name, result)
