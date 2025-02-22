from craft_text_detector import Craft
import cv2
from utilities import BaseDetector, load_dataset, run_benchmarks

class CRAFTDetector(BaseDetector):
    def __init__(self, name):
        self.name = name

    def predict(self, image_path):
        craft = Craft(output_dir="segmentation-benchmark/output/" + self.name, crop_type="poly", cuda= False)
        result = self.craft.detect_text(image_path)

        bounding_boxes = []
        for box in result["boxes"]:
            x_min = int(min(point[0] for point in box))
            y_min = int(min(point[1] for point in box))
            x_max = int(max(point[0] for point in box))
            y_max = int(max(point[1] for point in box))
            bounding_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))  # (x, y, width, height)

            craft.unload_craftnet_model()
            craft.unload_refinenet_model()
        return bounding_boxes


detector = CRAFTDetector("CRAFT")

dataset = load_dataset("./segmentation-benchmark/dataset/coco.json")
result = run_benchmarks(detector, dataset)

save_benchmark(detector.name, result)
