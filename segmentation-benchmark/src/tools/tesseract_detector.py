from tools.utilities import BaseDetector
from PIL import Image
import pytesseract


class TesseractDetector(BaseDetector):
    name = "tesseract"
    threshold = 0.5
    def __init__(self, name):
        self.name = name
    def predict(self, image_path):
        print(image_path)
        image = Image.open(image_path)

        # Use Tesseract to do OCR on the image
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Extract bounding boxes
        bounding_boxes = []
        for i in range(len(data['level'])):
            if int(data['conf'][i]) > 0:  # Only consider boxes with a certain confidence
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                bounding_boxes.append((x, y, w, h))
        return bounding_boxes

