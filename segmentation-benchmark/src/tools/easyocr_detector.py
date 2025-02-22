import easyocr
from PIL import Image

class EasyocrDetector:
    name = "easyocr"
    threshold = 0.5

    def __init__(self, name=None):
        if name:
            self.name = name
        self.reader = easyocr.Reader(['fa', 'en'])  # Initialize the EasyOCR reader

    def predict(self, image_path):
        print(image_path)
        image = Image.open(image_path)

        # Use EasyOCR to do detection on the image
        results = self.reader.readtext(image_path, text_threshold= self.threshold)

        # Extract bounding boxes
        bounding_boxes = []
        for (bbox, text, prob) in results:
            # bbox is a list of 4 points: top-left, top-right, bottom-right, bottom-left
            x0, y0 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            w, h = x2 - x0, y2 - y0
            bounding_boxes.append((x0, y0, w, h))

        return bounding_boxes

