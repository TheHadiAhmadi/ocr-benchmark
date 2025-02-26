import easyocr
from PIL import Image
from tools.utilities import BaseOCR

class Easyocr_OCR(BaseOCR):
    name = "easyocr"
    threshold = 0.5

    def __init__(self, name=None):
        if name:
            self.name = name
        self.reader = easyocr.Reader(['fa', 'en'])  # Initialize the EasyOCR reader

    def predict(self, image_path):
        image = Image.open(image_path)

        # Use EasyOCR to do detection on the image
        results = self.reader.readtext(image_path, text_threshold= self.threshold)

        text_confidence = []
        for (bbox, text, prob) in results:
            text_confidence.append((text, prob))

        return text_confidence
