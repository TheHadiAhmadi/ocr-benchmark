from tools.utilities import BaseOCR
from PIL import Image
import pytesseract


class Tesseract_OCR(BaseOCR):
    name = "tesseract"
    threshold = 0.5
    def __init__(self, name):
        self.name = name
    def predict(self, image_path):
        image = Image.open(image_path)

        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang="fas+eng")

        text_confidence = []
        for i in range(len(data['level'])):
            if int(data['conf'][i]) > 0:  # Only consider boxes with a certain confidence
                text = data['text'][i].strip()  # Extract the recognized text
                confidence = int(data['conf'][i])  # Extract the confidence score
                if text:  # Ensure there's text recognized
                    text_confidence.append((text, confidence))

        return text_confidence
