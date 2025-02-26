from paddleocr import PaddleOCR
from tools.utilities import BaseOCR
import re

# Function to check if a text is Persian
def is_persian(text):
    # Persian characters range
    persian_regex = re.compile(r'[\u0600-\u06FF]+')
    return bool(persian_regex.search(text))

# Reverse Persian text in the OCR result
def reverse_persian_text(text):
    if text:
        if is_persian(text):  # Check if the text is Persian
            return text[::-1]  # Reverse Persian text
        else:
            return text
    else:
        return text


class Paddle_OCR(BaseOCR):
    name: str = "paddle"
    def __init__(self, name):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="fa", rec_image_shape="3, 32, 256", det_db_box_thresh=0.6)
        self.name = name

    def predict(self, image_path):
        result = self.ocr.ocr(image_path, cls=True)

        if not result or not result[0]:  # Ensure result is valid
            return [("", 0.0)]  # Return empty text and confidence of 0

        text_confidence = []
        for line in result[0]:
            text = line[1][0]  # Extracted text
            confidence = line[1][1]  # Confidence score

            # Append the text and confidence to the list
            text_confidence.append((reverse_persian_text(text), confidence))

        return list(reversed(text_confidence))
