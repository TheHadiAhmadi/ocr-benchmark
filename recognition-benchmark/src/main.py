import json
from tools.tesseract_ocr import Tesseract_OCR
from tools.easyocr_ocr import Easyocr_OCR
from tools.paddle_ocr import Paddle_OCR
from tools.utilities import load_dataset, run_benchmarks, save_benchmark

# load coco file

dataset = load_dataset("./recognition-benchmark/dataset")
print(dataset)

tesseract1 = Tesseract_OCR('tesseract1')
paddle1 = Paddle_OCR('paddle1')
easyocr1 = Easyocr_OCR("easyocr")

def main():
    # detectors = [tesseract1, easyocr1, paddle1]
    detectors = [tesseract1, easyocr1, paddle1]
    # detectors = [paddle1]

    for detector in detectors:
        result = run_benchmarks(detector, dataset)
        save_benchmark(detector.name, result)

main()
