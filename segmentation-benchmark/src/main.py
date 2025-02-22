import json
# Main file
# import tools.paddle_detector
from tools.tesseract_detector import TesseractDetector
from tools.easyocr_detector import EasyocrDetector
from tools.paddle_detector import PaddleDetector
from tools.utilities import load_dataset, run_benchmarks

# load coco file

dataset = load_dataset("./segmentation-benchmark/dataset/coco.json")

tesseract1 = TesseractDetector('tesseract1')
paddle1 = PaddleDetector('paddle1')
easyocr1 = EasyocrDetector("easyocr 0.2")

easyocr1.threshold = 0.2

def main():
    # detectors = [tesseract1, easyocr1, paddle1]
    detectors = [tesseract1, easyocr1, paddle1]

    for detector in detectors:
        result = run_benchmarks(detector, dataset)
        save_benchmark(detector.name, result)

main()
