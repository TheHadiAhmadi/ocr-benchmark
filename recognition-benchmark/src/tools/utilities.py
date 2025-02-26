import os
import json
import re
import cv2
import json
import Levenshtein
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import time

def render_text(text):
    return text.replace('ي', 'ی').replace('١', '۱').replace('٢', '۲').replace('٣', '۳').replace('٤', '۴').replace('٥', '۵').replace('٦', '۶').replace('٧', '۷').replace('٨', '۸').replace('٩', '۹').replace('٠', '۰')

class BaseOCR:
    def __init__(self, name):
        self.name = name

    def run(self, image_path):
        pass  # Implement bounding box detection logic here

def generate_report(data):
    # return markdown table 
    pass

def load_dataset(dataset_folder):
    labels_dir = dataset_folder +"/" + "labels.json"
    with (open(labels_dir, 'r') as f):
        labels = json.loads(f.read())
    return labels


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
            return text;
    else:
        return text


def save_benchmark(name, result):
    metrics_file_path = "./recognition-benchmark/output/metrics.json"
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
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        print("Benchmarks saved to recognition-benchmark/output folder")


def run_benchmarks(detector, dataset):
    result = {
        "name": detector.name,
        "avg_confidence": 0.0,
        "avg_processing_time": 0.0,
        "avg_cer_text": 0.0,
        "avg_wer_text": 0.0,
        "avg_cer_number": 0.0,
        "avg_wer_number": 0.0,
        "images": {}
    }

    for img_name in dataset:

        ground_truth = render_text(dataset[img_name])
        result["images"][img_name] = {}
        result["images"][img_name]['text'] = ground_truth
        print("Processing " + img_name)
        img_path = "./recognition-benchmark/dataset/images/" + img_name + '.png'

        # Run ocr
        start_time = time.time()
        print(img_path)
        response = detector.predict(img_path)  # List of (text, confidence)

        recognized = render_text(" ".join([text for text, _ in response]))  # Concatenates all text values
        conf = sum([conf for _, conf in response]) / len(response) if response else 0


        processing_time = time.time() - start_time
        result['images'][img_name]['recognized'] = recognized
        result["images"][img_name]["processing_time"] = processing_time
        result['images'][img_name]['confidence'] = conf

        char_distance = Levenshtein.distance(ground_truth, render_text(recognized))
        char_error_rate = char_distance / max(1, len(ground_truth))

        expected_words = ground_truth.split(' ')
        recognized_words = render_text(recognized).split(' ')
        word_distance = Levenshtein.distance(" ".join(expected_words), " ".join(recognized_words))
        word_error_rate = word_distance / max(1, len(expected_words))

        result["images"][img_name]["cer"] = char_error_rate
        result["images"][img_name]["wer"] = word_error_rate


    cer_numbers = []
    wer_numbers = []
    cer_text = []
    wer_text = []

    # Iterate through the images in the result
    for img_name, img_data in result["images"].items():
        if img_name.startswith('n'):
            cer_numbers.append(img_data["cer"])
            wer_numbers.append(img_data["wer"])
        elif img_name.startswith('t'):
            cer_text.append(img_data["cer"])
            wer_text.append(img_data["wer"])

    # Calculate averages
    result["avg_cer_number"] = np.mean(cer_numbers) if cer_numbers else 0
    result["avg_wer_number"] = np.mean(wer_numbers) if wer_numbers else 0
    result["avg_cer_text"] = np.mean(cer_text) if cer_text else 0
    result["avg_wer_text"] = np.mean(wer_text) if wer_text else 0

    result["avg_processing_time"] = np.mean([img["processing_time"] for img in result["images"].values()])
    result["avg_confidence"] = np.mean([img["confidence"] for img in result["images"].values()])

    return result


