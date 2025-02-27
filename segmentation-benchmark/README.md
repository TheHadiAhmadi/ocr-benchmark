# Text segmentation benchmarks

## Similar benchmark:
https://rrc.cvc.uab.es/?ch=2&com=evaluation&task=1

## Download EAST model
to download east model you should run
```
wget https://raw.githubusercontent.com/oyyd/frozen_east_text_detection.pb/refs/heads/master/frozen_east_text_detection.pb
 mv frozen_east_text_detection.pb segmentation-benchmark/dataset/
```

## Python version
file `segmentation-benchmark/src/main.py` needs python 3.12 (I tested with 3.12.0) and file `segmentation-benchmark/src/tools/east_detector.py` needs another version (...)

```
conda create --name ocr_env python=3.12.9
conda activate ocr_env

conda install -c pytorch torch torchvision
conda install -c conda-forge opencv numpy pillow easyocr pytesseract scikit-learn 
pip install paddlepaddle paddleocr

conda create --name ocr_east_env python=3.10.16
conda activate ocr_east_env

conda install -c conda-forge numpy opencv pillow scikit-learn
```


1. to change dataset you need to add images to segmentation-benchmark/dataset/images folder.
2. to annotate images you can use roboflow.com site and copy annotated data(json) to segmentation-benchmark/dataset/annotations/{file_name}.json
3. to generate coco based dataset, run 
```
python segmentation-benchmark/scripts/generate_coco.py
```
it will create segmentation-benchmark/dataset/coco.json file

4. to run benchmarks, you can run
```
python segmentation-benchmark/src/main.py
```
output will be generated in segmentation-benchmark/output folder.


to generate html of outputs, run 
```
python segmentation-benchmark/scripts/generate_output.py
```
segmentation-benchmark/output/metrics_comparison.html file will be created which you can open in browser.

