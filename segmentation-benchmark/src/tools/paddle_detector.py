from paddleocr import PaddleOCR

class PaddleDetector:
    name: str
    def __init__(self, name):
        self.ocr = PaddleOCR(use_angle_cls=True, lang="fa", rec_image_shape="3, 32, 256", det_db_box_thresh=0.6)
        self.name = name

    def predict(self, image_path):
        result = self.ocr.ocr(image_path, cls=True)

        bboxes = []
        for line in result[0]:
            points = line[0]

            # Extract the top-left (x, y) and bottom-right (x, y) points
            x_min = min([point[0] for point in points])
            y_min = min([point[1] for point in points])
            x_max = max([point[0] for point in points])
            y_max = max([point[1] for point in points])

            # Calculate width and height
            w = x_max - x_min
            h = y_max - y_min

            # Append the bounding box in [x, y, w, h] format
            bboxes.append([x_min, y_min, w, h])

        return bboxes

