import json
import os
import cv2

def generate_coco(json_files, image_dir):
    """
    Convert ground truth files to COCO format.

    Args:
        json_files (list): List of paths to JSON files containing ground truth annotations.
        image_dir (str): Directory containing the images.

    Returns:
        dict: COCO-formatted dictionary.
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "words", "supercategory": "text"}]
    }

    annotation_id = 1  # Annotation IDs must be unique across all images

    for idx, json_file in enumerate(json_files, start=1):
        # Load the JSON file
        with open(json_file, 'r') as f:
            data = json.loads(f.read())

        # Extract image filename
        image_filename = os.path.splitext(os.path.basename(json_file))[0] + ".jpeg"
        image_path = os.path.join(image_dir, image_filename)

        # Get image dimensions (assuming images are in the same directory)
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        height, width, _ = img.shape
        
        item = {
            "id": idx,
            "file_name": image_filename,
            "width": width,
            "height": height
        }
        # Add image entry to COCO format
        coco_data["images"].append(item)
        # Process annotations
        for annotation in data["boxes"]:
            if annotation['label'] == 'words' or annotation['label'] == 'texts':  # Only consider 'words' label
                x = int(float(annotation['x']))
                y = int(float(annotation['y']))
                width = int(float(annotation['width']))
                height = int(float(annotation['height']))
                box = {
                    "id": annotation_id,
                    "image_id": idx,
                    "category_id": 1,  # Assuming 'words' is category 1
                    "bbox": [x - width/2, y - height / 2, width, height],
                    "area": width * height,
                    "iscrowd": 0
                }
                # Add annotation entry to COCO format
                coco_data["annotations"].append(box)
                annotation_id += 1

    return coco_data

# Example usage
if __name__ == "__main__":
    raw_path = "./segmentation-benchmark/dataset/annotations"

    files = [raw_path + "/" + f for f in os.listdir(raw_path) if f.endswith(".json")]
    
    image_dir = "./segmentation-benchmark/dataset/images"  # Replace with the directory containing the images

    coco_data = generate_coco(files, image_dir)

    # Save the COCO-formatted data to a JSON file
    output_file = "./segmentation-benchmark/dataset/coco.json"
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f"COCO-formatted data saved to {output_file}")
