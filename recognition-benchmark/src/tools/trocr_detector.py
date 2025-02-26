import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load the pre-trained TrOCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Set the model to evaluation mode
model.eval()

# Load your image (make sure it's in a suitable format)
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)

# Preprocess the image
pixel_values = processor(image, return_tensors="pt").pixel_values

# Forward pass to get predictions
with torch.no_grad():
    outputs = model.generate(pixel_values)

# Decode predictions and output bounding boxes
decoded_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display detected text
print("Detected Text:", decoded_text)

