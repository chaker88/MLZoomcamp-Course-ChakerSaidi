import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

# --- Load the ONNX model ---
session = ort.InferenceSession("hair_classifier_empty.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Helper functions ---

def download_image(url):
    """Download image from URL and return PIL Image"""
    with request.urlopen(url) as resp:
        buffer = resp.read()
    img = Image.open(BytesIO(buffer))
    return img

def prepare_image(img, target_size=(200, 200)):
    """Resize and convert to RGB"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    """Convert PIL image to normalized float32 numpy array with shape (1, C, H, W)"""
    arr = np.array(img).astype(np.float32) / 255.0  # ensure float32
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)   # add batch dimension
    return arr

# --- Lambda handler ---
def lambda_handler(event, context):
    """
    event = {
        "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    }
    """
    url = event.get("url")
    if url is None:
        return {"error": "No URL provided."}

    try:
        # Download and prepare image
        img = download_image(url)
        img = prepare_image(img)
        arr = preprocess_image(img)

        # Run ONNX inference
        result = session.run([output_name], {input_name: arr})[0][0][0]

        # Sigmoid for binary classification
        prob = 1 / (1 + np.exp(-result))
        label = "curly" if prob > 0.5 else "straight"

        return {
            "label": label,
            "probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}
    

    
if __name__ == "__main__":
    # Example test URL
    test_event = {
        "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
    }
    result = lambda_handler(test_event, None)
    print(result)
