import onnxruntime as ort
import numpy as np
from prepare_img import download_image, prepare_image

# --- Load the ONNX model ---
session = ort.InferenceSession("hair_classifier_v1.onnx")

# Get input/output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# --- Preprocessing function ---
def preprocess_image(img):
    """Convert PIL image to normalized float32 numpy array with shape (1, C, H, W)"""
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    arr = np.expand_dims(arr, axis=0)   # add batch dimension
    return arr

# --- Lambda handler ---
def lambda_handler(event, context):
    url = event["url"]  # assume the URL is provided
    img = download_image(url)
    img = prepare_image(img, target_size=(200, 200))
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
