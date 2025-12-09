# Use the existing Lambda Python 3.13 image with the model
FROM agrigorev/model-2025-hairstyle:v1

RUN pip install onnxruntime Pillow numpy

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]
