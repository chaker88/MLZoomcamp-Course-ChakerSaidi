import onnxruntime as ort

session = ort.InferenceSession("hair_classifier_v1.onnx")
print("Input names:", [i.name for i in session.get_inputs()])
print("Output names:", [o.name for o in session.get_outputs()])
print("done")