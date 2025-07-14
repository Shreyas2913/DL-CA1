import onnxruntime as ort
import numpy as np
import cv2
import os
import json

# Paths
frames_dir = "frames"
model_path = "xception_df.onnx"
output_json = "predictions.json"

# Load ONNX model
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

# Preprocess function
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (299, 299))
    img = img[:, :, ::-1]  # BGR to RGB
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # CHW
    img = np.expand_dims(img, axis=0)  # NCHW
    return img

# Run inference
predictions = {}
for fname in os.listdir(frames_dir):
    if fname.endswith(".jpg"):
        fpath = os.path.join(frames_dir, fname)
        img = preprocess_image(fpath)
        outputs = session.run(None, {"input": img})

        output_array = outputs[0]
        if isinstance(output_array, list):
            output_array = np.array(output_array)

        if output_array.ndim == 2 and output_array.shape[1] == 2:
            prob_fake = float(output_array[0, 1])  # class 1 = fake
            predictions[fname] = prob_fake
            print(f"✅ {fname}: {prob_fake:.4f}")
        else:
            raise ValueError(f"Unexpected output shape: {output_array.shape}")

# Save predictions
with open(output_json, "w") as f:
    json.dump(predictions, f, indent=2)
print(f"\n✅ All predictions saved to {output_json}")

