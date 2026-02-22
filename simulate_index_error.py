import os
import numpy as np
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
import sys

# Mock Flask/App variables
ENSEMBLE_DIR = r"C:\food_calorie_project\ensemble_models"
MODEL_PATHS = [
    os.path.join(ENSEMBLE_DIR, "ensemble_model1_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "ensemble_model3_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "model2_v2.keras")
]
CLASSES = ["chicken_curry", "donuts", "french_fries", "fried_rice", "ice_cream", "omelette", "pizza", "samosa"]

class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# Load models
ensemble_models = []
for p in MODEL_PATHS:
    if os.path.exists(p):
        print(f"Loading {p}...")
        try:
            m = tf.keras.models.load_model(p, custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D}, compile=False)
            ensemble_models.append(m)
        except Exception as e:
            print(f"Error loading {p}: {e}")

yolo_model = YOLO("yolov8n.pt")

def predict_ensemble(img_array):
    print("  Entering predict_ensemble...")
    if not ensemble_models:
        return "Unknown", 0.0
    img = Image.fromarray(img_array.astype('uint8')).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = (img / 127.5) - 1
    input_data = np.expand_dims(img, axis=0)
    all_preds = []
    for i, m in enumerate(ensemble_models):
        print(f"    Predicting with model {i}...")
        preds = m.predict(input_data)
        all_preds.append(preds)
    avg_preds = np.mean(all_preds, axis=0)
    if len(avg_preds.shape) > 1:
        avg_preds = avg_preds[0]
    confidence = float(np.max(avg_preds))
    idx = np.argmax(avg_preds)
    label = CLASSES[idx] if idx < len(CLASSES) else "Unknown"
    print(f"  ensemble result: {label} ({confidence})")
    return label, confidence

def detect_foods(img_path):
    print(f"Entering detect_foods with {img_path}...")
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    results = yolo_model(img_np)
    detections = []
    for result in results:
        boxes = result.boxes
        print(f"  Found {len(boxes)} boxes.")
        for i, box in enumerate(boxes):
            print(f"    Processing box {i}...")
            if len(box.xyxy) == 0:
                print("      Empty box.xyxy")
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            crop = img_np[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                print("      Empty crop")
                continue
            label, ensemble_conf = predict_ensemble(crop)
            detections.append({"label": label, "confidence": ensemble_conf})
    return detections

if __name__ == "__main__":
    test_img = "test_food.jpg.png" # Existing file
    if not os.path.exists(test_img):
        # Create a blank test image if not exists
        print("Creating blank test image...")
        Image.new("RGB", (640, 480), color="red").save(test_img)
    
    try:
        results = detect_foods(test_img)
        print("\nSUCCESS! Detected:", results)
    except Exception as e:
        import traceback
        traceback.print_exc()
