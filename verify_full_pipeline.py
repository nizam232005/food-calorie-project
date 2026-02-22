import os
import tensorflow as tf
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ------------------ CONFIG ------------------
ENSEMBLE_DIR = r"C:\food_calorie_project\ensemble_models"
MODEL_PATHS = [
    os.path.join(ENSEMBLE_DIR, "ensemble_model1_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "ensemble_model3_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "model2_v2.keras")
]

CLASSES = [
    "chicken_curry",
    "donuts",
    "french_fries",
    "fried_rice",
    "ice_cream",
    "omelette",
    "pizza",
    "samosa"
]

# ------------------ LOAD MODELS ------------------
print("Testing Model Loading...")
ensemble_models = []
for p in MODEL_PATHS:
    if os.path.exists(p):
        print(f"Loading: {p}")
        try:
            # We skip custom objects for the test to see if it loads at all
            m = tf.keras.models.load_model(p, compile=False)
            ensemble_models.append(m)
            print(f"Successfully loaded {p}")
        except Exception as e:
            print(f"Error loading {p}: {e}")
    else:
        print(f"Model not found: {p}")

# ------------------ TEST YOLO ------------------
print("\nTesting YOLOv8 Loading...")
try:
    yolo_model = YOLO("yolov8n.pt")
    print("YOLOv8 successfully loaded.")
except Exception as e:
    print(f"Error loading YOLO: {e}")

# ------------------ TEST NUTRITION ------------------
print("\nTesting Nutrition Service...")
from nutrition_service import get_nutrition
test_foods = ["chicken_curry", "masala dosa", "pizza", "samosa"]
for food in test_foods:
    nut = get_nutrition(food)
    print(f"{food}: {nut.get('calories')} kcal")

print("\n✅ Verification script finished.")
