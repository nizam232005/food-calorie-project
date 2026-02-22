import os
import sys
from ultralytics import YOLO
import torch

# Add current dir to path to import app (though we might not need all logic)
sys.path.append(os.getcwd())

MODEL_PATH = r"C:\food_calorie_project\yolo_models\yolo26n.pt"

def verify_yolo_load():
    print(f"Testing loading of: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"FAILED: Model file not found at {MODEL_PATH}")
        return False
    
    try:
        model = YOLO(MODEL_PATH)
        print("SUCCESS: Model loaded successfully.")
        
        # Print info
        print(f"Model Names: {model.names}")
        
        # Try a test inference if an image exists
        test_img = "temp_upload.jpg"
        if os.path.exists(test_img):
            print(f"Running inference on {test_img}...")
            results = model(test_img)
            print(f"Inference complete. Found {len(results[0].boxes)} objects.")
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls]
                print(f" - Detected: {name} ({conf:.2f})")
        else:
            print(f"No test image found at {test_img}, skipping inference test.")
            
        return True
    except Exception as e:
        print(f"FAILED: Error during load/inference: {e}")
        return False

if __name__ == "__main__":
    success = verify_yolo_load()
    sys.exit(0 if success else 1)
