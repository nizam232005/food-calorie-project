from ultralytics import YOLO
import os

model_path = r"c:\food_calorie_project\yolo_models\yolo26n.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"Model: {model_path}")
    print(f"Number of classes: {len(model.names)}")
    print(f"Classes: {model.names}")
else:
    print(f"Model not found at {model_path}")

model_path_s = r"c:\food_calorie_project\yolo_models\yolov8s.pt"
if os.path.exists(model_path_s):
    model_s = YOLO(model_path_s)
    print(f"\nModel: {model_path_s}")
    print(f"Number of classes: {len(model_s.names)}")
    print(f"Classes: {model_s.names}")
