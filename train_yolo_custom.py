import os
from ultralytics import YOLO

def train_custom_yolo():
    # 1. Load the pre-trained nano model (good starting point)
    model = YOLO('yolov8n.pt')

    # 2. Path to your data configuration file
    # Ensure food_data.yaml exists and points to your annotated images
    data_yaml = 'food_data.yaml'
    
    if not os.path.exists(data_yaml):
        print(f"Error: {data_yaml} not found. Please create it first.")
        return

    print("--- Starting YOLOv8 Training ---")
    # 3. Train the model
    # epochs: Number of passes over the dataset
    # imgsz: Image size (640 is standard)
    # batch: Batch size (reduce if you run out of memory)
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=16,
        name='food_detector_v1'
    )
    
    print("--- Training Complete ---")
    print(f"Results saved to: {results.save_dir}")
    print("You can find the best model weights at: runs/detect/food_detector_v1/weights/best.pt")

if __name__ == "__main__":
    train_custom_yolo()
