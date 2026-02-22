import os
import shutil

TRAIN_DIR = r"C:\food_calorie_project\dataset\food88\images\train"
VAL_DIR   = r"C:\food_calorie_project\dataset\food88\images\val"

CLASSES = [
    "pizza",
    "hamburger",
    "hot_dog",
    "donuts",
    "ice_cream",
    "samosa",
    "french_fries",
    "omelette"
]

def organize(folder):
    for cls in CLASSES:
        os.makedirs(os.path.join(folder, cls), exist_ok=True)

    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        if os.path.isfile(img_path):
            for cls in CLASSES:
                if img.startswith(cls):
                    shutil.move(img_path, os.path.join(folder, cls, img))
                    break

organize(TRAIN_DIR)
organize(VAL_DIR)

print("✅ Images organized into class folders")