import os
import shutil
import random

# PATHS
SOURCE_DIR = r"C:\food_calorie_project\dataset\food-101\images"
DEST_DIR   = r"C:\food_calorie_project\dataset\food88"

# Selected food classes (verified against standard Food-101)
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

IMAGES_PER_CLASS = 500
TRAIN_SPLIT = 0.8        # 80% train, 20% validation

# Create destination directories
for split in ["train", "val"]:
    os.makedirs(os.path.join(DEST_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, "labels", split), exist_ok=True)

class_id_map = {cls: idx for idx, cls in enumerate(CLASSES)}

print(f"Starting extraction for classes: {CLASSES}")

for food_class in CLASSES:
    class_path = os.path.join(SOURCE_DIR, food_class)
    
    if not os.path.exists(class_path):
        print(f"⚠️ Warning: Class directory not found: {class_path}")
        continue
        
    images = [img for img in os.listdir(class_path) if img.endswith(".jpg")]
    random.shuffle(images)
    images = images[:IMAGES_PER_CLASS]

    split_index = int(len(images) * TRAIN_SPLIT)
    train_images = images[:split_index]
    val_images = images[split_index:]

    print(f"Processing {food_class}: {len(train_images)} train, {len(val_images)} val")

    for img_name, split in [(img, "train") for img in train_images] + [(img, "val") for img in val_images]:
        src_img = os.path.join(class_path, img_name)
        # Rename image to include class for clarity and to avoid collisions
        dst_img_name = f"{food_class}_{img_name}"
        dst_img = os.path.join(DEST_DIR, "images", split, dst_img_name)

        shutil.copy(src_img, dst_img)

        # Create YOLO label (full image bounding box: class_id x_center y_center width height)
        label_name = dst_img_name.replace(".jpg", ".txt")
        label_path = os.path.join(DEST_DIR, "labels", split, label_name)

        with open(label_path, "w") as f:
            f.write(f"{class_id_map[food_class]} 0.5 0.5 1.0 1.0\n")

print("\n✅ Dataset extraction completed successfully!")
print(f"Data saved to: {DEST_DIR}")