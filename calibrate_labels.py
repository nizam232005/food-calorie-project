import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Patch for Keras 3
class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

MODEL_PATH = r"C:\food_calorie_project\model\keras_model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D})
except Exception as e:
    print(f"FAILED TO LOAD MODEL: {e}")
    exit()

# Folders to check in food88
train_dir = r"C:\food_calorie_project\dataset\food88\images\train"
folders = ["pizza", "hamburger", "hot_dog", "donuts", "ice_cream", "samosa", "french_fries", "omelette"]

print(f"{'Folder Name':<15} | {'Predicted Index':<15} | {'Confidence':<10}")
print("-" * 45)

for folder in folders:
    path = os.path.join(train_dir, folder)
    if not os.path.exists(path):
        print(f"{folder:<15} | Folder Not Found")
        continue
    
    # Take first 3 images to be sure
    imgs = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
    indices = []
    confs = []
    
    for img_name in imgs:
        img_path = os.path.join(path, img_name)
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_arr = np.array(img).astype(np.float32)
        # Standard TM normalization
        x = (img_arr / 127.5) - 1
        x = np.expand_dims(x, axis=0)
        
        preds = model.predict(x, verbose=0)[0]
        idx = np.argmax(preds)
        indices.append(idx)
        confs.append(preds[idx])
    
    avg_idx = round(sum(indices)/len(indices)) if indices else -1
    avg_conf = sum(confs)/len(confs) if confs else 0
    
    print(f"{folder:<15} | {avg_idx:<15} | {avg_conf:.2f}")

# Also check food_cnn_model.h5 just in case
print("\n--- Checking food_cnn_model.h5 ---")
MODEL_PATH_2 = r"C:\food_calorie_project\model\food_cnn_model.h5"
try:
    model2 = tf.keras.models.load_model(MODEL_PATH_2)
    for folder in folders:
        path = os.path.join(train_dir, folder)
        if not os.path.exists(path): continue
        img_path = os.path.join(path, os.listdir(path)[0])
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_arr = np.array(img).astype(np.float32) / 255.0 # CNN Usually uses 0-1
        x = np.expand_dims(img_arr, axis=0)
        preds = model2.predict(x, verbose=0)[0]
        idx = np.argmax(preds)
        print(f"{folder:<15} | {idx:<15} | {preds[idx]:.2f}")
except Exception as e:
    print(f"food_cnn_model.h5 check failed: {e}")
