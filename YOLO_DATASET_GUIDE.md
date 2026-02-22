# 🥗 YOLOv8 Food Detection Dataset Guide

To train YOLOv8 to identify multiple food items, you need a dataset where each image has a corresponding label file containing bounding box coordinates.

## 1. Where to Get a Dataset
Since the original Food-101 is for classification (no boxes), you should use a detection-enabled dataset. You selected this one:

- **[Food Detection (User Selected)](https://universe.roboflow.com/proyecto-dl-liq2i/food-detection-3gi6s)** - **Top Choice!** 

### How to Download from Roboflow:
1. **Click the link** above to go to the Roboflow Universe page.
2. **Sign in/Sign up** (it's free).
3. Click the blue **"Download Dataset"** button (usually in the top right).
4. For "Export Format", select **YOLOv8**.
5. Choose **"download zip to computer"** and click Continue.
6. Extract the zip file and rename the folder to `yolo_dataset`.
7. Move this folder into your project directory `C:/food_calorie_project/`.

---

**Self-Search Tip:** Go to [Roboflow Universe](https://universe.roboflow.com/search?q=food+detection) and search for "food detection". Look for projects with many images and a "YOLOv8" export option.

**Pro Tip:** When downloading from Roboflow, choose the **YOLOv8 Format**.

---

## 2. Folder Structure
Your `yolo_dataset` folder (specified in `food_data.yaml`) should look exactly like this:

```text
yolo_dataset/
├── images/
│   ├── train/ (image1.jpg, image2.jpg...)
│   └── val/   (image3.jpg...)
└── labels/
    ├── train/ (image1.txt, image2.txt...)
    └── val/   (image3.txt...)
```

### Important rules:
1. **Filename Match:** Each image (e.g., `pizza.jpg`) must have a corresponding text file (e.g., `pizza.txt`).
2. **Label Format:** Inside the `.txt` files, each line looks like this:
   `<class_id> <x_center> <y_center> <width> <height>` (all normalized between 0 and 1).

---

## 3. Training Steps
Once your data is in place:
1. Open `food_data.yaml` and verify the `path:` points to your `yolo_dataset` folder.
2. Ensure the `names:` section in `food_data.yaml` matches the order of classes in your dataset.
3. Run the training script:
   ```bash
   python train_yolo_custom.py
   ```

## 4. After Training
When training finishes, your new model will be saved at:
`runs/detect/food_detector_v1/weights/best.pt`

You should then update `app.py` to use this `best.pt` file instead of `yolov8n.pt`.
