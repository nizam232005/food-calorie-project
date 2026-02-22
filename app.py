from flask import Flask, render_template, request, redirect, session, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
from clarifai.client.model import Model
from nutrition_service import get_nutrition
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not found, attempting manual .env loading...")
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip("'").strip('"')

# ------------------ ENSEMBLE MODEL CONFIG ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENSEMBLE_DIR = os.path.join(BASE_DIR, "ensemble_models")
MODEL_PATHS = [
    os.path.join(ENSEMBLE_DIR, "ensemble_model1_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "ensemble_model3_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "model2_v2.keras")
]

# Classes matching the ensemble model training (Food101 + Indian)
CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries",
    "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque",
    "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai",
    "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza",
    "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen",
    "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops",
    "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara",
    "spring_rolls", "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
    "tiramisu", "tuna_tartare", "waffles"
]

# Patch for Keras 3 / Teachable Machine compatibility (if still needed)
class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

# Load ensemble models
ensemble_models = []
for p in MODEL_PATHS:
    if os.path.exists(p):
        print(f"Loading model: {p}")
        m = tf.keras.models.load_model(p, custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D})
        ensemble_models.append(m)
    else:
        print(f"⚠️ Warning: Model not found at {p}")

# ------------------ YOLO CONFIG ------------------
from ultralytics import YOLO
CUSTOM_MODEL_PATH = os.path.join(BASE_DIR, "yolo_models", "yolo26n.pt")

if os.path.exists(CUSTOM_MODEL_PATH):
    print(f"Loading CUSTOM YOLO model: {CUSTOM_MODEL_PATH}")
    yolo_model = YOLO(CUSTOM_MODEL_PATH)
    USING_CUSTOM_YOLO = True
else:
    print("Loading GENERIC YOLOv8n model (Custom model not found yet)")
    yolo_model = YOLO("yolov8n.pt")
    USING_CUSTOM_YOLO = False

# Classes from the custom YOLO dataset (20 classes)
YOLO_CLASSES = [
    'Aguacate', 'Ahuyama', 'Arepa', 'Arroz', 'Arroz con Pollo', 
    'Carne res', 'Chicharron', 'Chorizo', 'Criolla', 'Ensalada', 
    'Frijol', 'Habichuela', 'Huevo', 'Lentejas', 'Morcilla', 
    'Papa', 'Platano', 'Pollo', 'QR', 'Trucha'
]

# ------------------ CLARIFAI CONFIG ------------------
CLARIFAI_PAT = "1c8db2337b9e460a9386b6d3bbd8a2e7"
CLARIFAI_USER_ID = "clarifai"
CLARIFAI_APP_ID = "main"
CLARIFAI_MODEL_ID = "bd367be194cf45149e75f01d59f77ba7"


# ------------------ LOGMEAL CONFIG ------------------
LOGMEAL_API_KEY = os.getenv("LOGMEAL_API_KEY")
LOGMEAL_API_URL = "https://api.logmeal.es/v2/image/recognition/dish"

# ------------------ APP CONFIG ------------------
app = Flask(__name__)
app.secret_key = "super-secret-key"   # change later for production
DB_NAME = "database.db"
FOOD_RECOGNITION_API = os.getenv("FOOD_RECOGNITION_API", "logmeal").lower()
print(f"DEBUG: FOOD_RECOGNITION_API loaded as: {FOOD_RECOGNITION_API}")
print(f"DEBUG: LOGMEAL_API_KEY present: {LOGMEAL_API_KEY is not None}")

# ------------------ image recognition using Ensemble ------------------
def predict_ensemble(img_array):
    """Predict using ensemble of 3 models and average results"""
    if not ensemble_models:
        return "Unknown", 0.0
    
    # Preprocess image for MobileNetV2
    img = Image.fromarray(img_array.astype('uint8')).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32)
    # MobileNetV2 expects [-1, 1] for Teachable Machine variants or [0, 1] for others
    # Assuming the same preprocessing as before based on train_cnn.py
    img = (img / 127.5) - 1
    input_data = np.expand_dims(img, axis=0)

    all_preds = []
    for m in ensemble_models:
        preds = m.predict(input_data)
        all_preds.append(preds)
    
    # Average the predictions
    avg_preds = np.mean(all_preds, axis=0)
    
    # Ensure avg_preds is a 1D array of probabilities
    if len(avg_preds.shape) > 1:
        avg_preds = avg_preds[0]
        
    confidence = float(np.max(avg_preds))
    idx = np.argmax(avg_preds)
    
    if idx < len(CLASSES):
        label = CLASSES[idx]
    else:
        print(f"Index error: predicted index {idx} out of range for CLASSES (len {len(CLASSES)})")
        label = "Unknown"

    return label, confidence

# ------------------ YOLO Detection ------------------
def detect_foods(image_file):
    """Detect multiple food items using YOLO and return crops and boxes"""
    img = Image.open(image_file).convert("RGB")
    img_np = np.array(img)
    
    print(f"DEBUG: YOLO running on image...")
    results = yolo_model(img_np)
    detections = []
    
    if results:
        for r_idx, result in enumerate(results):
            boxes = result.boxes
            print(f"DEBUG: Result {r_idx} found {len(boxes)} boxes")
            for b_idx, box in enumerate(boxes):
                try:
                    # Check if boxes exist and have data
                    if hasattr(box, 'xyxy') and len(box.xyxy) > 0:
                        coords = box.xyxy[0].tolist()
                        if len(coords) >= 4:
                            x1, y1, x2, y2 = map(int, coords[:4])
                            crop = img_np[y1:y2, x1:x2]
                            conf = float(box.conf[0])
                            
                            # Hybrid Logic: Decide whether to use YOLO label, Ensemble, or external API
                            if USING_CUSTOM_YOLO and conf > 0.4:
                                # Custom YOLO is very reliable for its trained classes
                                class_id = int(box.cls[0])
                                if class_id < len(YOLO_CLASSES):
                                    label = YOLO_CLASSES[class_id]
                                    ensemble_conf = conf
                                else:
                                    # Fallback if class_id doesn't match custom classes
                                    print(f"DEBUG: YOLO class_id {class_id} not in custom set, using {FOOD_RECOGNITION_API}...")
                                    if FOOD_RECOGNITION_API == "logmeal":
                                        import io
                                        crop_img = Image.fromarray(crop)
                                        img_byte_arr = io.BytesIO()
                                        crop_img.save(img_byte_arr, format='JPEG')
                                        img_byte_arr.seek(0)
                                        label, ensemble_conf = predict_logmeal(img_byte_arr)
                                    elif FOOD_RECOGNITION_API == "clarifai":
                                        import io
                                        crop_img = Image.fromarray(crop)
                                        img_byte_arr = io.BytesIO()
                                        crop_img.save(img_byte_arr, format='JPEG')
                                        img_byte_arr.seek(0)
                                        label, ensemble_conf = predict_clarifai(img_byte_arr)
                                    else:
                                        label, ensemble_conf = predict_ensemble(crop)
                            elif FOOD_RECOGNITION_API == "logmeal":
                                # Convert crop back to bytes
                                import io
                                crop_img = Image.fromarray(crop)
                                img_byte_arr = io.BytesIO()
                                crop_img.save(img_byte_arr, format='JPEG')
                                img_byte_arr.seek(0)
                                
                                label, ensemble_conf = predict_logmeal(img_byte_arr)
                                if label == "Unauthorized":
                                    print("DEBUG: LogMeal unauthorized, falling back to Clarifai...")
                                    img_byte_arr.seek(0)
                                    label, ensemble_conf = predict_clarifai(img_byte_arr)
                                    # If Clarifai also fails or looks unknown, then local ensemble
                                    if label == "Unknown":
                                        label, ensemble_conf = predict_ensemble(crop)
                            elif FOOD_RECOGNITION_API == "clarifai":
                                import io
                                crop_img = Image.fromarray(crop)
                                img_byte_arr = io.BytesIO()
                                crop_img.save(img_byte_arr, format='JPEG')
                                img_byte_arr.seek(0)
                                label, ensemble_conf = predict_clarifai(img_byte_arr)
                            else:
                                # Default to local ensemble
                                label, ensemble_conf = predict_ensemble(crop)

                            if crop.size > 0:
                                detections.append({
                                    "label": label,
                                    "confidence": ensemble_conf,
                                    "box_conf": conf,
                                    "coords": [x1, y1, x2, y2]
                                })
                            else:
                                print(f"DEBUG: Box {b_idx} has zero area crop")
                        else:
                            print(f"DEBUG: Box {b_idx} has invalid coords length: {len(coords)}")
                    else:
                        print(f"DEBUG: Box {b_idx} has no xyxy data")
                except Exception as e:
                    print(f"DEBUG: Error processing box {b_idx}: {e}")
    else:
        print("DEBUG: YOLO returned no results")
            
    return detections

def predict_logmeal(image_file):
    """Predict using LogMeal API"""
    if not LOGMEAL_API_KEY:
        print("LOGMEAL ERROR: API Key not found in .env")
        return "Unknown", 0.0
        
    print("[LOGMEAL] Sending image to LogMeal API...")
    headers = {'Authorization': f'Bearer {LOGMEAL_API_KEY}'}
    
    # If image_file is already a bytes object or similar, handle it
    # But usually it's a file handler
    files = {'image': image_file}
    
    try:
        response = requests.post(LOGMEAL_API_URL, headers=headers, files=files)
        if response.status_code == 200:
            data = response.json()
            if 'recognition_results' in data and data['recognition_results']:
                top_result = data['recognition_results'][0]
                print(f"[LOGMEAL] Recognition successful: {top_result['name']} ({round(top_result['prob']*100, 2)}%)")
                return top_result['name'], top_result['prob']
            elif 'dishes' in data and data['dishes']:
                top_dish = data['dishes'][0]
                print(f"[LOGMEAL] Recognition successful (dish): {top_dish['name']} ({round(top_dish['prob']*100, 2)}%)")
                return top_dish['name'], top_dish['prob']
            else:
                print(f"[LOGMEAL] Warning: No food/dishes found in response: {data}")
        elif response.status_code in [401, 403]:
            print(f"LOGMEAL PERMISSION ERROR: {response.status_code}. Your account type may not have access to this endpoint.")
            print(f"LOGMEAL MESSAGE: {response.text}")
            return "Unauthorized", 0.0
        else:
            print(f"LOGMEAL ERROR: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"LOGMEAL ERROR: {e}")
        
    return "Unknown", 0.0

def predict_clarifai(image_file):
    image_bytes = image_file.read()
    image_file.seek(0) # Reset pointer
    
    clarifai_model = Model(
        user_id=CLARIFAI_USER_ID,
        app_id=CLARIFAI_APP_ID,
        model_id=CLARIFAI_MODEL_ID,
        pat=CLARIFAI_PAT
    )
    
    try:
        model_prediction = clarifai_model.predict_by_bytes(image_bytes, input_type="image")
        if not hasattr(model_prediction, 'outputs') or not model_prediction.outputs:
            print("CLARIFAI: No outputs found in response")
            return "Unknown", 0.0
            
        output = model_prediction.outputs[0]
        if not hasattr(output, 'data') or not output.data or not hasattr(output.data, 'concepts'):
            print("CLARIFAI: Output has no concept data")
            return "Unknown", 0.0
            
        concepts = output.data.concepts
        
        # Get the top concept
        if concepts and len(concepts) > 0:
            top_concept = concepts[0]
            return top_concept.name, top_concept.value
    except Exception as e:
        print(f"CLARIFAI ERROR: {e}")
        
    return "Unknown", 0.0

# ------------------ image recognition using API ------------------
API_KEY = "1e0927499e9a4f168c0a839a492deebc.sGBqS5H7PPYmZliP"

def food_api_predict(image_file):
    url = "https://api.calorieninjas.com/v1/imagedetection"
    headers = {"X-Api-Key": API_KEY}
    files = {"image": image_file}

    response = requests.post(url, headers=headers, files=files)

    print("STATUS CODE:", response.status_code)
    print("RAW RESPONSE:", response.text)   # 🔥 DEBUG LINE

    try:
        data = response.json()
    except Exception as e:
        print("JSON ERROR:", e)
        return "API error", "N/A"

    if "items" in data and len(data["items"]) > 0:
        food = data["items"][0].get("name", "Unknown")
        calories = data["items"][0].get("calories", "N/A")
        return food, calories

    return "Unknown food", "N/A"

# ------------------ DATABASE SETUP ------------------
def get_db():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    
    # List of columns that might be missing in older versions of the table
    optional_columns = [
        ("age", "INTEGER"),
        ("height", "INTEGER"),
        ("weight", "INTEGER"),
        ("gender", "TEXT"),
        ("activity_level", "TEXT"),
        ("goal", "TEXT"),
        ("diet", "TEXT")
    ]
    
    # Check existing columns
    c.execute("PRAGMA table_info(users)")
    existing_columns = [row[1] for row in c.fetchall()]
    
    for col_name, col_type in optional_columns:
        if col_name not in existing_columns:
            try:
                c.execute(f"ALTER TABLE users ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                # Column might already exist or table might be locked
                pass

    c.execute("""
    CREATE TABLE IF NOT EXISTS daily_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        date TEXT NOT NULL,
        calories REAL NOT NULL,
        food_name TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ------------------ HOME ------------------
@app.route("/")
def home():
    return render_template("home.html")

# ------------------ LOGIN ------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=?", (email,))
        user = c.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["user_name"] = user["name"]
            return redirect("/dashboard")
        else:
            return "❌ Invalid email or password"

    return render_template("login.html")

# ------------------ REGISTER ------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        hashed_password = generate_password_hash(request.form["password"])

        data = (
            request.form["name"],
            request.form["email"],
            hashed_password,
            request.form.get("age") or None,
            request.form.get("height") or None,
            request.form.get("weight") or None,
            request.form.get("gender"),
            request.form.get("activity_level"),
            request.form["goal"],
            request.form["diet"]
        )

        conn = get_db()
        c = conn.cursor()

        try:
            c.execute("""
                INSERT INTO users
                (name, email, password, age, height, weight, gender, activity_level, goal, diet)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, data)
            conn.commit()
        except sqlite3.IntegrityError:
            return "❌ Email already exists"
        finally:
            conn.close()

        return redirect("/")

    return render_template("register.html")

# ------------------ DASHBOARD ------------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/")

    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT age, height, weight, gender, activity_level, goal, diet
        FROM users WHERE id=?
    """, (session["user_id"],))
    user = dict(c.fetchone())
    conn.close()

    # Calculate personalized calorie prediction
    from calorie_predictor import predict_daily_calories
    prediction = predict_daily_calories(user)

    # Fetch recent history and today's totals
    from datetime import date
    today = date.today().isoformat()
    
    conn = get_db()
    c = conn.cursor()
    
    # Recent history (last 10 entries)
    c.execute("""
        SELECT food_name, calories, date 
        FROM daily_logs 
        WHERE user_id = ? 
        ORDER BY date DESC, id DESC 
        LIMIT 10
    """, (session["user_id"],))
    recent_history = c.fetchall()
    
    # Today's totals for macros
    # Note: We need to pull this from nutrition_service if not stored in history
    # For now, let's assume we want to show totals for whatever we have in logs
    c.execute("""
        SELECT SUM(calories) as total_cal
        FROM daily_logs 
        WHERE user_id = ? AND date = ?
    """, (session["user_id"], today))
    daily_sums = c.fetchone()
    
    conn.close()

    # Get meal recommendations
    from recommendation_service import get_meal_recommendations
    remaining_budget = max(0, (prediction.get('daily_calories', 2000) - (daily_sums['total_cal'] or 0)))
    recommendations = get_meal_recommendations(
        budget=remaining_budget,
        diet=user.get('diet', 'non-veg'),
        history=[row['food_name'] for row in recent_history]
    )

    return render_template(
        "dashboard.html",
        name=session["user_name"],
        user=user,
        prediction=prediction,
        target_calories=prediction.get('daily_calories', 2000),
        recent_history=recent_history,
        today_consumption=daily_sums['total_cal'] or 0,
        recommendations=recommendations
    )

# ------------------ LOG CALORIES ------------------
@app.route("/manual_entry")
def manual_entry():
    if "user_id" not in session:
        return redirect("/")
    from datetime import date
    return render_template("manual_entry.html", today_date=date.today().isoformat())

@app.route("/log_calories", methods=["POST"])
def log_calories():
    if "user_id" not in session:
        return redirect("/")
    
    user_id = session["user_id"]
    food_name = request.form.get("food_name", "Unknown")
    calories_str = request.form.get("calories", "").strip()
    portion_size = float(request.form.get("portion_size", 1.0))
    date = request.form.get("date") or __import__('datetime').date.today().isoformat()
    
    # Handle optional calories
    print(f"[DEBUG] Logging food: {food_name}, Portion: {portion_size}, Input Cal: {calories_str}")
    
    if not calories_str:
        # Auto-lookup if calories are not provided
        nutrition = get_nutrition(food_name, portion_size=portion_size)
        calories = nutrition.get('calories')
        print(f"[DEBUG] Lookup result for {food_name}: {calories}")
        
        if calories == 'N/A' or calories is None:
            calories = 0.0 # Default to 0 if not found
            print(f"[DEBUG] Food not found, defaulting to 0")
    else:
        try:
            calories = float(calories_str)
        except (ValueError, TypeError):
            calories = 0.0
    
    print(f"[DEBUG] Final calories to log: {calories}")
    
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO daily_logs (user_id, date, calories, food_name) VALUES (?, ?, ?, ?)", 
                  (user_id, date, float(calories), food_name))
        conn.commit()
        print(f"[DEBUG] Successfully logged to DB for date: {date}")
    except Exception as e:
        print(f"[DEBUG] DB Error in log_calories: {e}")
    finally:
        conn.close()
    
    return redirect("/dashboard")

# ------------------ API FOR PROGRESS DATA ------------------
@app.route("/api/progress_data")
def progress_data():
    if "user_id" not in session:
        return {"error": "Unauthorized"}, 401
    
    user_id = session["user_id"]
    conn = get_db()
    c = conn.cursor()
    
    # Get last 7 days of consumption
    c.execute("""
        SELECT date, SUM(CAST(calories AS REAL)) as total_calories 
        FROM daily_logs 
        WHERE user_id = ? 
        GROUP BY date 
        ORDER BY date ASC 
        LIMIT 30
    """, (user_id,))
    
    rows = c.fetchall()
    conn.close()
    
    data = {
        "dates": [row["date"] for row in rows],
        "consumption": [row["total_calories"] for row in rows]
    }
    
    return data

# ------------------ PREDICT ------------------

@app.route("/predict", methods=["POST"])
def predict_food():
    if "user_id" not in session:
        return redirect("/")

    image = request.files["image"]
    
    # Save image temporarily to read multiple times if needed
    temp_path = "temp_upload.jpg"
    image.save(temp_path)
    
    # 1. Multi-Food Detection using YOLO
    with open(temp_path, "rb") as f:
        detections = detect_foods(f)
    
    results = []
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    
    # 2. Process each detection
    if detections:
        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            
            # If YOLO/Ensemble confidence is low, can we re-check with primary API?
            if conf < 0.3:
                with open(temp_path, "rb") as f:
                    # Optional: Re-crop and send to API? 
                    # For now, we trust YOLO/Ensemble or fallback to single image API below
                    pass

            # Get nutrition
            nutrition = get_nutrition(label)
            
            # Aggregate totals (handling 'N/A')
            try:
                if nutrition.get('calories') != 'N/A':
                    total_calories += float(nutrition.get('calories', 0))
                if nutrition.get('protein') != 'N/A':
                    total_protein += float(nutrition.get('protein', 0))
                if nutrition.get('carbs') != 'N/A':
                    total_carbs += float(nutrition.get('carbs', 0))
                if nutrition.get('fat') != 'N/A':
                    total_fat += float(nutrition.get('fat', 0))
            except:
                pass

            results.append({
                "food": label,
                "confidence": round(conf * 100, 2),
                "nutrition": nutrition
            })
    else:
        # Fallback to designated API if YOLO finds nothing
        label, conf = "Unknown", 0.0
        
        print(f"DEBUG: YOLO found nothing. Falling back to {FOOD_RECOGNITION_API}...")
        
        with open(temp_path, "rb") as f:
            if FOOD_RECOGNITION_API == "clarifai":
                label, conf = predict_clarifai(f)
            elif FOOD_RECOGNITION_API == "logmeal":
                label, conf = predict_logmeal(f)
                if label == "Unauthorized":
                    print("DEBUG: LogMeal unauthorized, falling back to Clarifai...")
                    f.seek(0)
                    label, conf = predict_clarifai(f)
            elif FOOD_RECOGNITION_API == "ensemble":
                # Use the whole image for ensemble
                img = Image.open(temp_path).convert("RGB")
                img_array = np.array(img)
                label, conf = predict_ensemble(img_array)
            else:
                # Default fallback
                label, conf = predict_clarifai(f)

        # Final Fallback to ensemble if everything else failed
        if (label == "Unknown" or label == "Unauthorized"):
            print("DEBUG: Primary APIs failed, falling back to local Ensemble...")
            img = Image.open(temp_path).convert("RGB")
            img_array = np.array(img)
            label, conf = predict_ensemble(img_array)

        nutrition = get_nutrition(label)
        results.append({
            "food": label,
            "confidence": round(conf * 100, 2),
            "nutrition": nutrition
        })
        if nutrition.get('calories') != 'N/A':
            total_calories = float(nutrition.get('calories', 0))
        if nutrition.get('protein') != 'N/A':
            total_protein = float(nutrition.get('protein', 0))
        if nutrition.get('carbs') != 'N/A':
            total_carbs = float(nutrition.get('carbs', 0))
        if nutrition.get('fat') != 'N/A':
            total_fat = float(nutrition.get('fat', 0))

    return render_template(
        "result.html",
        results=results,
        total_calories=round(total_calories, 2),
        total_protein=round(total_protein, 2),
        total_carbs=round(total_carbs, 2),
        total_fat=round(total_fat, 2)
    )

# ------------------ AI CHATBOT ------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.json
    user_message = data.get("message")
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Fetch fresh context for the chatbot
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT goal, diet FROM users WHERE id=?", (session["user_id"],))
    user_profile = dict(c.fetchone())
    
    from datetime import date
    today = date.today().isoformat()
    c.execute("SELECT SUM(calories) FROM daily_logs WHERE user_id = ? AND date = ?", (session["user_id"], today))
    consumed = c.fetchone()[0] or 0
    
    c.execute("SELECT food_name FROM daily_logs WHERE user_id = ? ORDER BY id DESC LIMIT 5", (session["user_id"],))
    history = [row[0] for row in c.fetchall()]
    conn.close()

    from calorie_predictor import predict_daily_calories
    prediction = predict_daily_calories(user_profile)
    target = prediction.get('daily_calories', 2000)

    context = {
        "goal": user_profile.get('goal'),
        "diet": user_profile.get('diet'),
        "target": target,
        "consumed": consumed,
        "remaining": max(0, target - consumed),
        "history": history
    }

    from chat_service import get_chatbot_response
    bot_response = get_chatbot_response(user_message, context)
    
    return jsonify({"response": bot_response})

# ------------------ OCR LABEL SCANNER ------------------
@app.route("/api/scan_label", methods=["POST"])
def scan_label():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    image_file = request.files['image']
    from ocr_service import get_label_data
    try:
        nutrition_data = get_label_data(image_file)
        return jsonify(nutrition_data)
    except Exception as e:
        print(f"OCR ERROR: {e}")
        return jsonify({"error": "Failed to parse label"}), 500

# ------------------ LOGOUT ------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# ------------------ RUN ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")