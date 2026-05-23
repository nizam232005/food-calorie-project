<p align="center">
  <h1 align="center">🍽️ NutriSmart — AI-Powered Food & Calorie Tracker</h1>
  <p align="center">
    <em>Snap a photo. Identify the food. Track your nutrition. Achieve your goals.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Flask-2.3+-000000?logo=flask" alt="Flask">
    <img src="https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
    <img src="https://img.shields.io/badge/YOLOv8-ultralytics-00FFFF?logo=yolo" alt="YOLOv8">
    <img src="https://img.shields.io/badge/Gemini_AI-Google-4285F4?logo=google&logoColor=white" alt="Gemini">
    <img src="https://img.shields.io/badge/Deploy-Render-46E3B7?logo=render" alt="Render">
  </p>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [AI Models](#-ai-models)
- [API Endpoints](#-api-endpoints)
- [Deployment](#-deployment)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

**NutriSmart** is a full-stack AI-powered web application that identifies food from images and provides detailed nutritional information. It combines multiple computer vision models in an ensemble approach with external AI APIs to deliver accurate food recognition and calorie tracking, helping users manage their daily nutrition goals.

Whether you're trying to lose weight, gain muscle, or simply maintain a healthy diet, NutriSmart provides personalized calorie predictions and intelligent meal recommendations based on your profile.

---

## ✨ Key Features

### 🔍 Intelligent Food Recognition
- **Multi-model ensemble** — Averages predictions across 3 fine-tuned MobileNetV2 models (trained on Food-101 dataset) for robust accuracy
- **YOLOv8 object detection** — Detects and classifies multiple food items in a single image with bounding boxes
- **Dedicated Indian food classifier** — Specialized model trained on Indian cuisine categories
- **Gemini Vision fallback** — Uses Google Gemini AI as a fallback when local models have low confidence
- **LogMeal API integration** — Additional external API support for food recognition

### 📊 Comprehensive Nutrition Tracking
- **Hybrid nutrition lookup** — Local SQLite cache with 80+ pre-seeded foods + USDA FoodData Central API fallback + Gemini AI fallback
- **Macro breakdown** — Detailed calories, protein, carbs, fat, and fiber per serving
- **Portion size support** — Adjustable portion multipliers for accurate tracking
- **Daily progress tracking** — Visual charts showing consumption history over time

### 🤖 AI Nutrition Chatbot (RAG-Powered)
- **Retrieval-Augmented Generation** — Uses ChromaDB vector database with sentence-transformers for context-aware responses
- **Personalized advice** — Considers your goals, diet type, daily target, and recent food history
- **Auto-learning** — Automatically fetches and stores nutrition data for unknown foods via Gemini
- **Multi-word food understanding** — N-gram extraction handles complex food names like "chicken nuggets" or "french onion soup"

### 🎯 Personalized Calorie Predictions
- **ML-powered predictions** — Random Forest model trained on 10,000 synthetic profiles based on the Mifflin-St Jeor equation
- **Blended approach** — 70% ML prediction + 30% formula-based calculation for robustness
- **Goal-aware adjustments** — Supports weight loss (−500 kcal), maintenance, and muscle gain (+300 kcal)
- **Activity level factors** — Sedentary to extra-active multipliers

### 📱 Additional Features
- **OCR label scanning** — Scan nutrition labels from packaged food using EasyOCR
- **Smart meal recommendations** — Diet-preference-aware suggestions based on remaining calorie budget
- **User authentication** — Secure registration/login with hashed passwords
- **Daily food logging** — Manual entry and image-based logging with history
- **Responsive UI** — Modern glassmorphism design with dark mode

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                              │
│  Flask Templates (Jinja2) + Vanilla CSS + Chart.js           │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Flask Backend (app.py)                     │
│                                                              │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Auth &   │  │ Food Predict │  │ Nutrition Lookup       │ │
│  │ Sessions │  │ /predict     │  │ (Local → USDA → Gemini)│ │
│  └──────────┘  └──────┬───────┘  └────────────────────────┘ │
│                       │                                      │
│         ┌─────────────┼─────────────┐                        │
│         ▼             ▼             ▼                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ YOLOv8   │  │ Ensemble │  │ Indian   │                   │
│  │ Detector │  │ CNN (x3) │  │ Food CNN │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
│         │             │             │                        │
│         └─────────────┼─────────────┘                        │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │ Gemini Vision   │ (Low-confidence fallback)   │
│              │ LogMeal API     │                              │
│              └─────────────────┘                             │
│                                                              │
│  ┌──────────────┐  ┌────────────┐  ┌─────────────────────┐  │
│  │ RAG Chatbot  │  │ Calorie    │  │ OCR Label Scanner   │  │
│  │ (ChromaDB +  │  │ Predictor  │  │ (EasyOCR)           │  │
│  │  Gemini)     │  │ (RF Model) │  │                     │  │
│  └──────────────┘  └────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ SQLite   │ │ ChromaDB │ │ External │
    │ (Users,  │ │ (RAG     │ │ APIs     │
    │  Logs,   │ │  Vectors)│ │ (USDA,   │
    │  Nutri)  │ │          │ │  Gemini) │
    └──────────┘ └──────────┘ └──────────┘
```

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11, Flask 2.3+ |
| **Food Detection** | YOLOv8 (Ultralytics), TensorFlow/Keras (MobileNetV2 Ensemble) |
| **AI/LLM** | Google Gemini API (Vision + Chat) |
| **RAG Pipeline** | ChromaDB, Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **ML Prediction** | scikit-learn (Random Forest Regressor) |
| **OCR** | EasyOCR |
| **Database** | SQLite (users, logs, nutrition cache) |
| **Nutrition API** | USDA FoodData Central |
| **Frontend** | Jinja2 Templates, Vanilla CSS (Glassmorphism), Chart.js |
| **Deployment** | Docker, Render, Gunicorn |

---

## 📁 Project Structure

```
food_calorie_project/
│
├── app.py                      # Main Flask application & routes
├── calorie_predictor.py        # ML-based daily calorie prediction (Random Forest)
├── chat_service.py             # RAG-powered AI chatbot service
├── nutrition_service.py        # Hybrid nutrition lookup (Local → USDA → Gemini)
├── ocr_service.py              # Nutrition label OCR scanner (EasyOCR)
├── recommendation_service.py   # Smart meal recommendation engine
│
├── ensemble_models/            # Trained Keras models
│   ├── ensemble_model1_final_v2.keras
│   ├── ensemble_model3_final_v2.keras
│   ├── model2_v2.keras
│   └── indian_food_model.keras
│
├── yolo_models/                # YOLO object detection models
│   ├── yolo26n.pt              # Custom-trained YOLO model
│   └── yolov8s.pt              # YOLOv8 small model
│
├── nutrition_rag/              # RAG pipeline for AI chatbot
│   ├── rag_pipeline.py         # ChromaDB search & add functions
│   ├── build_db.py             # Script to build the vector database
│   └── nutrition_db/           # ChromaDB persistent storage
│
├── data/
│   ├── nutrition_data.json     # Nutrition dataset (80+ foods)
│   └── indian_food_classes.json # Indian food class labels
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Base layout template
│   ├── home.html               # Landing page
│   ├── login.html              # Login page
│   ├── register.html           # Registration with health profile
│   ├── dashboard.html          # Main dashboard with charts
│   ├── upload.html             # Image upload page
│   ├── result.html             # Food recognition results
│   └── manual_entry.html       # Manual food logging
│
├── static/
│   ├── style.css               # Application styles
│   └── charts.js               # Chart.js configurations
│
├── utils/                      # Utility functions
├── Dockerfile                  # Docker container configuration
├── render.yaml                 # Render deployment config
├── requirements.txt            # Python dependencies
├── food_data.yaml              # YOLO training data config
├── .env                        # Environment variables (not in repo)
└── .gitignore                  # Git ignore rules
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **pip** (Python package manager)
- A **Google Gemini API key** (free tier available at [Google AI Studio](https://aistudio.google.com/))
- *(Optional)* USDA API key from [FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nizam232005/food-calorie-project.git
   cd food-calorie-project
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   USDA_API_KEY=your_usda_api_key_here
   LOGMEAL_API_KEY=your_logmeal_api_key_here
   FOOD_RECOGNITION_API=ensemble
   ```

5. **Build the RAG database** *(first time only)*

   ```bash
   python nutrition_rag/build_db.py
   ```

6. **Run the application**

   ```bash
   python app.py
   ```

   The app will be available at **http://localhost:5000**

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key for Vision & Chat features |
| `USDA_API_KEY` | ⬜ Optional | USDA FoodData Central API key for nutrition lookups |
| `LOGMEAL_API_KEY` | ⬜ Optional | LogMeal API key for additional food recognition |
| `FOOD_RECOGNITION_API` | ⬜ Optional | Primary recognition engine: `ensemble` (default), `gemini`, or `logmeal` |
| `FLASK_DEBUG` | ⬜ Optional | Set to `true` for debug mode |
| `PORT` | ⬜ Optional | Server port (default: `5000`, Render uses `10000`) |

---

## 💡 Usage

### 1. Register & Set Up Your Profile
Create an account with your health details (age, height, weight, gender, activity level, fitness goal, and diet preference). The app uses this to calculate personalized calorie targets.

### 2. Snap & Analyze Food
Upload a photo of your meal from the dashboard. The AI pipeline will:
- Detect individual food items using YOLOv8
- Classify each item using the ensemble CNN models
- Fall back to Gemini Vision if confidence is low
- Fetch nutrition data for each detected item

### 3. Track Your Progress
- View your daily calorie consumption vs. target
- Browse your meal history with macro breakdowns
- Get smart meal recommendations for remaining calorie budget

### 4. Chat with the AI Nutritionist
Ask the chatbot anything about nutrition — it uses RAG to search the local database and provides personalized advice considering your goals, diet, and consumption history.

### 5. Scan Nutrition Labels
Use the OCR scanner to extract nutrition info directly from packaged food labels.

---

## 🧠 AI Models

### Ensemble CNN (Food-101 + Indian Foods)
- **Architecture**: MobileNetV2 (transfer learning)
- **Training Data**: Food-101 dataset (101 classes) + Indian food dataset
- **Ensemble Size**: 3 models with averaged predictions
- **Input**: 224×224 RGB images, normalized to [-1, 1]
- **Total classes**: 98 international food categories

### Indian Food Classifier
- **Dedicated model** for Indian cuisine categories
- **Classes**: Loaded dynamically from `data/indian_food_classes.json`
- **Used when**: Ensemble predicts an Indian food class

### YOLOv8 Object Detector
- **Custom-trained** on 20 food categories (Latin American cuisine)
- **Purpose**: Multi-food detection with bounding boxes in a single image
- **Fallback**: Generic YOLOv8n if custom model is unavailable

### Calorie Predictor (Random Forest)
- **Algorithm**: Random Forest Regressor (100 estimators)
- **Training**: 10,000 synthetic profiles based on Mifflin-St Jeor + Harris-Benedict equations
- **Features**: Age, height, weight, gender, activity level, goal
- **Output**: Blended prediction (70% ML + 30% formula)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Landing page |
| `GET/POST` | `/login` | User login |
| `GET/POST` | `/register` | User registration |
| `GET` | `/dashboard` | Main dashboard (auth required) |
| `POST` | `/predict` | Upload image → food recognition + nutrition |
| `POST` | `/log_calories` | Log a single food entry |
| `POST` | `/log_meal` | Log multiple food items at once |
| `GET` | `/api/progress_data` | Get daily calorie consumption data (JSON) |
| `POST` | `/api/chat` | Send message to AI chatbot (JSON) |
| `POST` | `/api/scan_label` | OCR scan a nutrition label (JSON) |
| `GET` | `/manual_entry` | Manual food entry form |
| `GET` | `/logout` | Clear session and logout |

---

## 🚢 Deployment

### Docker

```bash
docker build -t nutrismart .
docker run -p 10000:10000 --env-file .env nutrismart
```

### Render (Cloud)

The project includes a `render.yaml` for one-click deployment on [Render](https://render.com):

1. Connect your GitHub repository to Render
2. Set the environment variables in the Render dashboard
3. Deploy — Render will build the Docker image automatically

The production server uses **Gunicorn** with 1 worker and 120-second timeout to handle ML model loading.

---

## 🖼 Screenshots

> *Screenshots coming soon — run the app locally to explore the full UI!*

| Page | Description |
|------|-------------|
| **Home** | Modern landing page with glassmorphism cards |
| **Dashboard** | Calorie tracking, progress charts, meal recommendations |
| **Results** | Multi-food detection results with nutrition breakdown |
| **Chatbot** | AI nutrition assistant with personalized responses |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using Flask, TensorFlow, YOLOv8, and Google Gemini AI
</p>
