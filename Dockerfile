FROM python:3.11-slim

# Install system dependencies for OpenCV, EasyOCR, and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY ensemble_models/ ./ensemble_models/
COPY yolo_models/ ./yolo_models/
COPY model/ ./model/

# Copy application code
COPY app.py .
COPY chat_service.py .
COPY nutrition_service.py .
COPY calorie_predictor.py .
COPY ocr_service.py .
COPY recommendation_service.py .
COPY labels.txt .
COPY food_data.yaml .

# Copy templates and static files
COPY templates/ ./templates/
COPY static/ ./static/
COPY utils/ ./utils/

# Expose port
EXPOSE 10000

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "120", "--workers", "1", "app:app"]
