import os
import requests
from dotenv import load_dotenv
from PIL import Image
import numpy as np

load_dotenv()

LOGMEAL_API_KEY = os.getenv("LOGMEAL_API_KEY")
LOGMEAL_API_URL = "https://api.logmeal.es/v2/image/recognition/dish"

def test_logmeal(image_path):
    print(f"\n--- Testing LogMeal ---")
    if not LOGMEAL_API_KEY:
        print("Error: LOGMEAL_API_KEY not found in .env")
        return
    
    headers = {'Authorization': f'Bearer {LOGMEAL_API_KEY}'}
    with open(image_path, 'rb') as f:
        files = {'image': f}
        try:
            response = requests.post(LOGMEAL_API_URL, headers=headers, files=files)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("Response JSON:", response.json())
            else:
                print(f"Error Body: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")

if __name__ == "__main__":
    test_img = "temp_upload.jpg"
    if os.path.exists(test_img):
        test_logmeal(test_img)
    else:
        print(f"Error: {test_img} not found. Please upload an image first or provide a test image.")
