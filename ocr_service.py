import easyocr
import re
import numpy as np
from PIL import Image
import io

class OCRService:
    def __init__(self):
        # We initialize the reader for English (and maybe Marathi/Hindi if needed for Indian students)
        self.reader = easyocr.Reader(['en'])

    def scan_label(self, image_bytes):
        """
        Parses a nutrition label image and returns extracted calories/macros.
        """
        results = self.reader.readtext(image_bytes)
        full_text = " ".join([text for (bbox, text, prob) in results]).lower()
        
        # Dictionary to store results
        nutrition_data = {
            "calories": 0,
            "protein": 0,
            "carbs": 0,
            "fat": 0
        }

        # Improved Regex patterns for better real-world matching
        # Handles: "Calories: 200", "Energy=200", "Calorie (kcal) 200", etc.
        patterns = {
            "calories": r"(?:calories|energy|calori.*?)\D*?(\d+)",
            "protein": r"(?:protein|proteins)\D*?(\d+(?:\.\d+)?)",
            "carbs": r"(?:carbohydrate|total carbohydrate|carbs|carb)\D*?(\d+(?:\.\d+)?)",
            "fat": r"(?:total fat|fat|fats)\D*?(\d+(?:\.\d+)?)",
        }

        print(f"[OCR DEBUG] Extracted text: {full_text}")

        for key, pattern in patterns.items():
            match = re.search(pattern, full_text)
            if match:
                try:
                    val = float(match.group(1))
                    # Basic validation: calories shouldn't be suspiciously huge from a single scan error
                    if key == "calories" and val > 2000: continue 
                    nutrition_data[key] = val
                except:
                    pass

        return nutrition_data

# Global instance
_ocr_service = None

def get_label_data(image_file):
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    
    image_bytes = image_file.read()
    image_file.seek(0)
    return _ocr_service.scan_label(image_bytes)
