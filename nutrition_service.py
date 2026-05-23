"""
Hybrid Nutrition Lookup Service
Provides fast local lookups with API fallback to USDA FoodData Central
"""

import sqlite3
import requests
from typing import Dict, Optional
import os

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "nutrition.db")

# USDA FoodData Central API
USDA_API_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
# Note: USDA API works without key for basic searches, but rate limited
# For production, get free API key from: https://fdc.nal.usda.gov/api-key-signup.html


class NutritionService:
    """Manages nutrition data with local cache and API fallback"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_database()
        self._seed_initial_data()
    
    def _init_database(self):
        """Initialize the nutrition database"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
        CREATE TABLE IF NOT EXISTS nutrition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            food_name TEXT UNIQUE NOT NULL,
            calories REAL,
            protein REAL,
            carbs REAL,
            fat REAL,
            fiber REAL,
            serving_size TEXT,
            serving_weight_grams REAL,
            source TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Migration: Add diet_type if missing
        c.execute("PRAGMA table_info(nutrition)")
        columns = [row[1] for row in c.fetchall()]
        if 'diet_type' not in columns:
            try:
                c.execute("ALTER TABLE nutrition ADD COLUMN diet_type TEXT")
                print("[MIGRATION] Added diet_type column to nutrition table")
            except Exception as e:
                print(f"[ERROR] Migration failed: {e}")

        conn.commit()
        conn.close()
    
    def _seed_initial_data(self):
        """Populate database with common foods"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Continue to insert or ignore new items
        
        # Common foods database (per standard serving)
        initial_foods = [
            # Fruits
            # Fruits (Vegan)
            ("apple", 95, 0.5, 25, 0.3, 4.4, "1 medium", 182, "vegan", "local_cache"),
            ("banana", 105, 1.3, 27, 0.4, 3.1, "1 medium", 118, "vegan", "local_cache"),
            ("orange", 62, 1.2, 15, 0.2, 3.1, "1 medium", 131, "vegan", "local_cache"),
            ("grapes", 104, 1.1, 27, 0.2, 1.4, "1 cup", 151, "vegan", "local_cache"),
            ("strawberry", 49, 1.0, 12, 0.5, 3.0, "1 cup", 152, "vegan", "local_cache"),
            ("watermelon", 46, 0.9, 11, 0.2, 0.6, "1 cup diced", 152, "vegan", "local_cache"),
            ("mango", 99, 1.4, 25, 0.6, 2.6, "1 cup sliced", 165, "vegan", "local_cache"),
            
            # Vegetables (Vegan)
            ("broccoli", 55, 3.7, 11, 0.6, 2.4, "1 cup", 156, "vegan", "local_cache"),
            ("carrot", 52, 1.2, 12, 0.3, 3.6, "1 cup", 128, "vegan", "local_cache"),
            ("tomato", 22, 1.1, 4.8, 0.2, 1.5, "1 medium", 123, "vegan", "local_cache"),
            ("lettuce", 5, 0.5, 1.0, 0.1, 0.5, "1 cup", 36, "vegan", "local_cache"),
            ("potato", 164, 4.3, 37, 0.2, 4.4, "1 medium", 173, "vegan", "local_cache"),
            
            # Proteins
            ("chicken breast", 165, 31, 0, 3.6, 0, "3.5 oz", 100, "non-veg", "local_cache"),
            ("salmon", 206, 22, 0, 12, 0, "3.5 oz", 100, "non-veg", "local_cache"),
            ("egg", 72, 6.3, 0.4, 4.8, 0, "1 large", 50, "veg", "local_cache"),
            ("beef", 250, 26, 0, 15, 0, "3.5 oz", 100, "non-veg", "local_cache"),
            ("tuna", 132, 28, 0, 1.3, 0, "3.5 oz", 100, "non-veg", "local_cache"),
            
            # Grains & Carbs (Vegan mostly)
            ("rice", 206, 4.3, 45, 1.8, 0.6, "1 cup cooked", 158, "vegan", "local_cache"),
            ("bread", 79, 2.7, 15, 1.0, 0.8, "1 slice", 28, "vegan", "local_cache"),
            ("pasta", 221, 8.1, 43, 1.3, 2.5, "1 cup cooked", 140, "vegan", "local_cache"),
            ("oatmeal", 154, 6.0, 27, 2.6, 4.0, "1 cup cooked", 234, "vegan", "local_cache"),
            ("quinoa", 222, 8.1, 39, 3.6, 5.2, "1 cup cooked", 185, "vegan", "local_cache"),
            
            # Fast Food
            ("pizza", 285, 12, 36, 10, 2.5, "1 slice", 107, "veg", "local_cache"),
            ("hamburger", 354, 20, 30, 16, 1.5, "1 burger", 150, "non-veg", "local_cache"),
            ("french fries", 365, 3.8, 48, 17, 4.4, "medium serving", 117, "vegan", "local_cache"),
            ("french_fries", 365, 3.8, 48, 17, 4.4, "medium serving", 117, "vegan", "local_cache"),
            ("hot dog", 290, 10, 23, 17, 0.8, "1 hot dog", 98, "non-veg", "local_cache"),
            ("donut", 269, 3.2, 31, 15, 0.9, "1 donut", 60, "veg", "local_cache"),
            
            # Indian Foods
            ("samosa", 262, 3.5, 24, 17, 2.2, "1 samosa", 100, "veg", "local_cache"),
            ("naan", 262, 9.0, 45, 5.2, 2.0, "1 piece", 90, "veg", "local_cache"),
            ("biryani", 290, 8.5, 45, 8.5, 1.5, "1 cup", 200, "non-veg", "local_cache"),
            ("biriyani", 290, 8.5, 45, 8.5, 1.5, "1 cup", 200, "non-veg", "local_cache"),
            ("dal", 116, 8.0, 20, 0.5, 8.0, "1 cup", 198, "vegan", "local_cache"),
            ("roti", 120, 3.0, 23, 2.5, 3.5, "1 piece", 40, "vegan", "local_cache"),
            ("paneer tikka", 265, 15, 6, 21, 1.5, "5-6 pieces", 150, "veg", "local_cache"),
            ("butter chicken", 350, 25, 10, 25, 1.0, "1 cup", 240, "non-veg", "local_cache"),
            ("masala dosa", 360, 6, 50, 16, 4.0, "1 large dosa", 200, "veg", "local_cache"),
            ("idli", 58, 2, 12, 0.1, 0.5, "1 idli", 45, "vegan", "local_cache"),
            ("vada", 97, 2, 7, 7, 1.0, "1 vada", 50, "vegan", "local_cache"),
            ("tandoori chicken", 265, 30, 0, 15, 0, "1 piece", 150, "non-veg", "local_cache"),
            ("gulab jamun", 150, 2, 25, 5, 0.5, "1 piece", 50, "veg", "local_cache"),
            ("chicken_curry", 240, 25, 6, 13, 2.0, "1 cup", 200, "non-veg", "local_cache"),
            
            # Dairy
            ("milk", 149, 7.7, 12, 7.9, 0, "1 cup", 244, "veg", "local_cache"),
            ("yogurt", 149, 8.5, 11, 8.0, 0, "1 cup", 245, "veg", "local_cache"),
            ("cheese", 113, 7.0, 0.9, 9.3, 0, "1 oz", 28, "veg", "local_cache"),
            
            # Prepared dishes
            ("omelette", 154, 11, 1.1, 12, 0, "2 eggs", 122, "veg", "local_cache"),
            ("sandwich", 300, 15, 35, 12, 3.0, "1 sandwich", 150, "veg", "local_cache"),
            ("salad", 152, 5.0, 12, 10, 3.0, "1 bowl", 200, "vegan", "local_cache"),
            ("spaghetti", 221, 8.1, 43, 1.3, 2.5, "1 cup cooked", 140, "vegan", "local_cache"),
            
            # Additional classes from Ensemble/YOLO
            ("apple pie", 237, 1.9, 34, 11, 2.1, "1 slice", 100, "veg", "local_cache"),
            ("apple_pie", 237, 1.9, 34, 11, 2.1, "1 slice", 100, "veg", "local_cache"),
            ("baklava", 334, 6.0, 30, 21, 2.0, "1 piece", 60, "veg", "local_cache"),
            ("caesar salad", 190, 6.0, 8.0, 15, 2.0, "1 bowl", 150, "non-veg", "local_cache"),
            ("caesar_salad", 190, 6.0, 8.0, 15, 2.0, "1 bowl", 150, "non-veg", "local_cache"),
            ("cheesecake", 257, 4.4, 20, 18, 0.4, "1 slice", 80, "veg", "local_cache"),
            ("chocolate cake", 389, 3.5, 50, 20, 2.0, "1 slice", 100, "veg", "local_cache"),
            ("chocolate_cake", 389, 3.5, 50, 20, 2.0, "1 slice", 100, "veg", "local_cache"),
            ("cup cakes", 250, 2.5, 33, 12, 0.5, "1 cupcake", 60, "veg", "local_cache"),
            ("cup_cakes", 250, 2.5, 33, 12, 0.5, "1 cupcake", 60, "veg", "local_cache"),
            ("cupcake", 250, 2.5, 33, 12, 0.5, "1 cupcake", 60, "veg", "local_cache"),
            ("chocolate cupcake", 280, 2.8, 35, 14, 0.6, "1 cupcake", 65, "veg", "local_cache"),
            ("chocolate_cupcake", 280, 2.8, 35, 14, 0.6, "1 cupcake", 65, "veg", "local_cache"),
            ("french onion soup", 210, 10, 15, 12, 2.0, "1 bowl", 250, "non-veg", "local_cache"),
            ("french_onion_soup", 210, 10, 15, 12, 2.0, "1 bowl", 250, "non-veg", "local_cache"),
            ("fried rice", 333, 6.0, 45, 14, 2.0, "1 cup", 150, "non-veg", "local_cache"),
            ("fried_rice", 333, 6.0, 45, 14, 2.0, "1 cup", 150, "non-veg", "local_cache"),
            ("garlic bread", 350, 9.0, 46, 15, 2.0, "2 slices", 100, "veg", "local_cache"),
            ("garlic_bread", 350, 9.0, 46, 15, 2.0, "2 slices", 100, "veg", "local_cache"),
            ("greek salad", 106, 2.0, 6.0, 8.0, 2.0, "1 bowl", 150, "veg", "local_cache"),
            ("greek_salad", 106, 2.0, 6.0, 8.0, 2.0, "1 bowl", 150, "veg", "local_cache"),
            ("guacamole", 230, 3.0, 12, 21, 9.0, "0.5 cup", 100, "vegan", "local_cache"),
            ("hummus", 166, 8.0, 14, 10, 6.0, "100g", 100, "vegan", "local_cache"),
            ("ice cream", 207, 3.5, 24, 11, 0.7, "1 scoop", 100, "veg", "local_cache"),
            ("ice_cream", 207, 3.5, 24, 11, 0.7, "1 scoop", 100, "veg", "local_cache"),
            ("lasagna", 135, 15, 15, 6, 2.0, "1 piece", 150, "non-veg", "local_cache"),
            ("macaroni and cheese", 310, 12, 35, 14, 1.0, "1 cup", 180, "veg", "local_cache"),
            ("macaroni_and_cheese", 310, 12, 35, 14, 1.0, "1 cup", 180, "veg", "local_cache"),
            ("onion rings", 411, 3.0, 48, 23, 3.0, "6 pieces", 100, "veg", "local_cache"),
            ("onion_rings", 411, 3.0, 48, 23, 3.0, "6 pieces", 100, "veg", "local_cache"),
            ("pancakes", 227, 6.0, 28, 10, 1.0, "2 pieces", 100, "veg", "local_cache"),
            ("ramen", 436, 10, 52, 20, 2.0, "1 bowl", 300, "non-veg", "local_cache"),
            ("steak", 271, 25, 0, 19, 0, "4 oz", 112, "non-veg", "local_cache"),
            ("sushi", 300, 10, 60, 2.0, 2.0, "6 pieces", 200, "non-veg", "local_cache"),
            ("tacos", 226, 11, 18, 12, 3.0, "1 taco", 100, "non-veg", "local_cache"),
            ("waffles", 291, 7.9, 33, 14, 1.0, "1 waffle", 100, "veg", "local_cache"),
            
            # YOLO Classes (Spanish names)
            ("aguacate", 160, 2.0, 8.5, 14.7, 6.7, "100g", 100, "vegan", "local_cache"),
            ("ahuyama", 26, 1.0, 6.5, 0.1, 0.5, "100g", 100, "vegan", "local_cache"),
            ("arepa", 250, 4.0, 50, 2.0, 3.0, "1 unit", 100, "vegan", "local_cache"),
            ("arroz", 130, 2.7, 28, 0.3, 0.4, "100g", 100, "vegan", "local_cache"),
            ("arroz con pollo", 150, 8.0, 20, 4.0, 1.0, "100g", 100, "non-veg", "local_cache"),
            ("chicharron", 544, 61, 0, 31, 0, "100g", 100, "non-veg", "local_cache"),
            ("chorizo", 455, 24, 1.0, 38, 0, "100g", 100, "non-veg", "local_cache"),
            ("frijol", 143, 9.0, 26, 0.5, 9.0, "100g", 100, "vegan", "local_cache"),
            ("huevo", 155, 13, 1.1, 11, 0, "100g", 100, "veg", "local_cache"),
            ("lentejas", 116, 9.0, 20, 0.4, 8.0, "100g", 100, "vegan", "local_cache"),
            ("papa", 77, 2.0, 17, 0.1, 2.2, "100g", 100, "vegan", "local_cache"),
            ("platano", 122, 1.3, 32, 0.4, 2.3, "100g", 100, "vegan", "local_cache"),
            ("pollo", 239, 27, 0, 14, 0, "100g", 100, "non-veg", "local_cache"),
            ("trucha", 190, 21, 0, 11, 0, "100g", 100, "non-veg", "local_cache"),
        ]
        
        c.executemany("""
            INSERT OR REPLACE INTO nutrition
            (food_name, calories, protein, carbs, fat, fiber, serving_size, serving_weight_grams, diet_type, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, initial_foods)
        
        conn.commit()
        conn.close()
        print(f"[OK] Seeded nutrition database with {len(initial_foods)} foods")
    
    def get_nutrition(self, food_name: str, portion_size: float = 1.0) -> Dict:
        """
        Get nutrition information for a food item
        
        Args:
            food_name: Name of the food
            portion_size: Multiplier for serving size (1.0 = standard serving)
        
        Returns:
            Dictionary with nutrition data
        """
        food_name = food_name.lower().strip()
        
        # Try local cache first with exact name
        nutrition = self._lookup_local(food_name)
        
        # If not found, try replacing underscores with spaces
        if not nutrition and "_" in food_name:
            alt_name = food_name.replace("_", " ")
            print(f"[DEBUG] '{food_name}' not found, trying '{alt_name}'...")
            nutrition = self._lookup_local(alt_name)

        # If still not found, try removing underscores entirely
        if not nutrition and "_" in food_name:
            alt_name = food_name.replace("_", "")
            print(f"[DEBUG] '{food_name}' not found, trying '{alt_name}'...")
            nutrition = self._lookup_local(alt_name)
        
        # Fallback to API if not found locally
        if not nutrition:
            print(f"[API] '{food_name}' not in cache, querying USDA API...")
            nutrition = self._lookup_api(food_name)
            
            # Cache the result for future use
            if nutrition:
                self._cache_nutrition(food_name, nutrition)
        
        # Fallback to Gemini if USDA also failed
        if not nutrition:
            print(f"[GEMINI NUTRITION] USDA failed for '{food_name}', trying Gemini...")
            nutrition = self._lookup_gemini(food_name)
            if nutrition:
                self._cache_nutrition(food_name, nutrition)
        
        # Apply portion size multiplier
        if nutrition and portion_size != 1.0:
            for key in ['calories', 'protein', 'carbs', 'fat', 'fiber']:
                if key in nutrition and nutrition[key]:
                    nutrition[key] = round(nutrition[key] * portion_size, 1)
            nutrition['portion_multiplier'] = portion_size
        
        return nutrition or self._get_default_response(food_name)
    
    def _lookup_local(self, food_name: str) -> Optional[Dict]:
        """Search local database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        c.execute("""
            SELECT * FROM nutrition 
            WHERE LOWER(food_name) = ?
            LIMIT 1
        """, (food_name,))
        
        row = c.fetchone()
        conn.close()
        
        if row:
            return {
                'food_name': row['food_name'],
                'calories': row['calories'],
                'protein': row['protein'],
                'carbs': row['carbs'],
                'fat': row['fat'],
                'fiber': row['fiber'],
                'serving_size': row['serving_size'],
                'serving_weight_grams': row['serving_weight_grams'],
                'diet_type': row['diet_type'],
                'source': 'local_cache'
            }
        return None
    
    def _lookup_api(self, food_name: str) -> Optional[Dict]:
        """Query USDA FoodData Central API"""
        try:
            # Get API key from environment variable if available
            api_key = os.getenv('USDA_API_KEY')
            
            # Try to load API key from file if not in environment
            if not api_key:
                for key_file in ['.env', 'keys']:
                    if os.path.exists(key_file):
                        with open(key_file, 'r') as f:
                            for line in f:
                                if 'USDA_API_KEY' in line:
                                    api_key = line.split('=')[-1].strip().strip('"').strip("'")
                                    break
                        if api_key: break
            
            params = {
                'query': food_name,
                'pageSize': 1,
                'dataType': ['Survey (FNDDS)', 'Foundation', 'SR Legacy']
            }
            
            # Add API key if available
            if api_key:
                params['api_key'] = api_key
            
            response = requests.get(USDA_API_URL, params=params, timeout=5)
            
            # Check for API key error
            if response.status_code == 403:
                print("[WARNING] USDA API requires an API key.")
                print("          Get a free API key at: https://fdc.nal.usda.gov/api-key-signup.html")
                print("          Set it as environment variable: USDA_API_KEY=your_key_here")
                return None
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('foods') and len(data['foods']) > 0:
                    food = data['foods'][0]
                    # Map common nutrient names to our keys
                    nutrient_map = {
                        'Energy': ['Energy', 'Energy (kcal)', 'Total calories'],
                        'Protein': ['Protein'],
                        'Carbs': ['Carbohydrate, by difference', 'Total carbohydrate'],
                        'Fat': ['Total lipid (fat)', 'Fat'],
                        'Fiber': ['Fiber, total dietary', 'Fiber']
                    }
                    
                    found_nutrients = {}
                    for nutrient in food.get('foodNutrients', []):
                        name = nutrient.get('nutrientName')
                        value = nutrient.get('value', 0)
                        for key, variations in nutrient_map.items():
                            if name in variations:
                                found_nutrients[key] = value
                    
                    return {
                        'food_name': food.get('description', food_name),
                        'calories': found_nutrients.get('Energy', 0),
                        'protein': found_nutrients.get('Protein', 0),
                        'carbs': found_nutrients.get('Carbs', 0),
                        'fat': found_nutrients.get('Fat', 0),
                        'fiber': found_nutrients.get('Fiber', 0),
                        'serving_size': '100g',
                        'serving_weight_grams': 100,
                        'source': 'usda_api'
                    }
        except Exception as e:
            print(f"[ERROR] API Error: {e}")
        
        return None
    
    def _lookup_gemini(self, food_name: str) -> Optional[Dict]:
        """Query Gemini API for nutrition data as final fallback"""
        try:
            import json
            from google import genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("[GEMINI NUTRITION] No GEMINI_API_KEY found")
                return None
            
            client = genai.Client(api_key=api_key)
            
            prompt = f"""Give me the nutrition values per 100g for "{food_name.replace('_', ' ')}".
Return ONLY a JSON object with no extra text, no markdown, no backticks.
Format exactly like this:
{{"calories": 0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}}
Use realistic average values from nutritional databases."""

            models = ['gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-2.5-flash']
            
            for model_name in models:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                    raw = response.text.strip()
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    data = json.loads(raw)
                    
                    # Validate that we got real values (not all zeros)
                    if data.get('calories', 0) > 0:
                        print(f"[GEMINI NUTRITION] ✅ Got nutrition for '{food_name}' via {model_name}")
                        return {
                            'food_name': food_name,
                            'calories': data.get('calories', 0),
                            'protein': data.get('protein', 0),
                            'carbs': data.get('carbs', 0),
                            'fat': data.get('fat', 0),
                            'fiber': data.get('fiber', 0),
                            'serving_size': '100g',
                            'serving_weight_grams': 100,
                            'source': 'gemini_api'
                        }
                except Exception as e:
                    print(f"[GEMINI NUTRITION] Model {model_name} failed: {e}")
                    continue
                    
        except Exception as e:
            print(f"[GEMINI NUTRITION] Error: {e}")
        
        return None
    
    def _cache_nutrition(self, food_name: str, nutrition: Dict):
        """Save API result to local database"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT OR REPLACE INTO nutrition
                (food_name, calories, protein, carbs, fat, fiber, serving_size, serving_weight_grams, diet_type, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                food_name,
                nutrition.get('calories'),
                nutrition.get('protein'),
                nutrition.get('carbs'),
                nutrition.get('fat'),
                nutrition.get('fiber'),
                nutrition.get('serving_size'),
                nutrition.get('serving_weight_grams'),
                nutrition.get('diet_type', 'veg'), # Default to veg for unknown if needed
                'usda_api_cached'
            ))
            
            conn.commit()
            conn.close()
            print(f"[CACHED] Saved nutrition data for '{food_name}'")
        except Exception as e:
            print(f"[ERROR] Cache error: {e}")
    
    def _get_default_response(self, food_name: str) -> Dict:
        """Return default response when lookup fails"""
        return {
            'food_name': food_name,
            'calories': 'N/A',
            'protein': 'N/A',
            'carbs': 'N/A',
            'fat': 'N/A',
            'fiber': 'N/A',
            'serving_size': 'Unknown',
            'serving_weight_grams': None,
            'diet_type': 'unknown',
            'source': 'not_found'
        }

    def get_recommendations(self, calorie_limit: float, diet_preference: str, recent_foods: list = None) -> list:
        """Fetch smart recommendations from the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Filter logic:
        # 1. Matches diet preference (veg, vegan, non-veg)
        # 2. Calories fit within limit
        # 3. Randomize a bit but prioritize common/diverse items
        
        diet_query = "diet_type = ?"
        params = [diet_preference]
        
        if diet_preference == 'veg':
            diet_query = "diet_type IN ('veg', 'vegan')"
            params = []
        elif diet_preference == 'vegan':
            diet_query = "diet_type = 'vegan'"
            params = []
        else:
            # non-veg gets everything
            diet_query = "1=1"
            params = []

        query = f"""
            SELECT * FROM nutrition 
            WHERE {diet_query} AND calories <= ? 
            ORDER BY RANDOM() 
            LIMIT 5
        """
        params.append(calorie_limit)
        
        c.execute(query, params)
        rows = c.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]


# Global service instance
_service = None

def get_nutrition(food_name: str, portion_size: float = 1.0) -> Dict:
    """
    Convenience function to get nutrition data
    
    Usage:
        nutrition = get_nutrition("apple")
        nutrition = get_nutrition("pizza", portion_size=2.0)  # 2 slices
    """
    global _service
    if _service is None:
        _service = NutritionService()
    return _service.get_nutrition(food_name, portion_size)


if __name__ == "__main__":
    # Test the service
    print("Testing Nutrition Service\n" + "="*50)
    
    # Test local cache
    print("\n1. Testing local cache (apple):")
    result = get_nutrition("apple")
    print(f"   {result}")
    
    print("\n2. Testing your model classes (pizza):")
    result = get_nutrition("pizza")
    print(f"   {result}")
    
    print("\n3. Testing with portion size (2x french fries):")
    result = get_nutrition("french_fries", portion_size=2.0)
    print(f"   {result}")
    
    print("\n4. Testing API fallback (uncommon food):")
    result = get_nutrition("quinoa")
    print(f"   {result}")
