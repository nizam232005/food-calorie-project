import sys
import os
sys.path.append(os.getcwd())
from nutrition_service import get_nutrition

test_items = ["chocolate_cupcake", "cup_cakes", "chicken_curry", "aguacate", "chocolate cupcake"]

for item in test_items:
    result = get_nutrition(item)
    print(f"Item: {item:20} | Calories: {result.get('calories')}")
