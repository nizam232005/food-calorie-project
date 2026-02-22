import os
from dotenv import load_dotenv

load_dotenv()
print(f"USDA_API_KEY: {'Found' if os.getenv('USDA_API_KEY') else 'NOT FOUND'}")
print(f"LOGMEAL_API_KEY: {'Found' if os.getenv('LOGMEAL_API_KEY') else 'NOT FOUND'}")
print(f"GEMINI_API_KEY: {'Found' if os.getenv('GEMINI_API_KEY') else 'NOT FOUND'}")
print(f"Length of GEMINI_API_KEY if found: {len(os.getenv('GEMINI_API_KEY')) if os.getenv('GEMINI_API_KEY') else 'N/A'}")
