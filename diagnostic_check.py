import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value.strip("'").strip('"')

api = os.getenv("FOOD_RECOGNITION_API", "logmeal").lower()
print(f"DIAGNOSTIC: FOOD_RECOGNITION_API={api}")

log_key = os.getenv("LOGMEAL_API_KEY")
print(f"DIAGNOSTIC: LOGMEAL_API_KEY present: {log_key is not None}")

# Check if app.py has syntax errors
try:
    import app
    print("DIAGNOSTIC: app.py imported successfully (no syntax errors)")
except Exception as e:
    print(f"DIAGNOSTIC ERROR: app.py has errors: {e}")
