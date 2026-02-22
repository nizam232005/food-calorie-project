import os
import sys

# Manual .env loading
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value.strip("'").strip('"')

from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available models:")
for m in client.models.list():
    print(f"  - {m.name}")
