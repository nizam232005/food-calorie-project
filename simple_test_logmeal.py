import os
import requests

def get_keys():
    keys = {}
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    keys[k] = v
    return keys

def test_logmeal():
    keys = get_keys()
    api_key = keys.get("LOGMEAL_API_KEY")
    api_url = "https://api.logmeal.es/v2/image/recognition/dish"
    
    print(f"Testing LogMeal with key: {api_key[:5]}...")
    
    if not api_key:
        print("Error: No LogMeal API key found.")
        return

    headers = {'Authorization': f'Bearer {api_key}'}
    test_img = "temp_upload.jpg"
    
    if os.path.exists(test_img):
        with open(test_img, 'rb') as f:
            files = {'image': f}
            try:
                # Add a timeout to avoid hanging
                response = requests.post(api_url, headers=headers, files=files, timeout=10)
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text[:200]}")
            except Exception as e:
                print(f"Error: {e}")
    else:
        print(f"Error: {test_img} not found.")

if __name__ == "__main__":
    test_logmeal()
