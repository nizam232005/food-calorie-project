import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "nutrition_data.json")
DB_PATH = os.path.join(BASE_DIR, "nutrition_rag", "nutrition_db")

def build_database():
    print("Loading nutrition data...")
    with open(DATA_PATH, "r") as f:
        foods = json.load(f)

    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Delete existing collection if rebuilding
    try:
        client.delete_collection("nutrition")
    except:
        pass
    
    collection = client.get_or_create_collection("nutrition")

    print(f"Adding {len(foods)} foods to database...")
    for i, food in enumerate(foods):
        name = food["name"]
        
        # Create rich text description for better search
        text = (
            f"{name.replace('_', ' ')}: "
            f"{food['calories']} calories, "
            f"{food['protein']}g protein, "
            f"{food['carbs']}g carbs, "
            f"{food['fat']}g fat, "
            f"{food['fiber']}g fiber per 100g"
        )
        
        embedding = embedder.encode(text).tolist()
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[str(i)],
            metadatas=[food]
        )
    
    print(f"✅ Database built with {len(foods)} foods!")
    print(f"📁 Saved to: {DB_PATH}")

if __name__ == "__main__":
    build_database()