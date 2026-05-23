import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "nutrition_rag", "nutrition_db")
DATA_PATH = os.path.join(BASE_DIR, "data", "nutrition_data.json")

# Load models once
print("Loading RAG pipeline...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection("nutrition")
print("✅ RAG pipeline ready!")

def search_nutrition(query: str, n_results: int = 3):
    """Search nutrition database for relevant foods"""
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0], results["metadatas"][0]

def is_food_in_db(food_name: str, threshold: float = 0.7) -> bool:
    """Check if a specific food exists in the database with good confidence"""
    query_embedding = embedder.encode(food_name).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        include=["distances"]
    )
    if results["distances"] and results["distances"][0]:
        distance = results["distances"][0][0]
        # Lower distance = better match (ChromaDB uses L2 distance)
        return distance < 0.5
    return False

def add_food_to_db(food_data: dict):
    """Add a new food to ChromaDB and nutrition_data.json"""
    try:
        # Create text description
        name = food_data["name"]
        text = (
            f"{name.replace('_', ' ')}: "
            f"{food_data['calories']} calories, "
            f"{food_data['protein']}g protein, "
            f"{food_data['carbs']}g carbs, "
            f"{food_data['fat']}g fat, "
            f"{food_data['fiber']}g fiber per 100g"
        )

        # Get current count for new ID
        existing = collection.count()
        new_id = str(existing + 1)

        # Add to ChromaDB
        embedding = embedder.encode(text).tolist()
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[new_id],
            metadatas=[food_data]
        )

        # Also save to nutrition_data.json for persistence
        if os.path.exists(DATA_PATH):
            with open(DATA_PATH, "r") as f:
                all_foods = json.load(f)
            all_foods.append(food_data)
            with open(DATA_PATH, "w") as f:
                json.dump(all_foods, f, indent=2)

        print(f"✅ Auto-added new food: {name}")
        return True

    except Exception as e:
        print(f"❌ Failed to add food: {e}")
        return False