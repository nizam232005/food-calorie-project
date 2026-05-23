import os
import json
from google import genai
from typing import Dict
from nutrition_rag.rag_pipeline import search_nutrition, is_food_in_db, add_food_to_db

class ChatService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.models = [
                'gemini-2.0-flash-lite',
                'gemini-2.0-flash',
                'gemini-2.5-flash',
            ]
        else:
            print("GEMINI ERROR: GEMINI_API_KEY not found.")
            self.client = None

    def fetch_and_store_nutrition(self, food_name: str) -> bool:
        """Ask Gemini to find nutrition values and store in database"""
        prompt = f"""Give me the nutrition values per 100g for "{food_name}".
Return ONLY a JSON object with no extra text, no markdown, no backticks.
Format exactly like this:
{{"name": "{food_name.lower().replace(' ', '_')}", "calories": 0, "protein": 0.0, "carbs": 0.0, "fat": 0.0, "fiber": 0.0}}
Use realistic average values from nutritional databases."""

        for model_name in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                raw = response.text.strip()
                # Clean any markdown if present
                raw = raw.replace("```json", "").replace("```", "").strip()
                food_data = json.loads(raw)

                # Validate required fields
                required = ["name", "calories", "protein", "carbs", "fat", "fiber"]
                if all(k in food_data for k in required):
                    add_food_to_db(food_data)
                    return True
            except Exception as e:
                print(f"Failed to fetch nutrition for {food_name}: {e}")
                continue
        return False

    def _extract_food_names(self, message: str) -> list:
        """Extract potential food names from user message using n-grams.
        Handles multi-word foods like 'chicken nuggets', 'ice cream', 'french fries'.
        Returns candidates longest-first so multi-word matches are tried before single words.
        """
        # Common stop words to ignore
        stop_words = {
            'how', 'many', 'much', 'what', 'is', 'are', 'the', 'a', 'an', 'in',
            'of', 'for', 'and', 'or', 'to', 'can', 'do', 'does', 'about', 'tell',
            'me', 'my', 'i', 'it', 'its', 'calories', 'calorie', 'nutrition',
            'nutritional', 'value', 'values', 'info', 'information', 'per', 'have',
            'has', 'does', 'contain', 'good', 'bad', 'healthy', 'unhealthy',
            'protein', 'carbs', 'fat', 'fiber', 'give', 'show', 'please', 'thanks',
            'thank', 'you', 'hey', 'hi', 'hello', 'with', 'without', 'from',
            'this', 'that', 'these', 'those', 'some', 'any', 'eating', 'eat',
            'ate', 'had', 'having', 'today', 'yesterday', 'should', 'would',
            'could', 'need', 'want', 'like', 'compare', 'between', 'vs'
        }

        # Clean and tokenize
        clean_msg = message.lower().strip()
        # Remove punctuation
        for ch in '?!.,;:()[]{}"\'/':
            clean_msg = clean_msg.replace(ch, ' ')
        words = [w.strip() for w in clean_msg.split() if w.strip() and w.strip() not in stop_words]

        if not words:
            return []

        candidates = []

        # Generate n-grams: trigrams first, then bigrams, then unigrams
        # Longer matches are prioritized (e.g., "chicken nuggets" before "chicken")
        for n in range(min(3, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i + n])
                if len(phrase) > 3:  # Skip very short candidates
                    candidates.append(phrase)

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique

    def get_response(self, user_message: str, context: Dict) -> str:
        if not self.client:
            return "AI service not available. Please check your API key."

        # ---- STEP 1: RAG - Search our nutrition database ----
        try:
            docs, metadatas = search_nutrition(user_message, n_results=3)
            nutrition_context = "\n".join(docs)

            # Check if top result is a poor match - try to auto-add
            # Extract food names using n-grams (handles multi-word foods like "chicken nuggets")
            food_candidates = self._extract_food_names(user_message)
            for food_name in food_candidates:
                if not is_food_in_db(food_name):
                    print(f"Food '{food_name}' not in DB, fetching from Gemini...")
                    self.fetch_and_store_nutrition(food_name)
                    # Re-search after adding
                    docs, metadatas = search_nutrition(user_message, n_results=3)
                    nutrition_context = "\n".join(docs)
                    break

        except Exception as e:
            print(f"RAG search error: {e}")
            nutrition_context = "No specific nutrition data found."
            metadatas = []

        # ---- STEP 2: Build prompt with retrieved nutrition data ----
        prompt = f"""You are a professional Nutrition Assistant for the NutriSmart app.

RETRIEVED NUTRITION DATA FROM DATABASE:
{nutrition_context}

USER PROFILE:
- Goal: {context.get('goal', 'General Health')}
- Diet Type: {context.get('diet', 'Not specified')}
- Daily Calorie Target: {context.get('target', 2000)} kcal
- Calories Consumed Today: {context.get('consumed', 0)} kcal
- Calories Remaining: {context.get('remaining', 2000)} kcal
- Recent Foods: {", ".join(context.get('history', [])) if context.get('history') else "No recent logs"}

INSTRUCTIONS:
- Use the retrieved nutrition data above to answer accurately
- Be concise and friendly
- Use bullet points for lists
- Give personalized advice based on user profile
- If asked about a specific food, use the exact values from the database

User Question: {user_message}
"""

        # ---- STEP 3: Generate answer using Gemini ----
        for model_name in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                print(f"[RAG+GEMINI] Success with model: {model_name}")
                return response.text
            except Exception as e:
                print(f"[RAG+GEMINI] Model {model_name} failed: {e}")
                continue

        return "All AI models are currently at capacity. Please try again!"

# Global instance
_chat_service = None

def get_chatbot_response(message: str, context: Dict) -> str:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service.get_response(message, context)