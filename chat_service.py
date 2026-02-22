import os
from google import genai
from typing import Dict

class ChatService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            # Try multiple models in order - each has separate quota
            self.models = [
                'gemini-2.0-flash-lite',
                'gemini-2.0-flash',
                'gemini-2.5-flash',
                'gemini-flash-latest',
            ]
        else:
            print("GEMINI ERROR: GEMINI_API_KEY not found in environment.")
            self.client = None

    def get_response(self, user_message: str, context: Dict) -> str:
        if not self.client:
            return "I'm sorry, my brain isn't connected yet. Please check your API key."

        system_prompt = f"""
        You are a professional Nutrition and Fitness Assistant for the NutriSmart app.
        Your goal is to provide accurate, encouraging, and personalized advice.
        
        USER PROFILE & CONTEXT:
        - Goal: {context.get('goal', 'General Health')}
        - Diet Type: {context.get('diet', 'Not specified')}
        - Daily Calorie Target: {context.get('target', 2000)} kcal
        - Calories Consumed Today: {context.get('consumed', 0)} kcal
        - Calories Remaining: {context.get('remaining', 2000)} kcal
        - Recent History: {", ".join(context.get('history', [])) if context.get('history') else "No recent logs"}

        Be concise, use bullet points for lists, and always stay friendly!
        """

        # Try each model until one works
        for model_name in self.models:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=f"{system_prompt}\n\nUser Question: {user_message}"
                )
                print(f"[CHAT] Success with model: {model_name}")
                return response.text
            except Exception as e:
                print(f"[CHAT] Model {model_name} failed: {e}")
                continue
        
        return "All AI models are currently at capacity. Please try again in a minute!"

# Global instance
_chat_service = None

def get_chatbot_response(message: str, context: Dict) -> str:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service.get_response(message, context)
