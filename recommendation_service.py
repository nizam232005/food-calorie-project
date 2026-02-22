from nutrition_service import NutritionService

class RecommendationService:
    def __init__(self):
        self.nutrition_service = NutritionService()

    def suggest_next_meal(self, calorie_budget, diet_preference, recent_history=None):
        """
        Suggests 3-5 food items based on current budget and preference.
        """
        # We cap the calorie budget per meal (e.g., don't suggest a 2000 calorie meal if that's all left)
        # Usually a meal is 30-40% of daily total, or whatever is left.
        
        meal_limit = min(calorie_budget, 1000) # Max 1000 per suggestion
        
        recommendations = self.nutrition_service.get_recommendations(
            calorie_limit=meal_limit,
            diet_preference=diet_preference,
            recent_foods=recent_history
        )
        
        return recommendations

# Global instance
_rec_service = None

def get_meal_recommendations(budget, diet, history=None):
    global _rec_service
    if _rec_service is None:
        _rec_service = RecommendationService()
    return _rec_service.suggest_next_meal(budget, diet, recent_history=history)
