from __future__ import annotations
"""
Personalized Calorie Prediction Module
Uses Random Forest Regression trained on synthetic data based on Mifflin-St Jeor equation
"""

import numpy as np
import pickle
import os
from typing import Dict, Optional, Any

# Try to import sklearn, provide helpful error if not available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    # Define a dummy class to avoid NameError in type hints if scikit-learn is missing
    class RandomForestRegressor:
        pass
    print("Warning: scikit-learn not installed. Install with: pip install scikit-learn")

# Model file path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "calorie_model.pkl")

# Activity level multipliers (based on Harris-Benedict activity factors)
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,           # Little or no exercise
    'lightly_active': 1.375,    # Light exercise 1-3 days/week
    'moderately_active': 1.55,  # Moderate exercise 3-5 days/week
    'very_active': 1.725,       # Hard exercise 6-7 days/week
    'extra_active': 1.9         # Very hard exercise, physical job
}

# Goal adjustments (calories per day)
GOAL_ADJUSTMENTS = {
    'weight_loss': -500,    # Deficit for ~0.5kg/week loss
    'maintain': 0,          # No adjustment
    'muscle_gain': 300      # Surplus for muscle building
}

# Encoders for categorical features
GENDER_ENCODING = {'male': 0, 'female': 1}
ACTIVITY_ENCODING = {
    'sedentary': 0,
    'lightly_active': 1,
    'moderately_active': 2,
    'very_active': 3,
    'extra_active': 4
}
GOAL_ENCODING = {'weight_loss': 0, 'maintain': 1, 'muscle_gain': 2}


def calculate_bmr(age: int, height: float, weight: float, gender: str) -> float:
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.
    
    Args:
        age: Age in years
        height: Height in cm
        weight: Weight in kg
        gender: 'male' or 'female'
    
    Returns:
        BMR in calories/day
    """
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return bmr


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """
    Calculate Total Daily Energy Expenditure.
    
    Args:
        bmr: Basal Metabolic Rate
        activity_level: Activity level string
    
    Returns:
        TDEE in calories/day
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def generate_training_data(n_samples: int = 10000) -> tuple:
    """
    Generate synthetic training data based on Mifflin-St Jeor equation.
    
    Returns:
        X: Feature matrix
        y: Target values (daily calories)
    """
    np.random.seed(42)
    
    # Generate random user profiles
    ages = np.random.randint(18, 70, n_samples)
    heights = np.random.randint(150, 200, n_samples)  # cm
    weights = np.random.randint(45, 120, n_samples)   # kg
    genders = np.random.choice(['male', 'female'], n_samples)
    activity_levels = np.random.choice(list(ACTIVITY_ENCODING.keys()), n_samples)
    goals = np.random.choice(list(GOAL_ENCODING.keys()), n_samples)
    
    # Calculate target calories
    target_calories = []
    for i in range(n_samples):
        bmr = calculate_bmr(ages[i], heights[i], weights[i], genders[i])
        tdee = calculate_tdee(bmr, activity_levels[i])
        goal_adj = GOAL_ADJUSTMENTS[goals[i]]
        
        # Add some noise to simulate real-world variation
        noise = np.random.normal(0, 50)
        final_calories = max(1200, tdee + goal_adj + noise)  # Minimum 1200 calories
        target_calories.append(final_calories)
    
    # Encode features
    X = np.column_stack([
        ages,
        heights,
        weights,
        [GENDER_ENCODING[g] for g in genders],
        [ACTIVITY_ENCODING[a] for a in activity_levels],
        [GOAL_ENCODING[g] for g in goals]
    ])
    
    y = np.array(target_calories)
    
    return X, y


def train_model() -> RandomForestRegressor:
    """
    Train and save the Random Forest model.
    
    Returns:
        Trained model
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required. Install with: pip install scikit-learn")
    
    print("Generating training data...")
    X, y = generate_training_data()
    
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {MODEL_PATH}")
    return model


def load_model() -> Optional[RandomForestRegressor]:
    """
    Load the trained model from disk.
    
    Returns:
        Trained model or None if not found
    """
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None


def get_model() -> RandomForestRegressor:
    """
    Get the model, training if necessary.
    
    Returns:
        Trained model
    """
    model = load_model()
    if model is None:
        model = train_model()
    return model


def predict_daily_calories(user_data: Dict) -> Dict:
    """
    Predict daily calorie needs for a user.
    
    Args:
        user_data: Dictionary containing:
            - age: int
            - height: float (cm)
            - weight: float (kg)
            - gender: str ('male' or 'female')
            - activity_level: str (one of ACTIVITY_MULTIPLIERS keys)
            - goal: str ('weight_loss', 'maintain', or 'muscle_gain')
    
    Returns:
        Dictionary with:
            - daily_calories: int (predicted total)
            - bmr: int (Basal Metabolic Rate)
            - tdee: int (Total Daily Energy Expenditure)
            - goal_adjustment: int (calories added/subtracted for goal)
            - activity_multiplier: float
            - confidence: str ('high', 'medium', 'low')
    """
    # Extract user data with defaults (handling None values from DB)
    age = user_data.get('age') or 30
    height = user_data.get('height') or 170
    weight = user_data.get('weight') or 70
    gender = str(user_data.get('gender') or 'male').lower()
    activity_level = user_data.get('activity_level') or 'moderately_active'
    goal = user_data.get('goal') or 'maintain'
    
    # Validate inputs
    if age is None or height is None or weight is None:
        return {
            'daily_calories': None,
            'error': 'Missing required fields (age, height, or weight)',
            'confidence': 'none'
        }
    
    # Convert to numbers if strings
    try:
        age = int(age)
        height = float(height)
        weight = float(weight)
    except (ValueError, TypeError):
        return {
            'daily_calories': None,
            'error': 'Invalid numeric values',
            'confidence': 'none'
        }
    
    # Calculate formula-based values for transparency
    bmr = calculate_bmr(age, height, weight, gender)
    activity_multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    tdee = bmr * activity_multiplier
    goal_adjustment = GOAL_ADJUSTMENTS.get(goal, 0)
    
    # Use ML model for final prediction
    if SKLEARN_AVAILABLE:
        try:
            model = get_model()
            
            # Encode features
            gender_enc = GENDER_ENCODING.get(gender, 0)
            activity_enc = ACTIVITY_ENCODING.get(activity_level, 2)
            goal_enc = GOAL_ENCODING.get(goal, 1)
            
            X = np.array([[age, height, weight, gender_enc, activity_enc, goal_enc]])
            ml_prediction = model.predict(X)[0]
            
            # Blend ML prediction with formula (70% ML, 30% formula for robustness)
            formula_prediction = tdee + goal_adjustment
            daily_calories = int(0.7 * ml_prediction + 0.3 * formula_prediction)
            confidence = 'high'
        except Exception as e:
            print(f"ML prediction failed, using formula: {e}")
            daily_calories = int(tdee + goal_adjustment)
            confidence = 'medium'
    else:
        # Fallback to formula-only
        daily_calories = int(tdee + goal_adjustment)
        confidence = 'medium'
    
    # Ensure minimum calories
    daily_calories = max(1200, daily_calories)
    
    return {
        'daily_calories': daily_calories,
        'bmr': int(bmr),
        'tdee': int(tdee),
        'goal_adjustment': goal_adjustment,
        'activity_multiplier': activity_multiplier,
        'confidence': confidence,
        'breakdown': {
            'base_metabolism': int(bmr),
            'activity_bonus': int(tdee - bmr),
            'goal_adjustment': goal_adjustment
        }
    }


# Pre-load/train model on import
_model = None
def _init_model():
    global _model
    if SKLEARN_AVAILABLE and _model is None:
        try:
            _model = get_model()
        except Exception as e:
            print(f"Model initialization warning: {e}")

# Initialize on import (non-blocking)
try:
    _init_model()
except:
    pass


if __name__ == "__main__":
    # Test the predictor
    print("=" * 50)
    print("Testing Calorie Predictor")
    print("=" * 50)
    
    # Test case 1: Moderately active male
    test_user_1 = {
        'age': 25,
        'height': 175,
        'weight': 70,
        'gender': 'male',
        'activity_level': 'moderately_active',
        'goal': 'maintain'
    }
    result = predict_daily_calories(test_user_1)
    print(f"\nTest 1 - Active male (25yo, 175cm, 70kg):")
    print(f"  Daily Calories: {result['daily_calories']}")
    print(f"  BMR: {result['bmr']}")
    print(f"  TDEE: {result['tdee']}")
    print(f"  Confidence: {result['confidence']}")
    
    # Test case 2: Sedentary female wanting weight loss
    test_user_2 = {
        'age': 35,
        'height': 165,
        'weight': 65,
        'gender': 'female',
        'activity_level': 'sedentary',
        'goal': 'weight_loss'
    }
    result = predict_daily_calories(test_user_2)
    print(f"\nTest 2 - Sedentary female weight loss (35yo, 165cm, 65kg):")
    print(f"  Daily Calories: {result['daily_calories']}")
    print(f"  BMR: {result['bmr']}")
    print(f"  Goal Adjustment: {result['goal_adjustment']}")
    
    # Test case 3: Very active male wanting muscle gain
    test_user_3 = {
        'age': 22,
        'height': 180,
        'weight': 80,
        'gender': 'male',
        'activity_level': 'very_active',
        'goal': 'muscle_gain'
    }
    result = predict_daily_calories(test_user_3)
    print(f"\nTest 3 - Very active male muscle gain (22yo, 180cm, 80kg):")
    print(f"  Daily Calories: {result['daily_calories']}")
    print(f"  Breakdown: {result['breakdown']}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")
