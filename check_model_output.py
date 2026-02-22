import tensorflow as tf
import os

ENSEMBLE_DIR = r"C:\food_calorie_project\ensemble_models"
MODEL_PATHS = [
    os.path.join(ENSEMBLE_DIR, "ensemble_model1_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "ensemble_model3_final_v2.keras"),
    os.path.join(ENSEMBLE_DIR, "model2_v2.keras")
]

class PatchedDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(*args, **kwargs)

for p in MODEL_PATHS:
    if os.path.exists(p):
        print(f"Loading model: {p}")
        try:
            m = tf.keras.models.load_model(p, custom_objects={'DepthwiseConv2D': PatchedDepthwiseConv2D})
            print(f"Output shape: {m.output_shape}")
        except Exception as e:
            print(f"Error loading {p}: {e}")
    else:
        print(f"Model not found at {p}")
