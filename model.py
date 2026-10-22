"""
model.py
========
Loads trained models and provides prediction inference for student performance.

Corrected Label Mapping:
Target 'Class' is encoded alphabetically by LabelEncoder:
  0 -> 'H' (High)
  1 -> 'L' (Low)
  2 -> 'M' (Medium)
Previous bug where 1 was mapped to 'M' and 2 to 'L' has been corrected.
"""

import os
import sys
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

try:
    import ml_engine
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location('ml_engine', os.path.join(_CURRENT_DIR, 'ml_engine.py'))
    ml_engine = importlib.util.module_from_spec(spec)
    sys.modules['ml_engine'] = ml_engine
    spec.loader.exec_module(ml_engine)

ModelRegistry = ml_engine.ModelRegistry
validate_student_input = ml_engine.validate_student_input

# Corrected label map matching scikit-learn LabelEncoder on ['H', 'L', 'M']
label_map = {
    0: "H",  # High (Grade >= 90)
    1: "L",  # Low (Grade < 70)
    2: "M"   # Medium (Grade 70-89)
}

registry = ModelRegistry.get_instance(_CURRENT_DIR)


def predict_all_models(features):
    """
    Legacy-compatible prediction function accepting array-like [raisedHands, visitedResources, discussion, absences].
    Uses the corrected label map and validates ranges.
    """
    if isinstance(features, (list, tuple, np.ndarray)):
        if len(features) != 4:
            raise ValueError(f"Expected 4 features, got {len(features)}.")
        data = {
            "raisedHands": features[0],
            "visitedResources": features[1],
            "discussion": features[2],
            "absences": features[3]
        }
    elif isinstance(features, dict):
        data = features
    else:
        raise TypeError("Features must be a list/tuple of 4 numbers or a dict.")

    res = registry.predict_student(data)
    return res["predictions"]


def predict_student(data):
    """
    Full prediction pipeline returning model predictions, engagement score,
    evaluation tier, and validated input metadata.
    """
    return registry.predict_student(data)
