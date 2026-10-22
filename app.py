"""
app.py
======
Flask REST API for the AI-Powered Student Performance Analytics Platform.

Endpoints:
1. POST /predict
   - Existing prediction endpoint with corrected label mapping (0: 'H', 1: 'L', 2: 'M')
     and comprehensive input range validation.
2. GET /api/evaluation
   - Exposes comprehensive evaluation metrics (accuracy, precision, recall, F1)
     for all 5 models, comparison rankings, 3x3 confusion matrices, test set size,
     and methodology documentation.
3. GET /api/dataset/summary
   - Summarizes dataset class distributions and empirical engagement means.
4. POST /api/agent/query
   - AI Analytics Agent interface supporting controlled tool calling, evidence
     grounding, latency tracking, and 'correlation != causation' guardrails.
5. GET /api/health
   - System and model health check.
"""

import os
import sys
import json
import time

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

# Dynamically import ml_engine, tools, and agent
try:
    import ml_engine
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location('ml_engine', os.path.join(_CURRENT_DIR, 'ml_engine.py'))
    ml_engine = importlib.util.module_from_spec(spec)
    sys.modules['ml_engine'] = ml_engine
    spec.loader.exec_module(ml_engine)

try:
    import agent as agent_module
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location('agent', os.path.join(_CURRENT_DIR, 'agent.py'))
    agent_module = importlib.util.module_from_spec(spec)
    sys.modules['agent'] = agent_module
    spec.loader.exec_module(agent_module)

try:
    import tools as tools_module
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location('tools', os.path.join(_CURRENT_DIR, 'tools.py'))
    tools_module = importlib.util.module_from_spec(spec)
    sys.modules['tools'] = tools_module
    spec.loader.exec_module(tools_module)

ModelRegistry = ml_engine.ModelRegistry
LABEL_MAP = ml_engine.LABEL_MAP
AIAnalyticsAgent = agent_module.AIAnalyticsAgent

# Initialize Registry and Agent
registry = ModelRegistry.get_instance(_CURRENT_DIR)
ai_agent = AIAnalyticsAgent()

# Flask initialization with fallback
try:
    from flask import Flask, request, jsonify, make_response
    from flask_cors import CORS
    has_flask = True
    app = Flask(__name__)
    CORS(app)
except ImportError:
    has_flask = False
    app = None


# Core business logic handlers (decoupled for direct testability and Flask routing)
def handle_predict(data):
    if not data or not isinstance(data, dict):
        return {"error": "Request payload must be a JSON object."}, 400

    try:
        result = registry.predict_student(data)
        return {
            "predictions": result["predictions"],
            "engagementScore": result["engagementScore"],
            "evaluation": result["evaluation"],
            "validatedInput": result["validatedInput"]
        }, 200
    except ValueError as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}, 500


def handle_evaluation():
    if registry.evaluation_data is None:
        registry.load_evaluation()

    if registry.evaluation_data is None:
        return {"error": "Evaluation data not found. Please run train.py first."}, 500

    return registry.evaluation_data, 200


def handle_dataset_summary():
    try:
        dist = tools_module.get_prediction_distribution()
        eng = tools_module.get_engagement_summary()
        return {
            "distribution": dist,
            "engagement": eng
        }, 200
    except Exception as e:
        return {"error": f"Failed retrieving dataset summary: {str(e)}"}, 500


def handle_agent_query(data):
    if not data or not isinstance(data, dict):
        return {"error": "Request payload must be a JSON object containing a 'question' field."}, 400

    question = data.get("question", "").strip()
    if not question:
        return {"error": "The 'question' field cannot be empty."}, 400

    features = data.get("features")
    try:
        response = ai_agent.answer_query(question, student_features=features)
        return response, 200
    except Exception as e:
        return {"error": f"Agent execution failed: {str(e)}"}, 500


def handle_health():
    eval_loaded = registry.evaluation_data is not None
    models_count = len(registry.models)
    return {
        "status": "healthy",
        "models_loaded": models_count,
        "evaluation_available": eval_loaded,
        "agent_ready": True,
        "label_mapping_corrected": True,
        "class_mapping": LABEL_MAP
    }, 200


# Register Flask routes if Flask is available
if has_flask:
    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True)
        res, status = handle_predict(data)
        return jsonify(res), status

    @app.route("/api/evaluation", methods=["GET"])
    def get_evaluation():
        res, status = handle_evaluation()
        return jsonify(res), status

    @app.route("/api/dataset/summary", methods=["GET"])
    def get_summary():
        res, status = handle_dataset_summary()
        return jsonify(res), status

    @app.route("/api/agent/query", methods=["POST"])
    def agent_query():
        data = request.get_json(silent=True)
        res, status = handle_agent_query(data)
        return jsonify(res), status

    @app.route("/api/health", methods=["GET"])
    @app.route("/health", methods=["GET"])
    def health():
        res, status = handle_health()
        return jsonify(res), status


if __name__ == "__main__":
    if has_flask:
        print("=" * 60)
        print("Starting Student Performance Analytics Backend Server")
        print("Target-Label Mapping: 0 -> 'H', 1 -> 'L', 2 -> 'M' (Corrected)")
        print("Endpoints: /predict, /api/evaluation, /api/dataset/summary, /api/agent/query, /api/health")
        print("=" * 60)
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        print("Flask not installed in this environment. Core handlers are ready for direct WSGI/test execution.")
