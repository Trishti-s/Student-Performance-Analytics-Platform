# AI-Powered Student Performance Analytics Platform

> **Portfolio Summary**: Built an ML analytics platform with five classifiers and an LLM-powered tool-calling agent that retrieves evaluation metrics, compares model performance, and generates evidence-backed natural-language insights through a React dashboard.

---

## 🌟 System Architecture Overview

```
                      +-------------------------------------------------+
                      |              React 19 Dashboard                 |
                      |  - 🎯 Student Predictor & Activity Profiles    |
                      |  - 🤖 AI Analytics Agent & Evidence Inspector  |
                      |  - 📊 Model Benchmarks & Confusion Matrix View  |
                      +-----------------------+-------------------------+
                                              |  HTTP / REST (Axios)
                                              v
+---------------------------------------------------------------------------------------+
|                               Flask Backend REST API                                  |
|  POST /predict  |  GET /api/evaluation  |  GET /api/dataset/summary  |  POST /api/agent/query |
+-------------------------------------+-------------------------------------------------+
                                      |
       +------------------------------+-------------------------------+
       |                                                              |
       v                                                              v
+----------------------------------+        +-------------------------------------------+
|       Controlled AI Agent        |        |               Core ML Engine              |
|  - Tool Calling Query Router     |        |  5 Supervised Classifiers:                |
|  - Latency Tracker (ms)          |        |  1. Decision Tree (CART Gini)             |
|  - Privacy-Safe Audit Logger     |        |  2. Random Forest (Ensemble)              |
|  - Correlation != Causation Rule |        |  3. Logistic Regression (Softmax L2)      |
|  - Grounded Evidence Synthesizer |        |  4. Perceptron (One-vs-Rest)              |
+------------------+---------------+        |  5. MLP Neural Network (Sigmoid/Softmax)  |
                   |                        +---------------------+---------------------+
                   | Calls Bounded Tools                          |
                   v                                              v
+---------------------------------------------------------------------------------------+
|                                 5 Controlled Tools                                    |
|  1. compare_models(metric)            |  4. get_model_metrics(model_name)             |
|  2. get_prediction_distribution()     |  5. analyze_student_input(features)          |
|  3. get_engagement_summary()           |                                               |
+-------------------------------------+-------------------------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------------------+
|                 Evaluation Methodology & Single Source of Truth                       |
|  - Strict 70/30 Holdout Split (336 Train / 144 Test, random_state=42)                 |
|  - Single Source of Truth: model_evaluation.json & AI-Data.csv (480 Records)          |
+---------------------------------------------------------------------------------------+
```

---

## 🛠️ Key Architectural Upgrades & Fixes

### 1. Essential ML Correctness & Label Encoding Fix
- **Identified Critical Bug**: The original `train.py` applied `LabelEncoder` on the target `Class` (`['H', 'M', 'L']`). Alphabetical sorting in `LabelEncoder` assigns:
  $$\text{Index } 0 \to \text{'H'},\quad \text{Index } 1 \to \text{'L'},\quad \text{Index } 2 \to \text{'M'}$$
  However, `app.py` previously defined `label_map = {0: "H", 1: "M", 2: "L"}`. This caused an **inverted mapping bug** where every Medium student was misclassified as Low, and every Low student was misclassified as Medium.
- **Engineered Resolution**:
  - Established a unified, single source of truth across training, inference, and the agent:
    `LABEL_MAP = {0: "H", 1: "L", 2: "M"}`
  - Standardized `StudentAbsenceDays` binary encoding: `'Above-7' -> 0`, `'Under-7' -> 1`.
  - Added strict input validation for all features (`raisedHands`, `visitedResources`, `discussion`, `absences`) enforcing $0 \le x \le 100$ and returning informative HTTP 400 responses on invalid inputs.

### 2. Standalone Model Evaluation Dashboard & API
- **Test-Set Methodology**: Models are evaluated strictly on a held-out test set of **144 samples** (30% split, fixed seed $42$ on 480 total records). Metrics are **never** computed from ad-hoc user inputs.
- **Exposed Endpoints**:
  - `GET /api/evaluation`: Exposes accuracy, weighted/macro precision, recall, F1, 3x3 confusion matrices, test-set size, and methodology metadata.
  - `GET /api/dataset/summary`: Exposes overall class distributions and empirical engagement means.
  - Pre-calculated metrics are saved to `backend/model_evaluation.json` during training.

### 3. Controlled LLM Tool Calling Agent
- The AI Analytics Agent does not hallucinate answers; it accesses five deterministic Python tools:
  1. `compare_models(metric)`: Compares all 5 classifiers on test-set metrics (`accuracy`, `precision`, `recall`, `f1_score`) with ranking and score spread.
  2. `get_prediction_distribution()`: Summarizes H, M, and L distributions across the 480-sample dataset and 144-sample test set.
  3. `get_engagement_summary()`: Summarizes actual behavioral statistics across classes.
  4. `get_model_metrics(model_name)`: Retrieves detailed metrics, per-class breakdown, and confusion matrix for a specific classifier.
  5. `analyze_student_input(features)`: Runs all 5 models on student inputs, calculates the engagement score, and produces feature attribution.
- **Correlation $\neq$ Causation Guardrail**: Whenever engagement metrics are summarized, the agent explicitly enforces an analytical disclaimer stating that behavioral correlations do not prove direct causation.
- **Privacy & Latency Tracking**: Records execution time in milliseconds and logs tool calls without storing student names or personal identifiers (PII).

### 4. Upgraded React Dashboard
- Built with React 19, Recharts, and Axios.
- **Student Predictor Tab**: Features bounded numeric inputs, one-click at-risk/high-achiever profile loaders, an engagement index progress bar, 5 model prediction cards, a model agreement chart, and a quick "Explain with AI Agent" link.
- **AI Analytics Agent Tab**: Interactive query input with one-click suggestion chips, formatted evidence-backed answers, millisecond latency badge, collapsible raw tool evidence inspector, model comparison bar chart with metric selector (Accuracy, Precision, Recall, F1), and a 3x3 Confusion Matrix Inspector with class support and per-class metrics.

### 5. Automated Reliability & Verification Suite
- Comprehensive 29-test suite implemented with Python's built-in `unittest` framework:
  - `test_ml_correctness.py`: Label mapping, input validation, range rejection, profile prediction.
  - `test_evaluation_metrics.py`: Metrics integrity, confusion matrix dimensions, 144-sample test split.
  - `test_ai_agent.py`: Tool selection accuracy, metric reporting, out-of-scope query rejection, latency tracking, PII protection, correlation disclaimer enforcement.
  - `test_api_endpoints.py`: REST endpoint status codes, payload handling, healthchecks.

---

## 📊 Model Evaluation Benchmark Results (Held-Out Test Set: 144 Samples)

| Classifier | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Best Suited For |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **MLP Classifier** | **75.00%** | **75.13%** | **75.00%** | **74.82%** | Non-linear interaction modeling |
| **Random Forest** | **72.22%** | **72.78%** | **72.22%** | **72.39%** | Balanced multi-class generalization |
| **Logistic Regression** | **68.75%** | **70.42%** | **68.75%** | **67.64%** | Fast probabilistic baselines |
| **Decision Tree** | **68.06%** | **70.17%** | **68.06%** | **67.36%** | Interpretable decision rule branches |
| **Perceptron** | **64.58%** | **63.03%** | **64.58%** | **63.17%** | Linear decision boundary baseline |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Node.js 16+ & npm

### 1. Backend Setup & Test Execution

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run the full reliability test suite (29 tests)
python3 run_tests.py

# (Optional) Re-train models and generate model_evaluation.json
python3 train.py

# Start the Flask API server (runs on http://127.0.0.1:5000)
python3 app.py
```

### 2. Frontend Setup & Launch

```bash
cd frontend

# Install dependencies
npm install

# Start development server (runs on http://localhost:3000)
npm start
```

---

## 🔍 Sample AI Agent Queries

- *“Which model performed best on the test set?”*
- *“Compare precision and recall across all models.”*
- *“What is the distribution of H, M and L in the dataset?”*
- *“Explain the predictions for these input values.”*
- *“Summarize student engagement patterns in the dataset.”*
