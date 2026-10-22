# Student Performance Analytics Platform

An end-to-end machine learning application that predicts student performance, evaluates five classification models on a held-out test set, and presents data-driven insights through an interactive React dashboard. The platform features an LLM-powered analytics agent that uses controlled Python tool calling to retrieve verified evaluation metrics and answer user questions in natural language.

---

## Overview

Understanding student academic trajectory requires more than a single prediction. It involves validating model reliability, exploring behavioral engagement patterns, and communicating empirical findings clearly.

This project combines a Python machine learning backend with an interactive React dashboard to support real-time predictions, multi-model evaluation, and grounded AI analytics.

---

## Key Features

* **Student Performance Prediction**: Classifies student outcomes into **High (H)** , **Medium (M)** , and **Low (L)** categories using classroom participation and behavioral indicators.
* **Five Supervised ML Models**: Implements and benchmarks Decision Tree, Random Forest, Logistic Regression, Perceptron, and Multi-Layer Perceptron (MLP) Classifier.
* **Target Encoding Correctness**: Establishes a verified single source of truth for label mapping, eliminating index-inversion bugs between training and inference.
* **Proper Model Evaluation**: Evaluates all models on a strict 70/30 train/test holdout split (144 test samples) rather than training data or ad-hoc inputs.
* **Controlled AI Analytics Agent**: Enables natural-language queries backed by five deterministic Python tools. The agent cites verified metrics and enforces a strict *"correlation does not prove causation"* disclaimer on behavioral data.
* **Interactive React Dashboard**: Built with React and Recharts, offering visual prediction consensus, dynamic model comparison charts, and a times confusion matrix inspector.
* **REST API**: Flask backend providing endpoints for predictions, evaluation metadata, dataset summaries, and AI agent queries.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | React 19, Axios, Recharts, Custom CSS |
| **Backend** | Python 3.9+, Flask, Flask-CORS |
| **Machine Learning** | Scikit-learn, NumPy, Pandas, SciPy |
| **Testing & Reliability** | Python `unittest` (29 automated test cases) |
| **AI Analytics** | Controlled Python Tool Calling (Query Router, Latency Tracker, Privacy Logger) |

