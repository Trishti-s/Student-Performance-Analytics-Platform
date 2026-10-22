"""
tools.py
========
Controlled Python Tools for the AI Analytics Agent.

Provides strictly bounded, deterministic, evidence-grounded tools:
1. compare_models(metric='accuracy')
   - Compares all 5 classifiers across a specified metric on the test set.
2. get_prediction_distribution()
   - Summarizes class distribution (H/M/L) in both the full dataset and the test set.
3. get_engagement_summary()
   - Summarizes behavioral activity metrics across classes with strict correlation != causation disclaimers.
4. get_model_metrics(model_name)
   - Retrieves full evaluation metrics, confusion matrix, and per-class breakdown for a given model.
5. analyze_student_input(features)
   - Runs model predictions on user-provided features, calculates engagement score, and produces feature attribution.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Dynamic loader helper to ensure reliable import on 9p filesystems
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
LABEL_MAP = ml_engine.LABEL_MAP
CLASS_DESCRIPTIONS = ml_engine.CLASS_DESCRIPTIONS


def _get_evaluation():
    registry = ModelRegistry.get_instance(_CURRENT_DIR)
    if registry.evaluation_data is None:
        registry.load_evaluation()
    return registry.evaluation_data


def _get_dataset():
    csv_path = os.path.join(_CURRENT_DIR, 'AI-Data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Dataset not found at {csv_path}')
    return pd.read_csv(csv_path)


def compare_models(metric='accuracy'):
    """
    Compare all 5 trained models on the held-out test set according to a chosen metric.
    Supported metrics: 'accuracy', 'precision', 'recall', 'f1_score'.
    """
    metric_clean = str(metric).strip().lower().replace(' ', '_').replace('-', '_')
    metric_key_map = {
        'accuracy': 'accuracy',
        'acc': 'accuracy',
        'precision': 'precision_weighted',
        'precision_weighted': 'precision_weighted',
        'precision_macro': 'precision_macro',
        'recall': 'recall_weighted',
        'recall_weighted': 'recall_weighted',
        'recall_macro': 'recall_macro',
        'f1': 'f1_weighted',
        'f1_score': 'f1_weighted',
        'f1_weighted': 'f1_weighted',
        'f1_macro': 'f1_macro'
    }

    if metric_clean not in metric_key_map:
        valid_options = ['accuracy', 'precision', 'recall', 'f1_score']
        return {
            'error': f"Unsupported metric '{metric}'. Valid choices are: {', '.join(valid_options)}.",
            'supported_metrics': valid_options
        }

    target_key = metric_key_map[metric_clean]
    eval_data = _get_evaluation()
    models = eval_data.get('models', {})

    rankings = []
    for model_name, metrics in models.items():
        score = metrics.get(target_key, metrics.get('accuracy', 0.0))
        rankings.append({
            'model': model_name,
            'metric': metric_clean,
            'score': round(float(score), 4),
            'accuracy': round(float(metrics.get('accuracy', 0.0)), 4),
            'f1_weighted': round(float(metrics.get('f1_weighted', 0.0)), 4),
            'precision_weighted': round(float(metrics.get('precision_weighted', 0.0)), 4),
            'recall_weighted': round(float(metrics.get('recall_weighted', 0.0)), 4)
        })

    rankings.sort(key=lambda x: x['score'], reverse=True)
    for idx, item in enumerate(rankings):
        item['rank'] = idx + 1

    best = rankings[0]
    worst = rankings[-1]
    spread = round(best['score'] - worst['score'], 4)

    return {
        'metric_evaluated': metric_clean,
        'metric_type': 'Weighted' if 'weighted' in target_key else ('Macro' if 'macro' in target_key else 'Overall'),
        'test_set_size': eval_data.get('metadata', {}).get('test_size', 144),
        'best_model': {
            'model': best['model'],
            'score': best['score']
        },
        'lowest_model': {
            'model': worst['model'],
            'score': worst['score']
        },
        'score_spread': spread,
        'rankings': rankings,
        'methodology_summary': eval_data.get('metadata', {}).get('evaluation_methodology', '')
    }


def get_prediction_distribution():
    """
    Summarizes the actual distribution of High (H), Medium (M), and Low (L) performance
    labels across both the full 480-student dataset and the 144-sample test split.
    """
    df = _get_dataset()
    total = len(df)
    counts = df['Class'].value_counts().to_dict()

    eval_data = _get_evaluation()
    test_counts = eval_data.get('metadata', {}).get('test_class_distribution', {})
    test_total = eval_data.get('metadata', {}).get('test_size', 144)

    dataset_summary = []
    for cls in ['H', 'M', 'L']:
        cnt = int(counts.get(cls, 0))
        pct = round((cnt / total) * 100, 2)
        test_cnt = int(test_counts.get(cls, 0))
        test_pct = round((test_cnt / test_total) * 100, 2) if test_total > 0 else 0.0
        dataset_summary.append({
            'class': cls,
            'description': CLASS_DESCRIPTIONS.get(cls, ''),
            'dataset_count': cnt,
            'dataset_percentage': pct,
            'test_count': test_cnt,
            'test_percentage': test_pct
        })

    return {
        'total_dataset_records': total,
        'test_set_records': test_total,
        'classes': dataset_summary,
        'dominant_class': 'M (Medium-performing)',
        'observation': (
            'The dataset displays a class distribution where Medium (M) students form the plurality (approx 44%), '
            'followed by High (H) at approx 29.6% and Low (L) at approx 26.5%. Because of this slight imbalance, '
            'weighted and macro F1 scores provide a more dependable measure of model reliability than accuracy alone.'
        )
    }


def get_engagement_summary():
    """
    Summarizes empirical engagement metrics (raised hands, visited resources, discussions, absences)
    grouped by student performance class (H, M, L).
    Includes explicit analytical guardrails distinguishing correlation from causation.
    """
    df = _get_dataset()
    features = ['raisedhands', 'VisITedResources', 'Discussion']

    overall_stats = {}
    for f in features:
        overall_stats[f] = {
            'mean': round(float(df[f].mean()), 2),
            'median': round(float(df[f].median()), 2),
            'min': int(df[f].min()),
            'max': int(df[f].max())
        }

    grouped = df.groupby('Class')
    by_class = {}
    for cls in ['H', 'M', 'L']:
        sub = df[df['Class'] == cls]
        absence_counts = sub['StudentAbsenceDays'].value_counts().to_dict()
        under_7 = int(absence_counts.get('Under-7', 0))
        above_7 = int(absence_counts.get('Above-7', 0))
        sub_total = len(sub)

        by_class[cls] = {
            'description': CLASS_DESCRIPTIONS.get(cls, ''),
            'count': sub_total,
            'avg_raised_hands': round(float(sub['raisedhands'].mean()), 2),
            'avg_visited_resources': round(float(sub['VisITedResources'].mean()), 2),
            'avg_discussion': round(float(sub['Discussion'].mean()), 2),
            'absence_breakdown': {
                'under_7_days': under_7,
                'under_7_pct': round((under_7 / sub_total) * 100, 2) if sub_total > 0 else 0.0,
                'above_7_days': above_7,
                'above_7_pct': round((above_7 / sub_total) * 100, 2) if sub_total > 0 else 0.0
            }
        }

    return {
        'total_students_analyzed': len(df),
        'overall_activity_means': overall_stats,
        'class_breakdown': by_class,
        'key_empirical_findings': [
            'High-performing students (H) averaged 70.3 raised hands and 78.7 resource visits, with 97.2% absent under 7 days.',
            'Medium-performing students (M) averaged 48.9 raised hands and 60.6 resource visits, with 66.4% absent under 7 days.',
            'Low-performing students (L) averaged 16.9 raised hands and 18.3 resource visits, with 91.3% absent above 7 days.'
        ],
        'scientific_guardrail': (
            'CRITICAL ANALYTICAL DISTINCTION: While student engagement metrics (attendance, raised hands, '
            'and resource views) exhibit strong empirical correlation with academic outcomes, correlation does NOT '
            'prove causation. High resource visits may be driven by prior motivation, difficulty of the course material, '
            'or parental encouragement rather than acting as an isolated causal driver of higher marks.'
        )
    }


def get_model_metrics(model_name):
    """
    Returns detailed performance metrics, per-class evaluation, and the confusion matrix
    for a specific named model from the test set evaluation.
    """
    clean_name = str(model_name).strip().lower()
    name_alias_map = {
        'decision tree': 'Decision Tree',
        'decision_tree': 'Decision Tree',
        'dt': 'Decision Tree',
        'tree': 'Decision Tree',
        'random forest': 'Random Forest',
        'random_forest': 'Random Forest',
        'rf': 'Random Forest',
        'forest': 'Random Forest',
        'logistic regression': 'Logistic Regression',
        'logistic_regression': 'Logistic Regression',
        'logistic': 'Logistic Regression',
        'lr': 'Logistic Regression',
        'perceptron': 'Perceptron',
        'perc': 'Perceptron',
        'p': 'Perceptron',
        'mlp': 'MLP Classifier',
        'mlp classifier': 'MLP Classifier',
        'mlp_classifier': 'MLP Classifier',
        'neural network': 'MLP Classifier',
        'neural_network': 'MLP Classifier',
        'nn': 'MLP Classifier'
    }

    resolved_name = name_alias_map.get(clean_name)
    eval_data = _get_evaluation()
    available_models = list(eval_data.get('models', {}).keys())

    if not resolved_name or resolved_name not in eval_data.get('models', {}):
        return {
            'error': f"Model '{model_name}' not found. Available models are: {', '.join(available_models)}.",
            'available_models': available_models
        }

    metrics = eval_data['models'][resolved_name]
    return {
        'model_name': resolved_name,
        'overall_accuracy': metrics['accuracy'],
        'precision_weighted': metrics['precision_weighted'],
        'recall_weighted': metrics['recall_weighted'],
        'f1_weighted': metrics['f1_weighted'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
        'per_class_breakdown': metrics['per_class'],
        'confusion_matrix': {
            'labels': ['H (High)', 'L (Low)', 'M (Medium)'],
            'matrix': metrics['confusion_matrix'],
            'explanation': 'Rows represent actual ground-truth classes; columns represent predicted classes.'
        },
        'test_set_size': eval_data.get('metadata', {}).get('test_size', 144),
        'methodology': eval_data.get('metadata', {}).get('evaluation_methodology', '')
    }


def analyze_student_input(features):
    """
    Runs the multi-model prediction pipeline on a set of student features and provides
    detailed evidence and feature attribution.
    Expected features: {'raisedHands': 0-100, 'visitedResources': 0-100, 'discussion': 0-100, 'absences': 0-100 or 'Under-7'/'Above-7'}
    """
    registry = ModelRegistry.get_instance(_CURRENT_DIR)
    try:
        result = registry.predict_student(features)
    except ValueError as e:
        return {'error': str(e)}

    predictions = result['predictions']
    counts = {}
    for mod, pred in predictions.items():
        counts[pred] = counts.get(pred, 0) + 1

    consensus_label = max(counts.items(), key=lambda x: x[1])[0]
    consensus_agreement = counts[consensus_label]
    consensus_pct = round((consensus_agreement / len(predictions)) * 100, 1)

    # Attribution comparison against dataset averages
    val = result['validatedInput']
    attribution = []
    if val['raisedHands'] >= 60:
        attribution.append(f"Raised hands ({val['raisedHands']}) is well above average (46.8), aligning strongly with High-performance (H) profiles.")
    elif val['raisedHands'] <= 25:
        attribution.append(f"Raised hands ({val['raisedHands']}) is well below average (46.8), aligning with Low-performance (L) profiles.")

    if val['visitedResources'] >= 70:
        attribution.append(f"Visited resources ({val['visitedResources']}) is high (avg 54.8), typical of High-performance (H) students.")
    elif val['visitedResources'] <= 30:
        attribution.append(f"Visited resources ({val['visitedResources']}) is low (avg 54.8), typical of Low-performance (L) students.")

    if val['absenceCategory'] == 'Under-7':
        attribution.append(f"Attendance is strong ({val['absences']} absences, Under-7 days). In the dataset, 97% of High students had < 7 absences.")
    else:
        attribution.append(f"Absence count ({val['absences']} absences, Above-7 days) is elevated. In the dataset, 91% of Low students had >= 7 absences.")

    return {
        'validated_features': val,
        'predictions_by_model': predictions,
        'consensus_prediction': consensus_label,
        'consensus_description': CLASS_DESCRIPTIONS.get(consensus_label, ''),
        'model_agreement': f"{consensus_agreement} out of 5 models ({consensus_pct}% consensus)",
        'engagement_score': result['engagementScore'],
        'activity_evaluation': result['evaluation'],
        'feature_attribution_notes': attribution,
        'disclaimer': (
            'These predictions are model-generated classifications based on historical educational patterns. '
            'They do not constitute deterministic judgments of individual student capability.'
        )
    }


# Tool Registry Map for the Agent
CONTROLLED_TOOLS = {
    'compare_models': compare_models,
    'get_prediction_distribution': get_prediction_distribution,
    'get_engagement_summary': get_engagement_summary,
    'get_model_metrics': get_model_metrics,
    'analyze_student_input': analyze_student_input
}
