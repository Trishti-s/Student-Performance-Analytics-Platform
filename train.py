# train.py
# Training and Evaluation Script for Student Performance Prediction.
import os
import sys
import json
import pandas as pd
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

def train_and_evaluate():
    print('=' * 60)
    print('Starting Student Performance Model Training & Evaluation')
    print('=' * 60)

    csv_path = os.path.join(_CURRENT_DIR, 'AI-Data.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Dataset not found at {csv_path}')

    registry = ModelRegistry.get_instance(_CURRENT_DIR)
    eval_results = registry.train_and_save_all(csv_path)

    print()
    print('--- EVALUATION SUMMARY (Held-out Test Set: 144 samples / 30%) ---')
    for comp in eval_results['comparison']:
        m_name = comp['model']
        acc = comp['accuracy'] * 100
        prec = comp['precision'] * 100
        rec = comp['recall'] * 100
        f1 = comp['f1_score'] * 100
        print(f'{m_name:22} | Acc: {acc:6.2f}% | Precision: {prec:6.2f}% | Recall: {rec:6.2f}% | F1: {f1:6.2f}%')

    print()
    print('--- DETAILED CLASSIFICATION REPORTS ---')
    for model_name, metrics in eval_results['models'].items():
        print()
        print(f'>>> {model_name} <<<')
        print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
        print(f"Weighted Precision: {metrics['precision_weighted'] * 100:.2f}%")
        print(f"Weighted Recall: {metrics['recall_weighted'] * 100:.2f}%")
        print(f"Weighted F1: {metrics['f1_weighted'] * 100:.2f}%")
        print('Per-Class Breakdown:')
        for cls_name, cls_m in metrics['per_class'].items():
            print(f"  Class {cls_name}: Precision={cls_m['precision']*100:.1f}%, Recall={cls_m['recall']*100:.1f}%, F1={cls_m['f1_score']*100:.1f}%, Support={cls_m['support']}")
        print('Confusion Matrix (Rows: Ground Truth [H, L, M], Cols: Predicted [H, L, M]):')
        for row in metrics['confusion_matrix']:
            print(' ', row)

    print()
    print('=' * 60)
    print(f'All models trained. Evaluation saved to {os.path.join(_CURRENT_DIR, "model_evaluation.json")}.')
    print('=' * 60)
    return eval_results

if __name__ == '__main__':
    train_and_evaluate()
