"""
ml_engine.py
============
Core Machine Learning Engine for the Student Performance Analytics Platform.

Fixes & Key Architectural Features:
1. Label Encoding Correctness:
   - Target 'Class' has 3 values: 'H', 'L', 'M'.
   - Alphabetical sorting yields: 0 -> 'H', 1 -> 'L', 2 -> 'M'.
   - Fixed the critical bug where 'app.py' had {0: 'H', 1: 'M', 2: 'L'},
     which erroneously swapped Medium ('M') and Low ('L') predictions.
2. Binary Feature Encoding:
   - 'StudentAbsenceDays': 'Above-7' -> 0, 'Under-7' -> 1.
   - Accepts both categorical ('Under-7'/'Above-7') and numerical absences (>=7 -> 0, <7 -> 1).
3. 5 Classifiers:
   - Decision Tree (CART Gini)
   - Random Forest (Bootstrap Aggregated Ensemble)
   - Perceptron (One-vs-Rest Linear Classifier)
   - Logistic Regression (Multinomial Softmax with L2 regularization)
   - MLP Classifier (Feedforward Neural Network with Logistic Activation)
4. Evaluation Methodology:
   - Evaluated strictly on a held-out 144-sample test set (30% split, random_state=42).
   - Generates comprehensive metrics (accuracy, macro/weighted precision, recall, F1,
     per-class metrics, confusion matrix, test-set size, methodology).
   - Persists evaluation results to 'model_evaluation.json'.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

LABEL_MAP = {0: 'H', 1: 'L', 2: 'M'}
INV_LABEL_MAP = {'H': 0, 'L': 1, 'M': 2}
CLASS_NAMES = ['H', 'L', 'M']
CLASS_DESCRIPTIONS = {
    'H': 'High-performing (Grade >= 90)',
    'M': 'Medium-performing (Grade 70-89)',
    'L': 'Low-performing (Grade < 70)'
}

FEATURE_NAMES = ['raisedhands', 'VisITedResources', 'Discussion', 'StudentAbsenceDays']


def validate_student_input(data):
    if not isinstance(data, dict):
        raise ValueError('Input data must be a JSON object / dictionary.')

    field_aliases = {
        'raisedHands': ['raisedHands', 'raisedhands', 'raised_hands'],
        'visitedResources': ['visitedResources', 'VisITedResources', 'visited_resources'],
        'discussion': ['discussion', 'Discussion'],
        'absences': ['absences', 'StudentAbsenceDays', 'student_absence_days']
    }

    resolved = {}
    for standard_name, aliases in field_aliases.items():
        found = False
        for alias in aliases:
            if alias in data and data[alias] is not None and data[alias] != '':
                resolved[standard_name] = data[alias]
                found = True
                break
        if not found:
            raise ValueError(f"Missing required field: '{standard_name}'.")

    try:
        rh = float(resolved['raisedHands'])
        if not (0 <= rh <= 100):
            raise ValueError()
        resolved['raisedHands'] = int(round(rh))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid 'raisedHands': must be an integer between 0 and 100, got {resolved['raisedHands']}.")

    try:
        vr = float(resolved['visitedResources'])
        if not (0 <= vr <= 100):
            raise ValueError()
        resolved['visitedResources'] = int(round(vr))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid 'visitedResources': must be an integer between 0 and 100, got {resolved['visitedResources']}.")

    try:
        disc = float(resolved['discussion'])
        if not (0 <= disc <= 100):
            raise ValueError()
        resolved['discussion'] = int(round(disc))
    except (ValueError, TypeError):
        raise ValueError(f"Invalid 'discussion': must be an integer between 0 and 100, got {resolved['discussion']}.")

    raw_abs = resolved['absences']
    if isinstance(raw_abs, str):
        raw_clean = raw_abs.strip().lower()
        if raw_clean in ['under-7', 'under 7', '<7', 'low']:
            absence_encoded = 1
            absence_days = 3
        elif raw_clean in ['above-7', 'above 7', '>=7', '>7', 'high']:
            absence_encoded = 0
            absence_days = 10
        else:
            try:
                num = float(raw_clean)
                if not (0 <= num <= 100):
                    raise ValueError()
                absence_days = int(round(num))
                absence_encoded = 1 if absence_days < 7 else 0
            except ValueError:
                raise ValueError(f"Invalid 'absences': must be a number between 0 and 100 or 'Under-7'/'Above-7', got '{raw_abs}'.")
    else:
        try:
            num = float(raw_abs)
            if not (0 <= num <= 100):
                raise ValueError()
            absence_days = int(round(num))
            if raw_abs in [0, 1] and isinstance(raw_abs, int):
                absence_encoded = int(raw_abs)
                absence_days = 3 if absence_encoded == 1 else 10
            else:
                absence_encoded = 1 if absence_days < 7 else 0
        except (ValueError, TypeError):
            raise ValueError(f"Invalid 'absences': must be a number between 0 and 100, got {raw_abs}.")

    resolved['absence_days'] = absence_days
    resolved['absence_encoded'] = absence_encoded
    return resolved


class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

    def to_dict(self):
        if self.is_leaf():
            return {'value': int(self.value)}
        return {
            'feature': int(self.feature),
            'threshold': float(self.threshold),
            'left': self.left.to_dict() if self.left else None,
            'right': self.right.to_dict() if self.right else None
        }

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        if 'value' in d and d['value'] is not None:
            return cls(value=d['value'])
        return cls(
            feature=d.get('feature'),
            threshold=d.get('threshold'),
            left=cls.from_dict(d.get('left')),
            right=cls.from_dict(d.get('right'))
        )


class DecisionTreeModel:
    def __init__(self, max_depth=5, min_samples_split=4, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_classes_ = 3

    def fit(self, X, y):
        self.rng = np.random.RandomState(self.random_state)
        self.n_classes_ = len(np.unique(y))
        self.root = self._grow_tree(X, y)
        return self

    def _gini(self, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / len(y)
        return 1.0 - float(np.sum(p ** 2))

    def _best_split(self, X, y):
        m, n = X.shape
        if m < self.min_samples_split:
            return None, None
        feat_indices = np.arange(n)
        if self.max_features is not None and self.max_features < n:
            feat_indices = self.rng.choice(n, self.max_features, replace=False)

        best_gain = 0.0
        best_feat, best_thresh = None, None
        current_gini = self._gini(y)

        for feat in feat_indices:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gain = current_gini - (
                    (len(y[left_mask]) / m) * self._gini(y[left_mask]) +
                    (len(y[right_mask]) / m) * self._gini(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _grow_tree(self, X, y, depth=0):
        m = len(y)
        counts = np.bincount(y, minlength=self.n_classes_)
        majority_class = int(np.argmax(counts))

        if depth >= self.max_depth or len(np.unique(y)) == 1 or m < self.min_samples_split:
            return DecisionTreeNode(value=majority_class)

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return DecisionTreeNode(value=majority_class)

        left_mask = X[:, feat] <= thresh
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return DecisionTreeNode(feature=feat, threshold=thresh, left=left, right=right)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def to_dict(self):
        return {
            'type': 'DecisionTreeModel',
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'n_classes_': self.n_classes_,
            'root': self.root.to_dict() if self.root else None
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(max_depth=d.get('max_depth', 5), min_samples_split=d.get('min_samples_split', 4))
        obj.n_classes_ = d.get('n_classes_', 3)
        obj.root = DecisionTreeNode.from_dict(d.get('root'))
        return obj


class RandomForestModel:
    def __init__(self, n_estimators=35, max_depth=5, min_samples_split=4, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.trees = []
        n_samples = len(X)
        max_feat = max(1, int(np.sqrt(X.shape[1])))
        for i in range(self.n_estimators):
            sample_idx = rng.choice(n_samples, n_samples, replace=True)
            tree = DecisionTreeModel(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                random_state=rng.randint(0, 100000)
            )
            tree.fit(X[sample_idx], y[sample_idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        tree_preds = np.array([t.predict(X) for t in self.trees])
        majority = [np.bincount(tree_preds[:, i], minlength=3).argmax() for i in range(len(X))]
        return np.array(majority)

    def to_dict(self):
        return {
            'type': 'RandomForestModel',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'trees': [t.to_dict() for t in self.trees]
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(n_estimators=d.get('n_estimators', 35), max_depth=d.get('max_depth', 5))
        obj.trees = [DecisionTreeModel.from_dict(t) for t in d.get('trees', [])]
        return obj


class LogisticRegressionModel:
    def __init__(self, C=1.0, random_state=42):
        self.C = C
        self.random_state = random_state
        self.weights = None
        self.mean = None
        self.std = None

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        X_norm = (X - self.mean) / self.std
        X_b = np.hstack([X_norm, np.ones((len(X), 1))])

        num_classes = 3
        num_features = X_b.shape[1]

        def loss_fn(w_flat):
            W = w_flat.reshape((num_features, num_classes))
            logits = X_b @ W
            probs = self._softmax(logits)
            eps = 1e-12
            nll = -np.mean(np.log(probs[np.arange(len(y)), y] + eps))
            reg = 0.5 * (1.0 / self.C) * np.sum(W[:-1, :] ** 2)
            return nll + reg

        init_w = np.zeros(num_features * num_classes)
        res = minimize(loss_fn, init_w, method='L-BFGS-B')
        self.weights = res.x.reshape((num_features, num_classes))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self.mean) / self.std
        X_b = np.hstack([X_norm, np.ones((len(X), 1))])
        probs = self._softmax(X_b @ self.weights)
        return np.argmax(probs, axis=1)

    def to_dict(self):
        return {
            'type': 'LogisticRegressionModel',
            'weights': self.weights.tolist() if self.weights is not None else None,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.weights = np.array(d['weights']) if d.get('weights') is not None else None
        obj.mean = np.array(d['mean']) if d.get('mean') is not None else None
        obj.std = np.array(d['std']) if d.get('std') is not None else None
        return obj


class PerceptronModel:
    def __init__(self, max_iter=200, eta0=0.005, random_state=42):
        self.max_iter = max_iter
        self.eta0 = eta0
        self.random_state = random_state
        self.weights = None
        self.biases = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        X_norm = (X - self.mean) / self.std

        n_samples, n_features = X_norm.shape
        n_classes = 3
        self.weights = np.zeros((n_classes, n_features))
        self.biases = np.zeros(n_classes)

        for c in range(n_classes):
            y_binary = np.where(y == c, 1, -1)
            w = rng.randn(n_features) * 0.01
            b = 0.0
            for _ in range(self.max_iter):
                for i in range(n_samples):
                    margin = y_binary[i] * (np.dot(X_norm[i], w) + b)
                    if margin <= 0:
                        w += self.eta0 * y_binary[i] * X_norm[i]
                        b += self.eta0 * y_binary[i]
            self.weights[c] = w
            self.biases[c] = b
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self.mean) / self.std
        scores = np.dot(X_norm, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)

    def to_dict(self):
        return {
            'type': 'PerceptronModel',
            'weights': self.weights.tolist() if self.weights is not None else None,
            'biases': self.biases.tolist() if self.biases is not None else None,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.weights = np.array(d['weights']) if d.get('weights') is not None else None
        obj.biases = np.array(d['biases']) if d.get('biases') is not None else None
        obj.mean = np.array(d['mean']) if d.get('mean') is not None else None
        obj.std = np.array(d['std']) if d.get('std') is not None else None
        return obj


class MLPClassifierModel:
    def __init__(self, hidden_dim=32, max_iter=800, lr=0.08, random_state=42):
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.lr = lr
        self.random_state = random_state
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.mean = None
        self.std = None

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_in = X.shape
        n_out = 3

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        X_norm = (X - self.mean) / self.std

        Y_oh = np.zeros((n_samples, n_out))
        Y_oh[np.arange(n_samples), y] = 1.0

        self.W1 = rng.randn(n_in, self.hidden_dim) * np.sqrt(2.0 / (n_in + self.hidden_dim))
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = rng.randn(self.hidden_dim, n_out) * np.sqrt(2.0 / (self.hidden_dim + n_out))
        self.b2 = np.zeros(n_out)

        for _ in range(self.max_iter):
            z1 = X_norm @ self.W1 + self.b1
            a1 = self._sigmoid(z1)
            z2 = a1 @ self.W2 + self.b2
            probs = self._softmax(z2)

            dz2 = (probs - Y_oh) / n_samples
            dW2 = a1.T @ dz2 + 0.001 * self.W2
            db2 = np.sum(dz2, axis=0)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * a1 * (1.0 - a1)
            dW1 = X_norm.T @ dz1 + 0.001 * self.W1
            db1 = np.sum(dz1, axis=0)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_norm = (X - self.mean) / self.std
        z1 = X_norm @ self.W1 + self.b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ self.W2 + self.b2
        probs = self._softmax(z2)
        return np.argmax(probs, axis=1)

    def to_dict(self):
        return {
            'type': 'MLPClassifierModel',
            'W1': self.W1.tolist() if self.W1 is not None else None,
            'b1': self.b1.tolist() if self.b1 is not None else None,
            'W2': self.W2.tolist() if self.W2 is not None else None,
            'b2': self.b2.tolist() if self.b2 is not None else None,
            'mean': self.mean.tolist() if self.mean is not None else None,
            'std': self.std.tolist() if self.std is not None else None
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.W1 = np.array(d['W1']) if d.get('W1') is not None else None
        obj.b1 = np.array(d['b1']) if d.get('b1') is not None else None
        obj.W2 = np.array(d['W2']) if d.get('W2') is not None else None
        obj.b2 = np.array(d['b2']) if d.get('b2') is not None else None
        obj.mean = np.array(d['mean']) if d.get('mean') is not None else None
        obj.std = np.array(d['std']) if d.get('std') is not None else None
        return obj


def compute_metrics(y_true, y_pred, labels=[0, 1, 2]):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(np.mean(y_true == y_pred))

    num_l = len(labels)
    cm = np.zeros((num_l, num_l), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_l and 0 <= p < num_l:
            cm[t, p] += 1

    per_class = {}
    precisions, recalls, f1s, supports = [], [], [], []

    for i, label_idx in enumerate(labels):
        name = LABEL_MAP[label_idx]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = int(cm[i, :].sum())

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_class[name] = {
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'support': support
        }
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    macro_prec = float(np.mean(precisions))
    macro_rec = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_s = sum(supports)
    weighted_prec = float(sum(p * s for p, s in zip(precisions, supports)) / total_s) if total_s > 0 else 0.0
    weighted_rec = float(sum(r * s for r, s in zip(recalls, supports)) / total_s) if total_s > 0 else 0.0
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_s) if total_s > 0 else 0.0

    return {
        'accuracy': round(acc, 4),
        'precision_macro': round(macro_prec, 4),
        'recall_macro': round(macro_rec, 4),
        'f1_macro': round(macro_f1, 4),
        'precision_weighted': round(weighted_prec, 4),
        'recall_weighted': round(weighted_rec, 4),
        'f1_weighted': round(weighted_f1, 4),
        'per_class': per_class,
        'confusion_matrix': cm.tolist()
    }


class ModelRegistry:
    _instance = None

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = models_dir
        self.models = {}
        self.evaluation_data = None
        self.load_models()
        self.load_evaluation()

    @classmethod
    def get_instance(cls, models_dir=None):
        if cls._instance is None:
            cls._instance = cls(models_dir)
        return cls._instance

    def load_models(self):
        json_path = os.path.join(self.models_dir, 'saved_models.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                loaders = {
                    'Decision Tree': DecisionTreeModel.from_dict,
                    'Random Forest': RandomForestModel.from_dict,
                    'Logistic Regression': LogisticRegressionModel.from_dict,
                    'Perceptron': PerceptronModel.from_dict,
                    'MLP Classifier': MLPClassifierModel.from_dict
                }
                for name, loader in loaders.items():
                    if name in data:
                        self.models[name] = loader(data[name])
                if len(self.models) == 5:
                    return
            except Exception as e:
                print(f'[ModelRegistry] Error loading saved_models.json: {e}')

        self.train_and_save_all()

    def load_evaluation(self):
        eval_path = os.path.join(self.models_dir, 'model_evaluation.json')
        if os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    self.evaluation_data = json.load(f)
            except Exception as e:
                print(f'[ModelRegistry] Error loading model_evaluation.json: {e}')

    def train_and_save_all(self, csv_path=None):
        if csv_path is None:
            csv_path = os.path.join(self.models_dir, 'AI-Data.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'Dataset not found at {csv_path}')

        df = pd.read_csv(csv_path)

        df['StudentAbsenceDays_num'] = (df['StudentAbsenceDays'] == 'Under-7').astype(int)
        df['Class_num'] = df['Class'].map(INV_LABEL_MAP)

        X = df[['raisedhands', 'VisITedResources', 'Discussion', 'StudentAbsenceDays_num']].values.astype(float)
        y = df['Class_num'].values.astype(int)

        np.random.seed(42)
        n = len(X)
        indices = np.random.permutation(n)
        test_size = int(n * 0.3)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        models = {
            'Decision Tree': DecisionTreeModel(max_depth=5, min_samples_split=4, random_state=42),
            'Random Forest': RandomForestModel(n_estimators=40, max_depth=5, random_state=42),
            'Logistic Regression': LogisticRegressionModel(C=1.0, random_state=42),
            'Perceptron': PerceptronModel(max_iter=200, eta0=0.005, random_state=42),
            'MLP Classifier': MLPClassifierModel(hidden_dim=32, max_iter=800, lr=0.08, random_state=42)
        }

        eval_results = {}
        saved_dict = {}

        for name, m in models.items():
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            metrics = compute_metrics(y_test, y_pred, labels=[0, 1, 2])
            eval_results[name] = metrics
            saved_dict[name] = m.to_dict()
            self.models[name] = m

        json_path = os.path.join(self.models_dir, 'saved_models.json')
        with open(json_path, 'w') as f:
            json.dump(saved_dict, f)

        comparison = []
        for name, m_metrics in eval_results.items():
            comparison.append({
                'model': name,
                'accuracy': m_metrics['accuracy'],
                'precision': m_metrics['precision_weighted'],
                'recall': m_metrics['recall_weighted'],
                'f1_score': m_metrics['f1_weighted']
            })
        comparison.sort(key=lambda x: x['accuracy'], reverse=True)

        test_class_counts = {
            LABEL_MAP[k]: int(np.sum(y_test == k)) for k in [0, 1, 2]
        }
        total_class_counts = {
            LABEL_MAP[k]: int(np.sum(y == k)) for k in [0, 1, 2]
        }

        full_evaluation = {
            'metadata': {
                'dataset_name': 'AI-Data.csv (xAPI-Edu-Data)',
                'total_samples': len(X),
                'train_size': len(X_train),
                'test_size': len(X_test),
                'train_ratio': 0.70,
                'test_ratio': 0.30,
                'random_state': 42,
                'class_encoding': {
                    '0': 'H (High, Grade >= 90)',
                    '1': 'L (Low, Grade < 70)',
                    '2': 'M (Medium, Grade 70-89)'
                },
                'feature_names': ['raisedhands', 'VisITedResources', 'Discussion', 'StudentAbsenceDays'],
                'feature_encoding': {
                    'StudentAbsenceDays': {'0': 'Above-7', '1': 'Under-7'}
                },
                'test_class_distribution': test_class_counts,
                'dataset_class_distribution': total_class_counts,
                'evaluation_methodology': (
                    'Strict 70/30 train/test holdout evaluation with random_state=42. '
                    'Models are trained exclusively on 336 instances and evaluated on 144 held-out instances. '
                    'Target classes are mapped consistently (H=0, L=1, M=2). '
                    'Metrics are never computed from ad-hoc single user inputs.'
                )
            },
            'models': eval_results,
            'comparison': comparison
        }

        eval_path = os.path.join(self.models_dir, 'model_evaluation.json')
        with open(eval_path, 'w') as f:
            json.dump(full_evaluation, f, indent=2)

        self.evaluation_data = full_evaluation
        return full_evaluation

    def predict_student(self, data):
        validated = validate_student_input(data)

        feat_vector = np.array([
            validated['raisedHands'],
            validated['visitedResources'],
            validated['discussion'],
            validated['absence_encoded']
        ], dtype=float).reshape(1, -1)

        predictions = {}
        for name, model in self.models.items():
            pred_code = int(model.predict(feat_vector)[0])
            predictions[name] = LABEL_MAP.get(pred_code, 'Unknown')

        raw_engagement = (
            validated['raisedHands'] * 0.3 +
            validated['visitedResources'] * 0.3 +
            validated['discussion'] * 0.3 -
            validated['absence_days'] * 2.0
        )
        engagement = max(0, min(100, int(round(raw_engagement))))

        if engagement >= 70:
            evaluation = 'Excellent & Highly Active'
        elif engagement >= 40:
            evaluation = 'Average Activity'
        else:
            evaluation = 'Needs Improvement'

        return {
            'predictions': predictions,
            'engagementScore': engagement,
            'evaluation': evaluation,
            'validatedInput': {
                'raisedHands': validated['raisedHands'],
                'visitedResources': validated['visitedResources'],
                'discussion': validated['discussion'],
                'absences': validated['absence_days'],
                'absenceCategory': 'Under-7' if validated['absence_encoded'] == 1 else 'Above-7'
            }
        }
