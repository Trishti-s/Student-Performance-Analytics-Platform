"""
agent.py
========
AI Analytics Agent with Controlled Tool Calling.

Core Architectural Capabilities:
1. Controlled Tool Calling:
   - Provides access only to the 5 bounded tools in tools.py.
   - Strictly enforces tool input schema and parameter validation.
2. Grounded Evidence-Based Responses:
   - Never fabricates scores or metrics; all claims are backed by tool outputs.
   - Explicitly enforces the 'correlation does NOT equal causation' disclaimer on engagement queries.
3. Latency & Reliability Tracking:
   - Measures tool execution latency in milliseconds.
   - Logs tool events without storing student PII or sensitive inputs.
4. Robust Handling of Out-of-Scope & Malformed Queries:
   - Detects nonsensical, conversational, or invalid questions and guides the user to valid analytics queries.
"""

import os
import sys
import time
import re
import json
import logging

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

try:
    import tools
except ImportError:
    import importlib.util
    spec = importlib.util.spec_from_file_location('tools', os.path.join(_CURRENT_DIR, 'tools.py'))
    tools = importlib.util.module_from_spec(spec)
    sys.modules['tools'] = tools
    spec.loader.exec_module(tools)

# Setup privacy-safe logging (Logs metadata and tool names, NEVER student PII)
logger = logging.getLogger('AIAnalyticsAgent')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - [Agent Audit] - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Tool Declarations / Specifications
TOOL_DEFINITIONS = [
    {
        'name': 'compare_models',
        'description': 'Compare performance across all five classifiers on the held-out test set using a selected metric.',
        'parameters': {
            'metric': {
                'type': 'string',
                'description': 'Evaluation metric to compare: accuracy, precision, recall, or f1_score.',
                'default': 'accuracy'
            }
        }
    },
    {
        'name': 'get_prediction_distribution',
        'description': 'Summarize the distribution of performance labels (H, M, L) across the full dataset and test split.',
        'parameters': {}
    },
    {
        'name': 'get_engagement_summary',
        'description': 'Summarize empirical behavioral indicators (raised hands, visited resources, discussions, absences) grouped by class.',
        'parameters': {}
    },
    {
        'name': 'get_model_metrics',
        'description': 'Retrieve detailed metrics, per-class breakdown, and confusion matrix for a specific model.',
        'parameters': {
            'model_name': {
                'type': 'string',
                'description': 'Name of model: Decision Tree, Random Forest, Logistic Regression, Perceptron, or MLP Classifier.'
            }
        }
    },
    {
        'name': 'analyze_student_input',
        'description': 'Run model inference across all 5 models on student features, compute engagement score, and generate feature attribution.',
        'parameters': {
            'features': {
                'type': 'object',
                'description': 'Dictionary containing raisedHands, visitedResources, discussion, and absences.'
            }
        }
    }
]


class AIAnalyticsAgent:
    def __init__(self):
        self.tools = tools.CONTROLLED_TOOLS
        self.audit_log = []

    def _log_event(self, tool_name, latency_ms, status, error=None):
        """
        Privacy-safe audit logger: captures tool name, execution time, and status.
        Does NOT store student PII or feature values.
        """
        entry = {
            'timestamp': time.time(),
            'tool_name': tool_name,
            'latency_ms': round(latency_ms, 2),
            'status': status,
            'error': str(error) if error else None
        }
        self.audit_log.append(entry)
        logger.info(f"Tool: {tool_name} | Latency: {entry['latency_ms']}ms | Status: {status}")

    def route_query(self, question, student_features=None):
        if not question or not isinstance(question, str) or len(question.strip()) == 0:
            return None, {}, False, 'Question cannot be empty. Please provide an analytical question.'

        q = question.strip().lower()

        # Check for out-of-scope questions
        out_of_scope_patterns = [
            r'\bcapital of\b', r'\bweather\b', r'\bpoem\b', r'\bjoke\b',
            r'\bpresident\b', r'\bmovie\b', r'\bsong\b', r'\bcook\b',
            r'\brecipe\b', r'\bwho won\b', r'\bstock price\b'
        ]
        for pat in out_of_scope_patterns:
            if re.search(pat, q):
                return None, {}, False, (
                    "This question is outside the scope of the Student Performance Analytics Platform. "
                    "I can answer questions regarding model evaluation metrics, model comparisons, dataset distributions, "
                    "student engagement patterns, or explain predictions for student inputs."
                )

        # 1. Prediction explanation / student input analysis
        if any(term in q for term in ['these input', 'student input', 'this student', 'explain the prediction', 'explain prediction', 'input values', 'analyze student', 'for these input']):
            return 'analyze_student_input', {'features': student_features or {}}, True, None

        # 2. Specific single model query
        model_keywords = {
            'random forest': ['random forest', 'rf', 'forest'],
            'decision tree': ['decision tree', 'dt', 'tree'],
            'logistic regression': ['logistic regression', 'logistic', 'lr'],
            'perceptron': ['perceptron'],
            'mlp classifier': ['mlp', 'neural network', 'mlp classifier', 'neural net']
        }
        is_multi_compare = any(comp in q for comp in ['compare', 'all models', 'which model', 'across all', 'rank all', 'rank models'])
        if not is_multi_compare:
            for model_canon, aliases in model_keywords.items():
                if any(re.search(r'\b' + re.escape(alias) + r'\b', q) for alias in aliases):
                    return 'get_model_metrics', {'model_name': model_canon}, True, None

        # 3. Engagement & behavioral activity
        if any(term in q for term in ['engagement', 'attendance', 'absence', 'absent', 'raised hand', 'visited resource', 'discussion', 'behavior', 'active']):
            return 'get_engagement_summary', {}, True, None

        # 4. Class / Prediction Distribution
        if any(term in q for term in ['distribution', 'class balance', 'proportion', 'h, m and l', 'h, m, l', 'how many student', 'how many h', 'how many low', 'how many medium']):
            return 'get_prediction_distribution', {}, True, None

        # 5. Model comparison / best model
        if any(term in q for term in ['compare', 'best', 'highest', 'rank', 'which model', 'across all', 'performance of all', 'top model', 'lowest', 'metric', 'test set', 'evaluation', 'accuracy', 'precision', 'recall', 'f1']):
            metric = 'accuracy'
            if 'precision' in q and 'recall' in q:
                metric = 'precision'
            elif 'precision' in q:
                metric = 'precision'
            elif 'recall' in q:
                metric = 'recall'
            elif 'f1' in q:
                metric = 'f1_score'
            return 'compare_models', {'metric': metric}, True, None

        return None, {}, False, (
            "I could not determine a matching analytics tool for your question. "
            "Try asking: 'Which model performed best on the test set?', 'Compare precision and recall across all models', "
            "'What is the distribution of H, M and L in the dataset?', or 'Explain the predictions for these input values'."
        )

    def synthesize_answer(self, tool_name, tool_kwargs, tool_result, question):
        """
        Synthesizes a grounded, natural-language explanation from tool output.
        Strictly adheres to:
        1. Grounding in actual tool numbers (no hallucinations).
        2. Explicit 'correlation does not imply causation' disclosure for behavioral engagement.
        """
        if 'error' in tool_result:
            return f"Tool execution encountered an issue: {tool_result['error']}"

        q_lower = question.lower()

        if tool_name == 'compare_models':
            metric = tool_result.get('metric_evaluated', 'accuracy')
            best = tool_result['best_model']
            worst = tool_result['lowest_model']
            rankings = tool_result['rankings']

            lines = [
                f"Based on rigorous evaluation across the held-out test set ({tool_result['test_set_size']} samples, 30% split with fixed random seed 42):",
                "",
                f"**Top Performing Model**: **{best['model']}** achieved the highest {metric} of **{best['score'] * 100:.2f}%**.",
                f"**Lowest Performing Model**: **{worst['model']}** registered a {metric} of **{worst['score'] * 100:.2f}%** (spread of {tool_result['score_spread'] * 100:.2f}%).",
                "",
                "**Full Model Rankings on the Test Set**:",
            ]
            for r in rankings:
                lines.append(f"- **Rank {r['rank']} - {r['model']}**: {metric.capitalize()} = **{r['score'] * 100:.2f}%** (Accuracy: {r['accuracy'] * 100:.2f}%, F1-Score: {r['f1_weighted'] * 100:.2f}%)")

            lines.append("")
            lines.append("*Methodology Note: All metrics reflect holdout test evaluation and are never calculated from training data or single user inputs.*")
            return "\n".join(lines)

        elif tool_name == 'get_prediction_distribution':
            classes = tool_result['classes']
            total = tool_result['total_dataset_records']
            test_total = tool_result['test_set_records']

            lines = [
                f"The dataset comprises **{total} students**, split into a **336-sample training set** (70%) and a **{test_total}-sample test set** (30%).",
                "",
                "**Class Distribution Summary**:",
            ]
            for c in classes:
                lines.append(
                    f"- **Class {c['class']} ({c['description']})**: **{c['dataset_count']} students** in full dataset (**{c['dataset_percentage']}%**), "
                    f"and **{c['test_count']} students** in test set (**{c['test_percentage']}%**)."
                )

            lines.append("")
            lines.append(f"**Dominant Class**: {tool_result['dominant_class']}.")
            lines.append(f"**Key Observation**: {tool_result['observation']}")
            return "\n".join(lines)

        elif tool_name == 'get_engagement_summary':
            means = tool_result['overall_activity_means']
            by_class = tool_result['class_breakdown']

            lines = [
                f"Empirical engagement summary across **{tool_result['total_students_analyzed']} students** in the dataset:",
                "",
                "**Class-by-Class Behavioral Profile**:",
            ]
            for cls_name, info in by_class.items():
                lines.append(f"- **Class {cls_name} ({info['description']})**:")
                lines.append(f"  - Average Raised Hands: **{info['avg_raised_hands']}** / 100")
                lines.append(f"  - Average Visited Resources: **{info['avg_visited_resources']}** / 100")
                lines.append(f"  - Average Discussion Participation: **{info['avg_discussion']}** / 100")
                lines.append(f"  - Attendance: **{info['absence_breakdown']['under_7_pct']}%** had < 7 absences; **{info['absence_breakdown']['above_7_pct']}%** had >= 7 absences.")

            lines.append("")
            lines.append("**Key Findings**:" )
            for f in tool_result['key_empirical_findings']:
                lines.append(f"- {f}")

            lines.append("")
            lines.append(f"> ⚠️ **Important Analytical Distinction**: {tool_result['scientific_guardrail']}")
            return "\n".join(lines)

        elif tool_name == 'get_model_metrics':
            m_name = tool_result['model_name']
            cm = tool_result['confusion_matrix']['matrix']
            per_class = tool_result['per_class_breakdown']

            lines = [
                f"Detailed evaluation metrics for **{m_name}** evaluated on the held-out test set ({tool_result['test_set_size']} samples):",
                "",
                f"- **Overall Accuracy**: **{tool_result['overall_accuracy'] * 100:.2f}%**",
                f"- **Weighted F1-Score**: **{tool_result['f1_weighted'] * 100:.2f}%**",
                f"- **Weighted Precision**: **{tool_result['precision_weighted'] * 100:.2f}%**",
                f"- **Weighted Recall**: **{tool_result['recall_weighted'] * 100:.2f}%**",
                "",
                "**Per-Class Performance**:",
            ]
            for cls_key, metrics in per_class.items():
                lines.append(f"- **Class {cls_key}**: Precision = {metrics['precision'] * 100:.1f}%, Recall = {metrics['recall'] * 100:.1f}%, F1 = {metrics['f1_score'] * 100:.1f}% (Support: {metrics['support']} samples)")

            lines.append("")
            lines.append("**Confusion Matrix (Ground Truth Rows x Predicted Columns)**:")
            lines.append(f"- Ground Truth H: Predicted [H: {cm[0][0]}, L: {cm[0][1]}, M: {cm[0][2]}]")
            lines.append(f"- Ground Truth L: Predicted [H: {cm[1][0]}, L: {cm[1][1]}, M: {cm[1][2]}]")
            lines.append(f"- Ground Truth M: Predicted [H: {cm[2][0]}, L: {cm[2][1]}, M: {cm[2][2]}]")
            return "\n".join(lines)

        elif tool_name == 'analyze_student_input':
            feat = tool_result['validated_features']
            preds = tool_result['predictions_by_model']
            consensus = tool_result['consensus_prediction']
            desc = tool_result['consensus_description']

            lines = [
                f"Analysis of student activity: **Raised Hands**: {feat['raisedHands']}, **Visited Resources**: {feat['visitedResources']}, **Discussion**: {feat['discussion']}, **Absences**: {feat['absences']} ({feat['absenceCategory']}).",
                "",
                f"**Consensus Prediction**: **{consensus} ({desc})** with **{tool_result['model_agreement']}**.",
                f"**Calculated Engagement Score**: **{tool_result['engagement_score']} / 100** ({tool_result['activity_evaluation']}).",
                "",
                "**Individual Model Predictions**:",
            ]
            for m, p in preds.items():
                lines.append(f"- **{m}**: {p}")

            lines.append("")
            lines.append("**Feature Attribution & Dataset Comparison**:")
            for note in tool_result['feature_attribution_notes']:
                lines.append(f"- {note}")

            lines.append("")
            lines.append(f"> *Notice: {tool_result['disclaimer']}*")
            return "\n".join(lines)

        return json.dumps(tool_result, indent=2)

    def answer_query(self, question, student_features=None):
        """
        Main agent interface: routes question, calls tool, tracks latency,
        and generates evidence-backed natural language response.
        """
        start_time = time.perf_counter()

        tool_name, tool_kwargs, is_valid, rejection_reason = self.route_query(question, student_features)

        if not is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return {
                'answer': rejection_reason,
                'tool_calls': [],
                'evidence': None,
                'latency_ms': round(elapsed_ms, 2),
                'status': 'rejected',
                'correlation_disclaimer_included': False
            }

        tool_func = self.tools[tool_name]
        tool_start = time.perf_counter()
        try:
            tool_output = tool_func(**tool_kwargs)
            tool_elapsed_ms = (time.perf_counter() - tool_start) * 1000.0
            self._log_event(tool_name, tool_elapsed_ms, 'success')
            status = 'success'
            err_msg = None
        except Exception as e:
            tool_elapsed_ms = (time.perf_counter() - tool_start) * 1000.0
            self._log_event(tool_name, tool_elapsed_ms, 'error', error=e)
            tool_output = {'error': f'Failed executing {tool_name}: {str(e)}'}
            status = 'error'
            err_msg = str(e)

        answer_text = self.synthesize_answer(tool_name, tool_kwargs, tool_output, question)
        total_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        has_disclaimer = 'causation' in answer_text.lower() or 'correlation' in answer_text.lower()

        return {
            'answer': answer_text,
            'tool_calls': [
                {
                    'tool_name': tool_name,
                    'arguments': tool_kwargs,
                    'latency_ms': round(tool_elapsed_ms, 2),
                    'status': status,
                    'error': err_msg
                }
            ],
            'evidence': tool_output,
            'latency_ms': round(total_elapsed_ms, 2),
            'status': status,
            'correlation_disclaimer_included': has_disclaimer
        }
