"""
Evaluation metrics for TrialMatch AI.

- Flesch-Kincaid grade level (readability)
- BERTScore consistency (semantic similarity)
- Precision / Recall / F1 (eligibility classification)
"""

import logging

import textstat
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def flesch_kincaid_grade(text: str) -> float:
    """Return the Flesch-Kincaid grade level of *text*."""
    return textstat.flesch_kincaid_grade(text)


def bertscore_consistency(generated: str, reference: str) -> float:
    """
    Compute BERTScore F1 between *generated* and *reference* texts.
    Returns the mean F1 as a float in [0, 1].
    """
    try:
        from bert_score import score as bert_score_fn

        P, R, F = bert_score_fn(
            [generated], [reference], lang="en", verbose=False
        )
        return float(F.mean().item())
    except Exception as e:
        logger.warning(f"BERTScore computation failed: {e}")
        return float("nan")


def precision_recall_f1(y_pred: list[int], y_true: list[int]) -> dict:
    """
    Compute binary precision, recall, and F1.

    Labels: 1 = eligible, 0 = ineligible.
    Returns a dict with keys 'precision', 'recall', 'f1'.
    """
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
