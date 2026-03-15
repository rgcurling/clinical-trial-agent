"""
Evaluation metrics for TrialMatch AI.

- flesch_kincaid_grade: readability of generated cards
- bertscore_consistency: factual consistency of card vs. source trial text
- precision_at_k / ndcg_at_k: retrieval quality vs. TREC ground truth
- precision_recall_f1: helper for binary classification eval
"""

from __future__ import annotations
import math
import logging

import textstat

logger = logging.getLogger(__name__)

__all__ = [
    "flesch_kincaid_grade",
    "bertscore_consistency",
    "precision_at_k",
    "ndcg_at_k",
    "precision_recall_f1",
]


def flesch_kincaid_grade(text: str) -> float:
    """Return Flesch-Kincaid grade level using textstat."""
    return textstat.flesch_kincaid_grade(text)


def bertscore_consistency(generated: str, source: str) -> float:
    """
    Compute BERTScore F1 between a generated card and the source trial
    eligibility criteria text. Returns mean F1 as a float.

    Uses distilbert-base-uncased for speed. Falls back to 0.0 and logs
    a warning if bert_score is not installed.
    """
    try:
        from bert_score import score as bs_score
        P, R, F = bs_score(
            [generated],
            [source],
            lang="en",
            model_type="distilbert-base-uncased",
            rescale_with_baseline=True,
            verbose=False,
        )
        return float(F[0])
    except ImportError:
        logger.warning("bert_score not installed; returning 0.0 for BERTScore")
        return 0.0
    except Exception as e:
        logger.warning(f"BERTScore computation failed ({e}); returning 0.0")
        return 0.0


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """
    Precision@K: fraction of top-K retrieved NCT IDs that are in the relevant set.

    Args:
        retrieved: ordered list of NCT IDs returned by the system (ranked)
        relevant: set of NCT IDs judged relevant by TREC assessors
        k: cutoff rank
    Returns:
        float in [0, 1]
    """
    if not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for nct in top_k if nct in relevant_set)
    return hits / k


def ndcg_at_k(
    retrieved: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain at rank K.

    Args:
        retrieved: ordered list of NCT IDs returned by the system
        relevance_grades: dict mapping NCT ID → relevance grade (0, 1, or 2)
                          TREC uses 0=not relevant, 1=partially relevant, 2=highly relevant
        k: cutoff rank
    Returns:
        float in [0, 1]
    """
    def dcg(ranking: list[str], grades: dict[str, int], k: int) -> float:
        score = 0.0
        for i, nct in enumerate(ranking[:k], start=1):
            rel = grades.get(nct, 0)
            score += (2 ** rel - 1) / math.log2(i + 1)
        return score

    actual_dcg = dcg(retrieved, relevance_grades, k)

    # Ideal ranking: sort all known relevant NCT IDs by grade descending
    ideal_ranking = sorted(relevance_grades.keys(), key=lambda n: relevance_grades[n], reverse=True)
    ideal_dcg = dcg(ideal_ranking, relevance_grades, k)

    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def precision_recall_f1(
    predicted: list[str],
    ground_truth: list[str],
) -> dict[str, float]:
    """
    Binary precision, recall, F1 over NCT ID lists.

    Args:
        predicted: NCT IDs the system returned as eligible
        ground_truth: NCT IDs that are truly eligible
    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    predicted_set = set(predicted)
    truth_set = set(ground_truth)

    tp = len(predicted_set & truth_set)
    precision = tp / len(predicted_set) if predicted_set else 0.0
    recall = tp / len(truth_set) if truth_set else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}
