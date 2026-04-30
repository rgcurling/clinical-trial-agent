"""
Compare LLM matcher predictions against TREC physician ground truth.

Mapping:
  TREC grade 0         → not_relevant → negative class
  TREC grade 1 or 2   → relevant     → positive class
  Matcher score >= 0.5 → eligible     → predicted positive
  Matcher score < 0.5  → excluded     → predicted negative

Usage:
  cd trialmatch
  python compute_confusion_matrix.py                        # uses run_r1.json
  python compute_confusion_matrix.py --run results/run_r3.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RUN = "results/run_r1.json"
OUTPUT_PNG = "results/confusion_matrix.png"
OUTPUT_ERRORS = "results/matcher_errors.jsonl"


def _load_run(run_path: str) -> list[dict]:
    with open(run_path) as f:
        data = json.load(f)
    rows: list[dict] = []
    for topic in data.get("per_topic_results", []):
        if "error" in topic:
            continue
        patient_text = topic.get("topic_text", "")
        for trial in topic.get("top5", []):
            trec_grade = trial.get("trec_grade", -1)
            if trec_grade == -1:
                continue  # not judged — skip
            rows.append({
                "topic_id": topic["topic_id"],
                "nct_id": trial["nct_id"],
                "title": trial.get("title", ""),
                "overall_score": trial.get("overall_score", 0.0),
                "trec_grade": trec_grade,
                "patient_text": patient_text[:300],
            })
    return rows


def compute(run_path: str = DEFAULT_RUN) -> None:
    if not Path(run_path).exists():
        print(f"[Error] Run file not found: {run_path}")
        print("Run  python run_experiments.py  first, then re-run this script.")
        return

    rows = _load_run(run_path)
    if not rows:
        print("[Error] No judged patient-trial pairs found in run file.")
        return

    # Binarise
    y_true = np.array([1 if r["trec_grade"] > 0 else 0 for r in rows])
    y_pred = np.array([1 if r["overall_score"] >= 0.5 else 0 for r in rows])

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    tn, fp, fn, tp = cm.ravel()
    total = int(cm.sum())

    # ── Print matrix ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Confusion Matrix: Matcher vs. TREC Physician Judgments':^60}")
    print(f"{'='*60}")
    print(f"\n{'':>35}  Predicted")
    print(f"{'':>35}  {'Excluded':>10}  {'Eligible':>10}")
    print(f"  {'TREC  Not Relevant':>33}  {tn:>10}  {fp:>10}")
    print(f"  {'Grade Relevant':>33}  {fn:>10}  {tp:>10}")

    print(f"\n  Pairs evaluated  : {total}")
    print(f"  Overall Accuracy : {acc:.1%}")
    print(f"  Precision        : {prec:.1%}  (eligible class)")
    print(f"  Recall           : {rec:.1%}  (eligible class)")
    print(f"  F1 Score         : {f1:.1%}  (eligible class)")

    # ── Clinical interpretation ───────────────────────────────────────────────
    print(f"\n  Clinical Interpretation:")
    print(
        f"  - False Positives ({fp} cases): Matcher says eligible, physician says not relevant\n"
        f"    → Risk: Wasted patient time, screening visit for ineligible trial"
    )
    print(
        f"  - False Negatives ({fn} cases): Matcher says excluded, physician says relevant\n"
        f"    → Risk: Missed trial opportunities for the patient"
    )
    if acc < 0.80:
        print(
            f"\n  Recommended threshold: consider lowering eligibility cutoff from 0.5 to 0.4\n"
            f"  (current accuracy {acc:.1%} is below 80% — bias toward recall to reduce FN)"
        )
    else:
        print(f"\n  Accuracy {acc:.1%} meets production threshold (≥80%).")

    # ── Visualise ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))

        annot = np.array([
            [f"{cm[i, j]}\n({cm[i, j]/total*100:.1f}%)" for j in range(2)]
            for i in range(2)
        ])

        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=["Excluded\n(predicted)", "Eligible\n(predicted)"],
            yticklabels=["Not Relevant\n(TREC)", "Relevant\n(TREC)"],
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 14},
        )

        ax.set_title(
            f"Matcher vs TREC Physician Judgments\n"
            f"Acc={acc:.1%}  P={prec:.1%}  R={rec:.1%}  F1={f1:.1%}",
            fontsize=13,
            pad=14,
        )
        ax.set_ylabel("Actual label (TREC)", fontsize=12)
        ax.set_xlabel("Predicted label (Matcher)", fontsize=12)
        plt.tight_layout()

        os.makedirs(os.path.dirname(OUTPUT_PNG) or ".", exist_ok=True)
        fig.savefig(OUTPUT_PNG, dpi=300)
        plt.close(fig)
        print(f"\n  Heatmap saved → {OUTPUT_PNG}")

    except ImportError as e:
        print(f"\n  [Warning] Could not generate heatmap: {e}")
        print("  Install with:  pip install matplotlib seaborn")

    # ── Error case extraction ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_ERRORS) or ".", exist_ok=True)
    error_count = 0
    with open(OUTPUT_ERRORS, "w") as f:
        for row, yt, yp in zip(rows, y_true, y_pred):
            if yt != yp:
                error_type = "false_positive" if yp == 1 else "false_negative"
                f.write(json.dumps({
                    "error_type": error_type,
                    "topic_id": row["topic_id"],
                    "nct_id": row["nct_id"],
                    "matcher_score": row["overall_score"],
                    "trec_grade": row["trec_grade"],
                    "patient_text": row["patient_text"],
                    "trial_title": row["title"],
                }) + "\n")
                error_count += 1

    print(f"  Error cases ({error_count} FP+FN) saved → {OUTPUT_ERRORS}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compute matcher confusion matrix vs TREC qrels")
    p.add_argument(
        "--run",
        default=DEFAULT_RUN,
        metavar="PATH",
        help=f"Benchmark run JSON (default: {DEFAULT_RUN})",
    )
    args = p.parse_args()
    compute(args.run)
