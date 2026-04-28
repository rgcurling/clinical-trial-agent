"""
Build a confusion matrix comparing the matcher's eligibility predictions
against TREC physician relevance judgments for top-5 results.

Mapping:
  TREC grade 0         → "not_relevant"  → negative class
  TREC grade 1 or 2    → "relevant"      → positive class
  Matcher score >= 0.5 → "eligible"      → predicted positive
  Matcher score < 0.5  → "excluded"      → predicted negative

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


def _load_run(run_path: str) -> list[dict]:
    with open(run_path) as f:
        data = json.load(f)
    rows: list[dict] = []
    for topic in data.get("per_topic_results", []):
        if "error" in topic:
            continue
        for trial in topic.get("top5", []):
            trec_grade = trial.get("trec_grade", -1)
            if trec_grade == -1:
                continue  # not judged — skip
            rows.append({
                "topic_id": topic["topic_id"],
                "nct_id": trial["nct_id"],
                "overall_score": trial.get("overall_score", 0.0),
                "trec_grade": trec_grade,
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
    labels = ["not_relevant / excluded", "relevant / eligible"]

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\nConfusion Matrix  ({run_path})")
    print(f"  {'':>30}  Predicted")
    print(f"  {'':>30}  {'Excluded':>10}  {'Eligible':>10}")
    print(f"  {'Actual  Not Relevant':>30}  {cm[0, 0]:>10}  {cm[0, 1]:>10}")
    print(f"  {'Actual  Relevant':>30}  {cm[1, 0]:>10}  {cm[1, 1]:>10}")

    total = cm.sum()
    print(f"\n  Pairs evaluated : {total}")
    print(f"  Accuracy        : {acc:.3f}")
    print(f"  Precision       : {prec:.3f}  (eligible class)")
    print(f"  Recall          : {rec:.3f}  (eligible class)")
    print(f"  F1              : {f1:.3f}  (eligible class)")

    # ── Visualise ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 5))

        # Counts + percentages in each cell
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
            annot_kws={"size": 13},
        )

        ax.set_title(
            f"Matcher vs TREC Physician Judgments\n"
            f"Acc={acc:.2f}  P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}",
            fontsize=12,
            pad=12,
        )
        ax.set_ylabel("Actual label (TREC)", fontsize=11)
        ax.set_xlabel("Predicted label (Matcher)", fontsize=11)
        plt.tight_layout()

        os.makedirs(os.path.dirname(OUTPUT_PNG) or ".", exist_ok=True)
        fig.savefig(OUTPUT_PNG, dpi=300)
        plt.close(fig)
        print(f"\n  Heatmap saved → {OUTPUT_PNG}")

    except ImportError as e:
        print(f"\n  [Warning] Could not generate heatmap: {e}")
        print("  Install with:  pip install matplotlib seaborn")


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
