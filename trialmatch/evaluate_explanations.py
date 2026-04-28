"""
Compute Flesch-Kincaid grade and BERTScore F1 for top-5 explanations.

Usage:
  cd trialmatch
  python evaluate_explanations.py                        # uses run_r1.json
  python evaluate_explanations.py --run results/run_r3.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RUN = "results/run_r1.json"
OUTPUT_CSV = "results/explanation_metrics.csv"
FK_THRESHOLD = 8.0
BERTSCORE_THRESHOLD = 0.85


def evaluate(run_path: str = DEFAULT_RUN) -> None:
    if not Path(run_path).exists():
        print(f"[Error] Run file not found: {run_path}")
        print("Run  python run_experiments.py  first, then re-run this script.")
        return

    with open(run_path) as f:
        data = json.load(f)

    # Collect all top-5 records that have explanations
    records: list[dict] = []
    for topic in data.get("per_topic_results", []):
        if "error" in topic:
            continue
        for trial in topic.get("top5", []):
            explanation = trial.get("explanation", "").strip()
            eligibility = trial.get("eligibility_criteria", "").strip()
            if not explanation:
                continue
            records.append({
                "topic_id": topic["topic_id"],
                "nct_id": trial["nct_id"],
                "explanation_text": explanation,
                "eligibility_criteria": eligibility,
                "pre_computed_fk": trial.get("fk_grade"),
            })

    if not records:
        print(
            "[Error] No explanations found in run file.\n"
            "Re-run benchmark without --no-explanations to generate them."
        )
        return

    print(f"\nEvaluating {len(records)} explanations from {run_path}...\n")

    # ── FK grade ──────────────────────────────────────────────────────────────
    try:
        import textstat
        for r in records:
            if r["pre_computed_fk"] is not None:
                r["fk_grade"] = r["pre_computed_fk"]
            else:
                r["fk_grade"] = textstat.flesch_kincaid_grade(r["explanation_text"])
    except ImportError:
        print("[Warning] textstat not installed; using 0.0 for FK grade.  pip install textstat")
        for r in records:
            r["fk_grade"] = r["pre_computed_fk"] or 0.0

    # ── BERTScore ─────────────────────────────────────────────────────────────
    explanations = [r["explanation_text"] for r in records]
    references = [r["eligibility_criteria"] for r in records]

    bert_f1_scores: list[float] = []
    try:
        from bert_score import score as bert_score_fn
        print("Computing BERTScore (this may take 1-2 minutes on first run)...")
        _, _, F1 = bert_score_fn(explanations, references, lang="en", verbose=False)
        bert_f1_scores = F1.tolist()
    except ImportError:
        print("[Warning] bert-score not installed; using 0.0.  pip install bert-score")
        bert_f1_scores = [0.0] * len(records)
    except Exception as e:
        print(f"[Warning] BERTScore computation failed: {e}; using 0.0")
        bert_f1_scores = [0.0] * len(records)

    for r, bs in zip(records, bert_f1_scores):
        r["bertscore_f1"] = round(float(bs), 4)
        r["pass_fk"] = r["fk_grade"] <= FK_THRESHOLD
        r["pass_bertscore"] = r["bertscore_f1"] >= BERTSCORE_THRESHOLD

    # ── Summary statistics ────────────────────────────────────────────────────
    fk_vals = [r["fk_grade"] for r in records]
    bs_vals = [r["bertscore_f1"] for r in records]
    n = len(records)

    mean_fk = sum(fk_vals) / n
    mean_bs = sum(bs_vals) / n
    pct_fk_pass = sum(r["pass_fk"] for r in records) / n * 100
    pct_bs_pass = sum(r["pass_bertscore"] for r in records) / n * 100

    print(f"{'='*55}")
    print(f"{'Explanation Quality Metrics':^55}")
    print(f"{'='*55}")
    print(f"  Explanations evaluated : {n}")
    print(f"  Mean FK grade          : {mean_fk:.1f}  (target: ≤{FK_THRESHOLD:.0f})")
    print(f"  Mean BERTScore F1      : {mean_bs:.3f}  (target: ≥{BERTSCORE_THRESHOLD:.2f})")
    print(f"  % passing FK           : {pct_fk_pass:.1f}%")
    print(f"  % passing BERTScore    : {pct_bs_pass:.1f}%")
    print(f"{'='*55}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    fieldnames = [
        "topic_id", "nct_id", "explanation_text",
        "fk_grade", "bertscore_f1", "pass_fk", "pass_bertscore",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\n  Detailed CSV saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Evaluate explanation quality (FK + BERTScore)")
    p.add_argument(
        "--run",
        default=DEFAULT_RUN,
        metavar="PATH",
        help=f"Benchmark run JSON (default: {DEFAULT_RUN})",
    )
    args = p.parse_args()
    evaluate(args.run)
