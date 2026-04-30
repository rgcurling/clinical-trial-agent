"""
Validate explanation quality: readability (Flesch-Kincaid) and faithfulness (BERTScore).

Ensures patient-facing outputs are accessible (FK ≤8) and grounded in trial text (≥0.85).

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
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RUN = "results/run_r1.json"
OUTPUT_CSV = "results/explanation_metrics.csv"
OUTPUT_OUTLIERS = "results/explanation_outliers.txt"
FK_THRESHOLD = 8.0
BERTSCORE_THRESHOLD = 0.85


def evaluate(run_path: str = DEFAULT_RUN) -> None:
    if not Path(run_path).exists():
        print(f"[Error] Run file not found: {run_path}")
        print("Run  python run_experiments.py  first, then re-run this script.")
        return

    with open(run_path) as f:
        data = json.load(f)

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
                "explanation_length_chars": len(explanation),
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
            r["fk_grade"] = (
                r["pre_computed_fk"]
                if r["pre_computed_fk"] is not None
                else textstat.flesch_kincaid_grade(r["explanation_text"])
            )
    except ImportError:
        print("[Warning] textstat not installed; using 0.0.  pip install textstat")
        for r in records:
            r["fk_grade"] = r["pre_computed_fk"] or 0.0

    # ── BERTScore ─────────────────────────────────────────────────────────────
    explanations = [r["explanation_text"] for r in records]
    references = [r["eligibility_criteria"] for r in records]
    bert_f1_scores: list[float] = []

    try:
        from bert_score import score as bert_score_fn
        print("Computing BERTScore (may take 1-2 minutes on first run)...")
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
        r["combined_pass"] = r["pass_fk"] and r["pass_bertscore"]

    # ── Summary statistics ────────────────────────────────────────────────────
    fk_vals = [r["fk_grade"] for r in records]
    bs_vals = [r["bertscore_f1"] for r in records]
    n = len(records)

    mean_fk = sum(fk_vals) / n
    median_fk = median(fk_vals)
    mean_bs = sum(bs_vals) / n
    median_bs = median(bs_vals)
    pct_fk_pass = sum(r["pass_fk"] for r in records) / n * 100
    pct_bs_pass = sum(r["pass_bertscore"] for r in records) / n * 100
    pct_combined = sum(r["combined_pass"] for r in records) / n * 100

    print(f"{'='*60}")
    print(f"{'Explanation Quality Metrics':^60}")
    print(f"{'='*60}")
    print(f"  Explanations evaluated  : {n}")
    print()
    print(f"  Readability (Flesch-Kincaid Grade Level):")
    print(f"  - Mean   : {mean_fk:.1f}  (target: ≤{FK_THRESHOLD:.0f})")
    print(f"  - Median : {median_fk:.1f}")
    print(f"  - Pass rate: {pct_fk_pass:.1f}% meet threshold")
    print()
    print(f"  Faithfulness (BERTScore F1):")
    print(f"  - Mean   : {mean_bs:.3f}  (target: ≥{BERTSCORE_THRESHOLD:.2f})")
    print(f"  - Median : {median_bs:.3f}")
    print(f"  - Pass rate: {pct_bs_pass:.1f}% meet threshold")
    print()
    print(f"  Combined pass rate (both thresholds): {pct_combined:.1f}%")

    production_ready = pct_combined >= 85.0
    status = "PASS" if production_ready else "NEEDS IMPROVEMENT"
    print(f"\n  Production readiness: {status}")
    print(f"{'='*60}")

    # ── Outliers ──────────────────────────────────────────────────────────────
    worst_fk = sorted(records, key=lambda r: r["fk_grade"], reverse=True)[:3]
    worst_bs = sorted(records, key=lambda r: r["bertscore_f1"])[:3]

    os.makedirs(os.path.dirname(OUTPUT_OUTLIERS) or ".", exist_ok=True)
    with open(OUTPUT_OUTLIERS, "w", encoding="utf-8") as f:
        f.write("EXPLANATION OUTLIERS — TrialMatch AI M3\n")
        f.write("=" * 60 + "\n\n")

        f.write("TOP 3 WORST FK GRADE (too complex for patients)\n")
        f.write("-" * 60 + "\n")
        for r in worst_fk:
            f.write(f"Topic {r['topic_id']} / {r['nct_id']}  FK={r['fk_grade']:.1f}\n")
            f.write(r["explanation_text"][:400] + "\n\n")

        f.write("\nTOP 3 WORST BERTSCORE (potential hallucination risk)\n")
        f.write("-" * 60 + "\n")
        for r in worst_bs:
            f.write(
                f"Topic {r['topic_id']} / {r['nct_id']}  "
                f"BERTScore F1={r['bertscore_f1']:.3f}\n"
            )
            f.write(r["explanation_text"][:400] + "\n\n")

    print(f"\n  Outliers saved → {OUTPUT_OUTLIERS}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    fieldnames = [
        "topic_id", "nct_id", "explanation_length_chars", "explanation_text",
        "fk_grade", "bertscore_f1", "pass_fk", "pass_bertscore", "combined_pass",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"  Detailed CSV saved → {OUTPUT_CSV}")


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
