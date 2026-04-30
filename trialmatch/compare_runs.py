"""
Load the 4 experiment result files and print a comparison table.

Usage:
  cd trialmatch
  python compare_runs.py
  python compare_runs.py --results-dir results
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RUN_FILES = {
    "Baseline":   "results/run_baseline.json",
    "BiomedBERT": "results/run_r1.json",
    "Critic":     "results/run_r2.json",
    "Combined":   "results/run_r3.json",
}

OUTPUT_CSV = "results/comparison_table.csv"


def _pct_change(baseline: float, value: float) -> str:
    if baseline == 0:
        return "N/A"
    delta = (value - baseline) / baseline * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def load_results(results_dir: str = ".") -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    for label, rel_path in RUN_FILES.items():
        path = Path(results_dir) / rel_path if results_dir != "." else Path(rel_path)
        if not path.exists():
            print(f"  [missing] {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        loaded[label] = data
    return loaded


def compare(results_dir: str = ".") -> None:
    runs = load_results(results_dir)

    if not runs:
        print("No result files found. Run  python run_experiments.py  first.")
        return

    baseline_metrics = runs.get("Baseline", {}).get("metrics", {})
    base_p5 = baseline_metrics.get("p_at_5", 0.0)
    base_ndcg5 = baseline_metrics.get("ndcg_at_5", 0.0)
    base_runtime = runs.get("Baseline", {}).get("runtime_seconds", 0.0)
    base_cost = runs.get("Baseline", {}).get("estimated_cost_usd", 0.0)

    rows: list[dict] = []
    order = ["Baseline", "BiomedBERT", "Critic", "Combined"]

    for label in order:
        if label not in runs:
            rows.append({"Run": label, "status": "missing"})
            continue
        r = runs[label]
        m = r.get("metrics", {})
        p5 = m.get("p_at_5", 0.0)
        ndcg5 = m.get("ndcg_at_5", 0.0)
        runtime = r.get("runtime_seconds", 0.0)
        cost = r.get("estimated_cost_usd", 0.0)
        n_topics = len([t for t in r.get("per_topic_results", []) if "error" not in t])

        rows.append({
            "Run": label,
            "P@5": p5,
            "NDCG@5": ndcg5,
            "Latency (s)": runtime,
            "Cost ($)": cost,
            "Topics": n_topics,
            "vs Baseline (P@5)": "-" if label == "Baseline" else _pct_change(base_p5, p5),
            "vs Baseline (NDCG)": "-" if label == "Baseline" else _pct_change(base_ndcg5, ndcg5),
        })

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'TrialMatch AI — M3 Experimental Results':^80}")
    print(f"{'='*80}")

    header = f"{'Run':<14} {'P@5':>6} {'NDCG@5':>8} {'Time(s)':>9} {'Cost($)':>8} {'vs Base (P@5)':>14} {'vs Base (NDCG)':>15}"
    print(header)
    print(f"{'-'*80}")

    for row in rows:
        if row.get("status") == "missing":
            print(f"{row['Run']:<14}  (file not found)")
            continue
        print(
            f"{row['Run']:<14} "
            f"{row['P@5']:>6.3f} "
            f"{row['NDCG@5']:>8.3f} "
            f"{row['Latency (s)']:>9.1f} "
            f"{row['Cost ($)']:>8.2f} "
            f"{row['vs Baseline (P@5)']:>14} "
            f"{row['vs Baseline (NDCG)']:>15}"
        )

    print(f"{'='*80}")

    # Critic stats if available
    for label in order:
        if label not in runs:
            continue
        cs = runs[label].get("critic_stats", {})
        if cs.get("total_reviewed", 0) > 0:
            ar = cs.get("agreement_rate")
            ar_str = f"{ar:.1%}" if ar is not None else "N/A"
            print(
                f"\n  {label} critic stats: "
                f"agreement={ar_str}  "
                f"disagreements={cs.get('total_disagreements', 0)}  "
                f"overrides={cs.get('overrides', 0)}  "
                f"flags={cs.get('flags', 0)}"
            )

    # ── Save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    csv_rows = [r for r in rows if r.get("status") != "missing"]
    if csv_rows:
        fieldnames = [
            "Run", "P@5", "NDCG@5", "Latency (s)", "Cost ($)",
            "Topics", "vs Baseline (P@5)", "vs Baseline (NDCG)",
        ]
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Compare M3 experiment results")
    p.add_argument("--results-dir", default=".", help="Directory containing results/")
    args = p.parse_args()
    compare(args.results_dir)
