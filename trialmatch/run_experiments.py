"""
Run all 4 experimental configurations for M3 and save results.

Configs:
  Baseline  — TF-IDF,      no critic,  topics 26-40
  R1        — BiomedBERT,  no critic,  topics 26-40
  R2        — TF-IDF,      + critic,   topics 26-40
  R3        — BiomedBERT,  + critic,   topics 26-40

Usage:
  cd trialmatch
  python run_experiments.py
  python run_experiments.py --topic-range 26 30   # quick smoke-test
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# Allow running from trialmatch/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval.benchmark import run_trec_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOPIC_RANGE = (26, 40)
RESULTS_DIR = Path("results")

CONFIGS = [
    {
        "name": "Baseline",
        "retriever_type": "tfidf",
        "use_critic": False,
        "output_file": "results/run_baseline.json",
        "checkpoint": "results/ckpt_baseline.json",
    },
    {
        "name": "R1 (BiomedBERT)",
        "retriever_type": "biomedbert",
        "use_critic": False,
        "output_file": "results/run_r1.json",
        "checkpoint": "results/ckpt_r1.json",
    },
    {
        "name": "R2 (Critic)",
        "retriever_type": "tfidf",
        "use_critic": True,
        "output_file": "results/run_r2.json",
        "checkpoint": "results/ckpt_r2.json",
    },
    {
        "name": "R3 (Combined)",
        "retriever_type": "biomedbert",
        "use_critic": True,
        "output_file": "results/run_r3.json",
        "checkpoint": "results/ckpt_r3.json",
    },
]


def _is_complete(output_file: str, required_topics: int) -> bool:
    """Return True if the output file already has all required topics with correct config."""
    path = Path(output_file)
    if not path.exists():
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        valid = [r for r in data.get("per_topic_results", []) if "error" not in r]
        return len(valid) >= required_topics
    except Exception:
        return False


def run_all(topic_range: tuple[int, int] = TOPIC_RANGE, *, fresh: bool = False) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    summary: list[dict] = []
    n_required = topic_range[1] - topic_range[0] + 1

    if fresh:
        for cfg in CONFIGS:
            for f in [cfg["output_file"], cfg["checkpoint"]]:
                if Path(f).exists():
                    Path(f).unlink()
                    print(f"  [fresh] deleted {f}")

    print(f"\n{'='*70}")
    print(f"  TrialMatch AI — M3 Experiments  |  Topics {topic_range[0]}–{topic_range[1]}")
    print(f"{'='*70}\n")

    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] {cfg['name']}")
        print(f"  Retriever: {cfg['retriever_type']}  |  Critic: {cfg['use_critic']}")
        print(f"  Output: {cfg['output_file']}")
        print(f"  {'-'*60}")

        # Skip if already complete and not forcing fresh
        if not fresh and _is_complete(cfg["output_file"], n_required):
            print(f"  SKIPPED (already has {n_required} topics) — delete with --fresh to rerun")
            with open(cfg["output_file"]) as f:
                cached = json.load(f)
            summary.append({
                "name": cfg["name"],
                "output_file": cfg["output_file"],
                "p_at_5": cached["metrics"]["p_at_5"],
                "ndcg_at_5": cached["metrics"]["ndcg_at_5"],
                "runtime_seconds": cached.get("runtime_seconds", 0),
                "total_api_calls": cached.get("total_api_calls", 0),
                "estimated_cost_usd": cached.get("estimated_cost_usd", 0),
                "status": "ok",
            })
            continue

        wall_start = time.perf_counter()
        try:
            results = run_trec_benchmark(
                retriever_type=cfg["retriever_type"],
                use_critic=cfg["use_critic"],
                topic_range=topic_range,
                output_file=cfg["output_file"],
                generate_explanations=True,
                resume_from=cfg["checkpoint"] if not fresh else None,
            )
            wall_elapsed = time.perf_counter() - wall_start

            summary.append({
                "name": cfg["name"],
                "output_file": cfg["output_file"],
                "p_at_5": results["metrics"]["p_at_5"],
                "ndcg_at_5": results["metrics"]["ndcg_at_5"],
                "runtime_seconds": results["runtime_seconds"],
                "total_api_calls": results["total_api_calls"],
                "estimated_cost_usd": results["estimated_cost_usd"],
                "status": "ok",
            })
            print(
                f"\n  DONE: P@5={results['metrics']['p_at_5']:.3f}  "
                f"NDCG@5={results['metrics']['ndcg_at_5']:.3f}  "
                f"Runtime={wall_elapsed:.0f}s"
            )

        except Exception:
            wall_elapsed = time.perf_counter() - wall_start
            tb = traceback.format_exc()
            logger.error(f"Run '{cfg['name']}' crashed after {wall_elapsed:.0f}s:\n{tb}")
            _log_error(cfg["name"], tb)
            summary.append({
                "name": cfg["name"],
                "output_file": cfg["output_file"],
                "status": "error",
                "error": tb.splitlines()[-1],
            })
            print(f"\n  ERROR (continuing to next run): {tb.splitlines()[-1]}")

    # Print final table
    print(f"\n\n{'='*70}")
    print(f"{'All runs complete':^70}")
    print(f"{'='*70}")
    print(f"{'Run':<22} {'Status':>8} {'P@5':>8} {'NDCG@5':>8} {'Time(s)':>9} {'Cost($)':>8}")
    print(f"{'-'*70}")
    for s in summary:
        if s["status"] == "ok":
            print(
                f"{s['name']:<22} {'OK':>8} "
                f"{s['p_at_5']:>8.3f} {s['ndcg_at_5']:>8.3f} "
                f"{s['runtime_seconds']:>9.0f} {s['estimated_cost_usd']:>8.2f}"
            )
        else:
            print(f"{s['name']:<22} {'ERROR':>8}")
    print(f"{'='*70}")

    summary_path = RESULTS_DIR / "experiments_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved → {summary_path}")
    print("Run  python compare_runs.py  to generate the comparison table.\n")


def _log_error(run_name: str, tb: str) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "error_log.txt", "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"Run: {run_name}  |  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(tb)
        f.write("\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Run all 4 M3 experimental configurations")
    p.add_argument(
        "--fresh",
        action="store_true",
        default=False,
        help="Delete existing run files and start all configs from scratch",
    )
    p.add_argument(
        "--topic-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=[26, 40],
        help="Topic ID range (default: 26 40)",
    )
    args = p.parse_args()
    run_all(topic_range=tuple(args.topic_range), fresh=args.fresh)
