#!/usr/bin/env python3
"""
Evaluate explanation quality (Flesch-Kincaid + BERTScore).
Saves CSV to trialmatch/results/explanation_metrics.csv.

Usage:
    python scripts/evaluate_explanations.py                          # uses run_r1.json
    python scripts/evaluate_explanations.py --run results/run_r3.json
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trialmatch"))
os.chdir(ROOT / "trialmatch")

import evaluate_explanations as _ee  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/run_r1.json")
    args = p.parse_args()
    _ee.evaluate(args.run)
