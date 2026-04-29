#!/usr/bin/env python3
"""
Compute confusion matrix (matcher predictions vs TREC qrels).
Saves PNG heatmap to trialmatch/results/confusion_matrix.png.

Usage:
    python scripts/compute_confusion_matrix.py                         # uses run_r1.json
    python scripts/compute_confusion_matrix.py --run results/run_r3.json
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trialmatch"))
os.chdir(ROOT / "trialmatch")

import compute_confusion_matrix as _cm  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="results/run_r1.json")
    args = p.parse_args()
    _cm.compute(args.run)
