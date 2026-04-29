#!/usr/bin/env python3
"""
Generate comparison table from all 4 experimental runs.

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --results-dir trialmatch/results
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trialmatch"))
os.chdir(ROOT / "trialmatch")

import compare_runs as _cr  # noqa: E402

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    _cr.compare(args.results_dir)
