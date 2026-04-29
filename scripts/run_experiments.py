#!/usr/bin/env python3
"""
M3 experiment runner — delegates to trialmatch/run_experiments.py.

Usage:
    python scripts/run_experiments.py                          # full 26-40
    python scripts/run_experiments.py --topic-range 26 28     # smoke test
    python scripts/run_experiments.py --fresh                  # delete old runs first
"""
import argparse
import os
import sys
from pathlib import Path

# Make trialmatch importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trialmatch"))

import run_experiments as _runner  # noqa: E402


def main():
    p = argparse.ArgumentParser(parents=[argparse.ArgumentParser(add_help=False)])
    p.add_argument("--topic-range", nargs=2, type=int, metavar=("START", "END"), default=[26, 40])
    p.add_argument("--fresh", action="store_true", help="Delete existing run files before starting")
    args, _ = p.parse_known_args()

    if args.fresh:
        results_dir = ROOT / "trialmatch" / "results"
        for fname in ["run_baseline.json", "run_r1.json", "run_r2.json", "run_r3.json"]:
            path = results_dir / fname
            if path.exists():
                path.unlink()
                print(f"Deleted {path}")

    _runner.run_all(topic_range=tuple(args.topic_range))


if __name__ == "__main__":
    os.chdir(ROOT / "trialmatch")  # benchmark expects CWD = trialmatch/
    main()
