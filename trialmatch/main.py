#!/usr/bin/env python3
"""
TrialMatch AI — CLI entrypoint

Usage examples:
  python main.py --patient "58-year-old male with Stage IIIB NSCLC..."
  python main.py --patient-file data/sample_patients/patient_01.txt
  python main.py --patient-file data/sample_patients/patient_01.txt --output results/patient_01.json
  python main.py --eval-synthetic
  python main.py --benchmark --data-dir data/n2c2/
  python main.py --patient-file data/sample_patients/patient_01.txt --compare-models
  python main.py --clear-cache
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time

# Ensure trialmatch root is on the path when invoked from inside the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CACHE_DIR, MAX_TRIALS_TO_MATCH, SAMPLE_PATIENTS_DIR
from eval.benchmark import (
    print_n2c2_summary,
    print_synthetic_summary,
    run_n2c2_benchmark,
    run_synthetic_benchmark,
)
from pipeline.extractor import extract_patient_profile
from pipeline.explainer import generate_all_cards
from pipeline.matcher import ClaudeMatcher, GPT4oMatcher
from pipeline.models import PatientProfile
from pipeline.ranker import rank_trials
from pipeline.retriever import retrieve_trials

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Timer helper ──────────────────────────────────────────────────────────────

class _Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._start
        logger.info(f"[timing] {self.label}: {elapsed:.2f}s")


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    patient_text: str,
    matcher=None,
    label: str = "Claude",
) -> list[dict]:
    if matcher is None:
        matcher = ClaudeMatcher()

    print(f"\n{'='*60}")
    print(f"TrialMatch AI  |  Matcher: {label}")
    print(f"{'='*60}\n")

    # Stage 1 — Extract
    with _Timer("extraction"):
        profile: PatientProfile = extract_patient_profile(patient_text)
    print(f"Patient profile extracted:")
    print(f"  Age:        {profile.age}")
    print(f"  Conditions: {profile.conditions or '(none identified)'}")
    print(f"  Stage:      {profile.stage or '(not specified)'}")
    print(f"  Location:   {profile.location or '(not specified)'}")
    print()

    # Stage 2 — Retrieve
    condition = profile.conditions[0] if profile.conditions else patient_text.split()[0]
    with _Timer("retrieval"):
        trials = retrieve_trials(condition)
    print(f"Trials retrieved: {len(trials)}\n")

    # Stage 3 — Match
    with _Timer("matching"):
        match_results = matcher.match_trials(profile, trials)
    print(f"Trials matched: {len(match_results)}\n")

    # Stage 4 — Rank
    with _Timer("ranking"):
        ranked = rank_trials(match_results)
    print(f"Trials ranked (top {len(ranked)} returned):\n")

    # Stage 5 — Explain
    with _Timer("explanation"):
        cards = generate_all_cards(ranked)

    # Print cards
    for i, card in enumerate(cards, 1):
        print(f"{'─'*60}")
        print(f"Match #{i}  |  NCT: {card['nct_id']}  |  "
              f"Score: {card['match_score']:.2f}  |  FK Grade: {card['fk_grade']:.1f}")
        print(f"{'─'*60}")
        print(card["card_text"])
        print()

    return cards


# ── Output serialisation ──────────────────────────────────────────────────────

def save_results(cards: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cards, f, indent=2)
    logger.info(f"Results saved to {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trialmatch",
        description="TrialMatch AI — Agentic clinical trial matching with Claude",
    )

    input_group = p.add_mutually_exclusive_group()
    input_group.add_argument(
        "--patient",
        metavar="TEXT",
        help="Patient description as a free-text string",
    )
    input_group.add_argument(
        "--patient-file",
        metavar="PATH",
        help="Path to a .txt file containing the patient description",
    )
    input_group.add_argument(
        "--eval-synthetic",
        action="store_true",
        help="Run synthetic benchmark on all patients in data/sample_patients/",
    )
    input_group.add_argument(
        "--benchmark",
        action="store_true",
        help="Run n2c2 benchmark (requires --data-dir)",
    )
    input_group.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete all cached trial API responses",
    )

    p.add_argument(
        "--output",
        metavar="PATH",
        help="Save pipeline output to a JSON file at PATH",
    )
    p.add_argument(
        "--data-dir",
        metavar="DIR",
        help="n2c2 data directory (required with --benchmark)",
    )
    p.add_argument(
        "--compare-models",
        action="store_true",
        help="Run both Claude and GPT-4o matchers and compare results",
    )

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Clear cache ────────────────────────────────────────────────────────────
    if args.clear_cache:
        if os.path.isdir(CACHE_DIR):
            shutil.rmtree(CACHE_DIR)
            os.makedirs(CACHE_DIR, exist_ok=True)
            print(f"Cache cleared: {CACHE_DIR}")
        else:
            print("Cache directory does not exist; nothing to clear.")
        return

    # ── Synthetic eval ─────────────────────────────────────────────────────────
    if args.eval_synthetic:
        print("Running synthetic benchmark...")
        results = run_synthetic_benchmark()
        print_synthetic_summary(results)
        return

    # ── n2c2 benchmark ─────────────────────────────────────────────────────────
    if args.benchmark:
        if not args.data_dir:
            parser.error("--benchmark requires --data-dir")
        print(f"Running n2c2 benchmark on {args.data_dir}...")
        results = run_n2c2_benchmark(args.data_dir)
        print_n2c2_summary(results)
        return

    # ── Single patient run ─────────────────────────────────────────────────────
    patient_text = None
    if args.patient:
        patient_text = args.patient
    elif args.patient_file:
        with open(args.patient_file) as f:
            patient_text = f.read().strip()
    else:
        parser.print_help()
        sys.exit(0)

    if args.compare_models:
        # Run Claude
        claude_cards = run_pipeline(patient_text, ClaudeMatcher(), label="Claude")
        # Run GPT-4o
        try:
            gpt_cards = run_pipeline(patient_text, GPT4oMatcher(), label="GPT-4o")
            print("\n--- MODEL COMPARISON ---")
            print(f"{'NCT ID':<15} {'Claude Score':>14} {'GPT-4o Score':>13}")
            for cc, gc in zip(claude_cards, gpt_cards):
                print(
                    f"{cc['nct_id']:<15} {cc['match_score']:>14.3f} {gc['match_score']:>13.3f}"
                )
        except RuntimeError as e:
            print(f"\n[Warning] GPT-4o comparison skipped: {e}")
            gpt_cards = []
        if args.output:
            save_results({"claude": claude_cards, "gpt4o": gpt_cards}, args.output)
    else:
        cards = run_pipeline(patient_text, ClaudeMatcher(), label="Claude")
        if args.output:
            save_results(cards, args.output)


if __name__ == "__main__":
    main()
