"""
Benchmark runners for TrialMatch AI.

run_synthetic_benchmark  — end-to-end eval on sample_patients/, FK + BERTScore
run_trec_benchmark       — TREC Clinical Trials 2021 retrieval eval, P@5 + NDCG@5
print_synthetic_summary  — formatted table for synthetic results
print_trec_summary       — formatted table for TREC results

NOTE: main.py calls run_n2c2_benchmark / print_n2c2_summary for the --benchmark flag.
Those are aliased here to the TREC versions for backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

from config import SAMPLE_PATIENTS_DIR
from eval.metrics import (
    bertscore_consistency,
    flesch_kincaid_grade,
    ndcg_at_k,
    precision_at_k,
)
from pipeline.extractor import extract_patient_profile
from pipeline.explainer import generate_all_cards
from pipeline.matcher import ClaudeMatcher
from pipeline.ranker import rank_trials
from pipeline.retriever import retrieve_trials

logger = logging.getLogger(__name__)

# ── TREC data URLs (TrialGPT FTP mirror, publicly accessible) ────────────────

_TREC_2021_CORPUS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl"
)
_TREC_2021_TOPICS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_topics.json"
)
_TREC_2021_QRELS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_qrels.json"
)

_TREC_DATA_DIR = Path("data/trec_2021")


# ── TREC data download helpers ────────────────────────────────────────────────

def _download_if_missing(url: str, dest: Path) -> bool:
    """Download url to dest if dest doesn't exist. Returns True on success."""
    if dest.exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} → {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded {dest.name} ({dest.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")
        return False


def _ensure_trec_data() -> bool:
    """
    Download TREC 2021 corpus, topics, and qrels if not already present.
    Returns True if all files are available.
    """
    files = {
        "corpus": (_TREC_2021_CORPUS_URL, _TREC_DATA_DIR / "corpus.jsonl"),
        "topics": (_TREC_2021_TOPICS_URL, _TREC_DATA_DIR / "topics.json"),
        "qrels":  (_TREC_2021_QRELS_URL,  _TREC_DATA_DIR / "qrels.json"),
    }
    all_ok = True
    for name, (url, dest) in files.items():
        if not _download_if_missing(url, dest):
            logger.error(f"Could not obtain TREC {name} file.")
            all_ok = False
    return all_ok


# ── TREC data loaders ─────────────────────────────────────────────────────────

def _load_topics() -> dict[str, str]:
    """Return {topic_id: patient_note_text}."""
    path = _TREC_DATA_DIR / "topics.json"
    with open(path) as f:
        raw = json.load(f)
    # TREC topics format: list of {number, text} or dict keyed by topic id
    if isinstance(raw, list):
        return {str(t["number"]): t["text"] for t in raw}
    return {str(k): v for k, v in raw.items()}


def _load_qrels() -> dict[str, dict[str, int]]:
    """
    Return {topic_id: {nct_id: relevance_grade}}.
    Relevance grades: 0 = not relevant, 1 = partially relevant, 2 = highly relevant.
    """
    path = _TREC_DATA_DIR / "qrels.json"
    with open(path) as f:
        return json.load(f)


# ── TREC benchmark runner ─────────────────────────────────────────────────────

def run_trec_benchmark(max_topics: int = 10) -> list[dict]:
    """
    Evaluate TrialMatch AI against the TREC Clinical Trials 2021 benchmark.

    For each patient topic:
      1. Extract patient profile from the topic note
      2. Retrieve trials via ClinicalTrials.gov API
      3. Match and rank with ClaudeMatcher
      4. Compute Precision@5 and NDCG@5 vs. TREC relevance judgments

    Args:
        max_topics: number of TREC topics to evaluate (default 10 to limit API cost)

    Returns:
        list of per-topic result dicts
    """
    print("\nChecking TREC 2021 data...")
    if not _ensure_trec_data():
        print(
            "\n[Error] Could not download TREC 2021 data from FTP mirror.\n"
            "Manual download instructions:\n"
            "  Corpus: https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl\n"
            "  Topics: https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_topics.json\n"
            "  Qrels:  https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_qrels.json\n"
            f"Place all three files in: {_TREC_DATA_DIR.resolve()}\n"
        )
        return []

    topics = _load_topics()
    qrels = _load_qrels()
    matcher = ClaudeMatcher()
    results = []

    topic_ids = list(topics.keys())[:max_topics]
    print(f"Evaluating {len(topic_ids)} TREC topics (of {len(topics)} total)...\n")

    for topic_id in topic_ids:
        note_text = topics[topic_id]
        topic_qrels = qrels.get(topic_id, {})  # {nct_id: grade}
        relevant_ncts = [nct for nct, grade in topic_qrels.items() if grade > 0]

        logger.info(f"Topic {topic_id}: {len(relevant_ncts)} relevant trials in qrels")

        try:
            profile = extract_patient_profile(note_text)
            condition = profile.conditions[0] if profile.conditions else note_text.split()[0]
            trials = retrieve_trials(condition)
            match_results = matcher.match_trials(profile, trials)
            ranked = rank_trials(match_results)

            retrieved_ncts = [m.trial.nct_id for m in ranked]

            p5 = precision_at_k(retrieved_ncts, relevant_ncts, k=5)
            ndcg5 = ndcg_at_k(retrieved_ncts, topic_qrels, k=5)

            result = {
                "topic_id": topic_id,
                "retrieved_ncts": retrieved_ncts,
                "relevant_ncts": relevant_ncts,
                "precision_at_5": p5,
                "ndcg_at_5": ndcg5,
                "n_retrieved": len(retrieved_ncts),
                "n_relevant_in_qrels": len(relevant_ncts),
            }
        except Exception as e:
            logger.error(f"Topic {topic_id} failed: {e}")
            result = {
                "topic_id": topic_id,
                "error": str(e),
                "precision_at_5": 0.0,
                "ndcg_at_5": 0.0,
            }

        results.append(result)
        print(
            f"  Topic {topic_id}: P@5={result['precision_at_5']:.3f}  "
            f"NDCG@5={result['ndcg_at_5']:.3f}"
        )

    return results


def print_trec_summary(results: list[dict]) -> None:
    """Print a formatted summary table for TREC benchmark results."""
    if not results:
        print("\nNo TREC results to summarize.")
        return

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    print(f"\n{'='*65}")
    print(f"{'TREC Clinical Trials 2021 Benchmark':^65}")
    print(f"{'='*65}")
    print(f"{'Topic':<10} {'Retrieved':>10} {'Relevant':>10} {'P@5':>8} {'NDCG@5':>8}")
    print(f"{'-'*65}")

    for r in results:
        if "error" in r:
            print(f"{r['topic_id']:<10} {'ERROR':<10} {r['error'][:30]}")
            continue
        flag = " ⚠️" if r["precision_at_5"] == 0.0 else ""
        print(
            f"{r['topic_id']:<10} {r['n_retrieved']:>10} "
            f"{r['n_relevant_in_qrels']:>10} "
            f"{r['precision_at_5']:>8.3f} {r['ndcg_at_5']:>8.3f}{flag}"
        )

    if valid:
        mean_p5 = sum(r["precision_at_5"] for r in valid) / len(valid)
        mean_ndcg5 = sum(r["ndcg_at_5"] for r in valid) / len(valid)
        print(f"{'-'*65}")
        print(f"{'MEAN':<10} {' ':>10} {' ':>10} {mean_p5:>8.3f} {mean_ndcg5:>8.3f}")
        print(f"{'='*65}")
        print(f"\nTopics evaluated: {len(valid)}  |  Errors: {len(errors)}")
        print(f"Target: P@5 > ClinicalTrials.gov keyword baseline")

    if errors:
        print(f"\nFailed topics: {[r['topic_id'] for r in errors]}")


# ── Synthetic benchmark ───────────────────────────────────────────────────────

def run_synthetic_benchmark() -> list[dict]:
    """
    End-to-end eval on all patient .txt files in data/sample_patients/.
    Computes FK grade and BERTScore for each output card.
    Returns list of per-patient result dicts.
    """
    patient_dir = Path(SAMPLE_PATIENTS_DIR)
    patient_files = sorted(patient_dir.glob("*.txt"))

    if not patient_files:
        print(f"No patient files found in {patient_dir}")
        return []

    matcher = ClaudeMatcher()
    results = []

    for pf in patient_files:
        patient_text = pf.read_text().strip()
        logger.info(f"Synthetic eval: {pf.name}")

        try:
            profile = extract_patient_profile(patient_text)
            condition = profile.conditions[0] if profile.conditions else patient_text.split()[0]
            trials = retrieve_trials(condition)
            match_results = matcher.match_trials(profile, trials)
            ranked = rank_trials(match_results)
            cards = generate_all_cards(ranked)

            fk_scores = [c["fk_grade"] for c in cards]
            bert_scores = []
            for card, match in zip(cards, ranked):
                bs = bertscore_consistency(
                    card["card_text"],
                    match.trial.eligibility_criteria_raw,
                )
                bert_scores.append(bs)

            results.append({
                "patient_file": pf.name,
                "n_trials_returned": len(cards),
                "cards": cards,
                "mean_fk": sum(fk_scores) / len(fk_scores) if fk_scores else 0.0,
                "mean_bertscore": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
            })

        except Exception as e:
            logger.error(f"{pf.name} failed: {e}")
            results.append({
                "patient_file": pf.name,
                "error": str(e),
                "n_trials_returned": 0,
                "mean_fk": 0.0,
                "mean_bertscore": 0.0,
            })

    return results


def print_synthetic_summary(results: list[dict]) -> None:
    """Print formatted table of synthetic benchmark results."""
    if not results:
        print("No synthetic results to summarize.")
        return

    print(f"\n{'='*70}")
    print(f"{'Synthetic Patient Benchmark':^70}")
    print(f"{'='*70}")
    print(f"{'Patient':<20} {'Trials':>7} {'Mean FK':>9} {'BERTScore':>11}")
    print(f"{'-'*70}")

    for r in results:
        if "error" in r:
            print(f"{r['patient_file']:<20} {'ERROR':<7}  {r['error'][:35]}")
            continue
        fk_flag = " ⚠️" if r["mean_fk"] > 8 else ""
        print(
            f"{r['patient_file']:<20} {r['n_trials_returned']:>7} "
            f"{r['mean_fk']:>9.2f}{fk_flag} {r['mean_bertscore']:>11.3f}"
        )

    valid = [r for r in results if "error" not in r]
    if valid:
        mean_fk = sum(r["mean_fk"] for r in valid) / len(valid)
        mean_bs = sum(r["mean_bertscore"] for r in valid) / len(valid)
        print(f"{'-'*70}")
        print(f"{'MEAN':<20} {' ':>7} {mean_fk:>9.2f} {mean_bs:>11.3f}")
        print(f"{'='*70}")
        print(f"\nFK target: ≤ 8.0  |  BERTScore target: ≥ 0.85")
        fk_violations = sum(1 for r in valid if r["mean_fk"] > 8)
        if fk_violations:
            print(f"⚠️  {fk_violations} patient(s) exceed FK grade target")


# ── Backward-compatibility aliases (main.py imports these names) ──────────────

run_n2c2_benchmark = run_trec_benchmark
print_n2c2_summary = print_trec_summary
