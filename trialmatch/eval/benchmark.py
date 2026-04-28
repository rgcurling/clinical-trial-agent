"""
Benchmark runners for TrialMatch AI.

run_synthetic_benchmark  — end-to-end eval on sample_patients/, FK + BERTScore
run_trec_benchmark       — TREC Clinical Trials 2021 retrieval eval, P@5 + NDCG@5
print_synthetic_summary  — formatted table for synthetic results
print_trec_summary       — formatted table for TREC results

CLI usage (standalone):
  python eval/benchmark.py --retriever tfidf --topic-range 26 40
  python eval/benchmark.py --retriever biomedbert --use-critic --topic-range 26 40 \
      --output results/run_r3.json
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from config import SAMPLE_PATIENTS_DIR
from eval.metrics import (
    bertscore_consistency,
    flesch_kincaid_grade,
    ndcg_at_k,
    precision_at_k,
)
from pipeline.extractor import extract_patient_profile
from pipeline.explainer import generate_all_cards
from pipeline.matcher import ClaudeMatcher, critic_review, resolve_discrepancies
from pipeline.ranker import rank_trials
from pipeline.retriever import TfidfRetriever, BiomedBERTRetriever, retrieve_trials

logger = logging.getLogger(__name__)

# ── TREC data paths ───────────────────────────────────────────────────────────

_TREC_DATA_DIR = Path("data/trec_2021")
_TREC_2021_CORPUS_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl"
)
_TREC_2021_TOPICS_URL = "https://trec.nist.gov/data/trials/topics2021.xml"
_TREC_2021_QRELS_URL = "https://trec.nist.gov/data/trials/qrels2021.txt"

# Rough cost estimate per Claude API call (Sonnet 4, mixed input/output tokens)
_COST_PER_MATCH_CALL = 0.010   # ~1500 in + 400 out tokens
_COST_PER_CRITIC_CALL = 0.009  # ~2000 in + 200 out tokens
_COST_PER_EXPLAIN_CALL = 0.011  # ~800 in + 600 out tokens


# ── TREC data helpers ─────────────────────────────────────────────────────────

def _download_if_missing(url: str, dest: Path) -> bool:
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
    files = {
        "corpus": (_TREC_2021_CORPUS_URL, _TREC_DATA_DIR / "corpus.jsonl"),
        "topics": (_TREC_2021_TOPICS_URL, _TREC_DATA_DIR / "topics2021.xml"),
        "qrels":  (_TREC_2021_QRELS_URL,  _TREC_DATA_DIR / "qrels2021.txt"),
    }
    return all(_download_if_missing(url, dest) for _, (url, dest) in files.items())


def _load_topics(data_dir: Path) -> dict[str, str]:
    path = data_dir / "topics2021.xml"
    tree = ET.parse(path)
    root = tree.getroot()
    topics = {}
    for topic in root.findall("topic"):
        number = topic.get("number", "").strip()
        text = topic.text.strip() if topic.text else ""
        if number:
            topics[number] = text
    return topics


def _load_qrels(data_dir: Path) -> dict[str, dict[str, int]]:
    path = data_dir / "qrels2021.txt"
    qrels: dict[str, dict[str, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            topic_id, _iter, nct_id, relevance = parts[0], parts[1], parts[2], parts[3]
            qrels.setdefault(topic_id, {})[nct_id] = int(relevance)
    return qrels


# ── TREC benchmark ────────────────────────────────────────────────────────────

def run_trec_benchmark(
    max_topics: int = 10,
    *,
    retriever_type: str = "tfidf",
    use_critic: bool = False,
    topic_range: Optional[tuple[int, int]] = None,
    output_file: Optional[str] = None,
    corpus_path: Optional[Path] = None,
    generate_explanations: bool = True,
) -> dict:
    """
    Evaluate TrialMatch AI against TREC Clinical Trials 2021.

    Args:
        max_topics:           max topics to evaluate (ignored when topic_range is set)
        retriever_type:       'tfidf' or 'biomedbert'
        use_critic:           run second Claude reviewer after each match
        topic_range:          (start, end) inclusive topic-ID filter; overrides max_topics
        output_file:          save full results dict as JSON to this path
        corpus_path:          override default corpus location
        generate_explanations: generate patient-facing cards for top-5 (needed for Part 5)

    Returns dict with keys: config, metrics, per_topic_results, critic_stats,
    runtime_seconds, total_api_calls, estimated_cost_usd.
    """
    start_time = time.perf_counter()

    print("\nChecking TREC 2021 data...")
    if not _ensure_trec_data():
        print(
            "\n[Error] Could not obtain TREC 2021 data.\n"
            f"Place corpus.jsonl, topics2021.xml, qrels2021.txt in: "
            f"{_TREC_DATA_DIR.resolve()}\n"
        )
        return {}

    _corpus = corpus_path or (_TREC_DATA_DIR / "corpus.jsonl")
    topics = _load_topics(_TREC_DATA_DIR)
    qrels = _load_qrels(_TREC_DATA_DIR)

    # ── Select topics ─────────────────────────────────────────────────────────
    all_ids = list(topics.keys())
    if topic_range is not None:
        lo, hi = topic_range
        selected_ids = [t for t in all_ids if lo <= int(t) <= hi]
    else:
        selected_ids = all_ids[:max_topics]

    print(
        f"Retriever: {retriever_type}  |  Critic: {use_critic}  |  "
        f"Topics: {selected_ids[0]}–{selected_ids[-1]} ({len(selected_ids)} total)\n"
    )

    # ── Instantiate retriever ─────────────────────────────────────────────────
    if retriever_type == "biomedbert":
        retriever = BiomedBERTRetriever(_corpus)
    else:
        retriever = TfidfRetriever(_corpus)

    matcher = ClaudeMatcher()

    # ── Progress bar ──────────────────────────────────────────────────────────
    try:
        from tqdm import tqdm
        topic_iter = tqdm(selected_ids, desc="Topics", unit="topic")
    except ImportError:
        topic_iter = selected_ids

    per_topic_results: list[dict] = []
    api_calls = 0
    critic_agreements = 0
    critic_disagreements = 0
    critic_overrides = 0
    critic_flags = 0
    running_p5_sum = 0.0

    os.makedirs("results", exist_ok=True)

    for idx, topic_id in enumerate(topic_iter):
        note_text = topics[topic_id]
        topic_qrels = qrels.get(topic_id, {})
        relevant_ncts = [nct for nct, grade in topic_qrels.items() if grade > 0]

        try:
            profile = extract_patient_profile(note_text)
            api_calls += 1  # extractor may call Claude

            trials = retriever.retrieve(note_text, top_k=20)
            match_results = matcher.match_trials(profile, trials)
            api_calls += len(match_results)

            # Apply critic if requested
            if use_critic:
                reviewed: list = []
                for mr in match_results:
                    agent2 = critic_review(profile, mr.trial, mr)
                    api_calls += 1
                    resolved = resolve_discrepancies(
                        mr, agent2, topic_id=topic_id, nct_id=mr.trial.nct_id
                    )
                    reviewed.append(resolved)
                    if agent2.get("agree", True):
                        critic_agreements += 1
                    else:
                        critic_disagreements += 1
                        rec = agent2.get("recommendation", "")
                        if rec == "override":
                            critic_overrides += 1
                        elif rec == "flag_uncertain":
                            critic_flags += 1
                match_results = reviewed

            ranked = rank_trials(match_results)
            retrieved_ncts = [m.trial.nct_id for m in ranked]

            p5 = precision_at_k(retrieved_ncts, relevant_ncts, k=5)
            ndcg5 = ndcg_at_k(retrieved_ncts, topic_qrels, k=5)
            running_p5_sum += p5

            # Generate top-5 explanations
            top5_details: list[dict] = []
            if generate_explanations:
                cards = generate_all_cards(ranked)
                api_calls += len(cards)
            else:
                cards = [None] * len(ranked)

            for rank_i, (mr, card) in enumerate(zip(ranked, cards), 1):
                nct = mr.trial.nct_id
                detail = {
                    "rank": rank_i,
                    "nct_id": nct,
                    "title": mr.trial.title,
                    "overall_score": mr.overall_score,
                    "match_score": mr.match_score,
                    "met_criteria": mr.met_criteria,
                    "failed_criteria": mr.failed_criteria,
                    "uncertain_criteria": mr.uncertain_criteria,
                    "hard_exclusion": mr.hard_exclusion,
                    "exclusion_reason": mr.exclusion_reason,
                    "reasoning": mr.reasoning,
                    "critic_flagged": mr.critic_flagged,
                    "uncertain": mr.uncertain,
                    "trec_grade": topic_qrels.get(nct, -1),
                    "eligibility_criteria": mr.trial.eligibility_criteria_raw,
                }
                if card is not None:
                    detail["explanation"] = card.get("card_text", "")
                    detail["fk_grade"] = card.get("fk_grade", 0.0)
                top5_details.append(detail)

            result = {
                "topic_id": topic_id,
                "topic_text": note_text,
                "retrieved_ncts": retrieved_ncts,
                "relevant_ncts": relevant_ncts,
                "precision_at_5": p5,
                "ndcg_at_5": ndcg5,
                "n_retrieved": len(retrieved_ncts),
                "n_relevant_in_qrels": len(relevant_ncts),
                "top5": top5_details,
            }

        except Exception as exc:
            logger.error(f"Topic {topic_id} failed: {exc}")
            result = {
                "topic_id": topic_id,
                "error": str(exc),
                "precision_at_5": 0.0,
                "ndcg_at_5": 0.0,
                "top5": [],
            }

        per_topic_results.append(result)

        # Live progress line
        running_mean = running_p5_sum / (idx + 1)
        p5_val = result["precision_at_5"]
        print(
            f"  Topic {topic_id:>3}: P@5={p5_val:.3f}  "
            f"NDCG@5={result.get('ndcg_at_5', 0):.3f}  "
            f"(running P@5={running_mean:.3f})"
        )

        # Checkpoint every 10 topics
        if (idx + 1) % 10 == 0:
            ckpt = f"results/checkpoint_topic{topic_id}.json"
            with open(ckpt, "w") as f:
                json.dump(per_topic_results, f, indent=2)
            logger.info(f"Checkpoint saved → {ckpt}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    valid = [r for r in per_topic_results if "error" not in r]
    mean_p5 = sum(r["precision_at_5"] for r in valid) / len(valid) if valid else 0.0
    mean_ndcg5 = sum(r["ndcg_at_5"] for r in valid) / len(valid) if valid else 0.0
    runtime = time.perf_counter() - start_time

    n_critic_total = critic_agreements + critic_disagreements
    agreement_rate = (critic_agreements / n_critic_total) if n_critic_total > 0 else None
    avg_score_delta = None  # computed in compare_runs.py from disagreements log

    results_dict = {
        "config": {
            "retriever": retriever_type,
            "use_critic": use_critic,
            "topic_range": list(topic_range) if topic_range else None,
            "max_topics": max_topics,
            "generate_explanations": generate_explanations,
        },
        "metrics": {
            "p_at_5": round(mean_p5, 4),
            "ndcg_at_5": round(mean_ndcg5, 4),
        },
        "per_topic_results": per_topic_results,
        "critic_stats": {
            "total_reviewed": n_critic_total,
            "agreement_rate": round(agreement_rate, 4) if agreement_rate is not None else None,
            "total_disagreements": critic_disagreements,
            "overrides": critic_overrides,
            "flags": critic_flags,
        },
        "runtime_seconds": round(runtime, 1),
        "total_api_calls": api_calls,
        "estimated_cost_usd": round(
            api_calls * (
                _COST_PER_MATCH_CALL
                + (_COST_PER_CRITIC_CALL if use_critic else 0.0)
                + (_COST_PER_EXPLAIN_CALL if generate_explanations else 0.0)
            ) / 3,   # average of the three types
            2,
        ),
    }

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"\nResults saved → {output_file}")

    return results_dict


def print_trec_summary(results) -> None:
    """Print a formatted summary table. Accepts list[dict] or the new results dict."""
    # Handle both old list format and new dict format
    if isinstance(results, dict):
        per_topic = results.get("per_topic_results", [])
        metrics = results.get("metrics", {})
    else:
        per_topic = results
        metrics = {}

    if not per_topic:
        print("\nNo TREC results to summarize.")
        return

    valid = [r for r in per_topic if "error" not in r]
    errors = [r for r in per_topic if "error" in r]

    print(f"\n{'='*65}")
    print(f"{'TREC Clinical Trials 2021 Benchmark':^65}")
    print(f"{'='*65}")
    print(f"{'Topic':<10} {'Retrieved':>10} {'Relevant':>10} {'P@5':>8} {'NDCG@5':>8}")
    print(f"{'-'*65}")

    for r in per_topic:
        if "error" in r:
            print(f"{r['topic_id']:<10} {'ERROR':>10}  {r['error'][:30]}")
            continue
        flag = " *" if r["precision_at_5"] == 0.0 else ""
        print(
            f"{r['topic_id']:<10} {r.get('n_retrieved', 0):>10} "
            f"{r.get('n_relevant_in_qrels', 0):>10} "
            f"{r['precision_at_5']:>8.3f} {r['ndcg_at_5']:>8.3f}{flag}"
        )

    if valid:
        mean_p5 = metrics.get("p_at_5") or sum(r["precision_at_5"] for r in valid) / len(valid)
        mean_ndcg5 = metrics.get("ndcg_at_5") or sum(r["ndcg_at_5"] for r in valid) / len(valid)
        print(f"{'-'*65}")
        print(f"{'MEAN':<10} {' ':>10} {' ':>10} {mean_p5:>8.3f} {mean_ndcg5:>8.3f}")
        print(f"{'='*65}")
        print(f"\nTopics evaluated: {len(valid)}  |  Errors: {len(errors)}")

    if errors:
        print(f"Failed topics: {[r['topic_id'] for r in errors]}")


# ── Synthetic benchmark ───────────────────────────────────────────────────────

def run_synthetic_benchmark() -> list[dict]:
    """End-to-end eval on all patient .txt files in data/sample_patients/."""
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
            trials = retrieve_trials(condition, profile=profile)
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
        fk_flag = " *" if r["mean_fk"] > 8 else ""
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
            print(f"  {fk_violations} patient(s) exceed FK grade target")


# ── Backward-compatibility aliases ───────────────────────────────────────────

run_n2c2_benchmark = run_trec_benchmark
print_n2c2_summary = print_trec_summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog="benchmark",
        description="Run TREC Clinical Trials 2021 benchmark for TrialMatch AI",
    )
    p.add_argument(
        "--retriever",
        choices=["tfidf", "biomedbert"],
        default="tfidf",
        help="Retrieval method (default: tfidf)",
    )
    p.add_argument(
        "--use-critic",
        action="store_true",
        default=False,
        help="Enable critic agent (second Claude reviewer per trial)",
    )
    p.add_argument(
        "--topic-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive topic-ID range to evaluate, e.g. --topic-range 26 40",
    )
    p.add_argument(
        "--max-topics",
        type=int,
        default=10,
        help="Max topics to evaluate when --topic-range is not set (default: 10)",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Save results JSON to this path (default: results/run_<retriever>.json)",
    )
    p.add_argument(
        "--no-explanations",
        action="store_true",
        default=False,
        help="Skip explanation generation (faster, but Part 5 metrics unavailable)",
    )
    return p


if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Allow running as  python eval/benchmark.py  from trialmatch/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    args = _build_parser().parse_args()

    tr = tuple(args.topic_range) if args.topic_range else None
    out = args.output or f"results/run_{args.retriever}{'_critic' if args.use_critic else ''}.json"

    results = run_trec_benchmark(
        max_topics=args.max_topics,
        retriever_type=args.retriever,
        use_critic=args.use_critic,
        topic_range=tr,
        output_file=out,
        generate_explanations=not args.no_explanations,
    )
    print_trec_summary(results)
    print(
        f"\nRuntime: {results.get('runtime_seconds', 0):.1f}s  |  "
        f"API calls: {results.get('total_api_calls', 0)}  |  "
        f"Est. cost: ${results.get('estimated_cost_usd', 0):.2f}"
    )
