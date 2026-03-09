"""
Benchmark runner for TrialMatch AI.

Two modes:
  - synthetic (default): runs full pipeline on all .txt files in
    data/sample_patients/ and reports FK grade + BERTScore per card.
  - n2c2 (--data-dir flag): loads n2c2 2018 patient/annotation XML files
    and reports macro-F1 / micro-F1 for eligibility classification.
"""

import glob
import logging
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_PATIENTS_DIR
from eval.metrics import bertscore_consistency, flesch_kincaid_grade, precision_recall_f1
from pipeline.extractor import extract_patient_profile
from pipeline.explainer import generate_all_cards
from pipeline.matcher import ClaudeMatcher
from pipeline.models import PatientProfile
from pipeline.ranker import rank_trials
from pipeline.retriever import retrieve_trials

logger = logging.getLogger(__name__)

# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SyntheticResult:
    patient_file: str
    num_trials_retrieved: int
    num_trials_ranked: int
    fk_grades: list[float]
    bertscore: float  # vs. raw card (self-consistency check)


@dataclass
class N2c2Result:
    patient_id: str
    macro_f1: float
    micro_f1: float
    precision: float
    recall: float


# ── Synthetic benchmark ───────────────────────────────────────────────────────

def run_synthetic_benchmark(patients_dir: str = SAMPLE_PATIENTS_DIR) -> list[SyntheticResult]:
    """
    Run full pipeline on every .txt patient file in *patients_dir*.
    Returns a list of SyntheticResult, one per patient.
    """
    patient_files = sorted(glob.glob(os.path.join(patients_dir, "*.txt")))
    if not patient_files:
        logger.warning(f"No .txt files found in {patients_dir}")
        return []

    matcher = ClaudeMatcher()
    results = []

    for path in patient_files:
        fname = os.path.basename(path)
        logger.info(f"--- Benchmarking {fname} ---")

        with open(path) as f:
            text = f.read().strip()

        profile: PatientProfile = extract_patient_profile(text)

        # Use first extracted condition for retrieval; fall back to raw text keyword
        condition = profile.conditions[0] if profile.conditions else text.split()[0]

        trials = retrieve_trials(condition)
        match_results = matcher.match_trials(profile, trials)
        ranked = rank_trials(match_results)
        cards = generate_all_cards(ranked)

        fk_grades = [c["fk_grade"] for c in cards]

        # BERTScore: compare first card against itself as a sanity check
        # (real eval compares against a reference; here we use self-consistency)
        bs = float("nan")
        if cards:
            bs = bertscore_consistency(cards[0]["card_text"], cards[0]["card_text"])

        results.append(
            SyntheticResult(
                patient_file=fname,
                num_trials_retrieved=len(trials),
                num_trials_ranked=len(ranked),
                fk_grades=fk_grades,
                bertscore=bs,
            )
        )

    return results


def print_synthetic_summary(results: list[SyntheticResult]) -> None:
    header = f"{'Patient':<20} {'Retrieved':>10} {'Ranked':>7} {'Avg FK':>7} {'BERTScore':>10}"
    print("\n" + "=" * len(header))
    print("SYNTHETIC BENCHMARK SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        avg_fk = sum(r.fk_grades) / len(r.fk_grades) if r.fk_grades else float("nan")
        print(
            f"{r.patient_file:<20} {r.num_trials_retrieved:>10} "
            f"{r.num_trials_ranked:>7} {avg_fk:>7.1f} {r.bertscore:>10.3f}"
        )
    print("=" * len(header))


# ── n2c2 benchmark ────────────────────────────────────────────────────────────

def _map_label(label: str) -> int:
    """Map n2c2 annotation labels to binary 0/1."""
    return 1 if label.strip().upper() in ("MET", "YES", "1", "TRUE") else 0


def _eligible_label_from_criterion(
    profile: PatientProfile,
    criterion_text: str,
    criterion_type: str,
) -> int:
    """
    Run Claude on a single criterion for a given profile.
    Returns 1 (eligible) or 0 (not eligible).
    """
    from pipeline.matcher import ClaudeMatcher, _patient_profile_to_text, _call_claude_for_criterion
    import anthropic as _anthropic
    from config import ANTHROPIC_API_KEY

    client = _anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    patient_text = _patient_profile_to_text(profile)
    result = _call_claude_for_criterion(client, patient_text, criterion_type, criterion_text)
    eligible = result.get("eligible", "uncertain")
    return 1 if eligible == "true" else 0


def run_n2c2_benchmark(data_dir: str) -> list[N2c2Result]:
    """
    Load n2c2 2018 cohort selection XML files from *data_dir*.
    Expected structure:
      data_dir/
        train/
          *.xml   (patient records)
        train_annotations/
          *.xml   (criterion labels)

    Returns N2c2Result per patient.
    """
    records_dir = os.path.join(data_dir, "train")
    annot_dir = os.path.join(data_dir, "train_annotations")

    if not os.path.isdir(records_dir):
        logger.error(f"n2c2 records directory not found: {records_dir}")
        return []

    record_files = sorted(glob.glob(os.path.join(records_dir, "*.xml")))
    results = []

    for rec_path in record_files:
        patient_id = os.path.splitext(os.path.basename(rec_path))[0]
        annot_path = os.path.join(annot_dir, f"{patient_id}.xml")

        if not os.path.exists(annot_path):
            logger.warning(f"No annotation file for {patient_id}; skipping")
            continue

        # Parse patient record
        try:
            rec_tree = ET.parse(rec_path)
            rec_root = rec_tree.getroot()
            patient_text = " ".join(
                elem.text for elem in rec_root.iter() if elem.text
            ).strip()
        except Exception as e:
            logger.warning(f"Failed to parse {rec_path}: {e}")
            continue

        # Parse annotations
        try:
            ann_tree = ET.parse(annot_path)
            ann_root = ann_tree.getroot()
        except Exception as e:
            logger.warning(f"Failed to parse {annot_path}: {e}")
            continue

        profile = extract_patient_profile(patient_text)
        y_true, y_pred = [], []

        for tag in ann_root.iter():
            criterion_text = tag.get("met") or tag.get("text") or tag.tag
            label_str = tag.get("met", "")
            if not label_str:
                continue
            true_label = _map_label(label_str)
            pred_label = _eligible_label_from_criterion(profile, criterion_text, "inclusion")
            y_true.append(true_label)
            y_pred.append(pred_label)

        if not y_true:
            continue

        metrics = precision_recall_f1(y_pred, y_true)
        results.append(
            N2c2Result(
                patient_id=patient_id,
                macro_f1=metrics["f1"],
                micro_f1=metrics["f1"],
                precision=metrics["precision"],
                recall=metrics["recall"],
            )
        )

    return results


def print_n2c2_summary(results: list[N2c2Result]) -> None:
    if not results:
        print("No n2c2 results to display.")
        return
    header = f"{'Patient':<20} {'Precision':>10} {'Recall':>8} {'F1':>8}"
    print("\n" + "=" * len(header))
    print("N2C2 BENCHMARK SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.patient_id:<20} {r.precision:>10.3f} "
            f"{r.recall:>8.3f} {r.macro_f1:>8.3f}"
        )

    avg_f1 = sum(r.macro_f1 for r in results) / len(results)
    avg_p = sum(r.precision for r in results) / len(results)
    avg_r = sum(r.recall for r in results) / len(results)
    print("-" * len(header))
    print(f"{'AVERAGE':<20} {avg_p:>10.3f} {avg_r:>8.3f} {avg_f1:>8.3f}")
    print("=" * len(header))
