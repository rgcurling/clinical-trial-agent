"""
Generate a manual audit sheet for 3 diverse topics from the run_r1 results.

Selects topics 26, 30, 35 (or any 3 available) and formats a human-readable
audit sheet with physician grade, matcher scores, and review checkboxes.

Usage:
  cd trialmatch
  python select_audit_cases.py
  python select_audit_cases.py --run results/run_r3.json --topics 26 30 35
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RUN = "results/run_r1.json"
OUTPUT_PATH = "results/audit_sheet.txt"
DEFAULT_TOPIC_IDS = ["26", "30", "35"]

_GRADE_LABEL = {0: "0 — Not relevant", 1: "1 — Partially relevant", 2: "2 — Highly relevant"}
_SEP = "=" * 72


def _bullet_list(items: list[str], indent: int = 4) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- {it}" for it in items) if items else f"{' ' * indent}(none)"


def build_audit_sheet(run_path: str, topic_ids: list[str]) -> str:
    with open(run_path) as f:
        data = json.load(f)

    topic_map: dict[str, dict] = {
        r["topic_id"]: r
        for r in data.get("per_topic_results", [])
        if "error" not in r
    }

    # Fall back to first 3 available if requested topics missing
    available = list(topic_map.keys())
    selected: list[str] = []
    for tid in topic_ids:
        if tid in topic_map:
            selected.append(tid)
    if not selected:
        selected = available[:3]
        print(f"  [Warning] Requested topics not found; using {selected} instead.")

    lines: list[str] = []
    lines.append(_SEP)
    lines.append("  TrialMatch AI — Manual Audit Sheet")
    lines.append(f"  Run: {run_path}")
    lines.append(f"  Topics: {', '.join(selected)}")
    lines.append(_SEP)

    for topic_id in selected:
        topic = topic_map[topic_id]
        patient_text = topic.get("topic_text", "(not available)")

        # Try to extract a short condition label from the patient text (first line)
        first_line = patient_text.split("\n")[0][:80]

        lines.append("")
        lines.append(_SEP)
        lines.append(f"  AUDIT CASE: Topic {topic_id}")
        lines.append(f"  Patient note (first line): {first_line}...")
        lines.append(_SEP)
        lines.append("")
        lines.append("Patient Profile:")
        lines.append(patient_text[:800] + ("..." if len(patient_text) > 800 else ""))
        lines.append("")

        top5 = topic.get("top5", [])
        for trial in top5:
            rank = trial["rank"]
            nct_id = trial["nct_id"]
            title = trial.get("title", "(no title)")
            score = trial.get("overall_score", 0.0)
            trec_grade = trial.get("trec_grade", -1)
            grade_label = _GRADE_LABEL.get(trec_grade, "N/A — not judged")

            # Agreement / mismatch logic
            matcher_eligible = score >= 0.5
            trec_relevant = trec_grade > 0 if trec_grade != -1 else None
            if trec_relevant is None:
                agreement = "N/A (not in qrels)"
            elif matcher_eligible == trec_relevant:
                agreement = "MATCH"
            else:
                agreement = "MISMATCH"

            critic_note = ""
            if trial.get("uncertain"):
                critic_note = "  [CRITIC: overridden to 0.5 uncertain]"
            elif trial.get("critic_flagged"):
                critic_note = "  [CRITIC: flagged — 1 discrepancy]"

            lines.append(f"  {'─'*68}")
            lines.append(f"  TRIAL {rank}: {nct_id}")
            lines.append(f"  Title: {title[:70]}")
            lines.append(f"  {'─'*68}")
            lines.append(f"  Matcher Score       : {score:.2f}{critic_note}")
            lines.append(f"  TREC Physician Grade: {grade_label}")
            lines.append(f"  Agreement           : {agreement}")
            lines.append("")
            lines.append("  Matcher Assessment:")
            lines.append(f"    Met criteria:")
            lines.append(_bullet_list(trial.get("met_criteria", []), 6))
            lines.append(f"    Uncertain criteria:")
            lines.append(_bullet_list(trial.get("uncertain_criteria", []), 6))
            lines.append(f"    Failed/excluded criteria:")
            lines.append(_bullet_list(trial.get("failed_criteria", []), 6))
            if trial.get("hard_exclusion"):
                lines.append(f"    ** Hard exclusion: {trial.get('exclusion_reason', 'N/A')}")
            lines.append("")
            if trial.get("explanation"):
                lines.append("  Patient-facing explanation:")
                for expl_line in trial["explanation"].splitlines()[:8]:
                    lines.append(f"    {expl_line}")
                lines.append("")
            lines.append("  Manual Review:")
            lines.append("    [ ] Agree with matcher")
            lines.append("    [ ] Disagree — patient should be ELIGIBLE")
            lines.append("    [ ] Disagree — patient should be EXCLUDED")
            lines.append("    Notes: _________________________________________________")
            lines.append("")

        lines.append("")

    lines.append(_SEP)
    lines.append("End of audit sheet.")
    lines.append(_SEP)

    return "\n".join(lines)


def main(run_path: str = DEFAULT_RUN, topic_ids: list[str] = None) -> None:
    if topic_ids is None:
        topic_ids = DEFAULT_TOPIC_IDS

    if not Path(run_path).exists():
        print(f"[Error] Run file not found: {run_path}")
        print("Run  python run_experiments.py  first, then re-run this script.")
        return

    print(f"Generating audit sheet from {run_path}...")
    sheet = build_audit_sheet(run_path, topic_ids)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(sheet)

    print(f"Audit sheet saved → {OUTPUT_PATH}")
    print(f"  Topics included: {topic_ids}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate manual audit sheet for 3 topics")
    p.add_argument(
        "--run",
        default=DEFAULT_RUN,
        metavar="PATH",
        help=f"Benchmark run JSON (default: {DEFAULT_RUN})",
    )
    p.add_argument(
        "--topics",
        nargs="+",
        default=DEFAULT_TOPIC_IDS,
        metavar="ID",
        help="Topic IDs to include (default: 26 30 35)",
    )
    args = p.parse_args()
    main(run_path=args.run, topic_ids=args.topics)
