#!/usr/bin/env python3
"""
TrialMatch AI — Interactive Agent CLI

A patient (or clinician) enters a free-text profile and the agent
autonomously searches, ranks, evaluates, and explains matching trials.

Usage:
    cd trialmatch
    python agent_cli.py
    python agent_cli.py --patient-file data/sample_patients/patient_01.txt
    python agent_cli.py --location "Indianapolis, IN"
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

# Allow running from trialmatch/ or project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

from src.agents.orchestrator import AgentEvent, TrialMatchAgent  # noqa: E402

# ── Terminal colours ───────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

BLUE   = lambda t: _c("94", t)
GREEN  = lambda t: _c("92", t)
YELLOW = lambda t: _c("93", t)
CYAN   = lambda t: _c("96", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ── Display ────────────────────────────────────────────────────────────────────

def _print_banner():
    print()
    print(BOLD("━" * 60))
    print(BOLD("  TrialMatch AI  —  Clinical Trial Matching Agent"))
    print(BOLD("━" * 60))
    print(DIM("  Powered by Claude (Agent 1) + GPT-4o (Critic) + BiomedBERT"))
    print()

def _render_event(event: AgentEvent) -> None:
    if event.type == "thinking":
        print(DIM("\n🤔 Agent thinking:"))
        for line in event.content.splitlines():
            print(DIM(f"   {line}"))

    elif event.type == "tool_call":
        print(CYAN(f"\n⚙  {event.content}"))

    elif event.type == "tool_result":
        print(GREEN(f"   {event.content}"))

    elif event.type == "answer":
        print()
        print(BOLD("━" * 60))
        print(BOLD("  RESULTS"))
        print(BOLD("━" * 60))
        # Wrap long lines nicely
        for para in event.content.split("\n\n"):
            wrapped = textwrap.fill(para.strip(), width=70, subsequent_indent="  ")
            print(f"\n{wrapped}")
        print()
        print(BOLD("━" * 60))

    elif event.type == "error":
        print(YELLOW(f"\n⚠  {event.content}"))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="TrialMatch AI interactive agent")
    p.add_argument("--patient-file", metavar="PATH", help="Path to a patient .txt file")
    p.add_argument("--location", metavar="CITY", help="City/state for proximity filtering")
    args = p.parse_args()

    _print_banner()

    # Get patient text
    if args.patient_file:
        patient_text = Path(args.patient_file).read_text().strip()
        print(BOLD("Patient file:"), args.patient_file)
        print(DIM(patient_text[:200] + ("…" if len(patient_text) > 200 else "")))
    else:
        print(BOLD("Enter patient profile") + DIM(" (paste text, then press Enter twice):"))
        print()
        lines = []
        try:
            while True:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    break
                lines.append(line)
        except EOFError:
            pass
        patient_text = "\n".join(lines).strip()

    if not patient_text:
        print("No patient text provided. Exiting.")
        sys.exit(1)

    if args.location:
        print(BOLD("\nLocation:"), args.location)

    print()
    print(DIM("Starting agent…"))
    print()

    agent = TrialMatchAgent()
    for event in agent.run(patient_text, location=args.location):
        _render_event(event)


if __name__ == "__main__":
    main()
