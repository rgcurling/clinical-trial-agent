#!/usr/bin/env python3
"""
Single-config TREC benchmark runner.

Usage:
    python scripts/benchmark.py --retriever tfidf --topic-range 26 40 --output results/run_baseline.json
    python scripts/benchmark.py --retriever biomedbert --use-critic --topic-range 26 40 --output results/run_r3.json
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "trialmatch"))
os.chdir(ROOT / "trialmatch")

from eval.benchmark import _build_parser, run_trec_benchmark, print_trec_summary  # noqa: E402

if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.topic_range:
        tr = tuple(args.topic_range)
    elif args.topic_start is not None and args.topic_end is not None:
        tr = (args.topic_start, args.topic_end)
    else:
        tr = None

    out = args.output or f"results/run_{args.retriever}{'_critic' if args.use_critic else ''}.json"
    results = run_trec_benchmark(
        max_topics=args.max_topics,
        retriever_type=args.retriever,
        use_critic=args.use_critic,
        topic_range=tr,
        output_file=out,
        generate_explanations=not args.no_explanations,
        resume_from=args.resume_from or getattr(args, "resume_from_checkpoint", None),
    )
    print_trec_summary(results)
