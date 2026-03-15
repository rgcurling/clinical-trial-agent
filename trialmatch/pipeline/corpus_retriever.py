"""
Corpus-based retriever for TREC 2021 benchmark evaluation.

Instead of calling the live ClinicalTrials.gov API, loads the TREC 2021
corpus (~375k trials) from a local JSONL file and retrieves candidates
using BM25. This ensures retrieved NCT IDs are drawn from the same corpus
that TREC assessors judged, making P@5 and NDCG@5 scores meaningful.

The BM25 index is built once and cached to disk as a pickle file.
Subsequent loads take < 1s.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from pipeline.models import PatientProfile, Trial

logger = logging.getLogger(__name__)

_CORPUS_PATH = Path("data/trec_2021/corpus.jsonl")
_INDEX_CACHE  = Path("data/trec_2021/.bm25_index.pkl")


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


# ── Corpus parser ─────────────────────────────────────────────────────────────

def _parse_corpus_line(record: dict) -> Optional[Trial]:
    """
    Parse one JSONL record from the TrialGPT/TREC 2021 corpus into a Trial.

    Expected format (TrialGPT mirror):
        {
          "_id": "NCT...",
          "title": "...",
          "text": "Inclusion Criteria: ... Exclusion Criteria: ...",
          "metadata": {
              "conditions": [...],
              "phase": "Phase 2",
              "overall_status": "RECRUITING"   # optional
          }
        }
    """
    try:
        nct_id = record.get("_id") or record.get("nct_id", "")
        title  = record.get("title", "")
        text   = record.get("text", "")
        meta   = record.get("metadata", {})

        conditions = meta.get("conditions", [])
        if isinstance(conditions, str):
            conditions = [conditions]

        phase = meta.get("phase") or meta.get("phases")
        if isinstance(phase, list):
            phase = phase[0] if phase else None

        status = meta.get("overall_status", "RECRUITING")

        if not nct_id:
            return None

        return Trial(
            nct_id=nct_id,
            title=title,
            phase=phase,
            status=status,
            conditions=conditions,
            eligibility_criteria_raw=text,
            locations=[],
        )
    except Exception as e:
        logger.warning(f"Skipping malformed corpus record: {e}")
        return None


# ── BM25 index ────────────────────────────────────────────────────────────────

class CorpusRetriever:
    """
    BM25 retriever over the TREC 2021 trial corpus.

    First instantiation builds the index (~1-2 min for 375k trials) and
    caches it to data/trec_2021/.bm25_index.pkl. Subsequent loads take < 1s.

    Usage:
        retriever = CorpusRetriever()          # auto-detects corpus path
        trials = retriever.retrieve(profile, top_k=20)
    """

    def __init__(self, corpus_path: Path = _CORPUS_PATH):
        self.corpus_path = corpus_path
        self._trials: list[Trial] = []
        self._bm25 = None
        self._build_or_load()

    def _build_or_load(self) -> None:
        from rank_bm25 import BM25Okapi  # imported here so rank_bm25 is optional outside benchmark

        if _INDEX_CACHE.exists():
            logger.info("Loading BM25 index from cache...")
            with open(_INDEX_CACHE, "rb") as f:
                cached = pickle.load(f)
            self._trials = cached["trials"]
            self._bm25   = cached["bm25"]
            logger.info(f"Loaded {len(self._trials):,} trials from BM25 cache")
            return

        if not self.corpus_path.exists():
            raise FileNotFoundError(
                f"TREC corpus not found at {self.corpus_path.resolve()}\n"
                "Run the benchmark once with --benchmark to auto-download it,\n"
                "or download manually from:\n"
                "  https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl"
            )

        logger.info(f"Building BM25 index from {self.corpus_path} (one-time, ~2 min)...")
        trials: list[Trial] = []
        tokenized_corpus: list[list[str]] = []

        with open(self.corpus_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                if i > 0 and i % 50_000 == 0:
                    logger.info(f"  Indexed {i:,} records...")
                record = json.loads(line)
                trial  = _parse_corpus_line(record)
                if trial is None:
                    continue
                doc = f"{trial.title} {' '.join(trial.conditions)} {trial.eligibility_criteria_raw}"
                trials.append(trial)
                tokenized_corpus.append(_tokenize(doc))

        self._trials = trials
        self._bm25   = BM25Okapi(tokenized_corpus)

        logger.info(f"Indexed {len(trials):,} trials. Saving cache to {_INDEX_CACHE}...")
        _INDEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(_INDEX_CACHE, "wb") as f:
            pickle.dump({"trials": self._trials, "bm25": self._bm25}, f)
        logger.info("BM25 index cached.")

    def retrieve(
        self,
        profile: PatientProfile,
        top_k: int = 20,
    ) -> list[Trial]:
        """
        Return the top_k trials from the TREC corpus most relevant to the
        patient profile. Query is built from conditions + stage.
        """
        condition_text = " ".join(profile.conditions) if profile.conditions else ""
        stage_text     = profile.stage or ""
        query_text     = f"{condition_text} {stage_text}".strip()

        if not query_text:
            logger.warning("CorpusRetriever: empty query, returning empty list")
            return []

        tokens = _tokenize(query_text)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self._trials[i] for i in top_indices]
