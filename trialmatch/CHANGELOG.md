# Changelog

## [Unreleased] — 2026-03-30

### Changed

**matcher.py — Single API call per trial (was: one call per criterion)**
- Replaced the per-criterion Claude loop with a single call per trial. The new
  prompt asks Claude to return a structured JSON object covering all criteria at
  once: `overall_score`, `met_criteria`, `failed_criteria`, `uncertain_criteria`,
  `hard_exclusion`, `exclusion_reason`, and `reasoning`.
- `compute_match_score` now takes `(overall_score: float, hard_exclusion: bool)`
  and returns `0.0` if `hard_exclusion` is `True`, otherwise `overall_score`
  directly.
- Removed `GPT4oMatcher` and the `OPENAI_API_KEY` / `BASELINE_MODEL` imports.
- **Why:** Reduces API cost and latency by collapsing N criterion calls into 1.
  Also produces a more coherent overall assessment since the model can reason
  holistically across criteria rather than in isolation.

**models.py — Flattened MatchResult; removed CriterionResult**
- `CriterionResult` dataclass removed entirely.
- `MatchResult` fields replaced with: `overall_score`, `met_criteria` (list of
  strings), `failed_criteria`, `uncertain_criteria`, `hard_exclusion` (bool),
  `exclusion_reason` (str or None), `reasoning` (str). `match_score` and
  `uncertain_count` retained for downstream compatibility.

**retriever.py — Added corpus-based TF-IDF retriever**
- New function `retrieve_from_corpus(patient_text, corpus_path, top_k=20)` loads
  a TREC-format JSONL corpus and retrieves the top-k trials by TF-IDF cosine
  similarity using scikit-learn. The fitted vectorizer and sparse matrix are
  cached to disk alongside the corpus so fitting only happens once.
- Existing `retrieve_trials()` (live CT.gov API) is unchanged.

**ranker.py — Simplified exclusion filter**
- Hard-exclusion filter now reads `match.hard_exclusion` (boolean) directly
  instead of iterating over `criterion_results`. Unchanged sort logic.

**explainer.py — Updated for string-list criteria**
- `generate_trial_card()` now reads `match.met_criteria` and
  `match.uncertain_criteria` as plain `list[str]` instead of filtering
  `CriterionResult` objects.
- `_format_criteria_list` updated to accept `list[str]`.
- `CriterionResult` import removed.

**main.py — Added `--mode` flag; retriever is now injectable**
- New `--mode {inference,benchmark}` argument (default `inference`).
  - `inference`: uses live `retrieve_trials()` as before.
  - `benchmark`: uses `retrieve_from_corpus()` against `data/trec_2021/corpus.jsonl`.
- `run_pipeline()` accepts a `retriever` callable `(patient_text, profile) ->
  list[Trial]` so both modes share the same matching/ranking/explaining code.
- Removed `GPT4oMatcher` import and `--compare-models` flag (matcher removed).

**eval/benchmark.py — Replaced BM25 CorpusRetriever with TF-IDF retriever**
- `run_trec_benchmark()` now calls `retrieve_from_corpus(note_text, corpus_path,
  top_k=20)` directly, passing the raw patient topic text as the query.
- Removed the `CorpusRetriever` (BM25) instantiation; TF-IDF cache is managed
  inside `retrieve_from_corpus`.

**tests/test_matcher.py — Updated for new MatchResult shape**
- `CriterionResult` removed from imports and fixtures.
- `TestComputeMatchScore` tests updated to new `(overall_score, hard_exclusion)`
  signature.
- `test_criterion_results_populated` replaced with `test_met_criteria_populated`
  and `test_hard_exclusion_false_by_default`.
- Mock Claude response updated to return the new single-call JSON format.
