from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PatientProfile:
    raw_text: str
    conditions: list[str] = field(default_factory=list)
    stage: Optional[str] = None
    prior_treatments: list[str] = field(default_factory=list)
    biomarkers: list[str] = field(default_factory=list)
    age: Optional[int] = None
    location: Optional[str] = None
    exclusion_flags: list[str] = field(default_factory=list)


@dataclass
class Trial:
    nct_id: str
    title: str
    phase: Optional[str]
    status: str
    conditions: list[str]
    eligibility_criteria_raw: str
    locations: list[str] = field(default_factory=list)


@dataclass
class MatchResult:
    trial: Trial
    overall_score: float
    met_criteria: list[str]
    failed_criteria: list[str]
    uncertain_criteria: list[str]
    hard_exclusion: bool
    exclusion_reason: Optional[str]
    reasoning: str
    match_score: float          # 0.0 if hard_exclusion else overall_score
    uncertain_count: int        # len(uncertain_criteria)
    uncertain: bool = False     # critic set this when ≥2 discrepancies found
    critic_flagged: bool = False  # critic set this when exactly 1 discrepancy found
