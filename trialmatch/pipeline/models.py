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
class CriterionResult:
    criterion_text: str
    criterion_type: str          # "inclusion" or "exclusion"
    eligible: str                # "true", "false", "uncertain"
    confidence: float
    reasoning: str
    relevant_patient_info: str


@dataclass
class MatchResult:
    trial: Trial
    criterion_results: list[CriterionResult]
    match_score: float
    met_inclusion: int
    failed_inclusion: int
    triggered_exclusion: int
    uncertain_count: int
