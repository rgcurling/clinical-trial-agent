"""Pydantic request/response models for the TrialMatch REST API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    patient_text: str = Field(..., min_length=10, description="Free-text patient clinical note")
    location: Optional[str] = Field(None, description="City/state for proximity filtering, e.g. 'Indianapolis, IN'")
    max_trials: int = Field(5, ge=1, le=20, description="Number of top matches to return")
    use_critic: bool = Field(False, description="Run GPT-4o critic review on each match")
    status_filter: str = Field("RECRUITING", description="ClinicalTrials.gov overall status filter")


class PatientProfileOut(BaseModel):
    conditions: list[str]
    age: Optional[int]
    gender: Optional[str]
    biomarkers: list[str]
    stage: Optional[str]
    medications: list[str]
    performance_status: Optional[str]


class ClarifyingQuestion(BaseModel):
    criterion: str
    question: str


class TrialMatchOut(BaseModel):
    rank: int
    nct_id: str
    title: str
    phase: Optional[str]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    potential_score: float = Field(0.0, ge=0.0, le=1.0)
    met_criteria: list[str]
    failed_criteria: list[str]
    uncertain_criteria: list[str]
    clarifying_questions: list[ClarifyingQuestion] = []
    hard_exclusion: bool
    exclusion_reason: Optional[str]
    explanation: Optional[str]
    fk_grade: Optional[float]
    trial_url: str
    locations: list[str]
    critic_flagged: bool = False
    critic_override: bool = False


class MatchResponse(BaseModel):
    status: str
    patient_profile: PatientProfileOut
    matches: list[TrialMatchOut]
    n_candidates_retrieved: int
    n_candidates_matched: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    model: str
