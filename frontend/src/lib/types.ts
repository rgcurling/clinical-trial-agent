export interface PatientProfileOut {
  conditions: string[]
  age: number | null
  gender: string | null
  biomarkers: string[]
  stage: string | null
  medications: string[]
  performance_status: string | null
}

export interface TrialMatchOut {
  rank: number
  nct_id: string
  title: string
  phase: string | null
  overall_score: number
  met_criteria: string[]
  failed_criteria: string[]
  uncertain_criteria: string[]
  hard_exclusion: boolean
  exclusion_reason: string | null
  explanation: string
  fk_grade: number
  trial_url: string
  locations: string[]
  critic_flagged: boolean
  critic_override: boolean
}

export interface MatchResponse {
  status: string
  patient_profile: PatientProfileOut
  matches: TrialMatchOut[]
  n_candidates_retrieved: number
  n_candidates_matched: number
  processing_time_ms: number
}

export interface MatchRequest {
  patient_text: string
  location?: string
  max_trials?: number
  use_critic?: boolean
  status_filter?: string
}
