'use client'

import { useState } from 'react'
import { matchTrials } from '@/lib/api'
import type { MatchRequest, MatchResponse } from '@/lib/types'
import LoadingState from '@/components/LoadingState'
import PatientProfileCard from '@/components/PatientProfileCard'
import TrialCard from '@/components/TrialCard'

type AppState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: MatchResponse }
  | { status: 'error'; message: string }

const SAMPLE_NOTE =
  `58-year-old male with Stage IIIB non-small cell lung cancer (NSCLC).
EGFR mutation negative, PD-L1 expression 45%.
Previously treated with carboplatin and paclitaxel (2 cycles, completed 6 months ago).
Currently ECOG performance status 1. Located in Indianapolis, IN.
No prior immunotherapy. No active brain metastases.`

export default function Page() {
  const [appState, setAppState] = useState<AppState>({ status: 'idle' })
  const [patientText, setPatientText] = useState('')
  const [location, setLocation] = useState('')
  const [maxTrials, setMaxTrials] = useState(5)
  const [useCritic, setUseCritic] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!patientText.trim()) return
    setAppState({ status: 'loading' })

    const request: MatchRequest = {
      patient_text: patientText.trim(),
      max_trials: maxTrials,
      use_critic: useCritic,
    }
    if (location.trim()) request.location = location.trim()

    try {
      const data = await matchTrials(request)
      setAppState({ status: 'success', data })
    } catch (err) {
      setAppState({
        status: 'error',
        message: err instanceof Error ? err.message : 'An unexpected error occurred',
      })
    }
  }

  function reset() {
    setAppState({ status: 'idle' })
  }

  if (appState.status === 'loading') {
    return (
      <div className="min-h-screen">
        <Header />
        <LoadingState />
      </div>
    )
  }

  if (appState.status === 'error') {
    return (
      <div className="min-h-screen">
        <Header />
        <div className="max-w-lg mx-auto px-4 py-20 text-center">
          <div className="w-12 h-12 rounded-full bg-red-100 flex items-center justify-center mx-auto mb-4">
            <svg className="w-6 h-6 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <h2 className="text-lg font-semibold text-slate-800 mb-2">Something went wrong</h2>
          <p className="text-sm text-slate-500 mb-1">Make sure the API is running and reachable.</p>
          <p className="text-xs font-mono text-red-600 bg-red-50 border border-red-100 rounded px-3 py-2 mb-6 break-all">
            {appState.message}
          </p>
          <button
            onClick={reset}
            className="px-5 py-2 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      </div>
    )
  }

  if (appState.status === 'success') {
    const { data } = appState
    return (
      <div className="min-h-screen">
        <Header />
        <div className="max-w-2xl mx-auto px-4 py-8">
          <button
            onClick={reset}
            className="flex items-center gap-1.5 text-sm font-medium text-blue-600 hover:text-blue-800 mb-6"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            New Search
          </button>

          <PatientProfileCard profile={data.patient_profile} processingTime={data.processing_time_ms} />

          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-semibold text-slate-700">
              {data.matches.length === 0
                ? 'No trials found'
                : `${data.matches.length} matching trial${data.matches.length !== 1 ? 's' : ''}`}
            </h2>
            <p className="text-xs text-slate-400 tabular-nums">
              {data.n_candidates_retrieved} retrieved · {data.n_candidates_matched} matched
            </p>
          </div>

          {data.matches.length === 0 ? (
            <div className="text-center py-12 bg-white rounded-xl border border-slate-200">
              <p className="text-sm text-slate-500">
                No matching trials found. Try broadening the patient note or removing the location filter.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {data.matches.map((trial) => (
                <TrialCard key={trial.nct_id} trial={trial} />
              ))}
            </div>
          )}

          <p className="text-center text-xs text-slate-400 mt-8">
            Not for clinical decision-making. Results are AI-generated and must be reviewed by a qualified clinician.
          </p>
        </div>
      </div>
    )
  }

  // Idle — show the form
  return (
    <div className="min-h-screen">
      <Header />
      <div className="max-w-xl mx-auto px-4 py-10">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-slate-900 mb-3">
            Match patients to clinical trials
          </h1>
          <p className="text-slate-500 text-sm leading-relaxed">
            Paste a clinical note. AI extracts the patient profile, searches recruiting trials,
            evaluates eligibility criteria, and returns plain-English explanations.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5">
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              Patient Clinical Note <span className="text-red-500">*</span>
            </label>
            <textarea
              value={patientText}
              onChange={(e) => setPatientText(e.target.value)}
              placeholder={SAMPLE_NOTE}
              required
              rows={7}
              className="w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y font-mono leading-relaxed"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">
                Location <span className="text-slate-400 font-normal">(optional)</span>
              </label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                placeholder="e.g. Indianapolis, IN"
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">
                Max Results
              </label>
              <select
                value={maxTrials}
                onChange={(e) => setMaxTrials(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 bg-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              >
                {[1, 3, 5, 10, 15, 20].map((n) => (
                  <option key={n} value={n}>{n} trial{n !== 1 ? 's' : ''}</option>
                ))}
              </select>
            </div>
          </div>

          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={useCritic}
              onChange={(e) => setUseCritic(e.target.checked)}
              className="mt-0.5 w-4 h-4 rounded border-slate-300 text-blue-600 focus:ring-blue-500 focus:ring-offset-0"
            />
            <div>
              <p className="text-sm font-medium text-slate-700">Enable GPT-4o Critic Review</p>
              <p className="text-xs text-slate-500 mt-0.5">
                A second AI independently reviews eligibility assessments. More accurate, adds ~30 seconds.
              </p>
            </div>
          </label>

          <button
            type="submit"
            disabled={!patientText.trim()}
            className="w-full py-2.5 px-4 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 active:bg-blue-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Find Matching Trials
          </button>
        </form>

        <p className="text-center text-xs text-slate-400 mt-4">
          Not for clinical decision-making · Results must be reviewed by a qualified clinician
        </p>
      </div>
    </div>
  )
}

function Header() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
  return (
    <header className="bg-white border-b border-slate-200 px-4 py-3 sticky top-0 z-10">
      <div className="max-w-2xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center flex-shrink-0">
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
          <span className="font-bold text-slate-900">TrialMatch</span>
          <span className="text-slate-400 text-sm font-normal">AI</span>
        </div>
        <span className="text-xs text-slate-400 font-mono hidden sm:block truncate max-w-xs">
          {apiUrl}
        </span>
      </div>
    </header>
  )
}
