'use client'

import { useState } from 'react'
import Image from 'next/image'
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

function buildPatientText(condition: string, age: string, notes: string): string {
  const parts: string[] = []
  if (age) parts.push(`${age}-year-old`)
  parts.push(`with ${condition}`)
  if (notes.trim()) parts.push(`Notes: ${notes.trim()}`)
  return parts.join(' ') + '.'
}

export default function Page() {
  const [appState, setAppState] = useState<AppState>({ status: 'idle' })
  const [condition, setCondition] = useState('')
  const [age, setAge] = useState('')
  const [notes, setNotes] = useState('')
  const [location, setLocation] = useState('')
  const [maxTrials, setMaxTrials] = useState(3)
  const [useCritic, setUseCritic] = useState(false)

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!condition.trim()) return
    setAppState({ status: 'loading' })

    const patientText = buildPatientText(condition.trim(), age.trim(), notes)

    const request: MatchRequest = {
      patient_text: patientText,
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
                No matching trials found. Try broadening your search or removing the location filter.
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
            Research demo only — not for clinical decision-making. Results must be reviewed by a qualified clinician.
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
            Clinical Trial AI Agent
          </h1>
          <p className="text-slate-500 text-sm leading-relaxed">
            Enter a condition and we'll match you to actively recruiting trials.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="bg-white rounded-xl border border-slate-200 shadow-sm p-6 space-y-5">

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              Condition <span className="text-red-500">*</span>
            </label>
            <input
              type="text"
              value={condition}
              onChange={(e) => setCondition(e.target.value)}
              placeholder="e.g. lung cancer, Type 2 diabetes, Crohn's disease"
              required
              className="w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1.5">
                Age <span className="text-slate-400 font-normal">(optional)</span>
              </label>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value)}
                placeholder="e.g. 58"
                min={1}
                max={120}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

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
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 mb-1.5">
              Additional notes <span className="text-slate-400 font-normal">(optional)</span>
            </label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g. prior treatments tried, stage, biomarkers, anything else relevant"
              rows={3}
              className="w-full rounded-lg border border-slate-300 px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y leading-relaxed"
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
              {[1, 3, 5].map((n) => (
                <option key={n} value={n}>{n} trial{n !== 1 ? 's' : ''}</option>
              ))}
            </select>
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

          <p className="text-xs text-slate-500 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2.5">
            <span className="font-semibold text-slate-700">Research demo only.</span> Not medical advice — results must be reviewed by a qualified clinician before acting on them.
          </p>

          <button
            type="submit"
            disabled={!condition.trim()}
            className="w-full py-2.5 px-4 bg-blue-600 text-white text-sm font-semibold rounded-lg hover:bg-blue-700 active:bg-blue-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Search Trials
          </button>
        </form>
      </div>
    </div>
  )
}

function Header() {
  return (
    <header className="bg-white border-b border-slate-200 px-4 py-2 sticky top-0 z-10">
      <div className="max-w-2xl mx-auto flex items-center">
        <Image
          src="/logo-robot.png"
          alt="Clinical Trial Agent"
          width={96}
          height={96}
          className="object-contain"
          priority
        />
      </div>
    </header>
  )
}
