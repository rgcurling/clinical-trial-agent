'use client'

import { useEffect, useState } from 'react'

const STAGES = [
  {
    label: 'Extracting patient profile',
    tip: 'Identifying conditions, biomarkers, medications, and stage',
    delay: 2500,
  },
  {
    label: 'Searching ClinicalTrials.gov',
    tip: 'Querying for actively recruiting trials',
    delay: 7000,
  },
  {
    label: 'Evaluating eligibility criteria',
    tip: 'Claude is reviewing each trial — this is the longest step',
    delay: 38000,
  },
  {
    label: 'Generating plain-English explanations',
    tip: 'Translating medical criteria to grade-8 reading level',
    delay: 50000,
  },
  {
    label: 'Ranking and filtering results',
    tip: 'Applying exclusion filters and sorting by match score',
    delay: 55000,
  },
]

export default function LoadingState() {
  const [current, setCurrent] = useState(0)

  useEffect(() => {
    const timers = STAGES.map((stage, i) =>
      window.setTimeout(() => setCurrent(i + 1), stage.delay)
    )
    return () => timers.forEach(window.clearTimeout)
  }, [])

  return (
    <div className="flex flex-col items-center justify-center py-20 px-4">
      <div className="w-full max-w-sm">
        {/* Spinner */}
        <div className="flex justify-center mb-8">
          <div className="relative w-14 h-14">
            <div className="absolute inset-0 rounded-full border-4 border-blue-100" />
            <div className="absolute inset-0 rounded-full border-4 border-blue-600 border-t-transparent animate-spin" />
          </div>
        </div>

        <h2 className="text-center text-lg font-semibold text-slate-800 mb-1">
          Analyzing patient data
        </h2>
        <p className="text-center text-sm text-slate-500 mb-10">
          Usually 30–60 seconds depending on the number of trials
        </p>

        <div className="space-y-4">
          {STAGES.map((stage, i) => {
            const done = i < current
            const active = i === current
            return (
              <div key={i} className="flex items-start gap-3">
                <div className="mt-0.5 flex-shrink-0">
                  {done ? (
                    <span className="flex w-5 h-5 rounded-full bg-emerald-500 items-center justify-center">
                      <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                      </svg>
                    </span>
                  ) : active ? (
                    <span className="flex w-5 h-5 rounded-full border-2 border-blue-600 border-t-transparent animate-spin" />
                  ) : (
                    <span className="flex w-5 h-5 rounded-full border-2 border-slate-200" />
                  )}
                </div>
                <div>
                  <p className={`text-sm font-medium ${done ? 'text-slate-400 line-through' : active ? 'text-slate-800' : 'text-slate-400'}`}>
                    {stage.label}
                  </p>
                  {active && (
                    <p className="text-xs text-slate-500 mt-0.5">{stage.tip}</p>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
