import type { TrialMatchOut } from '@/lib/types'

interface Props {
  trial: TrialMatchOut
}

function parseExplanation(text: string) {
  const qualify: string[] = []
  const discuss: string[] = []
  let section = ''

  for (const raw of (text ?? '').split('\n')) {
    const line = raw.trim()
    if (/^WHY YOU MAY QUALIFY/i.test(line)) { section = 'qualify'; continue }
    if (/^THINGS TO CLARIFY/i.test(line)) { section = 'discuss'; continue }
    if (/^(TRIAL NAME|MATCH SCORE|LEARN MORE)/i.test(line)) { section = ''; continue }
    if (/^[-•*]\s+/.test(line)) {
      const item = line.replace(/^[-•*]\s+/, '')
      if (section === 'qualify') qualify.push(item)
      else if (section === 'discuss') discuss.push(item)
    }
  }
  return { qualify, discuss }
}

function scoreAccentColor(score: number) {
  if (score >= 0.7) return 'bg-emerald-500'
  if (score >= 0.4) return 'bg-amber-400'
  return 'bg-red-400'
}

function scoreBadgeStyle(score: number) {
  if (score >= 0.7) return 'bg-emerald-50 text-emerald-700 border-emerald-200'
  if (score >= 0.4) return 'bg-amber-50 text-amber-700 border-amber-200'
  return 'bg-red-50 text-red-700 border-red-200'
}

export default function TrialCard({ trial }: Props) {
  const pct = Math.round(trial.overall_score * 100)
  const { qualify, discuss } = parseExplanation(trial.explanation)

  return (
    <div className={`bg-white rounded-xl border shadow-sm overflow-hidden ${trial.hard_exclusion ? 'border-red-200 opacity-70' : 'border-slate-200'}`}>
      {/* Score bar along top */}
      <div className="h-1 bg-slate-100">
        <div
          className={`h-full ${scoreAccentColor(trial.overall_score)} transition-all`}
          style={{ width: `${pct}%` }}
        />
      </div>

      <div className="p-5">
        {/* Header row */}
        <div className="flex items-start gap-3 mb-3">
          <span className="flex-shrink-0 w-6 h-6 rounded-full bg-blue-600 text-white text-xs font-bold flex items-center justify-center mt-0.5">
            {trial.rank}
          </span>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              {trial.hard_exclusion && (
                <span className="px-2 py-0.5 text-xs font-semibold bg-red-100 text-red-700 rounded">
                  Excluded
                </span>
              )}
              {trial.critic_flagged && !trial.hard_exclusion && (
                <span className="px-2 py-0.5 text-xs font-semibold bg-amber-100 text-amber-700 rounded">
                  Critic Flagged
                </span>
              )}
              {trial.critic_override && (
                <span className="px-2 py-0.5 text-xs font-semibold bg-purple-100 text-purple-700 rounded">
                  Critic Override
                </span>
              )}
            </div>
            <h3 className="text-sm font-semibold text-slate-900 leading-snug">
              {trial.title}
            </h3>
          </div>

          {/* Score badge */}
          <span className={`flex-shrink-0 px-2.5 py-1 rounded-lg border text-sm font-bold tabular-nums ${scoreBadgeStyle(trial.overall_score)}`}>
            {pct}%
          </span>
        </div>

        {/* Meta row */}
        <div className="flex flex-wrap items-center gap-2 mb-4 text-xs text-slate-500 ml-9">
          <span className="font-mono text-slate-400">{trial.nct_id}</span>
          {trial.phase && (
            <span className="px-1.5 py-0.5 bg-blue-50 text-blue-600 border border-blue-100 rounded font-medium">
              {trial.phase}
            </span>
          )}
          {trial.locations.slice(0, 2).map((loc) => (
            <span key={loc} className="flex items-center gap-0.5">
              <svg className="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
              </svg>
              {loc}
            </span>
          ))}
          {trial.locations.length > 2 && (
            <span>+{trial.locations.length - 2} more</span>
          )}
        </div>

        {/* Hard exclusion reason */}
        {trial.hard_exclusion && trial.exclusion_reason && (
          <div className="ml-9 mb-4 p-3 bg-red-50 rounded-lg border border-red-100 text-sm text-red-700">
            <span className="font-semibold">Reason: </span>{trial.exclusion_reason}
          </div>
        )}

        {/* Explanation sections */}
        {!trial.hard_exclusion && (qualify.length > 0 || discuss.length > 0) && (
          <div className="ml-9 space-y-3 mb-4">
            {qualify.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
                  Why you may qualify
                </p>
                <ul className="space-y-1.5">
                  {qualify.map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                      <svg className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                      </svg>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {discuss.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-1.5">
                  Things to clarify with your doctor
                </p>
                <ul className="space-y-1.5">
                  {discuss.map((item, i) => (
                    <li key={i} className="flex items-start gap-2 text-sm text-slate-700">
                      <svg className="w-4 h-4 text-amber-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        {/* Uncertain criteria — expandable clarification panel */}
        {trial.uncertain_criteria.length > 0 && (
          <details className="ml-9 mb-3 group">
            <summary className="flex items-center gap-2 cursor-pointer list-none">
              <span className="text-xs font-medium text-amber-700 bg-amber-50 px-2 py-1 rounded border border-amber-100 select-none">
                ? {trial.uncertain_criteria.length} uncertain
              </span>
              {trial.potential_score > trial.overall_score && (
                <span className="text-xs text-slate-500">
                  Could reach <span className="font-semibold text-amber-600">{Math.round(trial.potential_score * 100)}%</span> if resolved
                </span>
              )}
              <svg
                className="w-3 h-3 text-slate-400 ml-auto transition-transform group-open:rotate-180"
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </summary>
            <div className="mt-2 space-y-2 pl-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Questions that would resolve uncertainty
              </p>
              {trial.clarifying_questions.length > 0
                ? trial.clarifying_questions.map((q, i) => (
                    <div key={i} className="p-2.5 bg-amber-50 rounded-lg border border-amber-100">
                      <p className="text-xs text-slate-500 mb-1">{q.criterion}</p>
                      <p className="text-sm text-slate-800 font-medium">→ {q.question}</p>
                    </div>
                  ))
                : trial.uncertain_criteria.map((c, i) => (
                    <div key={i} className="p-2.5 bg-amber-50 rounded-lg border border-amber-100">
                      <p className="text-sm text-slate-700">{c}</p>
                    </div>
                  ))
              }
            </div>
          </details>
        )}

        {/* Met criteria — expandable */}
        {trial.met_criteria.length > 0 && (
          <details className="ml-9 mb-3 group">
            <summary className="flex items-center gap-2 cursor-pointer list-none">
              <span className="text-xs font-medium text-emerald-700 bg-emerald-50 px-2 py-1 rounded border border-emerald-100 select-none">
                ✓ {trial.met_criteria.length} met
              </span>
              <svg
                className="w-3 h-3 text-slate-400 ml-auto transition-transform group-open:rotate-180"
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </summary>
            <div className="mt-2 space-y-1.5 pl-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Criteria met
              </p>
              {trial.met_criteria.map((c, i) => (
                <div key={i} className="flex items-start gap-2 p-2.5 bg-emerald-50 rounded-lg border border-emerald-100">
                  <svg className="w-4 h-4 text-emerald-500 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
                  </svg>
                  <p className="text-sm text-slate-700">{c}</p>
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Failed criteria — expandable */}
        {trial.failed_criteria.length > 0 && (
          <details className="ml-9 mb-3 group">
            <summary className="flex items-center gap-2 cursor-pointer list-none">
              <span className="text-xs font-medium text-red-700 bg-red-50 px-2 py-1 rounded border border-red-100 select-none">
                ✗ {trial.failed_criteria.length} failed
              </span>
              <svg
                className="w-3 h-3 text-slate-400 ml-auto transition-transform group-open:rotate-180"
                fill="none" viewBox="0 0 24 24" stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </summary>
            <div className="mt-2 space-y-1.5 pl-1">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                Criteria not met
              </p>
              {trial.failed_criteria.map((c, i) => (
                <div key={i} className="flex items-start gap-2 p-2.5 bg-red-50 rounded-lg border border-red-100">
                  <svg className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  <p className="text-sm text-slate-700">{c}</p>
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Footer: link */}
        <div className="flex items-center pt-3 border-t border-slate-100 ml-9">
          <div className="flex-1" />
          <a
            href={trial.trial_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs font-medium text-blue-600 hover:text-blue-800 flex items-center gap-1 hover:underline"
          >
            View on ClinicalTrials.gov
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
            </svg>
          </a>
        </div>
      </div>
    </div>
  )
}
