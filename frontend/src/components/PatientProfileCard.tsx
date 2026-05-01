import type { PatientProfileOut } from '@/lib/types'

interface Props {
  profile: PatientProfileOut
  processingTime: number
}

function Tag({ label, color }: { label: string; color: 'blue' | 'purple' | 'teal' | 'orange' | 'slate' }) {
  const styles = {
    blue: 'bg-blue-50 text-blue-700 border-blue-200',
    purple: 'bg-purple-50 text-purple-700 border-purple-200',
    teal: 'bg-teal-50 text-teal-700 border-teal-200',
    orange: 'bg-orange-50 text-orange-700 border-orange-200',
    slate: 'bg-slate-100 text-slate-600 border-slate-200',
  }
  return (
    <span className={`inline-block px-2 py-0.5 rounded border text-xs font-medium ${styles[color]}`}>
      {label}
    </span>
  )
}

export default function PatientProfileCard({ profile, processingTime }: Props) {
  const hasDemographics = profile.age !== null || profile.gender
  const hasConditions = profile.conditions.length > 0 || profile.stage
  const hasBiomarkers = profile.biomarkers.length > 0
  const hasMedications = profile.medications.length > 0

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
          <svg className="w-4 h-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
          Extracted Patient Profile
        </h2>
        <span className="text-xs text-slate-400 tabular-nums">
          {(processingTime / 1000).toFixed(1)}s
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {hasDemographics && (
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1.5">Demographics</p>
            <p className="text-sm text-slate-700">
              {[profile.age ? `Age ${profile.age}` : null, profile.gender].filter(Boolean).join(' · ')}
            </p>
          </div>
        )}

        {hasConditions && (
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1.5">Conditions</p>
            <div className="flex flex-wrap gap-1">
              {profile.conditions.map((c) => <Tag key={c} label={c} color="blue" />)}
              {profile.stage && <Tag label={profile.stage} color="purple" />}
            </div>
          </div>
        )}

        {hasBiomarkers && (
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1.5">Biomarkers</p>
            <div className="flex flex-wrap gap-1">
              {profile.biomarkers.map((b) => <Tag key={b} label={b} color="teal" />)}
            </div>
          </div>
        )}

        {hasMedications && (
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1.5">Prior Treatments</p>
            <div className="flex flex-wrap gap-1">
              {profile.medications.map((m) => <Tag key={m} label={m} color="orange" />)}
            </div>
          </div>
        )}

        {profile.performance_status && (
          <div>
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1.5">Performance</p>
            <p className="text-sm text-slate-700">{profile.performance_status}</p>
          </div>
        )}
      </div>
    </div>
  )
}
