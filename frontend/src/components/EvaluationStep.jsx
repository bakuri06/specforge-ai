import { useState } from 'react'

export default function EvaluationStep({
  readinessScore,
  evaluationFeedback,
  recommendedRounds,
  onSubmit,
  submitting,
}) {
  const [maxRounds, setMaxRounds] = useState(recommendedRounds ?? 1)

  const handleProceed = (event) => {
    event.preventDefault()
    onSubmit({ action: 'proceed', max_clarification_rounds: Number(maxRounds) })
  }

  const handleAbort = () => {
    onSubmit({ action: 'abort' })
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Requirements Readiness Check</h2>
        <p className="text-sm text-slate-500 mt-1">
          Before any clarifying questions are asked, here's how ready these requirements look.
        </p>
      </div>

      <div className="rounded-lg border border-slate-200 p-4">
        <div className="flex items-baseline gap-2">
          <span className="text-3xl font-bold text-slate-900">{readinessScore ?? '—'}</span>
          <span className="text-sm text-slate-500">/ 100 readiness score</span>
        </div>
        {evaluationFeedback && evaluationFeedback.length > 0 ? (
          <ul className="mt-3 space-y-1 text-sm text-slate-600 list-disc list-inside">
            {evaluationFeedback.map((item, index) => (
              <li key={index}>{item}</li>
            ))}
          </ul>
        ) : (
          <p className="mt-3 text-sm text-slate-500">No critical gaps found.</p>
        )}
      </div>

      <form onSubmit={handleProceed} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            Clarification rounds to allow (0 skips clarifying questions entirely)
          </label>
          <input
            type="number"
            min={0}
            max={5}
            value={maxRounds}
            onChange={(event) => setMaxRounds(event.target.value)}
            className="w-24 rounded-lg border border-slate-300 p-2 text-sm focus:border-indigo-500 focus:ring-indigo-500"
          />
          <p className="mt-1 text-xs text-slate-500">
            Recommended: {recommendedRounds ?? 1}
          </p>
        </div>

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={submitting}
            className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
          >
            {submitting ? 'Submitting...' : 'Proceed'}
          </button>
          <button
            type="button"
            onClick={handleAbort}
            disabled={submitting}
            className="rounded-lg border border-slate-300 px-5 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-50"
          >
            Abort &amp; Edit Requirements
          </button>
        </div>
      </form>
    </div>
  )
}
