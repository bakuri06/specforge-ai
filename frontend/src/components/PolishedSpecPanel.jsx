export default function PolishedSpecPanel({ polishedSpec }) {
  if (!polishedSpec) return null

  return (
    <details className="mb-6 rounded-lg border border-slate-200 bg-slate-50 p-4" open>
      <summary className="cursor-pointer text-sm font-semibold text-slate-700">
        Polished Requirements Blueprint
      </summary>
      <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap text-xs text-slate-600">
        {polishedSpec}
      </pre>
    </details>
  )
}
