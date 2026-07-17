export default function ExportStep({ formattedOutput, outputFormat, downloadHref, onRestart }) {
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Export Ready</h2>
        <p className="text-sm text-slate-500 mt-1">
          Format: <span className="font-medium text-slate-700">{outputFormat}</span>
        </p>
      </div>

      <pre className="max-h-96 overflow-auto rounded-lg bg-slate-900 p-4 text-xs text-slate-100 whitespace-pre-wrap">
        {formattedOutput}
      </pre>

      <div className="flex gap-3">
        <a
          href={downloadHref}
          download
          className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500"
        >
          Download File
        </a>
        <button
          type="button"
          onClick={onRestart}
          className="rounded-lg border border-slate-300 px-5 py-2.5 text-sm font-semibold text-slate-700 hover:bg-slate-50"
        >
          Start New Session
        </button>
      </div>
    </div>
  )
}
