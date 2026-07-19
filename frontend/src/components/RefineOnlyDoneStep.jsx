export default function RefineOnlyDoneStep({ polishedSpec, onRestart }) {
  const handleDownload = () => {
    const blob = new Blob([polishedSpec || ''], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'polished_spec.md'
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Requirements Refined</h2>
        <p className="text-sm text-slate-500 mt-1">
          The polished spec above is ready to download. No test matrix was generated
          for this session — start a new one if you need test cases too.
        </p>
      </div>

      <div className="flex gap-3">
        <button
          type="button"
          onClick={handleDownload}
          className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500"
        >
          Download Spec (.md)
        </button>
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
