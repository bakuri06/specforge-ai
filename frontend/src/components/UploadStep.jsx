import { useState } from 'react'

export default function UploadStep({ onSubmit, submitting, error }) {
  const [text, setText] = useState('')
  const [legacyTestCases, setLegacyTestCases] = useState('')
  const [files, setFiles] = useState([])
  const [legacyFiles, setLegacyFiles] = useState([])

  const handleSubmit = (event) => {
    event.preventDefault()
    onSubmit({ text, legacyTestCases, files, legacyFiles })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">
          Requirements (paste text)
        </label>
        <textarea
          value={text}
          onChange={(event) => setText(event.target.value)}
          rows={8}
          placeholder="Paste raw requirements, user stories, or a feature description..."
          className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:border-indigo-500 focus:ring-indigo-500"
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">
          Attachments (PDF, CSV, or screenshots)
        </label>
        <input
          type="file"
          multiple
          onChange={(event) => setFiles(Array.from(event.target.files))}
          className="block w-full text-sm text-slate-600"
        />
        {files.length > 0 && (
          <ul className="mt-2 text-xs text-slate-500 list-disc list-inside">
            {files.map((file) => (
              <li key={file.name}>{file.name}</li>
            ))}
          </ul>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-slate-700 mb-1">
          Legacy test cases (optional, paste or leave blank)
        </label>
        <textarea
          value={legacyTestCases}
          onChange={(event) => setLegacyTestCases(event.target.value)}
          rows={5}
          placeholder="Paste existing test cases to run delta analysis against..."
          className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:border-indigo-500 focus:ring-indigo-500"
        />
        <p className="mt-2 text-xs text-slate-500">Or upload a legacy CSV suite:</p>
        <input
          type="file"
          multiple
          accept=".csv,text/csv"
          onChange={(event) => setLegacyFiles(Array.from(event.target.files))}
          className="mt-1 block w-full text-sm text-slate-600"
        />
        {legacyFiles.length > 0 && (
          <ul className="mt-2 text-xs text-slate-500 list-disc list-inside">
            {legacyFiles.map((file) => (
              <li key={file.name}>{file.name}</li>
            ))}
          </ul>
        )}
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}

      <button
        type="submit"
        disabled={submitting}
        className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
      >
        {submitting ? 'Analyzing...' : 'Analyze Requirements'}
      </button>
    </form>
  )
}
