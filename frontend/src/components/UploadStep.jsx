import { useEffect, useState } from 'react'
import { getAvailableModels } from '../api/client.js'

const VISION_MODEL = 'qwen2.5vl:7b'

const MODEL_LABELS = {
  visionModel: 'Vision model (screenshot parsing)',
  reasoningModel: 'Reasoning model (BA/QA agents)',
  formatterModel: 'Formatter model (export compiler)',
}

export default function UploadStep({ onSubmit, submitting, error }) {
  const [text, setText] = useState('')
  const [legacyTestCases, setLegacyTestCases] = useState('')
  const [files, setFiles] = useState([])
  const [legacyFiles, setLegacyFiles] = useState([])

  // Reasoning/formatter dropdown options (vision is fixed, not driven by this)
  const [textModels, setTextModels] = useState(null)
  const [modelsError, setModelsError] = useState(null)
  const [modelsLoading, setModelsLoading] = useState(false)
  const [visionModel, setVisionModel] = useState(VISION_MODEL)
  const [reasoningModel, setReasoningModel] = useState('')
  const [formatterModel, setFormatterModel] = useState('')

  const loadModels = () => {
    setModelsLoading(true)
    setModelsError(null)
    getAvailableModels()
      .then((data) => {
        const nonVisionModels = (data.models || []).filter((m) => m !== VISION_MODEL)
        const pick = (preferred, current) =>
          nonVisionModels.includes(current)
            ? current
            : nonVisionModels.includes(preferred)
              ? preferred
              : nonVisionModels[0] || ''
        setTextModels(nonVisionModels)
        setReasoningModel((current) => pick(data.defaults?.reasoning_model, current))
        setFormatterModel((current) => pick(data.defaults?.formatter_model, current))
      })
      .catch(() => {
        // Ollama not reachable yet, or no models pulled — fall back silently
        // to whatever the backend has configured as defaults.
        setModelsError('Could not load available models from Ollama; using server defaults.')
      })
      .finally(() => setModelsLoading(false))
  }

  useEffect(() => {
    loadModels()
  }, [])

  const handleSubmit = (event) => {
    event.preventDefault()
    onSubmit({
      text,
      legacyTestCases,
      files,
      legacyFiles,
      visionModel,
      reasoningModel,
      formatterModel,
    })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {textModels && (
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-xs font-medium text-slate-500">
              Models pulled just now? Refresh the list below.
            </span>
            <button
              type="button"
              onClick={loadModels}
              disabled={modelsLoading}
              className="text-xs font-medium text-indigo-600 hover:text-indigo-500 disabled:opacity-50"
            >
              {modelsLoading ? 'Refreshing...' : 'Refresh models'}
            </button>
          </div>
          <div className="flex flex-row gap-4">
          {[
            ['visionModel', visionModel, setVisionModel, [VISION_MODEL]],
            ['reasoningModel', reasoningModel, setReasoningModel, textModels],
            ['formatterModel', formatterModel, setFormatterModel, textModels],
          ].map(([key, value, setValue, options]) => (
            <div key={key} className="min-w-0 flex-1">
              <label className="block text-xs font-medium text-slate-600 mb-1">
                {MODEL_LABELS[key]}
              </label>
              <select
                value={value}
                onChange={(event) => setValue(event.target.value)}
                className="w-full rounded border border-slate-300 py-1.5 px-2 text-sm focus:border-indigo-500 focus:ring-indigo-500"
              >
                {options.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))}
              </select>
            </div>
          ))}
          </div>
        </div>
      )}
      {modelsError && <p className="text-xs text-slate-400">{modelsError}</p>}

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
