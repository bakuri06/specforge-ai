import { useState } from 'react'

const CATEGORY_LABELS = {
  sunny_day: 'Sunny Day',
  rainy_day: 'Rainy Day',
  boundary: 'Boundaries',
  edge_case: 'Edge Cases',
}

const STATUS_BADGE = {
  new: 'bg-emerald-100 text-emerald-700',
  modified: 'bg-amber-100 text-amber-700',
  broken: 'bg-red-100 text-red-700',
  unchanged: 'bg-slate-100 text-slate-600',
}

function emptyScenario(category) {
  return {
    id: `custom-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    category,
    title: '',
    description: '',
    status: 'new',
    included: true,
  }
}

export default function ChecklistEditor({ testMatrix, onSubmit, submitting }) {
  const [matrix, setMatrix] = useState(testMatrix)
  const [outputFormat, setOutputFormat] = useState('testrail')

  const updateItem = (id, patch) => {
    setMatrix((prev) => prev.map((item) => (item.id === id ? { ...item, ...patch } : item)))
  }

  const addScenario = (category) => {
    setMatrix((prev) => [...prev, emptyScenario(category)])
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    onSubmit(matrix, outputFormat)
  }

  const categories = Object.keys(CATEGORY_LABELS)

  return (
    <form onSubmit={handleSubmit} className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">Test Strategy Matrix</h2>
        <p className="text-sm text-slate-500 mt-1">
          Check, uncheck, edit, or add scenarios before generating the final export.
        </p>
      </div>

      {categories.map((category) => {
        const items = matrix.filter((item) => item.category === category)
        return (
          <div key={category}>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-slate-700">
                {CATEGORY_LABELS[category]}
              </h3>
              <button
                type="button"
                onClick={() => addScenario(category)}
                className="text-xs font-medium text-indigo-600 hover:text-indigo-500"
              >
                + Add scenario
              </button>
            </div>

            <div className="space-y-2">
              {items.length === 0 && (
                <p className="text-xs text-slate-400 italic">No scenarios yet.</p>
              )}
              {items.map((item) => (
                <div
                  key={item.id}
                  className="flex gap-3 rounded-lg border border-slate-200 p-3 items-start"
                >
                  <input
                    type="checkbox"
                    checked={item.included}
                    onChange={(event) => updateItem(item.id, { included: event.target.checked })}
                    className="mt-1.5 h-4 w-4 rounded border-slate-300 text-indigo-600"
                  />
                  <div className="flex-1 space-y-1">
                    <input
                      value={item.title}
                      onChange={(event) => updateItem(item.id, { title: event.target.value })}
                      placeholder="Scenario title"
                      className="w-full rounded border border-transparent bg-transparent text-sm font-medium text-slate-900 focus:border-indigo-500 focus:bg-white focus:ring-indigo-500"
                    />
                    <textarea
                      value={item.description}
                      onChange={(event) =>
                        updateItem(item.id, { description: event.target.value })
                      }
                      rows={2}
                      placeholder="Steps / expected result"
                      className="w-full rounded border border-transparent bg-transparent text-xs text-slate-600 focus:border-indigo-500 focus:bg-white focus:ring-indigo-500"
                    />
                  </div>
                  <span
                    className={
                      'shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ' +
                      (STATUS_BADGE[item.status] || STATUS_BADGE.unchanged)
                    }
                  >
                    {item.status}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )
      })}

      <div className="flex items-center gap-4 border-t border-slate-200 pt-6">
        <label className="text-sm font-medium text-slate-700">Export format</label>
        <select
          value={outputFormat}
          onChange={(event) => setOutputFormat(event.target.value)}
          className="rounded-lg border border-slate-300 py-2 px-3 text-sm focus:border-indigo-500 focus:ring-indigo-500"
        >
          <option value="testrail">TestRail (Markdown)</option>
          <option value="qtest">qTest (CSV)</option>
          <option value="playwright">Playwright (TypeScript)</option>
        </select>

        <button
          type="submit"
          disabled={submitting}
          className="ml-auto rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
        >
          {submitting ? 'Generating...' : 'Sign Off & Generate'}
        </button>
      </div>
    </form>
  )
}
