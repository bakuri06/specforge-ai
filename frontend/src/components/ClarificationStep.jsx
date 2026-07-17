import { useState } from 'react'

export default function ClarificationStep({ title, description, questions, round, onSubmit, submitting }) {
  const [answers, setAnswers] = useState(questions.map(() => ''))

  const handleChange = (index, value) => {
    setAnswers((prev) => {
      const next = [...prev]
      next[index] = value
      return next
    })
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    onSubmit(answers)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-900">
          {title}
          {round > 1 && (
            <span className="ml-2 text-sm font-normal text-slate-400">
              (round {round})
            </span>
          )}
        </h2>
        <p className="text-sm text-slate-500 mt-1">{description}</p>
      </div>

      {questions.map((question, index) => (
        <div key={index}>
          <label className="block text-sm font-medium text-slate-700 mb-1">
            {index + 1}. {question}
          </label>
          <textarea
            value={answers[index]}
            onChange={(event) => handleChange(index, event.target.value)}
            rows={2}
            required
            className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:border-indigo-500 focus:ring-indigo-500"
          />
        </div>
      ))}

      <button
        type="submit"
        disabled={submitting}
        className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
      >
        {submitting ? 'Submitting...' : 'Submit Answers'}
      </button>
    </form>
  )
}
