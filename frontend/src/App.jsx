import { useState } from 'react'
import WizardStepper from './components/WizardStepper.jsx'
import UploadStep from './components/UploadStep.jsx'
import ClarificationStep from './components/ClarificationStep.jsx'
import ChecklistEditor from './components/ChecklistEditor.jsx'
import ExportStep from './components/ExportStep.jsx'
import PolishedSpecPanel from './components/PolishedSpecPanel.jsx'
import {
  startSession,
  clarifyRequirements,
  clarifyGaps,
  signOffChecklist,
  downloadUrl,
} from './api/client.js'

function stepKeyFor(session) {
  if (!session) return 'upload'
  if (session.formatted_output) return 'export'
  if (session.awaiting_input === 'ba_clarification') return 'refine'
  return 'matrix'
}

export default function App() {
  const [session, setSession] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const runAction = async (action) => {
    setBusy(true)
    setError(null)
    try {
      const result = await action()
      setSession(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const handleStart = (payload) => runAction(() => startSession(payload))

  const handleClarifyRequirements = (answers) =>
    runAction(() => clarifyRequirements(session.session_id, answers))

  const handleClarifyGaps = (answers) =>
    runAction(() => clarifyGaps(session.session_id, answers))

  const handleChecklistSubmit = (matrix, outputFormat) =>
    runAction(() => signOffChecklist(session.session_id, matrix, outputFormat))

  const handleRestart = () => {
    setSession(null)
    setError(null)
  }

  const stepKey = stepKeyFor(session)

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto max-w-3xl px-6 py-5">
          <h1 className="text-xl font-bold text-slate-900">SpecForge AI</h1>
          <p className="text-sm text-slate-500">
            Requirements-to-Test State Machine
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-3xl px-6 py-10">
        <div className="rounded-xl border border-slate-200 bg-white p-8 shadow-sm">
          <WizardStepper activeKey={stepKey} />

          <PolishedSpecPanel polishedSpec={session?.polished_spec} />

          {stepKey === 'upload' && (
            <UploadStep onSubmit={handleStart} submitting={busy} error={error} />
          )}

          {stepKey === 'refine' && session && (
            <ClarificationStep
              key={`ba-${session.ambiguity_round}-${session.ambiguity_questions.join('|')}`}
              title="A few things need clarifying"
              description="The BA Requirements Refiner flagged some ambiguity before it can produce a clean spec."
              questions={session.ambiguity_questions}
              round={session.ambiguity_round}
              onSubmit={handleClarifyRequirements}
              submitting={busy}
            />
          )}

          {stepKey === 'matrix' && session?.awaiting_input === 'gap_clarification' && (
            <ClarificationStep
              key={`gap-${session.gap_round}-${session.gap_questions.join('|')}`}
              title="Test coverage gaps found"
              description="The QA Test Matrix Builder needs a bit more detail before finalizing coverage."
              questions={session.gap_questions}
              round={session.gap_round}
              onSubmit={handleClarifyGaps}
              submitting={busy}
            />
          )}

          {stepKey === 'matrix' && session?.awaiting_input === 'checklist_signoff' && (
            <ChecklistEditor
              testMatrix={session.test_matrix}
              onSubmit={handleChecklistSubmit}
              submitting={busy}
            />
          )}

          {stepKey === 'export' && session && (
            <ExportStep
              formattedOutput={session.formatted_output}
              outputFormat={session.output_format}
              downloadHref={downloadUrl(session.session_id)}
              onRestart={handleRestart}
            />
          )}

          {error && stepKey !== 'upload' && (
            <p className="mt-4 text-sm text-red-600">{error}</p>
          )}
        </div>
      </main>
    </div>
  )
}
