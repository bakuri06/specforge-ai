import { useState } from 'react'
import WizardStepper from './components/WizardStepper.jsx'
import UploadStep from './components/UploadStep.jsx'
import EvaluationStep from './components/EvaluationStep.jsx'
import ClarificationStep from './components/ClarificationStep.jsx'
import ChecklistEditor from './components/ChecklistEditor.jsx'
import ExportStep from './components/ExportStep.jsx'
import PolishedSpecPanel from './components/PolishedSpecPanel.jsx'
import RefineOnlyDoneStep from './components/RefineOnlyDoneStep.jsx'
import {
  startSession,
  submitEvaluationDecision,
  clarifyRequirements,
  clarifyGaps,
  signOffChecklist,
  rewindSession,
  downloadUrl,
} from './api/client.js'

// Everything needed to decide whether a step's "go back" is valid already
// lives on the session response (ambiguity_round/gap_round stay meaningful
// after that step resolves, since _to_response always derives them from
// qa_history/gap_qa_history length regardless of awaiting_input) - no extra
// client-side tracking needed except for Upload's raw draft, which the
// backend never echoes back.
function backTargetsFor(session) {
  return {
    upload: true,
    evaluate: session?.readiness_score != null,
    refine: (session?.ambiguity_round ?? 1) > 1,
    matrix: (session?.test_matrix?.length ?? 0) > 0,
  }
}

// Maps a wizard-step key to the LangGraph node the backend should rewind to.
// "matrix" is only reachable as a go-back target while sitting on 'export'
// (checklist_signoff already resolved by then), so it's unambiguous.
const REWIND_TARGET_BY_STEP = {
  evaluate: 'evaluation_review',
  refine: 'ba_clarification',
  matrix: 'checklist_signoff',
}

function stepKeyFor(session) {
  if (!session) return 'upload'
  if (session.workflow_aborted) return 'aborted'
  if (session.formatted_output) return 'export'
  if (session.workflow_mode === 'refine_only' && session.polished_spec && !session.awaiting_input) {
    return 'refined'
  }
  if (session.awaiting_input === 'requirement_evaluation') return 'evaluate'
  if (session.awaiting_input === 'ba_clarification') return 'refine'
  return 'matrix'
}

export default function App() {
  const [session, setSession] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  // Only kept so "go back to Upload" can pre-fill the form - the backend
  // never echoes requirements_draft/out_of_scope_details/etc. back.
  const [uploadDraft, setUploadDraft] = useState(null)

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

  const handleStart = (payload) => {
    setUploadDraft(payload)
    runAction(() => startSession(payload))
  }

  const handleGoBack = (stepKey) => {
    if (stepKey === 'upload') {
      setSession(null)
      setError(null)
      return
    }
    const target = REWIND_TARGET_BY_STEP[stepKey]
    if (!target) return
    runAction(() => rewindSession(session.session_id, target))
  }

  const handleEvaluationDecision = (decision) =>
    runAction(() => submitEvaluationDecision(session.session_id, decision))

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
  const canGoBack = backTargetsFor(session)

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
          <WizardStepper activeKey={stepKey} onStepClick={handleGoBack} canGoBack={canGoBack} />

          <PolishedSpecPanel polishedSpec={session?.polished_spec} />

          {stepKey === 'upload' && (
            <UploadStep
              onSubmit={handleStart}
              submitting={busy}
              error={error}
              defaultValues={uploadDraft}
            />
          )}

          {stepKey === 'evaluate' && session && (
            <EvaluationStep
              readinessScore={session.readiness_score}
              evaluationFeedback={session.evaluation_feedback}
              recommendedRounds={session.recommended_clarification_rounds}
              onSubmit={handleEvaluationDecision}
              submitting={busy}
            />
          )}

          {stepKey === 'aborted' && (
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-slate-900">Session Aborted</h2>
              <p className="text-sm text-slate-500">
                You chose to stop here instead of proceeding. Edit your requirements
                and start a new session when ready.
              </p>
              <button
                type="button"
                onClick={handleRestart}
                className="rounded-lg bg-indigo-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-indigo-500"
              >
                Start New Session
              </button>
            </div>
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

          {stepKey === 'refined' && session && (
            <RefineOnlyDoneStep polishedSpec={session.polished_spec} onRestart={handleRestart} />
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
