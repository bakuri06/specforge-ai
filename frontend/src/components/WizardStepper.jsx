const STEPS = [
  { key: 'upload', label: 'Upload' },
  { key: 'evaluate', label: 'Readiness Check' },
  { key: 'refine', label: 'Refine Spec' },
  { key: 'matrix', label: 'Test Matrix' },
  { key: 'export', label: 'Export' },
]

export default function WizardStepper({ activeKey }) {
  const activeIndex = STEPS.findIndex((step) => step.key === activeKey)

  return (
    <ol className="flex items-center gap-2 mb-8">
      {STEPS.map((step, index) => {
        const isActive = index === activeIndex
        const isDone = index < activeIndex
        return (
          <li key={step.key} className="flex items-center gap-2 flex-1">
            <div
              className={
                'flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold ' +
                (isActive
                  ? 'bg-indigo-600 text-white'
                  : isDone
                    ? 'bg-indigo-200 text-indigo-800'
                    : 'bg-slate-200 text-slate-500')
              }
            >
              {index + 1}
            </div>
            <span
              className={
                'text-sm font-medium ' + (isActive ? 'text-slate-900' : 'text-slate-500')
              }
            >
              {step.label}
            </span>
            {index < STEPS.length - 1 && (
              <div className="h-px flex-1 bg-slate-200 ml-2" />
            )}
          </li>
        )
      })}
    </ol>
  )
}
