const STEPS = [
  { key: 'upload', label: 'Upload' },
  { key: 'evaluate', label: 'Readiness Check' },
  { key: 'refine', label: 'Refine Spec' },
  { key: 'matrix', label: 'Test Matrix' },
  { key: 'export', label: 'Export' },
]

export default function WizardStepper({ activeKey, onStepClick, canGoBack = {} }) {
  const activeIndex = STEPS.findIndex((step) => step.key === activeKey)

  return (
    <ol className="flex items-center gap-2 mb-8">
      {STEPS.map((step, index) => {
        const isActive = index === activeIndex
        const isDone = index < activeIndex
        const clickable = isDone && Boolean(canGoBack[step.key]) && Boolean(onStepClick)
        const circleClassName =
          'flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-semibold ' +
          (isActive
            ? 'bg-indigo-600 text-white'
            : isDone
              ? 'bg-indigo-200 text-indigo-800'
              : 'bg-slate-200 text-slate-500') +
          (clickable ? ' cursor-pointer hover:ring-2 hover:ring-indigo-400' : '')
        const labelClassName =
          'text-sm font-medium ' +
          (isActive ? 'text-slate-900' : 'text-slate-500') +
          (clickable ? ' cursor-pointer hover:text-indigo-600 hover:underline' : '')

        return (
          <li key={step.key} className="flex items-center gap-2 flex-1">
            {clickable ? (
              <button
                type="button"
                onClick={() => onStepClick(step.key)}
                title={`Go back to ${step.label} and resubmit`}
                className={circleClassName}
              >
                {index + 1}
              </button>
            ) : (
              <div className={circleClassName}>{index + 1}</div>
            )}
            {clickable ? (
              <button type="button" onClick={() => onStepClick(step.key)} className={labelClassName}>
                {step.label}
              </button>
            ) : (
              <span className={labelClassName}>{step.label}</span>
            )}
            {index < STEPS.length - 1 && (
              <div className="h-px flex-1 bg-slate-200 ml-2" />
            )}
          </li>
        )
      })}
    </ol>
  )
}
