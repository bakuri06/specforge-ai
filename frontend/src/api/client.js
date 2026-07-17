const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function handleResponse(response) {
  if (!response.ok) {
    const detail = await response.text()
    throw new Error(detail || `Request failed with status ${response.status}`)
  }
  return response.json()
}

export async function getAvailableModels() {
  const response = await fetch(`${API_BASE_URL}/api/models`)
  return handleResponse(response)
}

export async function startSession({
  text,
  legacyTestCases,
  files,
  legacyFiles = [],
  visionModel,
  reasoningModel,
  formatterModel,
}) {
  const formData = new FormData()
  formData.append('text', text)
  formData.append('legacy_test_cases', legacyTestCases)
  files.forEach((file) => formData.append('files', file))
  legacyFiles.forEach((file) => formData.append('legacy_files', file))
  if (visionModel) formData.append('vision_model', visionModel)
  if (reasoningModel) formData.append('reasoning_model', reasoningModel)
  if (formatterModel) formData.append('formatter_model', formatterModel)

  const response = await fetch(`${API_BASE_URL}/api/sessions/`, {
    method: 'POST',
    body: formData,
  })
  return handleResponse(response)
}

export async function getSession(sessionId) {
  const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`)
  return handleResponse(response)
}

export async function clarifyRequirements(sessionId, answers) {
  const response = await fetch(
    `${API_BASE_URL}/api/sessions/${sessionId}/clarify-requirements`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers }),
    },
  )
  return handleResponse(response)
}

export async function clarifyGaps(sessionId, answers) {
  const response = await fetch(
    `${API_BASE_URL}/api/sessions/${sessionId}/clarify-gaps`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers }),
    },
  )
  return handleResponse(response)
}

export async function signOffChecklist(sessionId, testMatrix, outputFormat) {
  const response = await fetch(
    `${API_BASE_URL}/api/sessions/${sessionId}/checklist-signoff`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test_matrix: testMatrix, output_format: outputFormat }),
    },
  )
  return handleResponse(response)
}

export function downloadUrl(sessionId) {
  return `${API_BASE_URL}/api/sessions/${sessionId}/download`
}
