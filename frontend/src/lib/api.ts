import type { MatchRequest, MatchResponse } from './types'

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export async function matchTrials(request: MatchRequest): Promise<MatchResponse> {
  const response = await fetch(`${API_URL}/match`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!response.ok) {
    let message = `API error (${response.status})`
    try {
      const body = await response.json()
      if (body.detail) message = String(body.detail)
    } catch { /* ignore parse error */ }
    throw new Error(message)
  }

  return response.json() as Promise<MatchResponse>
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_URL}/health`, {
      signal: AbortSignal.timeout(5000),
    })
    return response.ok
  } catch {
    return false
  }
}
