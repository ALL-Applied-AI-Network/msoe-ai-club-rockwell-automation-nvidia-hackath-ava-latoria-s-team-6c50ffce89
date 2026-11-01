export type ConnectionState = 'connecting' | 'live' | 'stale' | 'offline'

export interface DisplayValues {
  phase: string
  prediction: string | null
  confidence: number | null
  state_machine_step: string
  rl_correction: string | null
  selected_action: string | null
  grasp_result: 'pending' | 'verified' | 'failed'
  cycle_time_s: number | null
  placement_error_mm: number | null
  overall_success_rate: number | null
  model_status: string | null
  model_reason: string | null
}

export interface TimelineEvent extends DisplayValues {
  start_s: number
  end_s: number
  stage: string | null
  grab: number | null
}

export interface PlanningValues {
  status: 'ready' | 'waiting' | 'blocked'
  planner: 'deterministic' | 'gemini' | null
  instruction: string | null
  action: string | null
  source_color: string | null
  destination_color: string | null
  shape: string | null
  size_mm: number | null
  execution_skill: string | null
  tower_order: string[] | null
  requires_clarification: boolean
  summary: string | null
}

export interface CurrentContext {
  policy_name?: string | null
  reason?: string | null
  detections?: Record<string, unknown>
  detection_count?: number
  latency_ms?: number | null
  action_safe?: boolean | null
  action_target_count?: number
  camera_names?: string[]
  six_cube?: {
    enabled?: boolean
    move_index?: number
    move_count?: number
    current?: Record<string, unknown> | null
    verified_layers?: Array<Record<string, unknown>>
  }
  [key: string]: unknown
}

export interface DashboardSnapshot {
  schema_version: '1.0'
  run: {
    id: string
    source: 'simulation' | 'physical_robot'
    state: string
    phase: string
    mode: string
    updated_at: string | null
    result: string
    error: string | null
  }
  display: DisplayValues
  planning: PlanningValues
  current: CurrentContext | null
  progress: Record<string, number | null>
  metrics: Record<string, number | boolean | null>
  timeline: TimelineEvent[]
  timeline_basis: 'cumulative_cycle_time'
  media: {
    video_url: string | null
    video_content_type: string | null
    latest_frames: Record<string, string>
  }
  links: Record<string, string>
}

export interface SessionSummary {
  id: string
  source: 'simulation' | 'physical_robot'
  state: string
  phase: string
  mode: string
  result: string
  updated_at: string | null
  steps_recorded: number | null
  last_latency_ms: number | null
  last_action_safe: boolean | null
  completed_grabs: number | null
  verified_placements: number | null
  has_video: boolean
  is_current: boolean
}

export interface FeedbackRequest {
  run_id: string
  event_id: string
  kind: 'grasp' | 'stack'
  label: 'GOOD' | 'BAD'
  stage: string | null
  grab: number | null
  model_prediction: string | null
  model_confidence: number | null
}

export interface DryRunPlan {
  mode: 'dry_run'
  planning: PlanningValues
  plan: Record<string, unknown>
  evidence: {
    matching_skill: string | null
    nearby_sessions: number
    successful_sessions: number
    human_feedback_labels: number
  }
  safety_boundary: string
}

const configuredBase = import.meta.env.VITE_DASHBOARD_API_BASE?.trim() ?? ''
export const apiBase = configuredBase.replace(/\/$/, '')

export function apiUrl(path: string | null): string | null {
  if (!path) return null
  if (/^https?:\/\//i.test(path)) return path
  return `${apiBase}${path.startsWith('/') ? path : `/${path}`}`
}

export async function fetchDashboard(signal?: AbortSignal): Promise<DashboardSnapshot> {
  const response = await fetch(apiUrl('/api/dashboard')!, {
    credentials: 'include',
    cache: 'no-store',
    signal,
  })
  if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`)
  return (await response.json()) as DashboardSnapshot
}

export async function fetchSessions(signal?: AbortSignal): Promise<SessionSummary[]> {
  const response = await fetch(apiUrl('/api/sessions')!, {
    credentials: 'include', cache: 'no-store', signal,
  })
  if (!response.ok) throw new Error(`Session API returned ${response.status}`)
  const payload = (await response.json()) as { sessions?: SessionSummary[] }
  return payload.sessions ?? []
}

export async function submitFeedback(value: FeedbackRequest): Promise<void> {
  const response = await fetch(apiUrl('/api/feedback')!, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(value),
  })
  if (!response.ok) throw new Error(`Feedback API returned ${response.status}`)
}

export async function previewInstruction(instruction: string): Promise<DryRunPlan> {
  const response = await fetch(apiUrl('/api/plan')!, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ instruction }),
  })
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { error?: string } | null
    throw new Error(payload?.error ?? `Planning API returned ${response.status}`)
  }
  return (await response.json()) as DryRunPlan
}

export function timelineEventAt(
  snapshot: DashboardSnapshot,
  currentTime: number,
): TimelineEvent | null {
  const active = snapshot.timeline.find(
    (event) => currentTime >= event.start_s && currentTime < event.end_s,
  )
  if (active) return active
  const last = snapshot.timeline.at(-1)
  return last && currentTime >= last.end_s ? last : null
}
