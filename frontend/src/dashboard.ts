export type ConnectionState = 'connecting' | 'live' | 'offline'

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
  motion_segment: string | null
  motion_segment_index: number | null
  motion_segment_total: number | null
  gripper_command: string | null
  tip_error_mm: number | null
  wrist_joint_deg: number | null
  motion_evidence: string | null
  model_status: string | null
  model_reason: string | null
  model_input: string | null
  good_probability: number | null
  shifted_probability: number | null
  model_gate: string | null
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
  requires_clarification: boolean
  summary: string | null
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
  progress: Record<string, number | null>
  metrics: Record<string, number | boolean | null>
  timeline: TimelineEvent[]
  timeline_basis: 'cumulative_cycle_time' | 'recorded_controller_segments'
  media: {
    video_url: string | null
    video_content_type: string | null
    latest_frames: Record<string, string>
  }
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

export function displayForVideoTime(
  snapshot: DashboardSnapshot,
  currentTime: number,
): DisplayValues {
  if (!snapshot.timeline.length) return snapshot.display
  const active = snapshot.timeline.find(
    (event) => currentTime >= event.start_s && currentTime < event.end_s,
  )
  if (active) {
    return active
  }
  if (currentTime >= snapshot.timeline[snapshot.timeline.length - 1].end_s) {
    return snapshot.timeline[snapshot.timeline.length - 1]
  }
  return snapshot.display
}

export function timelineEventForVideoTime(
  snapshot: DashboardSnapshot,
  currentTime: number,
): TimelineEvent | null {
  return (
    snapshot.timeline.find(
      (event) => currentTime >= event.start_s && currentTime < event.end_s,
    ) ?? null
  )
}
