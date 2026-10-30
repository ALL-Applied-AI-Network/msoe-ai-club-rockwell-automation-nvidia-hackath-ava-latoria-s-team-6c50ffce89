import { useEffect, useState } from 'react'
import type { DashboardSnapshot } from '../../docs/frontend/liveStatus'

const apiBase = (
  import.meta.env.VITE_API_BASE_URL || 'http://localhost:8765'
).replace(/\/$/, '')

function percent(value: number | null | undefined) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

function measurement(
  value: number | null | undefined,
  unit: string,
) {
  return value == null ? '—' : `${value.toFixed(2)} ${unit}`
}

export default function LiveDemo() {
  const [dashboard, setDashboard] =
    useState<DashboardSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [videoTime, setVideoTime] = useState(0)
  const [videoError, setVideoError] = useState(false)

  useEffect(() => {
    let stopped = false
    let timer: ReturnType<typeof setTimeout> | undefined
    const controller = new AbortController()

    async function poll() {
      try {
        const response = await fetch(`${apiBase}/api/dashboard`, {
          credentials: 'include',
          cache: 'no-store',
          signal: controller.signal,
        })

        if (!response.ok) {
          throw new Error(`Backend returned HTTP ${response.status}`)
        }

        const data: DashboardSnapshot = await response.json()

        if (!stopped) {
          setDashboard(data)
          setError(null)
        }
      } catch (cause) {
        if (!stopped) {
          setError(
            cause instanceof Error
              ? cause.message
              : 'Cannot reach backend',
          )
        }
      } finally {
        if (!stopped) {
          timer = setTimeout(poll, 1000)
        }
      }
    }

    void poll()

    return () => {
      stopped = true
      controller.abort()

      if (timer !== undefined) {
        clearTimeout(timer)
      }
    }
  }, [])

    const videoPath = dashboard?.media.video_url

  const videoUrl = videoPath
    ? new URL(videoPath, `${apiBase}/`).href
    : '/fanuc_twin_sort_then_stack.webm'

  const timeline = dashboard?.timeline ?? []

  const matchingEvent = timeline.find(
    (event) =>
      videoTime >= event.start_s && videoTime < event.end_s,
  )

  const lastEvent = timeline[timeline.length - 1]

  const playbackEvent =
    matchingEvent ??
    (lastEvent && videoTime >= lastEvent.end_s ? lastEvent : null)

  // Recorded video uses its timeline; without video, show latest run data.
  // Hide values when disconnected to avoid presenting stale data as current.
  const display = error
    ? null
    : videoUrl
      ? playbackEvent
      : dashboard?.display

  const cards: Array<[string, string]> = [
    ['Current phase', display?.phase ?? '—'],
    ['Grasp quality prediction', display?.prediction ?? '—'],
    ['Confidence', percent(display?.confidence)],
    ['State-machine step', display?.state_machine_step ?? '—'],
    ['RL correction', display?.rl_correction ?? '—'],
    ['Selected action', display?.selected_action ?? '—'],
    ['Grasp result', display?.grasp_result ?? '—'],
    ['Cycle time', measurement(display?.cycle_time_s, 's')],
    [
      'Placement error',
      measurement(display?.placement_error_mm, 'mm'),
    ],
    ['Overall success rate', percent(display?.overall_success_rate)],
  ]

  const connectionLabel = error
    ? 'Disconnected'
    : !dashboard
      ? 'Connecting'
      : videoUrl
        ? 'Playback data'
        : 'Latest run data'

  return (
    <div className="demo-layout">
      <section className="demo-card">
        <h2>Simulation video</h2>

        {videoUrl ? (
          <video
            key={`${dashboard?.run.id}:${videoUrl}`}
            className="simulation-video"
            src={videoUrl}
            controls
            preload="metadata"
            onLoadedMetadata={(event) => {
              setVideoTime(event.currentTarget.currentTime)
              setVideoError(false)
            }}
            onTimeUpdate={(event) =>
              setVideoTime(event.currentTarget.currentTime)
            }
            onSeeked={(event) =>
              setVideoTime(event.currentTarget.currentTime)
            }
            onError={() => setVideoError(true)}
          />
        ) : (
          <div className="video-placeholder">
            <strong>Video coming soon</strong>
            <p>
              The recording will appear when the backend makes it
              available.
            </p>
          </div>
        )}

        {videoUrl && videoError && (
          <p className="error-message" role="alert">
            The recording could not be played. Check its URL and format.
          </p>
        )}

        <p className="video-note">
          {videoUrl
            ? 'Cards follow event boundaries calculated from recorded cycle times.'
            : 'Cards show the latest backend run data while awaiting the video.'}
        </p>
      </section>

      <section className="demo-card">
        <div className="panel-heading">
          <h2>Model and decisions</h2>
          <span className="status-badge">{connectionLabel}</span>
        </div>

        {error && (
          <p className="error-message" role="alert">
            Cannot update run data: {error}. Retrying automatically.
          </p>
        )}

        {!dashboard && !error && (
          <p className="panel-note">Connecting to the backend…</p>
        )}

        {dashboard && (
          <p className="panel-note">Run: {dashboard.run.id}</p>
        )}

        {dashboard?.run.error && (
          <p className="error-message" role="alert">
            Run error: {dashboard.run.error}
          </p>
        )}

        {videoUrl && !playbackEvent && !error && (
          <p className="panel-note">
            No recorded event is available at this playback time.
          </p>
        )}

        <dl className="decision-list">
          {cards.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </section>
    </div>
  )
}