import { useEffect, useMemo, useState } from 'react'
import './App.css'
import {
  apiUrl,
  displayForVideoTime,
  fetchDashboard,
  type ConnectionState,
  type DashboardSnapshot,
  type DisplayValues,
} from './dashboard'

const EMPTY_DISPLAY: DisplayValues = {
  phase: 'Waiting for run',
  prediction: null,
  confidence: null,
  state_machine_step: 'Waiting for telemetry',
  rl_correction: null,
  selected_action: null,
  grasp_result: 'pending',
  cycle_time_s: null,
  placement_error_mm: null,
  overall_success_rate: null,
}

function show(value: string | number | null, fallback = 'Not available'): string {
  return value === null || value === '' ? fallback : String(value).replaceAll('_', ' ')
}

function percent(value: number | null): string {
  return value === null ? 'Not scored yet' : `${(value * 100).toFixed(1)}%`
}

function App() {
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [message, setMessage] = useState('Connecting to the run data…')
  const [videoTime, setVideoTime] = useState(0)
  const [videoDuration, setVideoDuration] = useState(0)

  useEffect(() => {
    let active = true
    const controller = new AbortController()

    const refresh = async () => {
      try {
        const next = await fetchDashboard(controller.signal)
        if (!active) return
        setSnapshot(next)
        setConnection('live')
        setMessage(
          next.run.source === 'physical_robot'
            ? 'Live physical-robot telemetry'
            : next.run.mode === 'recorded'
              ? 'Recorded simulation data'
            : 'Live simulation data',
        )
      } catch (error) {
        if (!active || controller.signal.aborted) return
        setConnection('offline')
        setMessage(error instanceof Error ? error.message : 'Dashboard API is unavailable')
      }
    }

    void refresh()
    const interval = window.setInterval(() => void refresh(), 1000)
    return () => {
      active = false
      controller.abort()
      window.clearInterval(interval)
    }
  }, [])

  const display = useMemo(() => {
    if (!snapshot) return EMPTY_DISPLAY
    const timelineEnd = snapshot.timeline.at(-1)?.end_s ?? 0
    const timelineTime =
      videoDuration > 0 && timelineEnd > 0
        ? videoTime * (timelineEnd / videoDuration)
        : videoTime
    return displayForVideoTime(snapshot, timelineTime)
  }, [snapshot, videoDuration, videoTime])
  const plan = snapshot?.planning
  const videoUrl = apiUrl(snapshot?.media.video_url ?? null)

  return (
    <main className="demo-page">
      <header className="demo-header">
        <p className="eyebrow">Innovation Lab</p>
        <h1>Robot Demonstration</h1>
        <p>Watch the simulation alongside the model outputs and decisions.</p>
      </header>

      <div className="demo-layout">
        <section className="demo-card">
          <h2>Simulation video</h2>

          {videoUrl ? (
            <video
              className="simulation-video"
              controls
              preload="metadata"
              src={videoUrl}
              onTimeUpdate={(event) => setVideoTime(event.currentTarget.currentTime)}
              onLoadedMetadata={(event) => setVideoDuration(event.currentTarget.duration)}
            >
              Your browser does not support video playback.
            </video>
          ) : (
            <div className="video-placeholder">
              <strong>No recording found yet</strong>
              <span>Place an H.264 MP4 or WebM file in the selected run folder.</span>
            </div>
          )}

          <p className="video-note">
            The information on the right follows the run's recorded event timestamps as you
            play or seek.
          </p>
        </section>

        <section className="demo-card">
          <div className="panel-heading">
            <h2>Model and decisions</h2>
            <span className={`status-badge ${connection}`}>{message}</span>
          </div>

          {snapshot?.run.error && <p className="error-note">{snapshot.run.error}</p>}

          <dl className="decision-list">
            <div>
              <dt>Current phase</dt>
              <dd>{show(display.phase)}</dd>
            </div>

            <div>
              <dt>Grasp quality prediction</dt>
              <dd>{show(display.prediction, 'Not scored yet')}</dd>
            </div>

            <div>
              <dt>Confidence</dt>
              <dd>{percent(display.confidence)}</dd>
            </div>

            <div>
              <dt>State-machine step</dt>
              <dd>{show(display.state_machine_step)}</dd>
            </div>

            <div>
              <dt>RL correction</dt>
              <dd>{show(display.rl_correction, 'No correction requested')}</dd>
            </div>

            <div>
              <dt>Selected action</dt>
              <dd>{show(display.selected_action, 'Waiting')}</dd>
            </div>

            <div>
              <dt>Grasp result</dt>
              <dd className={`result-${display.grasp_result}`}>{show(display.grasp_result)}</dd>
            </div>

            <div>
              <dt>Cycle time</dt>
              <dd>{display.cycle_time_s === null ? 'Not available' : `${display.cycle_time_s.toFixed(1)} s`}</dd>
            </div>

            <div>
              <dt>Placement error</dt>
              <dd>{display.placement_error_mm === null ? 'Not available' : `${display.placement_error_mm.toFixed(1)} mm`}</dd>
            </div>

            <div>
              <dt>Overall success rate</dt>
              <dd>{percent(display.overall_success_rate)}</dd>
            </div>
          </dl>
        </section>
      </div>

      <section className="demo-card planning-card">
        <div className="panel-heading">
          <h2>Language planning stage</h2>
          <span className={`plan-badge ${plan?.status ?? 'waiting'}`}>
            {show(plan?.status ?? 'waiting')}
          </span>
        </div>
        <p className="planning-explanation">
          The planner turns the instruction into an allow-listed task. The state machine and
          safety checks still control every robot movement.
        </p>
        <dl className="planning-grid">
          <div>
            <dt>Instruction</dt>
            <dd>{show(plan?.instruction ?? null, 'Waiting for an instruction')}</dd>
          </div>
          <div>
            <dt>Planner used</dt>
            <dd>{show(plan?.planner ?? null, 'Not planned yet')}</dd>
          </div>
          <div>
            <dt>Validated plan</dt>
            <dd>{show(plan?.summary ?? null, 'No validated plan yet')}</dd>
          </div>
          <div>
            <dt>Safety handoff</dt>
            <dd>{plan?.status === 'ready' ? 'Approved for the state machine' : 'Robot motion remains blocked'}</dd>
          </div>
        </dl>
      </section>
    </main>
  )
}

export default App
