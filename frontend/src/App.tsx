import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import {
  apiUrl,
  fetchDashboard,
  timelineEventAt,
  type ConnectionState,
  type DashboardSnapshot,
  type DisplayValues,
  type TimelineEvent,
} from './dashboard'
import TrialDashboard from './TrialDashboard'

const RECORDED_TWIN_VIDEO = '/fanuc_twin_sort_then_stack.webm'

const EMPTY_DISPLAY: DisplayValues = {
  phase: 'Waiting for telemetry',
  prediction: null,
  confidence: null,
  state_machine_step: 'Waiting for telemetry',
  rl_correction: null,
  selected_action: null,
  grasp_result: 'pending',
  cycle_time_s: null,
  placement_error_mm: null,
  overall_success_rate: null,
  model_status: null,
  model_reason: null,
}

const WORKFLOW = [
  { key: 'observe', label: 'Observe', detail: 'Detect and localize' },
  { key: 'verify', label: 'Verify', detail: 'Score the grasp' },
  { key: 'decide', label: 'Decide', detail: 'Choose next action' },
  { key: 'execute', label: 'Execute', detail: 'Move within limits' },
]

function compact(value: string | null | undefined, fallback = 'Not available') {
  return value ? value.replaceAll('_', ' ') : fallback
}

function percent(value: number | null | undefined) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

function numberValue(value: number | boolean | null | undefined, suffix = '') {
  return typeof value === 'number' ? `${Number(value.toFixed(2))}${suffix}` : '—'
}

function dateTime(value: string | null | undefined) {
  if (!value) return 'No timestamp yet'
  const parsed = new Date(value)
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString()
}

function workflowIndex(display: DisplayValues) {
  const value = `${display.phase} ${display.state_machine_step}`.toLowerCase()
  if (/detect|locate|starting|observe/.test(value)) return 0
  if (/verify|dwell|check/.test(value)) return 1
  if (/recover|retry|decide|plan|correction/.test(value)) return 2
  return 3
}

function Icon({ name }: { name: 'camera' | 'model' | 'route' | 'shield' | 'robot' }) {
  const paths = {
    camera: <><path d="M4 8h16v10H4z"/><path d="m8 8 1.5-3h5L16 8"/><circle cx="12" cy="13" r="3"/></>,
    model: <><rect x="5" y="5" width="14" height="14" rx="2"/><path d="M9 9h6v6H9zM2 9h3m14 0h3M2 15h3m14 0h3M9 2v3m6-3v3M9 19v3m6-3v3"/></>,
    route: <><circle cx="6" cy="6" r="2"/><circle cx="18" cy="18" r="2"/><path d="M8 6h5a3 3 0 0 1 3 3v6m-9 3h9"/></>,
    shield: <><path d="M12 3 5 6v5c0 5 3 8 7 10 4-2 7-5 7-10V6z"/><path d="m9 12 2 2 4-5"/></>,
    robot: <><path d="M5 20h14M8 20v-4l4-2 2-5 3 1"/><circle cx="17" cy="10" r="2"/><circle cx="12" cy="14" r="2"/><path d="M17 8V5h3v4"/></>,
  }
  return <svg viewBox="0 0 24 24" aria-hidden="true">{paths[name]}</svg>
}

function App() {
  const [activeTab, setActiveTab] = useState<'live' | 'trials'>('live')
  const [snapshot, setSnapshot] = useState<DashboardSnapshot | null>(null)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [connectionMessage, setConnectionMessage] = useState('Connecting to run data')
  const [videoTime, setVideoTime] = useState(0)
  const [videoError, setVideoError] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const hasSnapshot = useRef(false)

  useEffect(() => {
    let mounted = true

    async function refresh() {
      const controller = new AbortController()
      const timeout = window.setTimeout(() => controller.abort(), 2500)
      try {
        const next = await fetchDashboard(controller.signal)
        if (!mounted) return
        hasSnapshot.current = true
        setSnapshot(next)
        setConnection('live')
        setConnectionMessage(
          next.run.source === 'physical_robot'
            ? 'Live lab telemetry'
            : next.run.mode === 'recorded'
              ? 'Recorded run telemetry'
              : 'Live simulation telemetry',
        )
      } catch (error) {
        if (!mounted) return
        setConnection((previous) => (previous === 'live' || hasSnapshot.current ? 'stale' : 'offline'))
        setConnectionMessage(
          error instanceof Error && error.name !== 'AbortError'
            ? 'Run API unavailable'
            : 'Run API timed out',
        )
      } finally {
        window.clearTimeout(timeout)
      }
    }

    void refresh()
    const interval = window.setInterval(() => void refresh(), 800)
    return () => {
      mounted = false
      window.clearInterval(interval)
    }
  }, [])

  const apiVideo = apiUrl(snapshot?.media.video_url ?? null)
  const videoUrl = apiVideo ?? RECORDED_TWIN_VIDEO
  const isSyncedRecording = Boolean(apiVideo && snapshot?.timeline.length)
  const activeEvent = useMemo(
    () => (snapshot && isSyncedRecording ? timelineEventAt(snapshot, videoTime) : null),
    [isSyncedRecording, snapshot, videoTime],
  )
  const display = activeEvent ?? snapshot?.display ?? EMPTY_DISPLAY
  const current = snapshot?.current
  const plan = snapshot?.planning
  const activeWorkflow = workflowIndex(display)
  const totalGrabs = snapshot?.progress.total_grabs
  const completedGrabs = snapshot?.progress.completed_grabs
  const progress =
    typeof snapshot?.progress.percent === 'number'
      ? snapshot.progress.percent
      : typeof completedGrabs === 'number' && typeof totalGrabs === 'number' && totalGrabs > 0
        ? (completedGrabs / totalGrabs) * 100
        : 0
  const detectionNames = current?.detections ? Object.keys(current.detections) : []
  const sixCube = current?.six_cube
  const verifiedLayers = sixCube?.verified_layers?.length ?? 0
  const modelLabel =
    display.prediction ??
    (display.model_status === 'not_evaluated'
      ? 'Not evaluated on this physical step'
      : 'Waiting for image-pair evaluation')
  const sourceLabel = snapshot
    ? snapshot.run.source === 'physical_robot'
      ? 'Physical FANUC cell'
      : 'Isaac Sim digital twin'
    : 'Recorded digital twin demo'

  function seekTo(event: TimelineEvent) {
    if (!videoRef.current || !isSyncedRecording) return
    videoRef.current.currentTime = event.start_s
    setVideoTime(event.start_s)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#overview" aria-label="Dashboard home">
          <img className="brand-logo" src="/stacksense-logo.png" alt="StackSense" />
        </a>
        <nav className="app-tabs" aria-label="Dashboard views">
          <button type="button" className={activeTab === 'live' ? 'active' : ''} onClick={() => setActiveTab('live')}>Run monitor</button>
          <button type="button" className={activeTab === 'trials' ? 'active' : ''} onClick={() => setActiveTab('trials')}>Results & review</button>
        </nav>
        <div className="topbar-meta">
          <span className={`connection-pill ${connection}`}><i />{connectionMessage}</span>
          <span className="read-only"><Icon name="shield" /> Monitor only</span>
        </div>
      </header>

      {activeTab === 'live' ? <main id="overview">
        <section className="hero">
          <div>
            <p className="eyebrow">CURRENT RUN</p>
            <h1>Robot run monitor</h1>
            <p className="hero-copy">
              Watch the run and review the current model prediction, robot step, safety status,
              and measured results in one place.
            </p>
          </div>
          <div className="run-identity">
            <span>Data source</span>
            <strong>{sourceLabel}</strong>
            <small>{snapshot ? `Run ${snapshot.run.id}` : 'Start the API for synchronized run data'}</small>
          </div>
        </section>

        <section className="run-strip" aria-label="Run overview">
          <div><span>Run state</span><strong>{compact(snapshot?.run.state, 'Demo only')}</strong></div>
          <div><span>Current phase</span><strong>{compact(display.phase)}</strong></div>
          <div><span>Safety</span><strong className={current?.action_safe === false ? 'danger' : 'good'}>{current?.action_safe === false ? 'Blocked' : current?.action_safe === true ? 'Targets validated' : 'No fault reported'}</strong></div>
          <div><span>Updated</span><strong>{dateTime(snapshot?.run.updated_at)}</strong></div>
        </section>

        <section className="workspace" aria-label="Video and live decision data">
          <article className="panel video-panel">
            <div className="panel-heading">
              <div><p className="section-kicker">VIDEO</p><h2>{apiVideo ? 'Run recording' : 'Digital twin recording'}</h2></div>
              <span className={`source-badge ${isSyncedRecording ? 'synced' : ''}`}>{isSyncedRecording ? 'Telemetry synced' : 'Video reference'}</span>
            </div>
            <div className="video-frame">
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                preload="metadata"
                onTimeUpdate={(event) => setVideoTime(event.currentTarget.currentTime)}
                onSeeked={(event) => setVideoTime(event.currentTarget.currentTime)}
                onError={() => setVideoError(true)}
                onLoadedData={() => setVideoError(false)}
              />
              <span className="video-label">{apiVideo ? snapshot?.run.source.replaceAll('_', ' ') : 'recorded twin'}</span>
            </div>
            {videoError && <p className="notice error">The recording could not be loaded in this browser.</p>}
            {!isSyncedRecording && (
              <p className="notice">
                This checked-in video is a visual reference. Start the dashboard API with a run
                folder to synchronize the panel with that run’s real telemetry.
              </p>
            )}
            {snapshot?.run.source === 'physical_robot' && !apiVideo && (
              <p className="notice">
                Live policy telemetry is connected. The lab video appears here after the operator
                recording is downloaded into the same telemetry folder.
              </p>
            )}
          </article>

          <article className="panel decision-panel">
            <div className="panel-heading">
              <div><p className="section-kicker">ROBOT STATUS</p><h2>Current step</h2></div>
              <span className={`result-badge ${display.grasp_result}`}>{compact(display.grasp_result)}</span>
            </div>

            <div className="workflow" aria-label="Closed-loop workflow">
              {WORKFLOW.map((step, index) => (
                <div className={index === activeWorkflow ? 'active' : index < activeWorkflow ? 'done' : ''} key={step.key}>
                  <span>{index + 1}</span><strong>{step.label}</strong><small>{step.detail}</small>
                </div>
              ))}
            </div>

            <div className="decision-grid">
              <div className="decision-card model-card">
                <span className="card-icon"><Icon name="model" /></span>
                <div><small>Grasp verifier</small><strong>{modelLabel}</strong><p>{display.confidence == null ? compact(display.model_reason, 'No score has been emitted') : `${percent(display.confidence)} confidence`}</p></div>
              </div>
              <div className="decision-card">
                <span className="card-icon"><Icon name="route" /></span>
                <div><small>State machine</small><strong>{compact(display.state_machine_step)}</strong><p>{compact(display.selected_action, 'Waiting for next permitted action')}</p></div>
              </div>
              <div className="decision-card">
                <span className="card-icon"><Icon name="shield" /></span>
                <div><small>Safety gate</small><strong>{snapshot?.run.error ? 'Stopped' : current?.action_safe === false ? 'Motion blocked' : 'Within policy limits'}</strong><p>{compact(current?.reason, snapshot?.run.error ?? 'No controller fault reported')}</p></div>
              </div>
              <div className="decision-card">
                <span className="card-icon"><Icon name="camera" /></span>
                <div><small>Perception</small><strong>{detectionNames.length ? `${detectionNames.length} tracked target${detectionNames.length === 1 ? '' : 's'}` : 'Waiting for detections'}</strong><p>{detectionNames.length ? detectionNames.join(' · ') : compact(current?.camera_names?.join(', '), 'No camera metadata yet')}</p></div>
              </div>
            </div>
          </article>
        </section>

        <section className="evidence-grid">
          <article className="panel plan-panel">
            <div className="panel-heading"><div><p className="section-kicker">TASK</p><h2>Instruction and plan</h2></div><span className={`plan-status ${plan?.status ?? 'waiting'}`}>{compact(plan?.status, 'waiting')}</span></div>
            <blockquote>{plan?.instruction ?? 'Waiting for a validated task instruction.'}</blockquote>
            <div className="plan-route">
              <div><span>Planner</span><strong>{compact(plan?.planner, 'Not selected')}</strong></div><b>→</b>
              <div><span>Validated skill</span><strong>{compact(plan?.execution_skill ?? plan?.action, 'No skill selected')}</strong></div><b>→</b>
              <div><span>Safety handoff</span><strong>{plan?.status === 'ready' ? 'Approved' : 'Motion remains blocked'}</strong></div>
            </div>
            {plan?.summary && <p className="plan-summary">{plan.summary}</p>}
          </article>

          <article className="panel metrics-panel">
            <div className="panel-heading"><div><p className="section-kicker">RESULTS</p><h2>Run metrics</h2></div><span className="evidence-label">Recorded values</span></div>
            <dl className="metric-list">
              <div><dt>Completed actions</dt><dd>{numberValue(completedGrabs)}<small>{typeof totalGrabs === 'number' ? ` of ${totalGrabs}` : ''}</small></dd></div>
              <div><dt>Verified placements</dt><dd>{numberValue(snapshot?.metrics.verified_placements)}</dd></div>
              <div><dt>Classifier accuracy</dt><dd>{percent(typeof snapshot?.metrics.classifier_accuracy === 'number' ? snapshot.metrics.classifier_accuracy : null)}</dd></div>
              <div><dt>Policy latency</dt><dd>{numberValue(current?.latency_ms, ' ms')}</dd></div>
              <div><dt>Placement error</dt><dd>{numberValue(display.placement_error_mm, ' mm')}</dd></div>
              <div><dt>Verified tower layers</dt><dd>{sixCube?.enabled ? verifiedLayers : '—'}<small>{sixCube?.move_count ? ` of ${sixCube.move_count}` : ''}</small></dd></div>
            </dl>
            <div className="progress-track" aria-label={`Run progress ${Math.round(progress)} percent`}><span style={{ width: `${Math.min(100, Math.max(0, progress))}%` }} /></div>
          </article>
        </section>

        <section className="panel timeline-panel">
          <div className="panel-heading"><div><p className="section-kicker">TIMELINE</p><h2>Run events</h2></div><span className="evidence-label">{snapshot?.timeline.length ?? 0} events</span></div>
          {snapshot?.timeline.length ? (
            <div className="timeline-list">
              {snapshot.timeline.slice(-12).map((event, index) => (
                <button type="button" onClick={() => seekTo(event)} className={activeEvent === event ? 'active' : ''} key={`${event.stage}-${event.grab}-${index}`} disabled={!isSyncedRecording}>
                  <span>{event.start_s.toFixed(1)}s</span><strong>{compact(event.state_machine_step)}</strong><small>{compact(event.prediction, compact(event.selected_action, 'No model score'))}</small><i className={event.grasp_result} />
                </button>
              ))}
            </div>
          ) : (
            <p className="empty-state">No event timeline is available yet. Live physical decisions still appear in the current-decision panel above.</p>
          )}
        </section>
      </main> : <TrialDashboard snapshot={snapshot} />}

      <footer><span>StackSense robot monitoring</span><span>View and review only — robot control remains separate</span></footer>
    </div>
  )
}

export default App
