import { useEffect, useMemo, useState } from 'react'
import {
  fetchSessions,
  previewInstruction,
  submitFeedback,
  type DashboardSnapshot,
  type DryRunPlan,
  type FeedbackRequest,
  type SessionSummary,
  type TimelineEvent,
} from './dashboard'
import { classifierEvaluation, modelSearch, rlEvaluation, trialMatrix } from './trialData'

type DashboardView = 'metrics' | 'grasps' | 'stacks' | 'runs' | 'instructions'
type ReviewLabel = 'GOOD' | 'BAD'

const EXAMPLE_INSTRUCTIONS = [
  'Place the red 10 mm cube on top of the blue 10 mm cube.',
  'Detect all six 10 mm cubes and build one tower in yellow, yellow, blue, blue, red, red order.',
]

const TWIN_VIDEO = { title: 'Digital twin sort and stack', camera: 'Isaac Sim', src: '/fanuc_twin_sort_then_stack.webm' }
const CSI_EPISODES = Array.from({ length: 12 }, (_, index) => ({
  title: `Episode ${index + 1}`,
  camera: 'Table camera',
  src: `/csi_episode_${index}_table.mp4`,
}))
const CSI_DETAIL_VIDEOS = [
  { title: 'Episode 2', camera: 'Side camera', src: '/csi_episode_1_side.mp4' },
  { title: 'Episode 2', camera: 'Wrist camera', src: '/csi_episode_1_wrist.mp4' },
]

function pct(value: number) { return `${(value * 100).toFixed(1)}%` }
function clean(value: string | null | undefined, fallback = 'Not available') { return value ? value.replaceAll('_', ' ') : fallback }
function eventId(kind: 'grasp' | 'stack', event: TimelineEvent) { return `${kind}-${event.stage ?? 'run'}-${event.grab ?? 0}-${event.start_s}` }

function ReviewQueue({ kind, events, reviews, onReview, syncStatus }: {
  kind: 'grasp' | 'stack'
  events: TimelineEvent[]
  reviews: Record<string, ReviewLabel>
  onReview: (kind: 'grasp' | 'stack', event: TimelineEvent, label: ReviewLabel) => void
  syncStatus: string
}) {
  const reviewed = events.filter((event) => reviews[eventId(kind, event)])
  return <section className="review-view">
    <div className="review-intro"><div><p className="section-kicker">REVIEW</p><h2>{kind === 'grasp' ? 'Grasp review' : 'Stack review'}</h2><p>{kind === 'grasp' ? 'Confirm whether the model result matches the observed pickup.' : 'Confirm tower stability and placement quality after each stack step.'}</p></div><div className="feedback-boundary"><strong>Human feedback</strong><span>Labels are saved for offline model improvement. They do not control the robot.</span></div></div>
    <div className="review-stats"><div><span>Available</span><strong>{events.length}</strong></div><div><span>Reviewed</span><strong>{reviewed.length}</strong></div><div><span>Good</span><strong>{reviewed.filter((event) => reviews[eventId(kind, event)] === 'GOOD').length}</strong></div><div><span>Bad</span><strong>{reviewed.filter((event) => reviews[eventId(kind, event)] === 'BAD').length}</strong></div></div>
    {events.length ? <div className="review-list">{events.map((event, index) => { const id = eventId(kind, event); const review = reviews[id]; return <article key={id}><div className="review-number">{String(index + 1).padStart(2, '0')}</div><div className="review-main"><span>{clean(event.stage)} · attempt {event.grab ?? index + 1}</span><strong>{kind === 'grasp' ? clean(event.prediction, 'No classifier score') : clean(event.grasp_result, 'Pending')}</strong><small>{kind === 'grasp' ? `${event.confidence == null ? '—' : pct(event.confidence)} confidence` : `Placement error ${event.placement_error_mm == null ? '—' : `${event.placement_error_mm.toFixed(2)} mm`}`}</small></div><div className="machine-label"><span>System result</span><strong>{clean(kind === 'grasp' ? event.prediction : event.grasp_result)}</strong></div><div className="review-actions"><button type="button" className={review === 'GOOD' ? 'good selected' : 'good'} onClick={() => onReview(kind, event, 'GOOD')}>✓ Good</button><button type="button" className={review === 'BAD' ? 'bad selected' : 'bad'} onClick={() => onReview(kind, event, 'BAD')}>× Bad</button></div></article> })}</div> : <div className="review-empty"><strong>No reviewable events yet</strong><p>Connect the API to a completed run. New events will appear automatically.</p></div>}
    <p className="feedback-status">{syncStatus}</p>
  </section>
}

function LabRuns({ sessions, error }: { sessions: SessionSummary[]; error: string | null }) {
  const [videoView, setVideoView] = useState<'twin' | 'episodes'>('twin')
  const visibleVideos = videoView === 'twin' ? [TWIN_VIDEO] : [...CSI_EPISODES, ...CSI_DETAIL_VIDEOS]
  return <section className="runs-view">
    <div className="review-intro"><div><p className="section-kicker">VIDEOS</p><h2>Run recordings</h2><p>Switch between the full digital-twin demonstration and the latest CSI robot trial.</p></div><div className="feedback-boundary"><strong>15 recordings available</strong><span>12 episode videos plus side and wrist views for detailed review.</span></div></div>
    <div className="video-tabs" role="tablist" aria-label="Run videos"><button type="button" className={videoView === 'twin' ? 'active' : ''} onClick={() => setVideoView('twin')}>Digital twin</button><button type="button" className={videoView === 'episodes' ? 'active' : ''} onClick={() => setVideoView('episodes')}>Latest CSI trial · 12 episodes</button></div>
    <div className={`video-library ${videoView === 'twin' ? 'single' : ''}`}>{visibleVideos.map((video) => <article className="video-card" key={video.src}><div><strong>{video.title}</strong><span>{video.camera}</span></div><video controls preload="metadata" src={video.src} /></article>)}</div>
    <div className="section-heading"><h2>Run history</h2><span>{sessions.length} sessions</span></div>
    {error && <p className="notice error">{error}</p>}
    {sessions.length ? <div className="run-cards">{sessions.map((session) => <article className={session.is_current ? 'current' : ''} key={session.id}><div className="run-card-top"><span className={`run-source ${session.source}`}>{session.source === 'physical_robot' ? 'CSI ARM' : 'ISAAC SIM'}</span>{session.is_current && <b>Connected</b>}</div><h3>{session.id}</h3><p>{clean(session.phase)} · {clean(session.state)}</p><dl><div><dt>Steps</dt><dd>{session.steps_recorded ?? session.completed_grabs ?? '—'}</dd></div><div><dt>Latency</dt><dd>{session.last_latency_ms == null ? '—' : `${session.last_latency_ms.toFixed(1)} ms`}</dd></div><div><dt>Safety</dt><dd>{session.last_action_safe == null ? '—' : session.last_action_safe ? 'Passed' : 'Blocked'}</dd></div><div><dt>Video</dt><dd>{session.has_video ? 'Ready' : 'Pending'}</dd></div></dl><footer><time>{session.updated_at ? new Date(session.updated_at).toLocaleString() : 'No timestamp'}</time><span className={session.result === 'success' ? 'success' : ''}>{clean(session.result)}</span></footer></article>)}</div> : <div className="review-empty"><strong>No lab sessions found</strong><p>Start the API from a remote telemetry session folder to populate this view.</p></div>}
  </section>
}

function InstructionStudio({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  const [instruction, setInstruction] = useState(EXAMPLE_INSTRUCTIONS[0])
  const [result, setResult] = useState<DryRunPlan | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function plan() {
    setLoading(true); setError(null)
    try { setResult(await previewInstruction(instruction)) }
    catch (cause) { setResult(null); setError(cause instanceof Error ? cause.message : 'Instruction could not be planned') }
    finally { setLoading(false) }
  }

  return <section className="instruction-view">
    <div className="review-intro"><div><p className="section-kicker">NEW INSTRUCTION</p><h2>Create a safe task plan</h2><p>Enter a request to map it to a validated robot skill and review the plan before testing.</p></div><div className="feedback-boundary"><strong>Preview only</strong><span>This page creates a plan. It does not move the robot.</span></div></div>
    <div className="instruction-workbench">
      <article className="instruction-editor"><label htmlFor="instruction">Robot instruction</label><textarea id="instruction" value={instruction} onChange={(event) => setInstruction(event.target.value)} rows={6} /><div className="example-row">{EXAMPLE_INSTRUCTIONS.map((example, index) => <button type="button" key={example} onClick={() => { setInstruction(example); setResult(null) }}>Load example {index + 1}</button>)}</div><button className="plan-button" type="button" onClick={plan} disabled={loading || !instruction.trim()}>{loading ? 'Checking instruction…' : 'Preview plan'}</button>{error && <p className="planner-error">{error}</p>}</article>
      <article className="plan-preview"><div className="preview-head"><span>Proposed plan</span><b className={result?.planning.status ?? 'waiting'}>{clean(result?.planning.status, 'waiting')}</b></div>{result ? <><blockquote>{result.planning.summary ?? instruction}</blockquote><div className="plan-flow"><div><span>Language planner</span><strong>{clean(result.planning.planner)}</strong></div><i>→</i><div><span>Allowlisted skill</span><strong>{clean(result.planning.execution_skill ?? result.planning.action)}</strong></div><i>→</i><div><span>State machine</span><strong>Approval required</strong></div></div><dl className="plan-evidence"><div><dt>Nearby run sessions</dt><dd>{result.evidence.nearby_sessions}</dd></div><div><dt>Successful sessions</dt><dd>{result.evidence.successful_sessions}</dd></div><div><dt>Human labels</dt><dd>{result.evidence.human_feedback_labels}</dd></div><div><dt>Current run</dt><dd>{snapshot?.run.id ?? 'Offline'}</dd></div></dl><p className="safety-boundary">{result.safety_boundary}</p></> : <div className="preview-empty"><strong>No plan generated</strong><p>Enter an instruction to see the schema-validated skill, evidence context, and safety handoff.</p></div>}</article>
    </div>
    <div className="instruction-explainer"><div><span>1</span><strong>Interpret</strong><p>Gemini or the deterministic fast path extracts only schema-approved task fields.</p></div><div><span>2</span><strong>Ground in evidence</strong><p>Prior run summaries and human labels show how much related evidence exists.</p></div><div><span>3</span><strong>Validate</strong><p>Unsupported objects, sizes, colors, or ambiguous wording are blocked.</p></div><div><span>4</span><strong>Hand off</strong><p>An approved skill still passes through state-machine and workspace safety checks.</p></div></div>
  </section>
}

function MetricsView() {
  const successLift = rlEvaluation.successRate - rlEvaluation.baselineSuccessRate
  const collapseReduction = 1 - rlEvaluation.collapseRate / rlEvaluation.baselineCollapseRate
  return <>
    <section className="headline-metrics"><article><span>Held-out classifier accuracy</span><strong>{pct(classifierEvaluation.accuracy)}</strong><small>{classifierEvaluation.testSamples} balanced test pairs</small></article><article><span>Unsafe GOOD rate</span><strong>{pct(classifierEvaluation.unsafeGoodRate)}</strong><small>Lower is better</small></article><article><span>RL success lift</span><strong>+{pct(successLift)}</strong><small>{pct(rlEvaluation.successRate)} vs {pct(rlEvaluation.baselineSuccessRate)}</small></article><article><span>Collapse reduction</span><strong>{pct(collapseReduction)}</strong><small>{pct(rlEvaluation.collapseRate)} vs {pct(rlEvaluation.baselineCollapseRate)}</small></article></section>
    <section className="trial-layout"><article className="panel comparison-panel"><div className="panel-heading"><div><p className="section-kicker">STACKING POLICY</p><h2>Recovery policy results</h2></div><span className="evidence-label">2,000 test episodes</span></div><div className="bar-comparison"><div className="bar-group"><div className="bar-title"><strong>Episode success</strong><span>Higher is better</span></div><div className="bar-row"><span>RL policy</span><i><b style={{ width: pct(rlEvaluation.successRate) }} /></i><strong>{pct(rlEvaluation.successRate)}</strong></div><div className="bar-row baseline"><span>Open loop</span><i><b style={{ width: pct(rlEvaluation.baselineSuccessRate) }} /></i><strong>{pct(rlEvaluation.baselineSuccessRate)}</strong></div></div><div className="bar-group"><div className="bar-title"><strong>Stack collapse</strong><span>Lower is better</span></div><div className="bar-row safe"><span>RL policy</span><i><b style={{ width: pct(rlEvaluation.collapseRate) }} /></i><strong>{pct(rlEvaluation.collapseRate)}</strong></div><div className="bar-row baseline"><span>Open loop</span><i><b style={{ width: pct(rlEvaluation.baselineCollapseRate) }} /></i><strong>{pct(rlEvaluation.baselineCollapseRate)}</strong></div></div></div><p className="scope-note"><b>Scope:</b> {rlEvaluation.scope}. The policy selects bounded recovery actions; it never commands joints directly.</p></article>
    <article className="panel confusion-panel"><div className="panel-heading"><div><p className="section-kicker">GRASP CLASSIFIER</p><h2>Held-out confusion matrix</h2></div><span className="evidence-label">Test set</span></div><div className="matrix-wrap"><div className="axis-y">Actual class</div><div className="matrix"><span /><b>GOOD</b><b>SHIFTED</b><b>GOOD</b><div className="matrix-hit"><strong>90</strong><small>correct</small></div><div><strong>0</strong><small>false shifted</small></div><b>SHIFTED</b><div className="matrix-warn"><strong>1</strong><small>unsafe good</small></div><div className="matrix-hit"><strong>89</strong><small>correct</small></div></div><div className="axis-x">Predicted class</div></div><div className="classifier-foot"><span>Macro F1 <strong>{pct(classifierEvaluation.macroF1)}</strong></span><span>Inference <strong>{classifierEvaluation.latencyMs.toFixed(2)} ms</strong></span></div></article></section>
    <section className="trial-layout lower"><article className="panel"><div className="panel-heading"><div><p className="section-kicker">TEST COVERAGE</p><h2>Simulation conditions</h2></div><span className="evidence-label">16 trials</span></div><div className="trial-table"><div className="trial-table-head"><span>Condition</span><span>Failure pressure</span><span>Mode</span><span>Variants</span></div>{trialMatrix.map((trial) => <div key={trial.name}><strong>{trial.name}</strong><span>{trial.risk}</span><span>{trial.modes}</span><b>{trial.variants}</b></div>)}</div></article><article className="panel"><div className="panel-heading"><div><p className="section-kicker">MODEL COMPARISON</p><h2>Tested classifiers</h2></div><span className="evidence-label">4 models</span></div><div className="model-list">{modelSearch.map((model, index) => <div className={model.selected ? 'selected' : ''} key={model.name}><span>{index + 1}</span><div><strong>{model.name}</strong><small>{pct(model.accuracy)} accuracy · {model.latency.toFixed(2)} ms</small></div>{model.selected && <b>Selected</b>}</div>)}</div><p className="scope-note">Selection balances accuracy, unsafe-GOOD behavior, and latency.</p></article></section>
  </>
}

export default function TrialDashboard({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  const [view, setView] = useState<DashboardView>('metrics')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [sessionError, setSessionError] = useState<string | null>(null)
  const [syncStatus, setSyncStatus] = useState('Human labels are saved to the active run when its API is connected.')
  const [reviews, setReviews] = useState<Record<string, ReviewLabel>>(() => { try { return JSON.parse(localStorage.getItem('stacksense-reviews') ?? '{}') as Record<string, ReviewLabel> } catch { return {} } })
  useEffect(() => { let mounted = true; async function refresh() { try { const next = await fetchSessions(); if (mounted) { setSessions(next); setSessionError(null) } } catch { if (mounted) setSessionError('The session index is unavailable. Current run data can still be reviewed.') } } void refresh(); const timer = window.setInterval(() => void refresh(), 3000); return () => { mounted = false; window.clearInterval(timer) } }, [])
  const graspEvents = snapshot?.timeline ?? []
  const stackEvents = useMemo(() => graspEvents.filter((event) => event.stage === 'stack'), [graspEvents])

  async function review(kind: 'grasp' | 'stack', event: TimelineEvent, label: ReviewLabel) {
    const id = eventId(kind, event); const next = { ...reviews, [id]: label }; setReviews(next); localStorage.setItem('stacksense-reviews', JSON.stringify(next))
    if (!snapshot) { setSyncStatus('Saved in this browser. Connect the run API to add it to retraining evidence.'); return }
    const payload: FeedbackRequest = { run_id: snapshot.run.id, event_id: id, kind, label, stage: event.stage, grab: event.grab, model_prediction: event.prediction, model_confidence: event.confidence }
    try { await submitFeedback(payload); setSyncStatus('Label recorded in dashboard_feedback.jsonl for offline review and retraining.') } catch { setSyncStatus('Saved in this browser; the feedback API is unavailable.') }
  }

  const fallbackSessions: SessionSummary[] = snapshot ? [{ id: snapshot.run.id, source: snapshot.run.source, state: snapshot.run.state, phase: snapshot.run.phase, mode: snapshot.run.mode, result: snapshot.run.result, updated_at: snapshot.run.updated_at, steps_recorded: typeof snapshot.metrics.steps_recorded === 'number' ? snapshot.metrics.steps_recorded : null, last_latency_ms: typeof snapshot.metrics.last_latency_ms === 'number' ? snapshot.metrics.last_latency_ms : null, last_action_safe: typeof snapshot.metrics.last_action_safe === 'boolean' ? snapshot.metrics.last_action_safe : null, completed_grabs: snapshot.progress.completed_grabs ?? null, verified_placements: typeof snapshot.metrics.verified_placements === 'number' ? snapshot.metrics.verified_placements : null, has_video: Boolean(snapshot.media.video_url), is_current: true }] : []

  return <main className="trials-page">
    <section className="trials-hero">
      <div><p className="eyebrow">RESULTS & REVIEW</p><h1>Project dashboard</h1><p>Review model results, robot runs, human feedback, and new instructions.</p></div>
      <div className="trial-count"><strong>16</strong><span>simulation trials</span><small>8 test conditions</small></div>
    </section>
    <div className="operations-layout">
      <aside className="operations-nav" aria-label="Results dashboard">
        <button className={view === 'metrics' ? 'active' : ''} onClick={() => setView('metrics')}><strong>Metrics</strong></button>
        <button className={view === 'grasps' ? 'active' : ''} onClick={() => setView('grasps')}><strong>Grasp review</strong></button>
        <button className={view === 'stacks' ? 'active' : ''} onClick={() => setView('stacks')}><strong>Stack review</strong></button>
        <button className={view === 'instructions' ? 'active' : ''} onClick={() => setView('instructions')}><strong>Instructions</strong></button>
        <button className={view === 'runs' ? 'active' : ''} onClick={() => setView('runs')}><strong>Videos & runs</strong></button>
        <div className="review-alert"><b>{Object.keys(reviews).length}</b><span>saved reviews</span></div>
      </aside>
      <div className="operations-content">{view === 'metrics' && <MetricsView />}{view === 'grasps' && <ReviewQueue kind="grasp" events={graspEvents} reviews={reviews} onReview={review} syncStatus={syncStatus} />}{view === 'stacks' && <ReviewQueue kind="stack" events={stackEvents} reviews={reviews} onReview={review} syncStatus={syncStatus} />}{view === 'instructions' && <InstructionStudio snapshot={snapshot} />}{view === 'runs' && <LabRuns sessions={sessions.length ? sessions : fallbackSessions} error={sessionError} />}</div>
    </div>
  </main>
}
