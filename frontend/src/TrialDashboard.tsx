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
import { PHYSICAL_DETAIL_VIDEOS, PHYSICAL_EPISODES, type PhysicalEpisode } from './demoData'

type DashboardView = 'metrics' | 'grasps' | 'stacks' | 'runs' | 'instructions'
type ReviewLabel = 'GOOD' | 'BAD'
type ReviewMap = Record<string, ReviewLabel>

const EXAMPLE_INSTRUCTIONS = [
  'Place the red 10 mm cube on top of the blue 10 mm cube.',
  'Detect all six 10 mm cubes and build one tower in yellow, yellow, blue, blue, red, red order.',
]

const TWIN_VIDEO = {
  title: 'Digital twin sort and stack',
  camera: 'Isaac Sim',
  src: '/fanuc_twin_sort_then_stack.webm',
}

function pct(value: number) { return `${(value * 100).toFixed(1)}%` }
function clean(value: string | null | undefined, fallback = 'Not available') {
  return value ? value.replaceAll('_', ' ') : fallback
}
function eventId(kind: 'grasp' | 'stack', event: TimelineEvent) {
  return `${kind}-${event.stage ?? 'run'}-${event.grab ?? 0}-${event.start_s}`
}
function episodeId(episode: PhysicalEpisode) { return `physical-episode-${episode.id}` }

function ReviewQueue({ kind, events, reviews, onReview, syncStatus }: {
  kind: 'grasp' | 'stack'
  events: TimelineEvent[]
  reviews: ReviewMap
  onReview: (kind: 'grasp' | 'stack', event: TimelineEvent, label: ReviewLabel) => void
  syncStatus: string
}) {
  const reviewed = events.filter((event) => reviews[eventId(kind, event)])
  const title = kind === 'grasp' ? 'Grasp review' : 'Stack review'
  return <section className="review-view">
    <div className="review-intro">
      <div><p className="section-kicker">DIGITAL-TWIN FEEDBACK</p><h2>{title}</h2><p>{kind === 'grasp' ? 'Confirm whether the recorded verifier result matches the visible pickup.' : 'Confirm tower stability and placement quality after every stack step.'}</p></div>
      <div className="feedback-boundary"><strong>Human-in-the-loop evidence</strong><span>Labels are stored for offline review and retraining. They never command the robot.</span></div>
    </div>
    <div className="review-stats">
      <div><span>Available</span><strong>{events.length}</strong></div><div><span>Reviewed</span><strong>{reviewed.length}</strong></div>
      <div><span>Good</span><strong>{reviewed.filter((event) => reviews[eventId(kind, event)] === 'GOOD').length}</strong></div>
      <div><span>Needs work</span><strong>{reviewed.filter((event) => reviews[eventId(kind, event)] === 'BAD').length}</strong></div>
    </div>
    <div className="review-list">{events.map((event, index) => {
      const id = eventId(kind, event); const review = reviews[id]
      return <article key={id}>
        <div className="review-number">{String(index + 1).padStart(2, '0')}</div>
        <div className="review-main"><span>{clean(event.stage)} · attempt {event.grab ?? index + 1}</span><strong>{kind === 'grasp' ? clean(event.prediction, 'No result') : clean(event.grasp_result)}</strong><small>{kind === 'grasp' ? 'Recorded image-pair verifier outcome' : 'Placement verification checkpoint'}</small></div>
        <div className="machine-label"><span>System result</span><strong>{clean(kind === 'grasp' ? event.prediction : event.grasp_result)}</strong></div>
        <div className="review-actions"><button type="button" className={review === 'GOOD' ? 'good selected' : 'good'} onClick={() => onReview(kind, event, 'GOOD')}>Good</button><button type="button" className={review === 'BAD' ? 'bad selected' : 'bad'} onClick={() => onReview(kind, event, 'BAD')}>Needs work</button></div>
      </article>
    })}</div>
    <p className="feedback-status">{syncStatus}</p>
  </section>
}

function EpisodeReviewButtons({ episode, review, onReview }: { episode: PhysicalEpisode; review: ReviewLabel | undefined; onReview: (episode: PhysicalEpisode, label: ReviewLabel) => void }) {
  return <div className="episode-feedback"><span>{review ? 'Human review saved' : 'Review this machine episode'}</span><div className="review-actions"><button type="button" className={review === 'GOOD' ? 'good selected' : 'good'} onClick={() => onReview(episode, 'GOOD')}>Successful</button><button type="button" className={review === 'BAD' ? 'bad selected' : 'bad'} onClick={() => onReview(episode, 'BAD')}>Needs work</button></div></div>
}

function PhysicalStackFeedback({ reviews, onReview, syncStatus }: { reviews: ReviewMap; onReview: (episode: PhysicalEpisode, label: ReviewLabel) => void; syncStatus: string }) {
  const reviewed = PHYSICAL_EPISODES.filter((episode) => reviews[episodeId(episode)])
  return <section className="review-view">
    <div className="review-intro"><div><p className="section-kicker">PHYSICAL STACK FEEDBACK</p><h2>Review machine outcomes</h2><p>Use the episode image and recording to label the visible placement outcome. These labels feed the machine metrics page.</p></div><div className="feedback-boundary"><strong>Human validation</strong><span>A still image is shown for quick review; open Episode videos when motion is needed to decide.</span></div></div>
    <div className="review-stats"><div><span>Episodes</span><strong>12</strong></div><div><span>Reviewed</span><strong>{reviewed.length}</strong></div><div><span>Successful</span><strong>{reviewed.filter((episode) => reviews[episodeId(episode)] === 'GOOD').length}</strong></div><div><span>Needs work</span><strong>{reviewed.filter((episode) => reviews[episodeId(episode)] === 'BAD').length}</strong></div></div>
    <div className="physical-feedback-list">{PHYSICAL_EPISODES.map((episode) => {
      const review = reviews[episodeId(episode)]
      return <article key={episode.id}>
        <img src={episode.thumbnail} alt={`Episode ${episode.id} machine view`} />
        <div className="physical-feedback-copy"><span>CSI physical robot</span><strong>{episode.title}</strong><small>{episode.durationS.toFixed(1)} seconds · table camera</small></div>
        <div className="review-state"><span>Human result</span><strong className={review === 'GOOD' ? 'good' : review === 'BAD' ? 'bad' : ''}>{review === 'GOOD' ? 'Successful' : review === 'BAD' ? 'Needs work' : 'Pending review'}</strong></div>
        <div className="review-actions"><button type="button" className={review === 'GOOD' ? 'good selected' : 'good'} onClick={() => onReview(episode, 'GOOD')}>Successful</button><button type="button" className={review === 'BAD' ? 'bad selected' : 'bad'} onClick={() => onReview(episode, 'BAD')}>Needs work</button></div>
      </article>
    })}</div>
    <p className="feedback-status">{syncStatus}</p>
  </section>
}

function LabRuns({ sessions, error, reviews, onReviewEpisode, syncStatus }: { sessions: SessionSummary[]; error: string | null; reviews: ReviewMap; onReviewEpisode: (episode: PhysicalEpisode, label: ReviewLabel) => void; syncStatus: string }) {
  const [videoView, setVideoView] = useState<'episodes' | 'twin'>('episodes')
  return <section className="runs-view">
    <div className="review-intro">
      <div><p className="section-kicker">ROBOT EVIDENCE</p><h2>Machine episodes</h2><p>Watch every CSI robot episode, record a human outcome, or switch to the complete synchronized digital-twin demonstration.</p></div>
      <div className="feedback-boundary"><strong>12 physical episodes · 281.3 seconds</strong><span>All table-camera recordings are local to this dashboard. Episode 2 also includes side and wrist views.</span></div>
    </div>
    <div className="video-tabs" role="tablist" aria-label="Run videos"><button type="button" className={videoView === 'episodes' ? 'active' : ''} onClick={() => setVideoView('episodes')}>Physical machine episodes</button><button type="button" className={videoView === 'twin' ? 'active' : ''} onClick={() => setVideoView('twin')}>Digital twin demonstration</button></div>
    {videoView === 'episodes' ? <>
      <div className="video-library">{PHYSICAL_EPISODES.map((episode) => <article className="video-card" key={episode.src}><div><strong>{episode.title}</strong><span>{episode.camera} · {episode.durationS.toFixed(1)} s</span></div><video controls preload="metadata" playsInline src={episode.src} /><EpisodeReviewButtons episode={episode} review={reviews[episodeId(episode)]} onReview={onReviewEpisode} /></article>)}</div>
      <div className="section-heading"><h2>Episode 2 camera detail</h2><span>Same attempt from two additional viewpoints</span></div>
      <div className="video-library detail-videos">{PHYSICAL_DETAIL_VIDEOS.map((video) => <article className="video-card" key={video.src}><div><strong>{video.title}</strong><span>{video.camera}</span></div><video controls preload="metadata" playsInline src={video.src} /></article>)}</div>
    </> : <div className="video-library single"><article className="video-card"><div><strong>{TWIN_VIDEO.title}</strong><span>{TWIN_VIDEO.camera} · synchronized on Run monitor</span></div><video controls preload="metadata" playsInline src={TWIN_VIDEO.src} /></article></div>}
    <p className="feedback-status">{syncStatus}</p>
    <div className="section-heading"><h2>Connected evidence sources</h2><span>{sessions.length} sources</span></div>
    {error && <p className="notice error">{error}</p>}
    <div className="run-cards">{sessions.map((session) => <article className={session.is_current ? 'current' : ''} key={session.id}><div className="run-card-top"><span className={`run-source ${session.source}`}>{session.source === 'physical_robot' ? 'CSI ARM' : 'ISAAC SIM'}</span>{session.is_current && <b>Active source</b>}</div><h3>{session.id}</h3><p>{clean(session.phase)} · {clean(session.state)}</p><dl><div><dt>Episodes / moves</dt><dd>{session.steps_recorded ?? session.completed_grabs ?? 'Recorded'}</dd></div><div><dt>Latency</dt><dd>{session.last_latency_ms == null ? 'Not exported' : `${session.last_latency_ms.toFixed(1)} ms`}</dd></div><div><dt>Safety</dt><dd>{session.last_action_safe == null ? 'Operator supervised' : session.last_action_safe ? 'Passed' : 'Blocked'}</dd></div><div><dt>Video</dt><dd>{session.has_video ? 'Available' : 'Pending'}</dd></div></dl><footer><time>{session.updated_at ? new Date(session.updated_at).toLocaleString() : 'Recorded evidence'}</time><span className={session.result === 'success' ? 'success' : ''}>{clean(session.result)}</span></footer></article>)}</div>
  </section>
}

function MachineMetrics({ reviews, openEpisodes }: { reviews: ReviewMap; openEpisodes: () => void }) {
  const labels = PHYSICAL_EPISODES.map((episode) => reviews[episodeId(episode)])
  const reviewed = labels.filter(Boolean).length
  const successful = labels.filter((label) => label === 'GOOD').length
  const needsWork = labels.filter((label) => label === 'BAD').length
  const pending = PHYSICAL_EPISODES.length - reviewed
  const successRate = reviewed ? successful / reviewed : null
  const totalDuration = PHYSICAL_EPISODES.reduce((total, episode) => total + episode.durationS, 0)
  const sorted = PHYSICAL_EPISODES.map((episode) => episode.durationS).sort((a, b) => a - b)
  const median = (sorted[5] + sorted[6]) / 2
  return <>
    <div className="metrics-scope"><div><p className="section-kicker">PHYSICAL MACHINE METRICS</p><h2>CSI episode evidence</h2><p>These numbers come from the 12 recorded machine episodes and your human validation—not the training or simulation test set.</p></div><button type="button" onClick={openEpisodes}>Review episode videos</button></div>
    <section className="headline-metrics"><article><span>Physical episodes</span><strong>12</strong><small>Table-camera recordings</small></article><article><span>Recorded machine time</span><strong>{totalDuration.toFixed(1)} s</strong><small>4 min 41.3 sec total</small></article><article><span>Median episode time</span><strong>{median.toFixed(1)} s</strong><small>Range 6.6–32.0 sec</small></article><article><span>Human validation</span><strong>{reviewed}/12</strong><small>{successRate == null ? 'Review episodes to calculate success rate' : `${pct(successRate)} successful among reviewed`}</small></article></section>
    <section className="machine-metrics-layout">
      <article className="panel episode-outcomes"><div className="panel-heading"><div><p className="section-kicker">OUTCOMES</p><h2>Human-validated episode results</h2></div><span className="evidence-label">Updates as you review</span></div><div className="outcome-summary"><div className="good"><strong>{successful}</strong><span>Successful</span></div><div className="bad"><strong>{needsWork}</strong><span>Needs work</span></div><div><strong>{pending}</strong><span>Pending review</span></div></div><div className="outcome-bar" aria-label={`${successful} successful, ${needsWork} needs work, ${pending} pending`}><span className="good" style={{ width: pct(successful / 12) }} /><span className="bad" style={{ width: pct(needsWork / 12) }} /><span className="pending" style={{ width: pct(pending / 12) }} /></div><p className="scope-note">Unreviewed episodes stay pending instead of being counted as successes. This prevents the dashboard from overstating physical-machine performance.</p></article>
      <article className="panel duration-panel"><div className="panel-heading"><div><p className="section-kicker">CYCLE TIME</p><h2>Recorded duration by episode</h2></div><span className="evidence-label">Machine video</span></div><div className="duration-list">{PHYSICAL_EPISODES.map((episode) => <div key={episode.id}><span>E{episode.id}</span><i><b style={{ width: pct(episode.durationS / 32) }} /></i><strong>{episode.durationS.toFixed(1)} s</strong></div>)}</div></article>
    </section>
  </>
}

function InstructionStudio({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  const [instruction, setInstruction] = useState(EXAMPLE_INSTRUCTIONS[0]); const [result, setResult] = useState<DryRunPlan | null>(null); const [error, setError] = useState<string | null>(null); const [loading, setLoading] = useState(false)
  async function plan() { setLoading(true); setError(null); try { setResult(await previewInstruction(instruction)) } catch (cause) { setResult(null); setError(cause instanceof Error ? cause.message : 'Instruction could not be planned') } finally { setLoading(false) } }
  return <section className="instruction-view"><div className="review-intro"><div><p className="section-kicker">NEW INSTRUCTION</p><h2>Create a safe task plan</h2><p>Enter a request to map it to a validated robot skill and review the plan before testing.</p></div><div className="feedback-boundary"><strong>Preview only</strong><span>This page creates a plan. It does not move the robot.</span></div></div><div className="instruction-workbench"><article className="instruction-editor"><label htmlFor="instruction">Robot instruction</label><textarea id="instruction" value={instruction} onChange={(event) => setInstruction(event.target.value)} rows={6} /><div className="example-row">{EXAMPLE_INSTRUCTIONS.map((example, index) => <button type="button" key={example} onClick={() => { setInstruction(example); setResult(null) }}>Load example {index + 1}</button>)}</div><button className="plan-button" type="button" onClick={plan} disabled={loading || !instruction.trim()}>{loading ? 'Checking instruction…' : 'Preview plan'}</button>{error && <p className="planner-error">{error}</p>}</article><article className="plan-preview"><div className="preview-head"><span>Proposed plan</span><b className={result?.planning.status ?? 'waiting'}>{clean(result?.planning.status, 'waiting')}</b></div>{result ? <><blockquote>{result.planning.summary ?? instruction}</blockquote><div className="plan-flow"><div><span>Language planner</span><strong>{clean(result.planning.planner)}</strong></div><i>→</i><div><span>Allowlisted skill</span><strong>{clean(result.planning.execution_skill ?? result.planning.action)}</strong></div><i>→</i><div><span>State machine</span><strong>Approval required</strong></div></div><dl className="plan-evidence"><div><dt>Nearby run sessions</dt><dd>{result.evidence.nearby_sessions}</dd></div><div><dt>Successful sessions</dt><dd>{result.evidence.successful_sessions}</dd></div><div><dt>Human labels</dt><dd>{result.evidence.human_feedback_labels}</dd></div><div><dt>Current run</dt><dd>{snapshot?.run.id ?? 'Offline'}</dd></div></dl><p className="safety-boundary">{result.safety_boundary}</p></> : <div className="preview-empty"><strong>No plan generated</strong><p>Enter an instruction to see the schema-validated skill, evidence context, and safety handoff.</p></div>}</article></div><div className="instruction-explainer"><div><span>1</span><strong>Interpret</strong><p>Gemini or the deterministic path extracts schema-approved task fields.</p></div><div><span>2</span><strong>Ground in evidence</strong><p>Prior run summaries and human labels show related evidence.</p></div><div><span>3</span><strong>Validate</strong><p>Unsupported objects, sizes, colors, or ambiguous wording are blocked.</p></div><div><span>4</span><strong>Hand off</strong><p>An approved skill still passes state-machine and workspace safety checks.</p></div></div></section>
}

export default function TrialDashboard({ snapshot }: { snapshot: DashboardSnapshot | null }) {
  const [view, setView] = useState<DashboardView>('runs'); const [sessions, setSessions] = useState<SessionSummary[]>([]); const [sessionError, setSessionError] = useState<string | null>(null); const [syncStatus, setSyncStatus] = useState('Reviews are saved in this browser and sent to the feedback API when it is connected.')
  const [reviews, setReviews] = useState<ReviewMap>(() => { try { return JSON.parse(localStorage.getItem('stacksense-reviews') ?? '{}') as ReviewMap } catch { return {} } })
  useEffect(() => { let mounted = true; async function refresh() { try { const next = await fetchSessions(); if (mounted) { setSessions(next); setSessionError(null) } } catch { if (mounted) setSessionError('The live session index is unavailable. The checked-in episode evidence remains available.') } } void refresh(); const timer = window.setInterval(() => void refresh(), 3000); return () => { mounted = false; window.clearInterval(timer) } }, [])
  const graspEvents = useMemo(() => snapshot?.timeline.filter((event) => event.state_machine_step === 'verify grasp') ?? [], [snapshot])
  function saveReview(id: string, label: ReviewLabel) { const next = { ...reviews, [id]: label }; setReviews(next); localStorage.setItem('stacksense-reviews', JSON.stringify(next)) }
  async function sendReview(payload: FeedbackRequest) { try { await submitFeedback(payload); setSyncStatus('Review saved to dashboard_feedback.jsonl for supervised offline improvement.') } catch { setSyncStatus('Review saved in this browser. The feedback API is currently unavailable.') } }
  async function review(kind: 'grasp' | 'stack', event: TimelineEvent, label: ReviewLabel) { const id = eventId(kind, event); saveReview(id, label); await sendReview({ run_id: snapshot?.run.id ?? 'digital-twin-replay', event_id: id, kind, label, stage: event.stage, grab: event.grab, model_prediction: event.prediction, model_confidence: event.confidence }) }
  async function reviewEpisode(episode: PhysicalEpisode, label: ReviewLabel) { const id = episodeId(episode); saveReview(id, label); await sendReview({ run_id: 'csi-remote-trial', event_id: id, kind: 'stack', label, stage: 'physical_episode', grab: episode.id, model_prediction: null, model_confidence: null }) }
  const physicalSession: SessionSummary = { id: 'CSI remote trial · 12 episodes', source: 'physical_robot', state: 'recorded', phase: 'operator-supervised', mode: 'physical', result: 'awaiting human review', updated_at: null, steps_recorded: 12, last_latency_ms: null, last_action_safe: true, completed_grabs: null, verified_placements: null, has_video: true, is_current: false }
  const replaySession: SessionSummary | null = snapshot ? { id: snapshot.run.id, source: snapshot.run.source, state: snapshot.run.state, phase: snapshot.run.phase, mode: snapshot.run.mode, result: snapshot.run.result, updated_at: snapshot.run.updated_at, steps_recorded: null, last_latency_ms: typeof snapshot.metrics.last_latency_ms === 'number' ? snapshot.metrics.last_latency_ms : null, last_action_safe: typeof snapshot.metrics.last_action_safe === 'boolean' ? snapshot.metrics.last_action_safe : true, completed_grabs: typeof snapshot.progress.completed_grabs === 'number' ? snapshot.progress.completed_grabs : null, verified_placements: typeof snapshot.metrics.verified_placements === 'number' ? snapshot.metrics.verified_placements : null, has_video: Boolean(snapshot.media.video_url), is_current: true } : null
  const evidenceSessions = [physicalSession, ...(sessions.length ? sessions : replaySession ? [replaySession] : [])]
  return <main className="trials-page"><section className="trials-hero"><div><p className="eyebrow">MACHINE EVIDENCE & FEEDBACK</p><h1>Episodes, metrics, and review</h1><p>Watch physical trials, validate outcomes, inspect digital-twin decisions, and preview new instructions.</p></div><div className="trial-count"><strong>12</strong><span>physical episodes</span><small>281.3 seconds recorded</small></div></section><div className="operations-layout"><aside className="operations-nav" aria-label="Evidence dashboard"><button className={view === 'runs' ? 'active' : ''} onClick={() => setView('runs')}><strong>Episode videos</strong></button><button className={view === 'metrics' ? 'active' : ''} onClick={() => setView('metrics')}><strong>Machine metrics</strong></button><button className={view === 'grasps' ? 'active' : ''} onClick={() => setView('grasps')}><strong>Grasp feedback</strong></button><button className={view === 'stacks' ? 'active' : ''} onClick={() => setView('stacks')}><strong>Stack feedback</strong></button><button className={view === 'instructions' ? 'active' : ''} onClick={() => setView('instructions')}><strong>Instructions</strong></button><div className="review-alert"><b>{Object.keys(reviews).length}</b><span>saved reviews</span></div></aside><div className="operations-content">{view === 'runs' && <LabRuns sessions={evidenceSessions} error={sessionError} reviews={reviews} onReviewEpisode={reviewEpisode} syncStatus={syncStatus} />}{view === 'metrics' && <MachineMetrics reviews={reviews} openEpisodes={() => setView('runs')} />}{view === 'grasps' && <ReviewQueue kind="grasp" events={graspEvents} reviews={reviews} onReview={review} syncStatus={syncStatus} />}{view === 'stacks' && <PhysicalStackFeedback reviews={reviews} onReview={reviewEpisode} syncStatus={syncStatus} />}{view === 'instructions' && <InstructionStudio snapshot={snapshot} />}</div></div></main>
}
