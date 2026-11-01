import { useState } from 'react'
import {
  previewInstruction,
  submitFeedback,
  type DashboardSnapshot,
  type DryRunPlan,
  type FeedbackRequest,
} from './dashboard'
import { PHYSICAL_EPISODES, type PhysicalEpisode } from './demoData'

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
function episodeId(episode: PhysicalEpisode) { return `physical-episode-${episode.id}` }
function evidenceId(kind: 'grasp' | 'stack', episode: PhysicalEpisode) { return `physical-${kind}-${episode.id}` }

function ratingText(review: ReviewLabel | undefined) {
  return review === 'GOOD' ? 'Successful' : review === 'BAD' ? 'Needs work' : 'Not rated'
}

function EpisodeReview({ episode, review, graspReview, stackReview, onReview }: {
  episode: PhysicalEpisode
  review: ReviewLabel | undefined
  graspReview: ReviewLabel | undefined
  stackReview: ReviewLabel | undefined
  onReview: (kind: 'episode' | 'grasp' | 'stack', episode: PhysicalEpisode, label: ReviewLabel) => void
}) {
  return <div className="episode-review-panel">
    <label>Episode rating
      <select value={review ?? ''} onChange={(event) => event.target.value && onReview('episode', episode, event.target.value as ReviewLabel)}>
        <option value="">Not rated</option><option value="GOOD">Successful</option><option value="BAD">Needs work</option>
      </select>
    </label>
    <details>
      <summary>How this episode did</summary>
      <div className="episode-explanation">
        <p><strong>Overall:</strong> {ratingText(review)}</p>
        <p><strong>Grasp:</strong> {ratingText(graspReview)} · <strong>Stack:</strong> {ratingText(stackReview)}</p>
        <p>{review === 'GOOD' ? 'The full attempt was approved after reviewing the physical-machine recording.' : review === 'BAD' ? 'The attempt needs follow-up. Use the grasp and stack feedback pages to identify the failed step.' : 'Review the full recording, then use the dedicated side-camera grasp and stack images before assigning a rating.'}</p>
      </div>
    </details>
  </div>
}

function PhysicalEvidenceFeedback({ kind, reviews, onReview, syncStatus }: { kind: 'grasp' | 'stack'; reviews: ReviewMap; onReview: (kind: 'grasp' | 'stack', episode: PhysicalEpisode, label: ReviewLabel) => void; syncStatus: string }) {
  const reviewed = PHYSICAL_EPISODES.filter((episode) => reviews[evidenceId(kind, episode)])
  const title = kind === 'grasp' ? 'Grasp feedback' : 'Stack feedback'
  return <section className="review-view">
    <div className="review-intro"><div><p className="section-kicker">PHYSICAL MACHINE FEEDBACK</p><h2>{title}</h2><p>{kind === 'grasp' ? 'Approve or deny the visible pickup in each physical episode.' : 'Approve or deny the visible placement result in each physical episode.'}</p></div><div className="feedback-boundary"><strong>Side-camera evidence</strong><span>Each image is captured from the physical side recording at the relevant grasp or placement moment.</span></div></div>
    <div className="review-stats"><div><span>Episodes</span><strong>{PHYSICAL_EPISODES.length}</strong></div><div><span>Reviewed</span><strong>{reviewed.length}</strong></div><div><span>Approved</span><strong>{reviewed.filter((episode) => reviews[evidenceId(kind, episode)] === 'GOOD').length}</strong></div><div><span>Denied</span><strong>{reviewed.filter((episode) => reviews[evidenceId(kind, episode)] === 'BAD').length}</strong></div></div>
    <div className="physical-feedback-list">{PHYSICAL_EPISODES.map((episode) => {
      const review = reviews[evidenceId(kind, episode)]
      return <article key={episode.id}>
        <img src={kind === 'grasp' ? episode.graspThumbnail : episode.stackThumbnail} alt={`${episode.title} ${kind} from the side camera`} />
        <div className="physical-feedback-copy"><span>CSI side camera</span><strong>{episode.title} · {kind}</strong><small>{kind === 'grasp' ? 'Pickup checkpoint' : 'Placement checkpoint'} from the physical-machine recording</small></div>
        <div className="review-state"><span>Human result</span><strong className={review === 'GOOD' ? 'good' : review === 'BAD' ? 'bad' : ''}>{review === 'GOOD' ? 'Approved' : review === 'BAD' ? 'Denied' : 'Pending review'}</strong></div>
        <div className="review-actions"><button type="button" className={review === 'GOOD' ? 'good selected' : 'good'} onClick={() => onReview(kind, episode, 'GOOD')}>Approve</button><button type="button" className={review === 'BAD' ? 'bad selected' : 'bad'} onClick={() => onReview(kind, episode, 'BAD')}>Deny</button></div>
      </article>
    })}</div>
    <p className="feedback-status">{syncStatus}</p>
  </section>
}

function LabRuns({ reviews, onReview, syncStatus }: { reviews: ReviewMap; onReview: (kind: 'episode' | 'grasp' | 'stack', episode: PhysicalEpisode, label: ReviewLabel) => void; syncStatus: string }) {
  const [videoView, setVideoView] = useState<'episodes' | 'twin'>('episodes')
  return <section className="runs-view">
    <div className="video-tabs" role="tablist" aria-label="Run videos"><button type="button" className={videoView === 'episodes' ? 'active' : ''} onClick={() => setVideoView('episodes')}>Physical machine episodes</button><button type="button" className={videoView === 'twin' ? 'active' : ''} onClick={() => setVideoView('twin')}>Digital twin demonstration</button></div>
    {videoView === 'episodes' ? <>
      <div className="video-library">{PHYSICAL_EPISODES.map((episode) => <article className="video-card" key={episode.src}><div><strong>{episode.title}</strong><span>{episode.camera} · {episode.durationS.toFixed(1)} s</span><b className={`episode-rating ${reviews[episodeId(episode)] === 'GOOD' ? 'good' : reviews[episodeId(episode)] === 'BAD' ? 'bad' : ''}`}>{ratingText(reviews[episodeId(episode)])}</b></div><video controls preload="metadata" playsInline src={episode.src} /><EpisodeReview episode={episode} review={reviews[episodeId(episode)]} graspReview={reviews[evidenceId('grasp', episode)]} stackReview={reviews[evidenceId('stack', episode)]} onReview={onReview} /></article>)}</div>
    </> : <div className="video-library single"><article className="video-card"><div><strong>{TWIN_VIDEO.title}</strong><span>{TWIN_VIDEO.camera} · synchronized on Run monitor</span></div><video controls preload="metadata" playsInline src={TWIN_VIDEO.src} /></article></div>}
    <p className="feedback-status">{syncStatus}</p>
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
  const median = sorted[Math.floor(sorted.length / 2)]
  const episodeCount = PHYSICAL_EPISODES.length
  return <>
    <div className="metrics-scope"><div><p className="section-kicker">PHYSICAL MACHINE METRICS</p><h2>CSI episode evidence</h2><p>These numbers come from the 11 valid recorded machine episodes and your human validation—not the training or simulation test set.</p></div><button type="button" onClick={openEpisodes}>Review episode videos</button></div>
    <section className="headline-metrics"><article><span>Physical episodes</span><strong>{episodeCount}</strong><small>Episode 0 excluded: incorrect parts</small></article><article><span>Recorded machine time</span><strong>{totalDuration.toFixed(1)} s</strong><small>{(totalDuration / 60).toFixed(1)} minutes of valid evidence</small></article><article><span>Median episode time</span><strong>{median.toFixed(1)} s</strong><small>Range {sorted[0].toFixed(1)}–{sorted.at(-1)!.toFixed(1)} sec</small></article><article><span>Human validation</span><strong>{reviewed}/{episodeCount}</strong><small>{successRate == null ? 'Review episodes to calculate success rate' : `${pct(successRate)} successful among reviewed`}</small></article></section>
    <section className="machine-metrics-layout">
      <article className="panel episode-outcomes"><div className="panel-heading"><div><p className="section-kicker">OUTCOMES</p><h2>Human-validated episode results</h2></div><span className="evidence-label">Updates as you review</span></div><div className="outcome-summary"><div className="good"><strong>{successful}</strong><span>Successful</span></div><div className="bad"><strong>{needsWork}</strong><span>Needs work</span></div><div><strong>{pending}</strong><span>Pending review</span></div></div><div className="outcome-bar" aria-label={`${successful} successful, ${needsWork} needs work, ${pending} pending`}><span className="good" style={{ width: pct(successful / episodeCount) }} /><span className="bad" style={{ width: pct(needsWork / episodeCount) }} /><span className="pending" style={{ width: pct(pending / episodeCount) }} /></div><p className="scope-note">Unreviewed episodes stay pending instead of being counted as successes. This prevents the dashboard from overstating physical-machine performance.</p></article>
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
  const [view, setView] = useState<DashboardView>('runs'); const [syncStatus, setSyncStatus] = useState('Reviews are saved in this browser and sent to the feedback API when it is connected.')
  const [reviews, setReviews] = useState<ReviewMap>(() => { try { return JSON.parse(localStorage.getItem('stacksense-reviews') ?? '{}') as ReviewMap } catch { return {} } })
  function saveReview(id: string, label: ReviewLabel) { const next = { ...reviews, [id]: label }; setReviews(next); localStorage.setItem('stacksense-reviews', JSON.stringify(next)) }
  async function sendReview(payload: FeedbackRequest) { try { await submitFeedback(payload); setSyncStatus('Review saved to dashboard_feedback.jsonl for supervised offline improvement.') } catch { setSyncStatus('Review saved in this browser. The feedback API is currently unavailable.') } }
  async function reviewPhysical(kind: 'episode' | 'grasp' | 'stack', episode: PhysicalEpisode, label: ReviewLabel) { const id = kind === 'episode' ? episodeId(episode) : evidenceId(kind, episode); saveReview(id, label); await sendReview({ run_id: 'csi-remote-trial', event_id: id, kind: kind === 'grasp' ? 'grasp' : 'stack', label, stage: `physical_${kind}`, grab: episode.id, model_prediction: null, model_confidence: null }) }
  const validDuration = PHYSICAL_EPISODES.reduce((total, episode) => total + episode.durationS, 0)
  return <main className="trials-page"><section className="trials-hero"><div><p className="eyebrow">MACHINE EVIDENCE & FEEDBACK</p><h1>Episodes, metrics, and review</h1><p>Watch physical trials, validate outcomes, inspect digital-twin decisions, and preview new instructions.</p></div><div className="trial-count"><strong>{PHYSICAL_EPISODES.length}</strong><span>valid physical episodes</span><small>{validDuration.toFixed(1)} seconds recorded</small></div></section><div className="operations-layout"><aside className="operations-nav" aria-label="Evidence dashboard"><button className={view === 'runs' ? 'active' : ''} onClick={() => setView('runs')}><strong>Episode videos</strong></button><button className={view === 'metrics' ? 'active' : ''} onClick={() => setView('metrics')}><strong>Machine metrics</strong></button><button className={view === 'grasps' ? 'active' : ''} onClick={() => setView('grasps')}><strong>Grasp feedback</strong></button><button className={view === 'stacks' ? 'active' : ''} onClick={() => setView('stacks')}><strong>Stack feedback</strong></button><button className={view === 'instructions' ? 'active' : ''} onClick={() => setView('instructions')}><strong>Instructions</strong></button><div className="review-alert"><b>{Object.keys(reviews).length}</b><span>saved reviews</span></div></aside><div className="operations-content">{view === 'runs' && <LabRuns reviews={reviews} onReview={reviewPhysical} syncStatus={syncStatus} />}{view === 'metrics' && <MachineMetrics reviews={reviews} openEpisodes={() => setView('runs')} />}{view === 'grasps' && <PhysicalEvidenceFeedback kind="grasp" reviews={reviews} onReview={reviewPhysical} syncStatus={syncStatus} />}{view === 'stacks' && <PhysicalEvidenceFeedback kind="stack" reviews={reviews} onReview={reviewPhysical} syncStatus={syncStatus} />}{view === 'instructions' && <InstructionStudio snapshot={snapshot} />}</div></div></main>
}
