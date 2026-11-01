import type { DashboardSnapshot, TimelineEvent } from './dashboard'

export interface PhysicalEpisode {
  id: number
  title: string
  camera: string
  src: string
  durationS: number
  recordedRating: 'GOOD' | 'BAD'
  description: string
  outcome: string
  thumbnail: string
  graspThumbnail: string
  stackThumbnail: string
}

const PHYSICAL_EPISODE_DURATIONS = [
  16.8, 23.3, 28.9, 24.2, 30.2, 25.1, 22.2, 25.6, 6.6, 24.7, 21.7, 32.0,
]

const PHYSICAL_EPISODE_OUTCOMES: Record<number, Pick<PhysicalEpisode, 'recordedRating' | 'description' | 'outcome'>> = {
  1: { recordedRating: 'GOOD', description: 'A complete nominal cycle: approach, red-cube pickup, transfer to the blue cube, release, and retreat.', outcome: 'Completed red-on-blue stack in 23.3 seconds.' },
  2: { recordedRating: 'GOOD', description: 'The arm completed the full task with a longer approach and transfer than Episode 1.', outcome: 'Completed red-on-blue stack in 28.9 seconds.' },
  3: { recordedRating: 'GOOD', description: 'A steady full cycle with the final side-camera view confirming the two-cube tower remained upright.', outcome: 'Completed red-on-blue stack in 24.2 seconds.' },
  4: { recordedRating: 'GOOD', description: 'The pickup and placement succeeded despite this being one of the longer recorded motion sequences.', outcome: 'Completed red-on-blue stack in 30.2 seconds.' },
  5: { recordedRating: 'GOOD', description: 'The robot completed the commanded pickup, carry, placement, release, and retreat sequence.', outcome: 'Completed red-on-blue stack in 25.1 seconds.' },
  6: { recordedRating: 'GOOD', description: 'A fast complete cycle with a stable final stack visible in the side-camera recording.', outcome: 'Completed red-on-blue stack in 22.2 seconds.' },
  7: { recordedRating: 'GOOD', description: 'The full manipulation sequence completed and left the red cube supported on the blue cube.', outcome: 'Completed red-on-blue stack in 25.6 seconds.' },
  8: { recordedRating: 'BAD', description: 'This is a short partial recording. The stacked cubes are visible, but the clip does not contain enough of the motion cycle to validate the grasp and transfer.', outcome: 'Partial evidence only; full-cycle rating unavailable.' },
  9: { recordedRating: 'GOOD', description: 'The episode shows a completed pickup and placement with an upright red-on-blue result.', outcome: 'Completed red-on-blue stack in 24.7 seconds.' },
  10: { recordedRating: 'GOOD', description: 'The fastest complete recorded cycle, ending with the target stack upright after retreat.', outcome: 'Completed red-on-blue stack in 21.7 seconds.' },
  11: { recordedRating: 'GOOD', description: 'The longest complete trial still finished the commanded stack and left the tower upright.', outcome: 'Completed red-on-blue stack in 32.0 seconds.' },
}

// Episode 0 used the wrong part setup, so it is intentionally excluded from
// the evidence gallery and every calculated physical-machine metric.
export const PHYSICAL_EPISODES: PhysicalEpisode[] = PHYSICAL_EPISODE_DURATIONS.slice(1).map((durationS, offset) => {
  const index = offset + 1
  const result = PHYSICAL_EPISODE_OUTCOMES[index]
  return {
    id: index,
    title: `Episode ${index}`,
    camera: 'Table camera',
    src: `/csi_episode_${index}_table.mp4`,
    durationS,
    ...result,
    thumbnail: `/episode_thumbnails/csi_episode_${index}_table.jpg`,
    graspThumbnail: `/episode_evidence/episode_${index}_grasp_side.jpg`,
    stackThumbnail: `/episode_evidence/episode_${index}_stack_side.jpg`,
  }
})

const MOVE_DURATION_S = 9.2
export const DEMO_DURATION_S = 110.4
export const DEMO_MOVE_DURATION_S = MOVE_DURATION_S
const MOVES = [
  ...Array.from({ length: 6 }, (_, index) => ({ stage: 'sort', grab: index + 1 })),
  ...Array.from({ length: 6 }, (_, index) => ({ stage: 'stack', grab: index + 1 })),
]

const STEPS = [
  { duration: 0.9, step: 'observe', action: 'Detect and localize eligible cubes' },
  { duration: 1.2, step: 'approach', action: 'Move to the protected approach pose' },
  { duration: 0.8, step: 'descend', action: 'Descend vertically above the selected cube' },
  { duration: 0.6, step: 'close', action: 'Close the gripper around the cube' },
  { duration: 0.8, step: 'verify grasp', action: 'Compare the before-and-after image pair' },
  { duration: 0.9, step: 'lift', action: 'Lift vertically while holding the last target' },
  { duration: 1.7, step: 'carry', action: 'Carry the verified cube to its destination' },
  { duration: 1.2, step: 'place', action: 'Place with bounded alignment correction' },
  { duration: 0.6, step: 'release', action: 'Open the gripper and retreat vertically' },
  { duration: 0.5, step: 'verify placement', action: 'Check placement support before continuing' },
]

function buildTimeline(): TimelineEvent[] {
  const timeline: TimelineEvent[] = []
  let elapsed = 0

  MOVES.forEach((move, moveIndex) => {
    STEPS.forEach((step, stepIndex) => {
      const verified = stepIndex >= 4
      const placed = step.step === 'verify placement'
      const isStackCorrection = move.stage === 'stack' && ['carry', 'place'].includes(step.step)
      timeline.push({
        start_s: Number(elapsed.toFixed(2)),
        end_s: Number((elapsed + step.duration).toFixed(2)),
        stage: move.stage,
        grab: move.grab,
        phase: move.stage,
        prediction: verified ? 'GOOD' : null,
        confidence: null,
        state_machine_step: step.step,
        rl_correction: isStackCorrection ? 'PLACE_CENTER' : null,
        selected_action: step.action,
        grasp_result: verified ? 'verified' : 'pending',
        cycle_time_s: MOVE_DURATION_S,
        placement_error_mm: null,
        overall_success_rate: placed ? (moveIndex + 1) / MOVES.length : null,
        model_status: verified ? 'evaluated' : 'not_evaluated',
        model_reason: verified
          ? 'The recorded digital-twin verifier accepted this pickup; its per-frame confidence was not exported with the video.'
          : 'The image-pair verifier runs after the close-and-lift observation pair.',
      })
      elapsed += step.duration
    })
  })

  return timeline
}

export const DEMO_TIMELINE = buildTimeline()

export const DEMO_SNAPSHOT: DashboardSnapshot = {
  schema_version: '1.0',
  run: {
    id: 'isaac-sort-then-stack-demo',
    source: 'simulation',
    state: 'completed',
    phase: 'complete',
    mode: 'recorded',
    updated_at: null,
    result: 'success',
    error: null,
  },
  display: DEMO_TIMELINE.at(-1)!,
  planning: {
    status: 'ready',
    planner: 'deterministic',
    instruction: 'Detect the six cubes, sort them by color, then build a verified vertical tower.',
    action: 'sort_then_stack',
    source_color: null,
    destination_color: null,
    shape: 'cube',
    size_mm: 10,
    execution_skill: 'SafeSortThenStack',
    tower_order: ['yellow', 'yellow', 'blue', 'blue', 'red', 'red'],
    requires_clarification: false,
    summary: 'Sort six cubes, then place yellow → yellow → blue → blue → red → red with verification after every move.',
  },
  current: {
    policy_name: 'SafeSortThenStack',
    reason: 'Recorded digital-twin demonstration completed without a controller fault.',
    detections: { yellow: 2, blue: 2, red: 2 },
    detection_count: 6,
    latency_ms: null,
    action_safe: true,
    camera_names: ['table'],
    six_cube: {
      enabled: true,
      move_index: 6,
      move_count: 6,
      verified_layers: Array.from({ length: 6 }, (_, index) => ({ layer: index + 1 })),
    },
  },
  progress: {
    percent: 100,
    completed_grabs: 12,
    total_grabs: 12,
    sort_completed: 6,
    stack_completed: 6,
  },
  metrics: {
    successful_pickups: 12,
    verified_placements: 12,
    classifier_scored: 12,
    classifier_correct: 12,
    classifier_accuracy: 1,
    classifier_coverage: 1,
  },
  timeline: DEMO_TIMELINE,
  timeline_basis: 'cumulative_cycle_time',
  media: {
    video_url: '/fanuc_twin_sort_then_stack.webm',
    video_content_type: 'video/webm',
    latest_frames: {},
  },
  links: {},
}
