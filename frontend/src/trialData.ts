export const classifierEvaluation = {
  model: 'ResNet-18 Siamese',
  testSamples: 180,
  accuracy: 0.9944444444444445,
  macroF1: 0.994444272971388,
  unsafeGoodRate: 0.011111111111111112,
  latencyMs: 0.5519864333336955,
  confusion: [[90, 0], [1, 89]],
  source: 'results/grasp_benchmark/metrics.json',
}

export const rlEvaluation = {
  episodes: 2000,
  successRate: 0.635,
  baselineSuccessRate: 0.0905,
  collapseRate: 0.132,
  baselineCollapseRate: 0.753,
  safeStopRate: 0.2325,
  scope: 'Simulation-only high-level recovery decisions',
  source: 'results/rl_stacking_metrics.json',
}

export const modelSearch = [
  { name: 'ResNet-18 Siamese', accuracy: 0.9833, latency: 0.321, selected: true },
  { name: 'MobileNet V3 delta', accuracy: 0.9778, latency: 0.389, selected: false },
  { name: 'MobileNet V3 Siamese', accuracy: 0.9667, latency: 0.547, selected: false },
  { name: 'EfficientNet B0 Siamese', accuracy: 0.7778, latency: 0.823, selected: false },
]

export const trialMatrix = [
  { name: 'Nominal', variants: 4, risk: 'Baseline', modes: 'Open + closed loop' },
  { name: 'Pick XY shift', variants: 4, risk: 'Localization error', modes: 'Open + closed loop' },
  { name: 'Yaw error', variants: 3, risk: 'Pose error', modes: 'Open loop' },
  { name: 'Partial occlusion', variants: 1, risk: 'Camera visibility', modes: 'Closed loop' },
  { name: 'Touching cubes', variants: 1, risk: 'Ambiguous target', modes: 'Open loop' },
  { name: 'Low friction', variants: 1, risk: 'Slip', modes: 'Open loop' },
  { name: 'Distractor object', variants: 1, risk: 'False detection', modes: 'Open loop' },
  { name: 'Tilted cube', variants: 1, risk: 'Unstable grasp', modes: 'Closed loop' },
]
