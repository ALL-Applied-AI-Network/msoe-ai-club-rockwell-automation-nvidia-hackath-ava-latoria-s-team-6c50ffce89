export type RunState = "running" | "completed" | "failed";
export type RunPhase = "starting" | "sort" | "stack" | "complete" | "error";
export type RunMode = "demo" | "fast" | "physics";

export interface GraspEvent {
  stage: "sort" | "stack";
  grab: number;
  detection_id: number | null;
  tracked_object: string | null;
  predicted_class: string | null;
  confidence: number | null;
  probabilities: Record<string, number> | null;
  physics_pickup_success: boolean;
  classifier_correct: boolean | null;
  pickup_verified: boolean;
  placement_verified: boolean;
  placement_error_mm: number | null;
  cycle_time_s: number | null;
  visual_gate: string | null;
  camera_quality: { valid: boolean | null; reason: string | null };
  rl: {
    action: string | null;
    safe_stop: boolean;
    grasp_retries: number;
    placement_retries: number;
  };
}

export interface LiveStatus {
  schema_version: "1.0";
  run_id: string;
  updated_at: string;
  state: RunState;
  phase: RunPhase;
  mode: RunMode;
  progress: {
    completed_grabs: number;
    total_grabs: number;
    percent: number;
    sort_completed: number;
    stack_completed: number;
  };
  current: GraspEvent | null;
  metrics: {
    successful_pickups: number;
    verified_placements: number;
    classifier_scored: number;
    classifier_correct: number;
    classifier_accuracy: number | null;
    classifier_coverage: number | null;
  };
  events: GraspEvent[];
  summary: Record<string, unknown> | null;
  error: string | null;
  artifacts: {
    metrics_json: string;
    metrics_markdown: string;
    sort_report: string;
    stack_report: string;
  };
}
