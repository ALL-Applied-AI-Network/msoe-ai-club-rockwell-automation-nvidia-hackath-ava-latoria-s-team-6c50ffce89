export type RunState = "running" | "completed" | "failed";
export type RunPhase = "starting" | "sort" | "stack" | "complete" | "error";
export type RunMode = "demo" | "fast" | "physics" | "integrated";

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
    actions: string[];
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

export interface DashboardSnapshot {
  schema_version: "1.0";
  run: {
    id: string;
    source: "simulation" | "physical_robot";
    state: string;
    phase: string;
    mode: string;
    updated_at: string | null;
    result: string;
    error: string | null;
  };
  display: {
    phase: string;
    prediction: string | null;
    confidence: number | null;
    state_machine_step: string;
    rl_correction: string | null;
    selected_action: string | null;
    grasp_result: "pending" | "verified" | "failed";
    cycle_time_s: number | null;
    placement_error_mm: number | null;
    overall_success_rate: number | null;
    model_status: string | null;
    model_reason: string | null;
  };
  planning: {
    status: "ready" | "waiting" | "blocked";
    planner: "deterministic" | "gemini" | null;
    instruction: string | null;
    action: string | null;
    source_color: string | null;
    destination_color: string | null;
    shape: string | null;
    size_mm: number | null;
    execution_skill: string | null;
    tower_order: string[] | null;
    requires_clarification: boolean;
    summary: string | null;
  };
  current: GraspEvent | Record<string, unknown> | null;
  progress: Record<string, number | null>;
  metrics: Record<string, number | boolean | null>;
  timeline: Array<{
    start_s: number;
    end_s: number;
    stage: string | null;
    grab: number | null;
    phase: string;
    prediction: string | null;
    confidence: number | null;
    state_machine_step: string;
    rl_correction: string | null;
    selected_action: string | null;
    grasp_result: "pending" | "verified" | "failed";
    cycle_time_s: number | null;
    placement_error_mm: number | null;
    overall_success_rate: number | null;
    model_status: string | null;
    model_reason: string | null;
  }>;
  timeline_basis: "cumulative_cycle_time";
  media: {
    video_url: string | null;
    video_content_type: string | null;
    latest_frames: Record<string, string>;
  };
  links: {
    status: string;
    events: string;
    metrics: string;
    artifacts: string;
  };
}
