# Values and video contract

The complete example payload is frontend/public/dashboard.template.json;
the executable contract and validation are frontend/src/dashboardContract.ts.
Use JSON numbers without quotes, units, percent signs, NaN, or Infinity.
Use null for unavailable values, never zero as a placeholder. Zero is displayed.
All DisplayValues keys must exist in display AND in every timeline entry.

| Field | Input example | Display / meaning |
| --- | --- | --- |
| schema_version | "1.0" | Required contract version |
| run.id | "stack-001" | Run identifier; change when switching runs |
| run.error | null | String for an actual run error |
| media.video_url | "/media/recording.webm" | API-relative path in backend mode; HTTP(S) absolute URLs also work |
| display.phase | "stack" | Current phase |
| display.prediction | "secure" | Exact model class label; example label is not a required classifier enum |
| display.confidence | 0.963 | 96.3%; valid range 0–1 |
| display.state_machine_step | "verify" | Actual reported state-machine step |
| display.rl_correction | "dx=1.2 mm; dy=-0.5 mm" | Human-readable adjustment with units; null if unknown |
| display.selected_action | "PROCEED" | Actual policy action; do not invent it |
| display.grasp_result | "Pickup verified" | Reported verification result |
| display.cycle_time_s | 12.5 | 12.50 s; nonnegative seconds |
| display.placement_error_mm | 2.4 | 2.40 mm; convert meters ×1000 before sending |
| display.overall_success_rate | 0.833 | 83.3%; valid range 0–1 |
| timeline[].start_s | 5 | Seconds measured from recording start, inclusive |
| timeline[].end_s | 12.5 | Seconds measured from recording start, exclusive |

The other timeline fields have exactly the same names/types as display.
Timeline entries contain full values, not partial changes. Entries must be
sorted, nonoverlapping, and have end_s > start_s >= 0. Gaps are permitted and
display blank cards. Values after the last event are also blank. At a shared
boundary the next event is selected. At the exact video end the final interval
is no longer selected if its end_s equals the duration.

## Map older LiveStatus data in the backend

| Dashboard field | Possible existing source / required decision |
| --- | --- |
| run.id / run.error | run_id / error |
| display.phase | phase |
| prediction / confidence | current.predicted_class / current.confidence |
| cycle_time_s / placement_error_mm | current fields with the same names |
| selected_action | current.rl.action, after confirming it represents the policy action you want shown |
| state_machine_step | Requires an actual state-machine trace; older LiveStatus has no equivalent field |
| rl_correction | Requires actual correction details; action alone is not a displacement |
| grasp_result | Map pickup_verified and camera validity with an explicit team-agreed label |
| overall_success_rate | Define numerator/denominator first; no direct equivalent in older LiveStatus |
| media / timeline | Requires recording metadata and timestamps; absent from older LiveStatus |

If camera_quality.valid is false, supply prediction="NO SCORE" and confidence=null;
do not present an invalid classifier observation as success. Keep physics pickup
verification distinct from classifier confidence. Define overall_success_rate
consistently (for example verified placements / completed attempts); use null
when the denominator is zero. Classifier accuracy is a different measurement.
Do not derive state-machine steps or timestamps from cycle durations alone.
If logs only identify completed grasps, describe those intervals as completed
grasp results instead of claiming they identify every intermediate step.

## Backend and recording checklist

- Return JSON with Content-Type: application/json. Update run data atomically.
- Allow the frontend's exact origin with credentialed CORS when cross-origin;
  this frontend sends credentials: include. Do not put secrets in VITE variables.
- Serve the video bytes with an appropriate video MIME type and byte-range
  support for seeking. A GitHub blob page is HTML, not a video URL.
- Use a browser-playable WebM or MP4. For cross-origin authenticated media,
  verify the video URL's own authentication works in the target browser.
- Keep media.video_url and timeline tied to run.id. When publishing a new run,
  update them together. A video without a matching timeline shows blank cards.
- Without video, display supplies latest run values. During playback, only the
  matching timeline event supplies values. Invalid responses hide stale cards.
- With a video load failure, cards are blank to avoid implying synchronization.

## Acceptance examples

confidence=0 must show 0.0%; confidence=null must show an em dash.
confidence=96.3 must be rejected with a fraction-format message.
cycle_time_s=12.5 must show 12.50 s; placement_error_mm=2.4 shows 2.40 mm.
Seek to 4.9 s, 5.0 s and 12.5 s with the sample intervals: first, second,
then no event respectively. Replace these samples with real evidence before demo.
