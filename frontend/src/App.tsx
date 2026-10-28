import './App.css'

function App() {
  // Sample values for layout testing.
  // Later, these will come from the backend JSON.
  const sample = {
    phase: 'Sorting',
    prediction: 'GOOD',
    confidence: 0.94,
    step: 'Verify grasp',
    correction: 'No correction needed',
    action: 'Move object to sorting destination',
    graspResult: 'Pickup verified',
  }

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

          <video className="simulation-video" controls preload="metadata">
            <source src="/demo.mp4" type="video/mp4" />
            Your browser does not support video playback.
          </video>

          <p className="video-note">
            Recorded simulation. Use the controls to play, pause, or seek.
          </p>
        </section>

        <section className="demo-card">
          <div className="panel-heading">
            <h2>Model and decisions</h2>
            <span className="sample-badge">Sample data</span>
          </div>

          <p className="sample-note">
            These example outputs are not connected to the video or backend.
          </p>

          <dl className="decision-list">
            <div>
              <dt>Current phase</dt>
              <dd>{sample.phase}</dd>
            </div>

            <div>
              <dt>Grasp quality prediction</dt>
              <dd>{sample.prediction}</dd>
            </div>

            <div>
              <dt>Confidence</dt>
              <dd>{(sample.confidence * 100).toFixed(1)}%</dd>
            </div>

            <div>
              <dt>State-machine step</dt>
              <dd>{sample.step}</dd>
            </div>

            <div>
              <dt>RL correction</dt>
              <dd>{sample.correction}</dd>
            </div>

            <div>
              <dt>Selected action</dt>
              <dd>{sample.action}</dd>
            </div>

            <div>
              <dt>Grasp result</dt>
              <dd>{sample.graspResult}</dd>
            </div>
          </dl>
        </section>
      </div>
    </main>
  )
}
<section className="workflow-section" aria-labelledby="workflow-title">
  <h2 id="workflow-title">How the system works</h2>
  <p className="workflow-intro">
    The system uses feedback to decide whether to continue, recover,
    or stop safely.
  </p>

  <ol className="workflow-list">
    <li>
      <h3>Observe</h3>
      <p>Capture camera observations of the object and grasp.</p>
    </li>
    <li>
      <h3>Verify</h3>
      <p>
        Classify grasp quality and check confidence and camera validity.
      </p>
    </li>
    <li>
      <h3>Decide</h3>
      <p>
        Use the state machine to continue the task or select a bounded
        recovery step.
      </p>
    </li>
    <li>
      <h3>Place and check</h3>
      <p>
        Apply permitted placement adjustments and verify the result
        before continuing.
      </p>
    </li>
  </ol>

  <p className="project-footnote">
    This page presents a simulation demonstration. The interface does
    not command the physical robot.
  </p>
</section>

export default App