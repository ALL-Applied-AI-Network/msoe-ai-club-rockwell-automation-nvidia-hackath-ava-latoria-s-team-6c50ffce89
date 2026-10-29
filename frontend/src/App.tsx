import LiveDemo from './LiveDemo'
import './App.css'

function App() {
  return (
    <main className="demo-page">
      <header className="demo-header">
        <p className="eyebrow">
          MSOE · Rockwell Automation + NVIDIA Innovation Lab
        </p>
        <h1>Vision-Guided Robotic Sorting and Stacking</h1>
        <p>
          Camera-based grasp verification and recovery decisions to help
          a robot pick, sort, and stack objects more reliably.
        </p>
      </header>

      <section className="project-overview" aria-label="Project overview">
        <article className="demo-card">
          <h2>The challenge</h2>
          <p>
            Objects can shift, slip, or be picked up incorrectly. Without
            checking the result, a robot can carry that mistake into
            sorting or stacking.
          </p>
        </article>

        <article className="demo-card">
          <h2>Our approach</h2>
          <p>
            Camera observations help classify grasp quality. A state
            machine coordinates the task and recovery steps, while
            reinforcement learning provides bounded placement
            adjustments in simulation.
          </p>
        </article>

        <article className="demo-card">
          <h2>The demonstration</h2>
          <p>
            Watch the simulation alongside classifier predictions,
            confidence, recovery decisions, and grasp results. The panel
            uses backend run data and follows the recorded event timeline
            when video is available.
          </p>
        </article>
      </section>

      <LiveDemo />

      <section className="workflow-section" aria-labelledby="workflow-title">
        <h2 id="workflow-title">How the system works</h2>
        <p className="workflow-intro">
          Feedback helps the system decide whether to continue, recover,
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
              Classify grasp quality and check confidence and camera
              validity.
            </p>
          </li>
          <li>
            <h3>Decide</h3>
            <p>
              Use the state machine to continue or select a bounded
              recovery step.
            </p>
          </li>
          <li>
            <h3>Place and check</h3>
            <p>
              Apply permitted placement adjustments and verify the result.
            </p>
          </li>
        </ol>

        <p className="project-footnote">
          This interface presents simulation results and does not command
          the physical robot.
        </p>
      </section>
    </main>
  )
}

export default App