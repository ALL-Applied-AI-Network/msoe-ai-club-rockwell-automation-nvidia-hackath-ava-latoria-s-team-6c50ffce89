import { useState } from 'react'
import LiveDemo from './LiveDemo'
import './App.css'

const processes = [
  {
    number: '01',
    title: 'Observe',
    subtitle: 'Camera observations',
    description:
      'Camera observations provide the visual evidence used to evaluate a grasp.',
    details: [
      'The simulation supplies observations of the object and manipulation.',
      'The backend handles image processing and camera-validity checks.',
      'This dashboard displays the results; it does not analyze video frames.',
    ],
    code:
      'LiveDemo.tsx requests /api/dashboard and stores the response in dashboard state.',
  },
  {
    number: '02',
    title: 'Verify',
    subtitle: 'Grasp assessment',
    description:
      'The backend reports the predicted grasp quality and its confidence.',
    details: [
      'Prediction describes the classifier’s assessment.',
      'Confidence describes the model’s certainty about that prediction.',
      'The reported grasp result is a separate field; confidence alone does not establish success.',
    ],
    code:
      'The decision panel reads prediction, confidence, and grasp_result from the selected timeline event.',
  },
  {
    number: '03',
    title: 'Decide',
    subtitle: 'Task and recovery',
    description:
      'The state machine coordinates the task and reports the next action.',
    details: [
      'State-machine information identifies the reported manipulation step.',
      'Selected action describes the backend’s decision when available.',
      'RL correction displays any adjustment reported by the backend.',
    ],
    code:
      'The frontend displays state_machine_step, selected_action, and rl_correction. It does not issue robot commands.',
  },
  {
    number: '04',
    title: 'Place & review',
    subtitle: 'Outcome and measurements',
    description:
      'Recorded measurements help explain the outcome of each manipulation cycle.',
    details: [
      'Cycle time shows the recorded duration in seconds.',
      'Placement error shows the reported offset in millimeters.',
      'Overall success rate summarizes the result supplied by the backend.',
    ],
    code:
      'During playback, LiveDemo.tsx finds the timeline event whose start_s and end_s contain the video’s current time.',
  },
]

export default function App() {
  const [collapsed, setCollapsed] = useState(false)
  const [expandedProcess, setExpandedProcess] = useState<number | null>(null)

  return (
    <div className={`dashboard-shell ${collapsed ? 'is-collapsed' : ''}`}>
      <aside className="sidebar">
        <div className="sidebar-top">
          <a className="brand" href="#overview" aria-label="Innovation Lab home">
            <span className="brand-icon">IL</span>

            {!collapsed && (
              <span className="brand-copy">
                <strong>Innovation Lab</strong>
                <small>Robotics workspace</small>
              </span>
            )}
          </a>

          <button
            className="sidebar-toggle"
            type="button"
            onClick={() => setCollapsed(!collapsed)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-expanded={!collapsed}
            aria-controls="sidebar-navigation"
          >
            {collapsed ? '›' : '‹'}
          </button>
        </div>

        {!collapsed && <p className="navigation-label">WORKSPACE</p>}

        <nav id="sidebar-navigation" className="sidebar-navigation">
          <a href="#overview" title="Overview">
            <span className="navigation-icon" aria-hidden="true">◈</span>
            <span className={collapsed ? 'visually-hidden' : ''}>
              Overview
            </span>
          </a>

          <a href="#simulation" title="Simulation and run data">
            <span className="navigation-icon" aria-hidden="true">▷</span>
            <span className={collapsed ? 'visually-hidden' : ''}>
              Simulation & run
            </span>
          </a>

          <a href="#workflow" title="System workflow">
            <span className="navigation-icon" aria-hidden="true">▦</span>
            <span className={collapsed ? 'visually-hidden' : ''}>
              System workflow
            </span>
          </a>
        </nav>

        <div className="sidebar-footer">
          <span className="workspace-avatar">MS</span>

          {!collapsed && (
            <div>
              <strong>MSOE AI Lab</strong>
              <small>Simulation dashboard</small>
            </div>
          )}
        </div>
      </aside>

      <main className="dashboard-main">
        <header className="topbar">
          <span>Workspace / Robotics</span>
          <span className="environment-label">
            <span aria-hidden="true" />
            Simulation environment
          </span>
        </header>

        <section id="overview" className="dashboard-heading">
          <div>
            <p className="eyebrow">VISION-GUIDED ROBOTICS</p>
            <h1>Sorting & stacking</h1>
            <p className="heading-description">
              Explore the simulation, inspect run results, and follow the
              verification workflow.
            </p>
          </div>

          <a className="primary-link" href="#workflow">
            Explore workflow <span aria-hidden="true">↗</span>
          </a>
        </section>

        <section className="project-overview" aria-label="Project overview">
          <article className="overview-card">
            <span className="overview-symbol" aria-hidden="true">◎</span>
            <div>
              <h2>The challenge</h2>
              <p>Objects can shift or slip during pickup and placement.</p>
            </div>
          </article>

          <article className="overview-card">
            <span className="overview-symbol" aria-hidden="true">◈</span>
            <div>
              <h2>Our approach</h2>
              <p>Use visual verification and bounded recovery decisions.</p>
            </div>
          </article>

          <article className="overview-card">
            <span className="overview-symbol" aria-hidden="true">▤</span>
            <div>
              <h2>The demonstration</h2>
              <p>Review the recording alongside reported backend results.</p>
            </div>
          </article>
        </section>

        <section id="simulation" aria-labelledby="simulation-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">RUN EXPLORER</p>
              <h2 id="simulation-title">Simulation & insights</h2>
            </div>
            <span className="section-tag">Recorded run / backend data</span>
          </div>

          <LiveDemo />
        </section>

        <section
          id="workflow"
          className="workflow-section"
          aria-labelledby="workflow-title"
        >
          <div className="section-heading">
            <div>
              <p className="eyebrow">BEHIND THE PROCESS</p>
              <h2 id="workflow-title">How the system works</h2>
            </div>
            <span className="section-hint">Click a card to explore</span>
          </div>

          <div className="process-grid">
            {processes.map((process, index) => {
              const isExpanded = expandedProcess === index

              return (
                <article
                  className={`process-card ${isExpanded ? 'is-expanded' : ''}`}
                  key={process.number}
                >
                  <button
                    className="process-trigger"
                    type="button"
                    aria-expanded={isExpanded}
                    aria-controls={`process-detail-${index}`}
                    onClick={() =>
                      setExpandedProcess(isExpanded ? null : index)
                    }
                  >
                    <span className="process-topline">
                      <span className="process-number">{process.number}</span>
                      <span className="process-expand" aria-hidden="true">
                        {isExpanded ? '−' : '+'}
                      </span>
                    </span>

                    <span className="process-title">{process.title}</span>
                    <span className="process-subtitle">
                      {process.subtitle}
                    </span>
                    <span className="process-description">
                      {process.description}
                    </span>

                    <span className="process-action">
                      {isExpanded ? 'Close details ↑' : 'Explore process ↗'}
                    </span>
                  </button>

                  <div
                    id={`process-detail-${index}`}
                    className="process-detail"
                    hidden={!isExpanded}
                  >
                    <ul>
                      {process.details.map((detail) => (
                        <li key={detail}>{detail}</li>
                      ))}
                    </ul>

                    <div className="code-explanation">
                      <h3>What the frontend code does</h3>
                      <p>{process.code}</p>
                    </div>
                  </div>
                </article>
              )
            })}
          </div>
        </section>

        <footer className="page-footer">
          MSOE · Rockwell Automation · NVIDIA
          <span>
            Simulation results interface · Does not command the physical robot
          </span>
        </footer>
      </main>
    </div>
  )
}