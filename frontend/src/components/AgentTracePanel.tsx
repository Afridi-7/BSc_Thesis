import type { AgentTraceStep } from "../types";

interface Props {
  trace?: AgentTraceStep[];
  mode?: string;
}

function summarise(step: AgentTraceStep): string {
  if (step.type === "tool_call") {
    const args = step.args ?? {};
    const argStr = Object.keys(args).length
      ? Object.entries(args)
          .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
          .join(", ")
      : "";
    return `→ ${step.name ?? "?"}(${argStr})`;
  }
  if (step.type === "tool_result") {
    return `← ${step.name ?? "?"}`;
  }
  return "thought";
}

export default function AgentTracePanel({ trace, mode }: Props) {
  // No analysis run yet — show neutral placeholder.
  if (mode === undefined && (trace === undefined || trace.length === 0)) {
    return (
      <div className="card">
        <h2>Agent Trace</h2>
        <p className="empty">Run an analysis to view the reasoning trace.</p>
      </div>
    );
  }

  if (mode && mode !== "agent") {
    return (
      <div className="card">
        <h2>Agent Trace</h2>
        <p className="empty">
          Linear reasoning mode is active. Set
          <code> reasoning.mode: agent </code>
          in <code>config.yaml</code> to enable the LangChain ReAct agent and
          see its tool-use trace here.
        </p>
      </div>
    );
  }

  if (!trace || trace.length === 0) {
    return (
      <div className="card">
        <h2>Agent Trace</h2>
        <p className="empty">Agent produced no intermediate steps for this run.</p>
      </div>
    );
  }

  return (
    <div className="card">
      <h2>Agent Trace ({trace.length} steps)</h2>
      <p className="card-subtitle">
        ReAct trace from the LangChain agent. Each step shows the tool the LLM
        decided to invoke, its arguments, and the observation it received back.
      </p>
      <ol className="agent-trace">
        {trace.map((step, i) => (
          <li key={i} className={`trace-step trace-${step.type}`}>
            <details open={i < 3}>
              <summary>
                <span className="trace-index">#{i + 1}</span>
                <span className="trace-summary">{summarise(step)}</span>
              </summary>
              {step.type === "tool_call" && step.args && (
                <pre className="trace-body">
                  {JSON.stringify(step.args, null, 2)}
                </pre>
              )}
              {(step.type === "tool_result" || step.type === "thought") &&
                step.content && (
                  <pre className="trace-body">{step.content}</pre>
                )}
            </details>
          </li>
        ))}
      </ol>
    </div>
  );
}
