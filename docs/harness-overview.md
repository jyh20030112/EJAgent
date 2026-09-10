# EJAgent: Agent Harness

EJAgent is a Python Agent Harness for building agents that can act across
multiple tasks with controlled access to tools, useful context, recoverable
state, and evidence about their progress. The project brings these concerns
together around the model's decisions; the model–tool loop is one component of
that system.

The distribution remains `ejagent-core`, and `AgentHarness` is the application
entry point. `RuntimeKernel` is the existing class name for its single-Run
execution kernel. In documentation, **Harness** names the surrounding system
and **Kernel** names that component.

## Responsibilities

| Concern | Harness responsibility | Current building blocks |
| --- | --- | --- |
| Continuity | Preserve accepted Conversation across tasks and recover committed state | `AgentHarness`, snapshots, `SessionStore` |
| Context | Assemble the information and instructions needed for each decision | `ContextPipeline`, Skills, derived compaction, transient instructions |
| Capabilities | Connect model providers and tools, and own their resource lifecycle | `ModelPort`, `ToolExecutor`, `ManagedResource` |
| Control | Admit work, accept user intervention, and bound execution | Run serialization, cancellation, Steering, FIFO Follow-ups, `RunLimits` |
| Evidence and feedback | Observe environment changes, assess progress, and inform subsequent decisions | Optional trajectory monitor, host evaluator, Context projection |
| Accountability | Keep outcomes, side effects, failures, and commit decisions inspectable | `RunOutcome`, `RunAudit`, observers, JSONL journal |

These responsibilities span the assembled Harness. They do not imply that all
logic belongs in the `AgentHarness` class. The class owns lifecycle and accepted
state; the Kernel, adapters, and host-supplied evaluator retain their own
boundaries.

## Decision and feedback flow

```text
User task / accepted Conversation
  → Harness admits a Run and captures its starting revision
  → Context pipeline assembles the next decision view
  → Model proposes a response or Tool actions
  → Kernel executes the actions within configured controls and limits
  → Optional monitor requests fresh environment evaluation
  → Assessment can inform the next model Context
  → Harness coordinates the outcome's Conversation/Audit commit
```

Conversation, Audit, and model Context remain separate. Summaries and feedback
can change what a model sees without rewriting accepted history. Failed and
cancelled Runs remain auditable; they do not advance the Conversation revision.
External Tool side effects are not rolled back by a failed Conversation commit.

## Implemented capabilities and remaining policy work

The current implementation supports state recovery, lifecycle management,
context composition, tools, live controls, and optional online trajectory
feedback. The host evaluator supplies the actual environment facts and the
Requirement/Constraint verdicts. Coverage and recurrence are calculated from
those inputs; model claims do not automatically become verified facts.

The public [evaluation module](evaluation.md) provides Run-bound acceptance
criteria, deterministic file/probe checks, evidence invalidation, and limited
feedback when evidence is unavailable, plus an optional `ModelJudge` for explicit
semantic criteria. Hosts
can register domain evidence sources and verification methods, and retain full
reports in a separate JSONL journal.

Trajectory assessments provide observation and Context feedback. The independent
`CompletionPolicy` defaults to observation. Enabling enforcement retries rejected
text or tool completions within the same Run, bounded by retry and Run limits;
rejected final prose remains in Audit instead of committed Conversation.
Dynamic [task planning](task-planning.md) can generate a task definition and bind
acceptance to registered capabilities before execution. The actor revises execution
steps through a version-checked `update_plan` tool after checkpoint feedback.
Acceptance remains fixed within the Run. Workspace and command sources support
version-bound code verification. Automatic Action denial and mandatory replanning
remain separate policy work.
Completion feedback semantics and the policy choice are described in
[ADR 0001](adr/0001-failed-completion-audit-continues-run.md).

Each `AgentHarness` currently represents one logical agent. Multi-agent
coordination and serializing an active Run for arbitrary pause/resume are not
implemented. `continue_run()` starts a new Run from committed history. These
are current capability boundaries, not a definition of the project's entire
long-term scope.

## Explore the Harness

Start with the [usage guide](usage-guide.md) for application composition or the
[Streamlit example](../examples/streamlit_app.py) for interactive controls and
feedback. The example evaluator checks only probe completion and overlap; a
different application must supply its own domain evaluation.

For implementation details, read the
[Harness architecture](runtime-kernel-harness-design.md),
[class and execution guide](core-classes-and-runtime-flow.md), and
[trajectory integration](trajectory-runtime-readiness.md). The experiment
documents preserve the evidence and assumptions of their recorded phases;
they are not a substitute for the current capability descriptions above.
