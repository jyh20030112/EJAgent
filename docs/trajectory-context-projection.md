# Harness Trajectory State and Feedback

## Status

The Harness supports internal, opt-in trajectory Context projection under
`ejagent._trajectory`. A host composes `AgentHarness` with a monitor and
`TrajectoryContextPipeline` to provide checkpoint state and optional feedback
at each decision. The
projection is not a stable top-level API and gives no trajectory code authority
to admit Actions or terminate Runs. The Streamlit example enables this
composition by default; the library does not.

The entry gates are executable in
[`phase2_evidence.py`](../experiments/trajectory/phase2_evidence.py), with the
historical v1 result in
[`2026-09-01-phase2-summary.json`](../experiments/trajectory/results/2026-09-01-phase2-summary.json).
That artifact records the former event-gated policy. Running the script now
checks the v2 state/feedback policy without rewriting the historical result.

## Module and Interface

`TrajectoryContextProjector` is the pure deep module:

```python
projection = projector.project(frame)
```

The `TrajectoryContextFrame` joins four evaluator-owned inputs for one
Decision Boundary:

- the stable Goal anchor;
- one explicit `TrajectoryCheckpoint`;
- its `ProgressSnapshot`;
- one `TrajectoryContextEvent` determining whether the state needs an
  additional intervention instruction.

`TrajectoryContextPipeline` is the small adapter at the existing
`ContextPipeline` seam. It builds the host's base Context first, obtains a
frame for the current `run_id` and `turn`, and appends one disposable
`TransientInstruction`:

```text
RuntimeKernel._build_context
  -> host-selected base ContextPipeline
  -> host TrajectoryContextSource(ContextRequest)
  -> TrajectoryContextProjector.project(frame)
  -> TransientInstruction (observed state or explicit unavailable status)
  -> next model request
```

RuntimeKernel depends on the stable `ContextPipeline` and optional
`TrajectoryMonitor` Interfaces, and has no dependency on trajectory detection
or Fact-model internals.

`OnlineTrajectoryMonitor` now produces a `TrajectoryUpdate` immediately after
each captured semantic boundary. An optional synchronous update sink can stage
`update.to_context_frame(...)` in `TrajectoryContextBuffer`; the buffer keys
frames by exact `(run_id, turn)`. This composition connects online assessment
to the existing pipeline without making Conversation an event store or
exposing the current event before the next model decision.

## Fact validity

An `EnvironmentFact` is immutable. Its checkpoint-relative `FactValidity` is
one of:

| Validity | Meaning | Model projection |
| --- | --- | --- |
| `current` | Freshness condition still holds | May appear as current truth |
| `invalidated` | A named checkpoint or event made it historical | May appear only in an explicit invalidation delta |
| `stale` | Its freshness condition no longer holds | Projection fails closed |
| `unknown` | Freshness cannot currently be established | Projection fails closed |

A current Fact carries its subject, predicate, frozen value, claim scope,
source, observation time, checkpoint, Evidence reference, freshness condition,
and authority. Complete capture is required for model-facing projection.
State fingerprints remain controller-only.

## State and feedback visibility

State visibility is independent of intervention visibility. A suspected cycle
does not hide current Facts, requirement/constraint verdicts, progress, or the
revisable plan. The suspicion remains in controller assessment and audit.

| Event | State visible | Optional feedback |
| --- | --- | --- |
| `FactsUpdated` | yes | none |
| `ProgressEvaluated` | yes | none |
| `CycleSuspected` | yes | none; suspicion stays controller-only |
| `CycleConfirmed` | yes | exhausted Action path and replan request |
| `ConstraintViolated` | yes | violated item and required recovery boundary |
| `ExternalStateChanged` | yes | invalidated Facts and refreshed current State |
| `CompletionAuditFailed` | yes | unmet items, missing Evidence, continue-current-Run instruction |
| `EvaluationUnavailable` | explicit unavailable status | gather missing Evidence; no current success claim |

Observations are exposed at the next Decision Boundary. A frame for turn 2
cannot be used for turn 1, turn 3, or another Run. Missing frames produce
`state_status: "unavailable"` with a null checkpoint. Incomplete evaluation also
produces unavailable status, retaining its checkpoint and invalidation metadata.
Neither case supplies current Facts, verdicts, or a completion score.

## Context schema v2

The `trajectory_context` envelope now uses `ejagent.trajectory-context.v2`.
Current state fields remain at the envelope level, including `goal_anchor`,
`checkpoint`, `current_facts`, `invalidated_facts`, `requirements`, `constraints`,
`revisable_plan`, and `progress`. `run_id`, `for_turn`, and `state_status` identify
the projection's decision and availability. All invalidated Facts from the
checkpoint are included as metadata, without their old values.

The v1 fields `event`, `event_id`, `event_evidence_refs`, `affected_items`,
`recent_causal_actions`, and `instruction` move into optional `feedback`.
Ordinary state and suspected cycles use `feedback: null` and instruction source
`trajectory:state`; interventions retain `trajectory:<event>` sources. A missing
observation has no event ID or causal evidence and supplies only its unavailable
event and instruction. Consumers must check the schema and nullable feedback.

Pipeline metadata separates `trajectory_context_visible` from
`trajectory_feedback_visible`. When a frame exists, `trajectory_event` retains
the actual controller event even when its warning is hidden from the Actor.
Projections remain transient; they do not enter committed conversation history.

## Phase-2 entry gates

| Gate | Evidence |
| --- | --- |
| More than FS-001 | deployment-routing supplies a second confirmed period-two failure; six additional domains exercise controls |
| Provenance/freshness/invalidation | typed Facts are required by projection; stale or unknown validity fails closed |
| Concurrent causal attribution | unattributed batches return `causally_ambiguous`; complete batch paths may be assessed |
| False-positive review | productive wait, exploration, legitimate retry, and regress-then-recover all remain `no_cycle` |
| Completion Audit Run semantics | [ADR 0001](adr/0001-failed-completion-audit-continues-run.md) chooses same-Run feedback while budget remains |

## Current projection boundaries

- no default-on trajectory pipeline in the library or stable top-level export;
- no universal Fact collector or durable Fact store;
- no default-on capture or enforcement inside `RuntimeKernel`;
- no domain-independent Completion verifier; the host evaluator supplies its
  authoritative Requirement and Constraint verdicts;
- no Action denial, cancellation, or termination policy;
- no projection of detector thresholds, fingerprints, stale values, or the full
  trajectory log.

The host owns domain Fact capture and evaluation. `OnlineTrajectoryMonitor`
generates events from those evaluations and analysis; the host connects its
update sink to the Context source. The projection is a Harness feedback
capability whose correctness depends on these domain inputs.

The internal capture/event/context seam has passed the
[Harness integration gates](trajectory-runtime-readiness.md) and is now wired into
`RuntimeKernel` when a host explicitly supplies a monitor. This does not enable
enforcement by itself.

`TrajectoryContextBuffer` keys frames by Run and turn. Reads do not consume a
frame, so rebuilding Context for the same turn returns the same staged input;
`close_run()` removes its frames. Rebuilding a turn preserves the checkpoint and
Fact observation times; projection does not run an evaluator, refresh evidence,
or call a Planner. Hosts remain responsible for capturing decision-boundary
observations rather than treating an older successful evaluation as current.

Completion enforcement is configured separately through `CompletionPolicy`.
When approval is required, a rejected completion continues the same Run while
budget remains. Advisory-only assessments can target a next turn that never
occurs. Applications should distinguish recorded assessments from instructions
actually delivered to the model, as the Streamlit example does. Context v2
does not change cycle thresholds or acceptance rules.
