# Environment Facts and Event-Context Exposure

This document develops the information boundaries for the EJAgent Harness.
Use the [Harness overview](harness-overview.md) for current project scope and
[trajectory integration](trajectory-runtime-readiness.md) for implemented wiring.

## Status

This is an exploratory information-boundary specification. It defines what
must be recorded, evaluated, and potentially shown to a model when designing
trajectory feedback and future enforcement policies.

Event names and payloads in this document are conceptual. They are not current
public EJAgent contracts.

The first timing and context-condition controls have been executed; see
[Trajectory Pre-implementation Experiment Report](trajectory-experiment-report.md).
The internal Phase-2 projection now implements the decision-relevant subset of
this specification; see
[Trajectory Context Projection](trajectory-context-projection.md). Its event
names remain internal and are not stable public contracts.

## Purpose

An Agent Harness needs two different views of execution:

- a complete, lossless-enough trajectory for audit and evaluation;
- a bounded, decision-relevant Context projection for the next model call.

Recording an item does not imply that the model should see it. Model context is
a projection selected at a decision boundary, not a copy of the event log.

## Authority model

The following order is the default preference for claims about the
environment:

```text
direct deterministic environment read
  > deterministic verifier
  > identified external system
  > independent semantic evaluator
  > actor self-report
```

This ordering is claim-relative rather than global. A focused unit test can be
authoritative for the behavior it covers without being authoritative for an
entire Goal.

### Fact validity

An Environment Fact is current only when all of these remain valid:

- its source can be identified;
- its evidence remains available or reproducible;
- its scope covers the claim being made;
- the environment checkpoint it describes has not been invalidated for that
  subject;
- any freshness condition required by the source still holds.

A later mutation does not erase an old Fact. It changes that Fact from current
evidence into historical evidence.

### Conceptual fact envelope

Before choosing implementation types, experiments should be able to capture:

| Field | Meaning |
| --- | --- |
| subject | Environment entity described by the Fact |
| predicate | Property or relationship asserted |
| value | Frozen asserted value |
| scope | Claims for which the Fact is relevant |
| source | Tool, verifier, or external system that produced the Evidence |
| observed_at | Observation time |
| checkpoint | Environment version against which the Fact holds |
| evidence_ref | Reference to the supporting raw Observation or artifact |
| freshness | Condition under which the Fact remains current |
| authority | Maximum claim scope this Fact can support |

Confidence is not a substitute for provenance. Non-deterministic semantic
judgments should remain Assessments rather than being silently promoted to
Environment Facts.

## State construction

A State is a Goal-relative projection, not an attempt to serialize the entire
world. For a coding task it might select:

- worktree and relevant-file identities;
- changed files and public-interface shape;
- focused and broad verifier results;
- build or service status;
- Requirement satisfaction;
- Constraint violations;
- externally visible side effects.

Different domains will require different projections. The Core vocabulary must
not assume that all environments are repositories, browsers, or databases.

### State Fingerprint rules

A State Fingerprint may cheaply identify possible recurrence, but:

1. fingerprint equality is not by itself cycle proof;
2. the underlying Facts must be compared before intervention;
3. the projection version must be part of the identity;
4. missing or stale Facts must not be treated as equality;
5. a fingerprint must not be presented to the model as an environment fact.

## Consumers

Four consumers need different projections:

| Consumer | Needs |
| --- | --- |
| Trajectory recorder | Ordered Actions, Observations, Facts, States, costs, and decisions |
| Evaluator | Goal, Requirements, Constraints, comparable States, and scoped Evidence |
| Kernel controller | Proposed Action, current State, recent Assessments, budgets, and policy history |
| Model | Stable objective plus the smallest current evidence needed for the next decision |

An after-commit Observer may consume durable Audit, but it cannot govern the
Run that already finished.

## Conceptual event sequence

```text
RunStarted
  -> BaselineObserved
  -> ContextProjected
  -> ActionProposed
  -> ActionAdmitted | ActionDenied
  -> ActionCompleted
  -> ObservationRecorded
  -> FactsUpdated
  -> StateCheckpointed
  -> ProgressEvaluated
  -> ContextProjected | CompletionAudited | RunTerminated
```

Several Actions may belong to one concurrent Tool batch. Their completion
events preserve actual completion order, while the resulting model messages
may preserve proposal order. Both orderings are relevant and must not be
silently conflated.

## Event-context exposure matrix

The matrix describes what a subsequent model call may see. It does not require
every event to cause a model call. The configured trajectory pipeline projects
checkpoint state independently of optional intervention feedback; see the
[v2 context policy](trajectory-context-projection.md).

| Event | Recorder stores | Evaluator receives | Next model projection | Notes |
| --- | --- | --- | --- | --- |
| `RunStarted` | Goal/task identity, configuration, baseline reference | Goal, Requirements, Constraints, baseline Facts | stable objective, completion conditions, relevant baseline State | A new Run is not necessarily a new Goal |
| `ActionProposed` | semantic Action and raw Tool proposal | normally nothing until Evidence exists | nothing by default | Controller may inspect before execution |
| `ActionDenied` | denied Action, current State, decision reason | policy decision and relevant Facts | actionable denial, current available decision space | Do not expose hidden thresholds or unrelated policy state |
| `ActionCompleted` | Action identity, timing, side-effect class, raw result reference | Observation metadata | bounded result or normalized Observation | Completion does not imply progress |
| `FactsUpdated` | added, changed, invalidated, and confirmed Facts | current scoped Facts | relevant Fact Delta with provenance | Stale Facts must be removed or visibly marked historical |
| `StateCheckpointed` | State, projection version, fingerprint, causal Actions | previous and current State | current State summary when decision-relevant | Fingerprint stays controller-facing |
| `VerificationCompleted` | verifier command/configuration and raw Evidence | scoped verifier Facts | Requirement/Constraint changes and failing evidence | Evidence scope must match any completion claim |
| `ProgressEvaluated` | current/best snapshots, delta, cost window, assessment source | all assessment inputs | concise Task/Epistemic Progress, Regression, blockers | One zero Delta is not automatically stagnation |
| `CycleSuspected` | candidate period and matching fingerprints | underlying Facts and progress window | current Facts, verdicts, and progress; no suspicion warning | Suspicion stays controller-only and does not suppress state |
| `CycleConfirmed` | compared checkpoints, repeated path, exhausted evidence, costs | confirmed recurrence inputs | Goal anchor, equivalent States, repeated Actions, no-progress evidence, replan request | Explain evidence, not only “loop detected” |
| `ConstraintViolated` | violated Constraint and authoritative Evidence | violation and affected Requirements | high-priority violation, impact, required recovery boundary | A hard violation may block before another model call |
| `ExternalStateChanged` | source event and invalidated Facts | refreshed State inputs | changed current Facts and invalidations | Conversation memory cannot override the external change |
| `CompletionProposed` | actor claim and supporting references | original Goal, every Requirement and Constraint, current Facts | no completion confirmation yet | Actor claim starts an audit; it is not completion |
| `CompletionAudited` | per-item verdict and Evidence coverage | audit inputs and verdict | missing evidence and unmet items, or final verified summary | Plan completion is not an input requirement |
| `RunTerminated` | terminal status, reason, partial State and cost | optional post-run evaluation | no next-call projection unless another Run starts | Partial failed trajectories remain auditable |

## Model-visible Context layers

When a model call is necessary, its Context should be assembled from explicit
layers in priority order.

### 1. Stable objective

- original Goal or current Run task;
- Requirements relevant to this Run;
- hard and soft Constraints;
- evidence required for completion.

The objective must not be rewritten to match the current Plan.

### 2. Current authoritative State

- current, relevant Environment Facts;
- each Fact's concise source and checkpoint;
- active verifier results;
- current Constraint violations and blockers.

Historical Facts must not appear as current merely because they remain in the
Conversation.

### 3. Current Plan

- current strategy and active step;
- hypotheses on which it depends;
- hypotheses already refuted by Evidence.

The Plan must be visibly represented as revisable strategy, not truth.

### 4. Recent causal window

The smallest recent chain needed to understand the current decision:

```text
Action -> Observation -> State Delta -> Progress Delta
```

Repeated raw outputs should be referenced or summarized rather than copied.

### 5. Progress summary

- current and best verified Requirement coverage;
- latest Task and Epistemic Progress;
- Regression and Constraint status;
- blockers and justified waits;
- cost over the relevant window.

Progress must cite Evidence or its evaluator source.

### 6. Intervention

Only when Harness policy intervenes:

- what was denied or detected;
- which checkpoints and Facts support the decision;
- which path is exhausted in the current State;
- whether the model should retry, gather missing Evidence, recover a
  Constraint, or replan from the Goal.

The model needs the consequence and supporting evidence, not the detector's
complete internal scoring formula.

## Projection rules

1. **Show current truth, not the whole log.** Preserve the whole trajectory for
   audit; project only current and causally relevant information.
2. **Prefer deltas after the baseline.** Re-send a full State only when facts
   were invalidated broadly or the model is replanning from the Goal.
3. **Keep raw Evidence reachable.** Summaries must retain references that permit
   inspection when details matter.
4. **Mark provenance and freshness.** An unlabeled historical fact is more
   dangerous than an omitted fact.
5. **Separate Fact from Assessment.** “Test X failed” and “the Plan is
   regressing” must not share an authority level.
6. **Do not count narration as Progress.** A status restatement, Plan rewrite,
   or completion claim creates no Progress without Evidence.
7. **Do not intervene on one quiet step.** Reads, waits, and diagnosis can have
   zero Task Progress while producing justified Epistemic Progress.
8. **Replan from Goal and current State.** A confirmed exhausted path should
   not merely mutate the previous Plan locally.

## Fact invalidation

Every Action should declare or allow conservative inference of its possible
effect scope:

- pure Observation: normally preserves State Facts;
- environment mutation: invalidates Facts about affected subjects;
- broad or unknown mutation: invalidates the enclosing State projection;
- external asynchronous change: invalidates Facts named by the source event;
- verifier: normally adds Evidence without mutating the domain State, though
  test setup side effects must be represented explicitly.

Until effect scope is trustworthy, broad invalidation is safer than presenting
stale Facts as current.

## Checkpoint policy candidates

The first experiments should compare, not yet choose among:

- after every Action;
- after every side-effecting Action;
- after a complete concurrent Tool batch;
- after a verifier;
- every fixed Action window;
- when recurrence is suspected;
- before accepting a Completion Claim.

Checkpoint frequency affects cost, freshness, and causal attribution. It is a
policy decision, not part of the definition of Environment Fact.

## Current EJAgent boundary

Today, EJAgent provides useful but incomplete inputs for this model:

- RunAudit records model deltas, Assistant messages, Tool starts, arguments,
  Tool results, controls, and terminal facts.
- ContextRequest separates committed messages, current-Run pending messages,
  and transient instructions.
- Context pipelines may create disposable projections without rewriting
  Conversation.
- the repeat guard only recognizes consecutive equal Tool-name/argument
  signatures and runs before the new Result and State are known;
- a text-only Assistant response currently ends the Run without an independent
  Completion Audit;
- no stable public contract distinguishes Environment Fact, State, Assessment,
  or Progress; the opt-in Phase-2 module now models the subset needed for its
  experiment and Context projection.

These statements describe the baseline to measure. They do not commit the
project to adding all concepts to the Core public API.

## First observation-only study

Before Harness enforcement, the study should:

1. replay RunAudit into normalized Action and Observation records;
2. attach experiment-owned environment checkpoints to those records;
3. derive candidate Facts with explicit provenance;
4. compare failing and healthy scenarios using the same State projection;
5. calculate assessments without altering Tool execution or model Context;
6. report which facts would have changed a decision and at which event;
7. measure false positives before any Action is denied.

An after-commit Observer can support offline analysis, but cannot provide
in-Run control. That limitation is useful during the observation phase because
it prevents the experiment from changing the trajectory it is measuring.

## Open questions

- Is a Goal scoped to one Run, one Session, or an independently persisted
  objective?
- Which component owns Requirement and Constraint definitions?
- How is an environment checkpoint identified across processes?
- Which facts survive a new Run and which must be freshly verified?
- How are concurrent Action completions attributed to one State Delta?
- Can tools reliably declare read/write/idempotency and effect scope?
- Which Assessments require an independent evaluator rather than rules?
- What Context budget is reserved for Facts, Evidence, and intervention?
- When may a raw Observation be summarized without losing decision-relevant
  evidence?
- Does a failed Completion Audit continue the same Run or create a new Run?
- Which trajectory data is durable, and which is intentionally ephemeral?
- How are secrets and unrelated environment data excluded from Facts and
  Audit?
