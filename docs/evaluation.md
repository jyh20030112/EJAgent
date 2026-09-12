# Task evaluation and completion policy

`ejagent.evaluation` implements host-declared acceptance criteria, versioned
evidence reads, deterministic checks, optional semantic judging, and trajectory
feedback. Deterministic evaluation remains the default. Completion enforcement
is an independent, explicit Harness setting.

## Bind criteria to a Run

Configure evidence and methods once, then give each Run its own immutable plan.
A plan can also be generated from a query by the optional
[task planner](task-planning.md), which binds criteria to this same registry:


```python
from ejagent.evaluation import (
    EvaluationCriterion, EvaluationMonitor, EvaluationPlan,
    FileEvidenceSource, GoalEvaluator, file_exists, json_fields,
)
from ejagent.harness import AgentHarness

plan = EvaluationPlan(
    goal="Write result.json with an answer field",
    version="result.v1",
    requirements=(
        EvaluationCriterion("exists", "File exists", "exists", ("result",)),
        EvaluationCriterion("shape", "JSON has answer", "shape", ("result",)),
    ),
    artifact_refs=("result.json",),
)
evaluator = GoalEvaluator(
    sources={"result": FileEvidenceSource("result.json")},
    verifiers={"exists": file_exists, "shape": json_fields("answer")},
)
monitor = EvaluationMonitor(evaluator)
# model and tools are the application's existing ModelPort and ToolExecutor.
harness = AgentHarness(
    agent_id="writer", model=model, tools=tools,
    trajectory=monitor, context=monitor.context_pipeline(),
)
async with harness:
    outcome = await harness.run(plan.goal, evaluation_plan=plan)
```

Use `monitor.context_pipeline(base=existing_pipeline)` to retain Skills or
compaction. `run()`, `continue_run()`, and `follow_up()` accept `evaluation_plan`;
omitting it produces `not_evaluated`, with no requirements or fabricated score in observation mode. Enforcement requires
both a monitor and a plan.
Follow-ups and restored conversations do not inherit the preceding Run's plan.
Changing criteria, order, goal, or version within one Run is a protocol error.

## Evidence and verification

Each `EvaluationCriterion` names a registered async verification method and its
required evidence keys. The verifier receives immutable `VerificationRequest`
data and a cancellation token, and returns `CheckResult`:

| Status | Meaning | Trajectory value |
| --- | --- | --- |
| `pass` | Valid evidence satisfies the condition | `True` |
| `fail` | Valid evidence disproves the condition | `False` |
| `unknown` | Evidence or verification is unavailable | `None` |
| `conflict` | Supporting observations disagree | `None` |

Known verdicts must cite declared evidence keys. Reports include criterion IDs,
methods, reasons, references, source versions, and missing evidence. Malformed
integration results raise `EvaluationProtocolError`; source I/O errors, timeouts,
and explicit `EvidenceUnavailable` yield unknown and can recover next checkpoint.

`FileEvidenceSource` reads a bounded regular UTF-8 file. Observed absence fails
`file_exists`; unreadable or undecodable content is unknown. `json_fields()` checks
JSON syntax and top-level fields. `ProbeEvidenceSource.record()` accepts completed
intervals with an explicit Run ID; `boolean_field()` checks probe completion or
overlap, and retains conflict when independent sources disagree.

Custom `EvidenceSource` adapters implement `revision()`, `read()`, and
`close_run()`. Revisions must change on relevant mutations, including changes
that restore previous content. Reads must be read-only and return a bounded
`EvidenceSnapshot`. A revision check brackets the entire collection, followed
by another check after verification. An inconsistent snapshot remains unknown.
These checks observe a point in time; they do not lock an external environment.

Rules depend only on their declared evidence and fixed criterion. Acquire any
external state through a source, so cached results cannot hide undeclared
dependencies. `$completion` is reserved for the proposed final response; it is
unknown before completion or when the 65,536-character candidate was truncated.
Set `completion_only=True` for checks that are meaningful only at completion.
They remain unknown before that point, without making otherwise complete
environment capture unavailable or suppressing its cycle analysis.
Signals also expose at most 128 current tool-batch receipt references and an
`observations_complete` flag. These refer to Audit records by Run and tool-call
ID; host readers resolve raw artifacts. They are not proof of task success.

## Feedback, cost, and lifecycle

The evaluator captures baseline, completed tool batches, and proposed completion
(including a tool's `COMPLETE` result when a plan is bound). Valid dependency
versions permit cached checks; changed versions invalidate old evidence and
dependent results. Completion covers all requirements and constraints.

Semantic evidence identity excludes capture time, location, and observation ID.
Repeated equivalent reads add no novelty. Coverage, progress, and cycle detection
remain the responsibility of the existing trajectory analyzer. Incomplete
evaluation projects only missing items and invalidations, with no asserted current
facts or numeric completion score. Feedback remains transient model Context.

`timeout_seconds`, `max_concurrency`, and `max_evidence_bytes` bound evaluator
operations. Async adapters must cooperate with cancellation; blocking work should
be isolated. Built-in file reads use worker threads and byte limits; cancellation
stops awaiting them but cannot interrupt an OS read already in progress.
`EvaluationCost` records per-checkpoint reads, revision checks, verifier calls,
cache hits, and elapsed milliseconds. Sum these across reports for Run totals;
they do not increment Actor action, request, or token counts. Trajectory elapsed
time already measures the whole Run, including evaluation waits; do not add
evaluator elapsed time to that wall-clock value again.

Pass `report_sink=JsonlEvaluationJournal(path)` to retain full reports and bounded
observed content independently of SessionStore. Audit receipts carry
`evaluation_report_ref`; matching journal records resolve evidence after the
original artifact changes. Logs can contain application data; the host owns their
location and retention. Without a sink, only the latest active report is exposed
by `latest_report(run_id)`. Run completion, cancellation, and exceptions clear
caches, probe records, and temporary Context frames. Standalone callers must
call `close_run(run_id)` in a `finally` block.

## Optional model judge

Mark only semantic criteria with `semantic=True`, then pass a `ModelJudge` to
`GoalEvaluator(semantic_judge=...)`. Ordinary criteria never invoke the judge:

```python
from ejagent.evaluation import JudgeLimits, ModelJudge

quality = EvaluationCriterion(
    "answer_quality", "The JSON answer explains why 2 + 2 equals 4",
    "answer_quality", ("result",), semantic=True,
    guard_method="shape", completion_only=True,
)
judge = ModelJudge(judge_model, limits=JudgeLimits(
    max_requests=8, max_tokens=16_384, max_output_tokens=1024,
    timeout_seconds=30, max_concurrency=2,
))
# Add quality to the Run's fixed plan before admission.
# Reuse the file source and shape verifier from the first example.
evaluator = GoalEvaluator(
    sources={"result": FileEvidenceSource("result.json")},
    verifiers={"shape": json_fields("answer")}, semantic_judge=judge,
)
```

A `guard_method` is a necessary deterministic precondition. A failed, unknown,
or conflicting guard prevents the model request and preserves that conclusion.
A passed shape check permits a separate semantic judgment; it does not establish
answer correctness by itself. Model judgments never manufacture environment Facts.

The judge uses its own `ModelPort` requests, instructions, and budget. It receives
the goal, task, one criterion, and only that criterion's evidence. Artifact text
is data, and the judge receives no tools. Structured responses must use the
expected criterion ID, status, short reason, existing evidence references, and
missing-evidence list. Invalid structures, unsupported citations, and tool requests
produce unknown. This validation checks protocol and traceability, not the truth
of every semantic judgment; application-specific calibration still matters.

### Output format and recovery

Judge requests default to `response_format={"type": "json_object"}`. The
OpenAI-compatible adapter forwards `ModelRequest.response_format` unchanged as a
detached JSON object. Callers may supply a provider-supported strict schema via
`ModelJudge(model, response_format={"type": "json_schema", "json_schema": ...})`.
JSON mode alone does not validate fields or evidence references; local validation
always runs. A strict Pydantic `JudgeOutput` model defines fields, types, enums,
and self-contained verdict consistency rules; its generated JSON Schema is
included in the system prompt. Extra fields and implicit type conversions are
rejected. Current criterion/evidence binding remains a separate check after
schema validation. See [shared structured output recovery](structured-output.md).
Set `response_format=None` for prompt-only providers. The native
Anthropic adapter currently rejects this passthrough option explicitly; configure
its Judge with `response_format=None` instead of silently dropping the option.
Actor requests omit it unless explicitly configured by their caller.

A response consisting entirely of one complete JSON Markdown fence (with a
`json` label or no label) is unwrapped and validated normally. Valid `pass`,
`fail`, `unknown`, and `conflict` judgments retain their meaning. Surrounding
prose, multiple blocks, partial JSON, duplicate keys, invalid fields, and invented
references are never guessed or repaired. Recovering a fence does not make an
extra model request or reject the Actor's completion.

After a fenced or invalid response, the next actual Judge request in that Run
receives a `TransientInstruction` with source `judge:output_format`. It requests
raw JSON with the required fields without changing the criterion or evidence.
The instruction survives cache hits, clears after a valid plain JSON response,
counts toward the prompt byte limit, and is removed when the Run closes. It
never enters Actor Context or committed Conversation.

Within a retry sequence, an additional temporary `UserMessage` supplies the JSON
syntax location or Pydantic field-path errors and a bounded excerpt of the failed
response. This message is explicitly diagnostic data; the steer itself contains
fixed instructions. The original criterion/evidence payload remains unchanged.
Diagnostic excerpts are not carried into another evaluation or saved in the
evaluation report. Only the generic reminder can survive until the next decision.

`JudgeLimits.max_format_retries` defaults to **1** (set **0** to disable). Invalid
JSON or response fields may trigger one internal Judge retry over the same
evidence snapshot. All attempts share one evaluation timeout, request/token
budgets, output cap, and cancellation. A valid negative or insufficient-evidence
judgment, provider error, refusal, or truncated output is not retried internally.
OpenAI `finish_reason` and Anthropic `stop_reason` are exposed through
`ModelResponseCompleted.finish_reason`. Known truncation reasons produce unknown
even if the partial output parses as JSON; adjust explicit output/context limits
instead of repeatedly requesting the same truncated response. Missing content
that violates the model-message protocol remains an unavailable response.

`EvaluationReport.judge_attempts` records each admitted request's criterion,
request/retry index, format-steering flag, outcome, short reason, and provider
finish reason. Outcomes distinguish `valid`, `recovered_markdown`, `invalid_json`,
`invalid_schema`, `truncated`, `refused`, and unavailable/interrupted responses.
The JSONL journal and Streamlit evidence inspector include these records; cache
hits do not replay old attempts. No raw response or hidden reasoning is stored
in these diagnostics. Internal retries consume Judge usage, not Actor turns or
completion-retry allowances.

Harness automatically manages the judge model exposed by the evaluation monitor,
deduplicating shared resources. Standalone users manage the ModelPort lifecycle.
Custom providers must honor `ModelRequest.max_output_tokens` and report usage.
Built-in OpenAI-compatible and Anthropic adapters forward the output cap.

Request and token budgets are per Run; judge requests serialize within a Run and
respect the configured concurrency across Runs. `max_tokens` uses actual reported
input/output usage. One in-flight request can exceed the remaining total budget;
its verdict becomes unknown, its full reported cost remains recorded, and no
further request is admitted. This is not a prepaid hard billing cap. Missing usage
also blocks further requests for that Run. Prompt/response byte limits, timeouts,
and cancellation bound local processing. A new Run starts fresh budgets.

Report costs contain `model_requests`, `model_input_tokens`,
`model_output_tokens`, and `model_unreported_requests`. Sum known reported tokens
with Actor usage for a combined reported total; unreported requests mean the
actual total is incomplete. Interrupted evaluations emit an unknown report to the
configured sink before cleanup, retaining request counts and usage uncertainty.
Such a report may lack an accepted checkpoint receipt; associate it by Run ID.

## Independent completion enforcement

```python
from ejagent.evaluation import CompletionMode, CompletionPolicy

harness = AgentHarness(
    agent_id="writer", model=model, tools=tools,
    trajectory=monitor, context=monitor.context_pipeline(),
    completion_policy=CompletionPolicy(CompletionMode.ENFORCE, max_retries=2),
)
```

`OBSERVE` is the default and preserves advisory completion behavior. `ENFORCE`
accepts text or tool completion only when the monitor explicitly allows it. Fail,
unknown, conflict, and capture failure cannot silently approve completion. A
rejected claim is audited and omitted from committed Conversation; tool calls
and results retain their protocol ordering. The next decision receives both
completion feedback and the monitor's staged Context within the same Run.

`max_retries` counts additional completion attempts. Actor turn/token limits and
cancellation still apply. Exhaustion produces a failed Run with
`StopReason.COMPLETION_AUDIT_FAILED`, without advancing the session revision. A
broken monitor stops retries immediately; a temporary evidence failure can recover.
Tool `REJECT`/`CANCEL` controls remain terminal. See [ADR 0001](adr/0001-failed-completion-audit-continues-run.md)
for the policy choice and its evidence.

The Streamlit example now uses this formal evaluator. **Semantic completion
review** enables an independent final-summary judge; **Require completion
approval** enables enforcement. Configure judge request/token budgets and
completion retries before starting. Demo mode uses a deterministic judge stand-in;
provider mode creates a separate model adapter. The Trajectory tab shows item
reasons, evidence versions, missing evidence, Actor/judge costs, and actual model
Context deliveries. Reports persist under the session folder's `evaluations/`
directory. Disabling trajectory feedback removes evaluation and enforcement.

In demo mode, **Run completion recovery** deliberately proposes completion before
collecting evidence, then recovers through same-Run feedback (three Actor turns).

Run the standalone artifact example without credentials:

```bash
uv run python examples/evaluate_artifact.py result.json
uv run python examples/evaluate_artifact.py result.json --journal evaluations.jsonl
```

## Code workspace verification

`WorkspaceEvidenceSource` captures explicit UTF-8 paths and their versions.
`CommandEvidenceSource` executes a host-configured command against those dependencies
and caches its result for unchanged checkpoints within the Run. Use
`command_succeeded` to verify exit status; file changes invalidate old results,
and changes during execution make the evidence unavailable. Command timeouts and
output bounds are configurable. See the [planning guide](task-planning.md#workspace-and-command-evidence)
and [feature example](../examples/planned_feature.py) for a complete integration.
