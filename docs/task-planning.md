# Dynamic tasks and execution plans

`AgentHarness(planner=...)` can prepare a task from each user query, bind its
acceptance conditions to host verification capabilities, and expose a revisable
execution plan to the actor. This is opt-in: existing callers supplying an
`evaluation_plan` retain their existing behavior.

## Two plans with different owners

| Value | Producer | Changes during execution |
| --- | --- | --- |
| Original query | User | Retained as the Run's user message |
| `TaskDefinition.task` and acceptance goal | Planner, validated by the host adapter | Fixed for this Run |
| `EvaluationPlan` | Planner selects host capabilities; adapter binds methods and evidence | Fixed for this Run |
| `ExecutionPlan` | Planner proposes initial steps; actor proposes updates | Harness assigns versions after validation |
| Evidence and acceptance verdicts | Host sources and verifiers; optional semantic Judge | Refreshed at checkpoints |

Schemas and verification implementations remain predefined. Task text, selected
criteria, step descriptions, and subsequent revisions are generated per query.
A completed execution step is an actor claim, not proof that a requirement passed.
Changing a goal or weakening acceptance requires a separately prepared task; it
cannot be done through `update_plan`.

## Configure the capability catalog

```python
from ejagent.contracts import EvaluationCriterion
from ejagent.planning import ModelTaskPlanner, VerificationCapability

planner = ModelTaskPlanner(
    planning_model,  # A separate, configured ModelPort instance.
    capabilities=(
        VerificationCapability(
            "regression_tests",
            EvaluationCriterion(
                "template", "The host regression command succeeds",
                "command", ("tests",),
            ),
            required=True,
        ),
    ),
    environment={"scope": "Describe the workspace and what these tests verify"},
)
```

The planner receives the query, conversation content, actual tool definitions,
host environment information, and the capability catalog. It returns structured
JSON containing task, goal, criteria, steps, unknowns, and unsupported goals.
Capability IDs bind to host-owned methods, evidence keys, semantic flags, guards,
and completion-only settings. The model cannot introduce executable verifiers or
new evidence sources. `required=True` prevents omission of that host capability;
`constraint=True` places selected conditions in the acceptance constraint set.

The host condition remains in each criterion's description. The planner can add
its task-specific description, but this does not extend what the underlying
verifier proves. A passing command proves its configured checks passed; it does
not automatically verify every natural-language claim. Catalog design and any
semantic checks must match the application's actual goals.

Unknown capabilities, omitted mandatory capabilities, unsupported goals, invalid
JSON, incomplete output, and steps that fail to cover requirements reject
preparation before actor actions. `EvaluationMonitor.validate_plan()` also rejects
unconfigured sources/methods or a missing semantic Judge. Custom monitors own
validation of their own capabilities.

`ModelTaskPlanner` makes one model request per preparation. Default bounds are
60 seconds, 4,096 output tokens, 64 KiB prompt, and 32 KiB response. The default
`response_format` is passed through as `{"type": "json_object"}`; use `None` for
providers that require prompt-only JSON. A single complete Markdown JSON fence is
accepted; partial JSON, duplicate keys and surrounding prose are rejected. There
is no automatic planning retry or automatic environment investigation. Supply
investigation results in `environment` or use a custom `TaskPlanner`.

## Connect preparation, feedback, and execution

```python
from ejagent import AgentHarness
from ejagent.contracts import CompletionMode, CompletionPolicy

harness = AgentHarness(
    agent_id="coding-agent",
    model=actor_model,
    tools=workspace_tools,
    planner=planner,
    trajectory=monitor,
    context=monitor.context_pipeline(),
    completion_policy=CompletionPolicy(CompletionMode.ENFORCE),
)
async with harness:
    outcome = await harness.run("Add the requested feature and preserve compatibility")
```

Each `run(query)` or queued `follow_up(query)` without an explicit evaluation plan
gets fresh preparation. Supplying `evaluation_plan=` bypasses automatic planning.
`continue_run()` has no new query and does not invoke the planner; supply its
acceptance plan explicitly if completion enforcement is enabled.

The model loop is:

```text
query → planner → validated task and initial execution plan
  → baseline checkpoint → actor context
  → actor actions → completed tool batch → refreshed evaluation
  → latest feedback and plan → actor may call update_plan → next action
  → proposed completion → independent acceptance gate
```

The plan is projected in a temporary `planning` instruction after the configured
Context pipeline. It contains the task, acceptance conditions, unknowns, current
plan, and latest checkpoint ID. Trajectory feedback also exposes the current
plan in `revisable_plan`. These instructions do not become committed conversation
messages. Historical tool messages can contain older proposals, but the current
projection identifies the authoritative version.

## Update a plan at a decision boundary

The Harness registers `update_plan` only for prepared tasks. Its arguments are:

```json
{
  "expected_version": 1,
  "based_on_checkpoint": "run-id:cp2",
  "reason": "The boundary-case test failed; correct the interval calculation",
  "steps": [
    {
      "id": "implement",
      "description": "Handle touching and disjoint intervals, then verify",
      "requirement_ids": ["regression"],
      "status": "in_progress"
    }
  ]
}
```

The model must copy the actual version and checkpoint from its current context.
Updates require a nonempty reason, unique step IDs, known criterion references,
and coverage of every acceptance requirement. At most one step may be in progress;
other statuses are `pending`, `completed`, and `blocked`. There are at most 64
steps. The host assigns the next version atomically: competing updates against
the same version cannot both succeed. Stale versions/checkpoints, additional goal
fields, and malformed updates produce tool errors without changing the plan.

The actor chooses when a revision is useful. Checkpoints provide opportunities
and feedback; there is no mandatory extra replanning model call on every turn or
policy forcing a plan update before every action. Completion enforcement still
uses evidence even if the actor marks all steps completed. Failed acceptance
adds `completion_audit` feedback and permits bounded recovery within the Run.

## Workspace and command evidence

`WorkspaceEvidenceSource(root, paths)` reads explicitly configured UTF-8 paths,
including known absence, with a total size bound. It does not discover a repository
or index code. `CommandEvidenceSource(workspace, command)` runs a fixed argument
vector in that workspace and exposes exit code, stdout, stderr, truncation status,
and a dependency revision. `command_succeeded` checks the observed exit status.

A command result is cached per Run and dependency revision. File changes or a host
call to `invalidate()` force a fresh command; a file change during execution makes
the evidence unavailable. Cancellation/timeouts stop and reap the subprocess;
output capture is bounded. Configure the evaluator's timeout to accommodate the
command timeout. Include every relevant dependency in `paths` and explicitly
invalidate external dependencies. This is an observation interface, not a process
sandbox: host verification commands execute with the application's permissions.

## Inspect the examples and audit

```bash
# Real configured LLM for both planner and actor (.env configuration).
uv run python examples/planned_feature.py

# Scripted model replies, actual Python edits and regression tests.
uv run python examples/planned_feature.py --demo

uv run streamlit run examples/streamlit_app.py
```

The feature example creates an isolated directory under `.ejagent-sessions/`,
adds `overlap_seconds` to a small Python module, and verifies existing behavior
and boundary cases. It saves full planner/actor request contexts, the generated
code, test evidence, JSONL session audit, `result.json`, and readable `contexts.md`.
The offline scenario deliberately misses a boundary case and exercises completion
rejection followed by plan revision and repair. Its test catalog covers this
fixture, not arbitrary software development requests.

In Streamlit, select the provider mode and enable **Dynamic task planning**.
The planner selects applicable probe criteria from the query instead of always
binding the fixed three-probe plan. The trajectory view displays task definition
and current execution plan. Available verification still covers only probes and
optional summary review; unsupported user goals fail preparation.

Audit records include `planning_started`, `planning_usage`, `task_planned`,
`execution_plan_updated`, `execution_plan_rejected`, and `planning_failed` as
applicable. Planner usage is separate from actor `RunUsage` and Judge costs;
missing usage is reported as unavailable. Failures after a model response retain
reported model usage as well. `last_task_definition` and `last_execution_plan`
provide in-process inspection (including a live plan during execution). Durable
copies are retained in the Run audit, including failed Runs. They do not implement
mid-Run resume or automatically restore an active plan after process restart.
