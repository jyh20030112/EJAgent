<div align="center">
  <img src="assets/ejagent-mascot.svg" width="180" alt="EJAgent mascot hatching from an egg">
  <h1>EJAgent Core</h1>
  <p><em>Hatch an agent of your own — a Python Agent Harness for context, tools, state, control, and evidence-driven feedback.</em></p>
  <p>
    <a href="https://github.com/jyh20030112/EJAgent/stargazers"><img src="https://img.shields.io/github/stars/jyh20030112/EJAgent" alt="GitHub stars"></a>
    <a href="https://pypi.org/project/ejagent-core/"><img src="https://img.shields.io/pypi/v/ejagent-core" alt="PyPI version"></a>
    <a href="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml"><img src="https://github.com/jyh20030112/EJAgent/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2ea44f" alt="MIT license"></a>
    <br>
    <img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
    <img src="https://img.shields.io/badge/asyncio-native-4B8BBE" alt="asyncio native">
    <img src="https://img.shields.io/badge/MCP-ready-7C3AED" alt="MCP ready">
  </p>
  <p><strong>English</strong> · <a href="README_zh-CN.md">简体中文</a></p>
</div>

---

EJAgent Core is a Python **Agent Harness** that brings together context,
tools, state, control, and evaluation around an agent's decisions. It helps an
agent carry work across tasks, recover committed state after restarts, accept
user intervention, and use verified environment feedback when choosing its
next action.

Use it to build assistants, coding agents, research agents, and task-oriented
applications. Model providers, capabilities, context strategies, storage, and
evaluators can be composed for the application's domain.

## What You Can Build

- **Highly customized agents** — replace the model provider, tool backend,
  context strategy, storage layer, and observers independently.
- **Stateful assistants** — keep typed conversation history across multiple
  tasks and continue from the latest committed state.
- **Durable agents** — persist sessions to an append-only journal and recover
  them after a process restart.
- **Tool-using agents** — expose Python functions, compose multiple tool
  executors, or connect MCP services through one consistent interface.
- **Controllable agents** — cancel active work, steer the next model step,
  queue follow-up tasks, and enforce turn or token limits.
- **Context-aware agents** — inject local Skills, derive summaries for long
  conversations, or implement your own context policy.
- **Observable systems** — capture structured results, failures, token usage,
  model events, and tool activity without coupling observers to execution.
- **Agents with trajectory feedback** — connect a host evaluator to assess
  Requirement satisfaction, Constraints, and repeated State/Action patterns,
  then project relevant feedback into the next model Context.
- **Provider-flexible applications** — use OpenAI-compatible endpoints,
  Anthropic, or implement a provider adapter for another model API.

## Why EJAgent Core

A useful Harness manages what the model can see and do, what persists between
tasks, and how execution evidence informs later decisions. EJAgent keeps
accepted Conversation, execution Audit, and temporary model Context separate,
so feedback and summaries can evolve without rewriting committed history.

`AgentHarness` is the application entry point and owns lifecycle, accepted
state, controls, and commit coordination. Its `RuntimeKernel` executes one Run;
Context pipelines, tools, providers, and optional trajectory evaluation supply
the surrounding capabilities. See the [Harness overview](docs/harness-overview.md)
for responsibilities and current implementation boundaries.

## Install

EJAgent Core requires Python 3.12 or newer.

```bash
uv add ejagent-core
```

Add an optional integration when needed:

```bash
uv add 'ejagent-core[anthropic]'  # Anthropic
uv add 'ejagent-core[mcp]'        # MCP
```

## Quick Start

Configure an OpenAI-compatible endpoint:

```env
MODEL_API_KEY=sk-xxxxxxxx
MODEL_URL=https://api.example.com/v1
CHAT_MODEL=your-model
```

Create a stateful agent:

```python
from ejagent.contracts import SystemMessage
from ejagent.harness import AgentHarness
from ejagent.providers import ModelConfig, OpenAIModelPort
from ejagent.tools import FunctionToolExecutor

model = OpenAIModelPort(ModelConfig.from_env())
harness = AgentHarness(
    agent_id="assistant",
    model=model,
    tools=FunctionToolExecutor(),
    initial_messages=(SystemMessage("Answer precisely."),),
)

async with harness:
    await harness.run("Remember that my project is EJAgent.")
    answer = await harness.run("What is my project?")
    print(answer.result.output)
```

The same agent can be upgraded without changing its calling style:

```python
from ejagent.context import SkillsContextPipeline
from ejagent.storage import JsonlSessionStore
from ejagent.tools import McpToolExecutor

harness = AgentHarness(
    agent_id="assistant",
    model=model,
    tools=McpToolExecutor("mcp_config.json"),
    context=SkillsContextPipeline("skills"),
    store=JsonlSessionStore(".ejagent-sessions"),
)
```

## Evaluate Task Results

Use `ejagent.evaluation` to bind an immutable `EvaluationPlan` to each Run,
read versioned evidence, and evaluate acceptance criteria. `GoalEvaluator`
combines deterministic `Verifier` checks with an optional `ModelJudge` for
explicit semantic criteria. A semantic criterion can declare `guard_method`:
the deterministic guard must return `pass` before the LLM is called. Both paths
produce a shared evaluation report; deterministic checks make no model requests.

Planner and Judge JSON outputs use strict Pydantic schemas and bounded format
retries with temporary correction context; see [structured output recovery](docs/structured-output.md).

The library implements evidence collection, verification orchestration, result
validation, and reporting. Built-in sources cover files, workspaces, commands,
and probes. The host configures sources, registers verification capabilities,
and supplies domain-specific rules where needed. `EvaluationMonitor` connects
reports to trajectory analysis. Missing evidence remains unknown, and changed
evidence invalidates previous conclusions.

Opt into `CompletionPolicy(CompletionMode.ENFORCE)` to retry rejected completions
within the same Run. Observation remains the default.
See the [evaluation guide](docs/evaluation.md) for Harness wiring, custom checks,
report logs, and the credential-free `examples/evaluate_artifact.py` example.

## Keep State Visible at Every Decision

When configured through `EvaluationMonitor.context_pipeline()`, the trajectory
pipeline supplies checkpoint state at each model decision: current facts,
requirement and constraint verdicts, and progress. Suspected cycles suppress
only the warning; state remains visible. Actionable events such as confirmed
cycles or failed completion audits add optional `feedback`.

Missing or incomplete observations produce explicit unavailable status without
replaying old success or coverage. Projection itself performs no extra evaluation
or Planner call, and its instructions remain outside committed conversation
history. See the [v2 context schema and migration notes](docs/trajectory-context-projection.md).

## Generate Tasks and Revise Execution Plans

Supply `AgentHarness(planner=ModelTaskPlanner(...))` to derive a task, goal,
acceptance criteria, and initial execution steps from each query. The planner
selects host-registered verification capabilities. The actor can call `update_plan`
after checkpoint feedback; the Harness validates versions and records revisions,
while acceptance conditions remain fixed for the Run.

```bash
uv run python examples/planned_feature.py         # Configured LLM
uv run python examples/planned_feature.py --demo  # Scripted replies, real tests
```

The example saves actual model contexts and adds a Python feature in an isolated
workspace. See [dynamic task planning](docs/task-planning.md) for composition,
workspace evidence, limits, and audit inspection. Streamlit provider mode also
supports **Dynamic task planning** for query-specific probe validation.

## Explore the Harness in Streamlit

The repository includes an interactive app for exploring Harness behavior.
It runs without credentials in deterministic demo mode, or against the
OpenAI-compatible endpoint configured above.

```bash
uv sync --locked --extra streamlit
uv run streamlit run examples/streamlit_app.py
```

Use the app to inspect JSONL recovery, concurrent Tool timing, cancellation,
Steering, FIFO Follow-ups, Run limits, revisions, usage, and durable Audit
records. Harness settings are captured when you click **Start**; stop the
session before changing them.

**Trajectory feedback** is enabled by default in this app and can be disabled
before starting. By default, the evaluator checks three requirements from actual probe
records for each Run: A completes, B completes, and a completed A/B pair overlaps.
It does not grade arbitrary chat tasks. The **Trajectory** tab shows checkpoint
progress, cycle assessments, completion advice, and the transient instructions
included in model contexts. Detailed views cover the latest Run in the current
session; full reports persist in the session folder's `evaluations/` directory,
with associated receipts in durable Audit.

For a deterministic feedback demonstration, choose demo mode and click
**Controls → Run trajectory recovery**. The model alternates single probes until
the analyzer confirms a cycle, then responds to that Context by requesting both
probes together. Allow at least eight turns and 160 demo tokens. Ordinary parallel
validation remains available, and the same evaluator and Context wiring apply to
the real Provider mode. Enable **Semantic completion review** for a separate
final-answer judge and **Require completion approval** for bounded same-Run
retries. Demo mode uses a deterministic judge stand-in. The Trajectory tab shows
item reasons, evidence versions, missing evidence, and separate Actor/judge costs.
With approval required, **Run completion recovery** demonstrates a rejected claim
followed by verified completion in the same Run (three demo turns).

## Customize Every Boundary

| You want to change               | Extension point    |
| -------------------------------- | ------------------ |
| Model provider or protocol       | `ModelPort`        |
| Local or remote tool backend     | `ToolExecutor`     |
| Context selection and projection | `ContextPipeline`  |
| Long-history summarization       | `ContextCompactor` |
| Session persistence              | `SessionStore`     |
| Logging, tracing, or metrics     | `RunObserver`      |
| Task and initial plan generation | `TaskPlanner`      |
| Evaluation evidence collection  | `EvidenceSource`   |
| Deterministic acceptance rules   | `Verifier`         |
| Online trajectory observation   | `TrajectoryMonitor` in `ejagent.kernel` |

These are narrow, provider-neutral contracts. Implement only the part your
application needs, then compose it through `AgentHarness`. The built-in online
low-level monitor and trajectory Context adapter live in the internal
`ejagent._trajectory` package. Applications can use the public
`ejagent.evaluation` module for evidence sources, deterministic checks, and
semantic judging through `ModelJudge`.

## Built-in Capabilities

- OpenAI-compatible and Anthropic streaming model adapters
- Python function tools, composite tool executors, and MCP tools
- Local Skill discovery and explicit Skill activation
- Derived context compaction without rewriting conversation history
- Ephemeral in-process sessions and durable JSONL sessions
- Concurrent Tool execution with deterministic Conversation ordering
- Cooperative cancellation, live steering, and FIFO follow-ups
- Structured audit records and normalized usage accounting
- Revision-based, idempotent session commits with cross-process file locking
- Deterministic and optional LLM acceptance evaluation with shared reports
- Optional online trajectory assessment with per-decision state and optional feedback

Native user input currently supports text only: `run(task: str)`,
`follow_up(task: str)`, and `UserMessage.content` accept strings. Image and mixed
text/image messages are not implemented; image URLs or Base64 strings remain
text rather than visual model input.

Each `AgentHarness` currently manages one logical agent. Multi-agent coordination
and arbitrary mid-Run pause/resume are not implemented. Trajectory-based Action
denial and mandatory replanning remain policy work. Actor-proposed execution plan
updates are supported for prepared tasks. Completion enforcement is
available as an explicit, independently configured policy.

## Documentation

- [Agent Harness Overview](docs/harness-overview.md) — project scope,
  responsibilities, feedback flow, and current capabilities.
- [Full Usage Guide](docs/usage-guide.md) — installation, configuration, and
  recipes for every built-in capability.
- [Harness Classes and Execution Flow](docs/core-classes-and-runtime-flow.md) — the
  internal model and complete Run lifecycle.
- [Agent Harness Architecture](docs/runtime-kernel-harness-design.md) — normative
  architectural boundaries and invariants.
- [Trajectory Integration](docs/trajectory-runtime-readiness.md) — online
  observation, Context feedback, and enforcement boundaries.

## Development

```bash
uv sync --locked --all-extras --group dev
uv run ruff check src tests examples benchmarks
uv run ruff format --check src tests examples benchmarks
uv run mypy
uv run python -m unittest discover -s tests -p 'test*.py' -q
uv build
```
