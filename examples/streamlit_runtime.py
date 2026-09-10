from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Mapping, Sequence
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import TypeVar

from ejagent._trajectory import TrajectoryContextPipeline, TrajectoryUpdate
from ejagent.contracts import (
    AssistantMessage,
    CancellationToken,
    ContextRequest,
    ContextView,
    ControlReceipt,
    ConversationMessage,
    JsonObject,
    ModelPort,
    ModelRequest,
    ModelResponseCompleted,
    ModelStreamEvent,
    ModelTextDelta,
    ModelUsage,
    RunAudit,
    RunLimits,
    RunOutcome,
    RunResult,
    SystemMessage,
    ToolCall,
    ToolDefinition,
    ToolExecutionResult,
    ToolResultMessage,
    TransientInstruction,
    UserMessage,
    thaw_json_value,
)
from ejagent.contracts.planning import ExecutionPlan, TaskDefinition
from ejagent.evaluation import (
    CheckResult,
    CompletionMode,
    CompletionPolicy,
    EvaluationCriterion,
    EvaluationMonitor,
    EvaluationPlan,
    EvaluationReport,
    EvaluationStatus,
    GoalEvaluator,
    JsonlEvaluationJournal,
    JudgeLimits,
    ModelJudge,
    ProbeEvidenceSource,
    VerificationRequest,
    boolean_field,
)
from ejagent.harness import AgentHarness, HarnessStatus
from ejagent.kernel import CheckpointSignal
from ejagent.planning import ModelTaskPlanner, VerificationCapability
from ejagent.storage import JsonlSessionStore
from ejagent.tools import FunctionTool, FunctionToolExecutor

_ResultT = TypeVar("_ResultT")

ModelFactory = Callable[[], ModelPort]
PROBE_GOAL = (
    "Validate this Run's probes: A completes, B completes, and a completed A/B "
    "pair overlaps in time. This assessment covers only probe validation, not "
    "other user goals."
)
TRAJECTORY_DEMO_TASK = (
    "Demonstrate trajectory recovery. Alternate parallel_probe_a and "
    "parallel_probe_b, one tool per turn. When trajectory context confirms a "
    "cycle, change the plan: call both tools together, then report the result."
)

COMPLETION_DEMO_TASK = (
    "Demonstrate completion recovery. First propose completion before running "
    "the probes. After completion-audit feedback, run both probes together and "
    "return a supported summary."
)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Configuration captured when a validation runtime starts."""

    agent_id: str = "streamlit-validation"
    store_root: Path = Path(".ejagent-sessions")
    max_turns: int = 20
    max_tokens: int | None = None
    max_repeated_tool_calls: int = 3
    probe_delay_seconds: float = 1.5
    trajectory_enabled: bool = True
    semantic_review: bool = False
    dynamic_planning: bool = False
    completion_enforced: bool = False
    completion_max_retries: int = 2
    judge_max_requests: int = 8
    judge_max_tokens: int = 16_384
    judge_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError("agent_id must not be empty")
        object.__setattr__(self, "agent_id", self.agent_id.strip())
        object.__setattr__(self, "store_root", Path(self.store_root).expanduser())
        if self.probe_delay_seconds <= 0:
            raise ValueError("probe_delay_seconds must be greater than zero")
        if not isinstance(self.dynamic_planning, bool):
            raise TypeError("dynamic_planning must be a boolean")
        if not isinstance(self.trajectory_enabled, bool):
            raise TypeError("trajectory_enabled must be a boolean")
        if not isinstance(self.semantic_review, bool) or not isinstance(
            self.completion_enforced, bool
        ):
            raise TypeError("evaluation options must be boolean")
        if not self.trajectory_enabled and (
            self.semantic_review or self.completion_enforced or self.dynamic_planning
        ):
            raise ValueError(
                "planning, semantic review and enforcement require trajectory feedback"
            )
        CompletionPolicy(max_retries=self.completion_max_retries)
        JudgeLimits(
            max_requests=self.judge_max_requests,
            max_tokens=self.judge_max_tokens,
            timeout_seconds=self.judge_timeout_seconds,
        )
        RunLimits(
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            max_repeated_tool_calls=self.max_repeated_tool_calls,
        )

    @property
    def limits(self) -> RunLimits:
        return RunLimits(
            max_turns=self.max_turns,
            max_tokens=self.max_tokens,
            max_repeated_tool_calls=self.max_repeated_tool_calls,
        )


@dataclass(frozen=True, slots=True)
class ProbeExecution:
    """Timing captured for one validation tool invocation."""

    call_id: str
    tool_name: str
    started_at: datetime
    finished_at: datetime | None = None
    elapsed_seconds: float | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Detached view read by the Streamlit script thread."""

    agent_id: str
    status: HarnessStatus
    revision: int
    messages: tuple[ConversationMessage, ...]
    last_result: RunResult | None
    latest_outcome: RunOutcome | None
    audits: tuple[RunAudit, ...]
    pending_follow_ups: int
    controls: tuple[ControlReceipt, ...]
    probes: tuple[ProbeExecution, ...]
    last_error: str | None
    trajectory_updates: tuple[TrajectoryUpdate, ...] = ()
    trajectory_contexts: tuple[TrajectoryContextDelivery, ...] = ()
    evaluation_reports: tuple[EvaluationReport, ...] = ()
    task_definition: TaskDefinition | None = None
    execution_plan: ExecutionPlan | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryContextDelivery:
    """The trajectory instruction included in a built model ContextView."""

    run_id: str
    turn: int
    instruction: TransientInstruction


class _InspectableTrajectoryPipeline:
    def __init__(self, pipeline: TrajectoryContextPipeline) -> None:
        self._pipeline = pipeline
        self.deliveries: list[TrajectoryContextDelivery] = []

    async def start(self) -> None:
        await self._pipeline.start()

    async def shutdown(self) -> None:
        await self._pipeline.shutdown()

    async def build(
        self,
        request: ContextRequest,
        *,
        cancellation: CancellationToken,
    ) -> ContextView:
        view = await self._pipeline.build(request, cancellation=cancellation)
        for message in view.messages:
            if isinstance(message, TransientInstruction) and (
                message.source.startswith("trajectory:")
                or message.source == "completion_audit"
            ):
                self.deliveries.append(
                    TrajectoryContextDelivery(request.run_id, request.turn, message)
                )
        return view


@dataclass(slots=True)
class _ActiveProbe:
    call_id: str
    tool_name: str
    started_at: datetime
    started_monotonic: float


@dataclass(slots=True)
class _ProbeRecorder:
    evidence: ProbeEvidenceSource | None = None
    run_id: str | None = None
    active: dict[str, _ActiveProbe] = field(default_factory=dict)
    completed: list[ProbeExecution] = field(default_factory=list)

    def start(self, call: ToolCall) -> None:
        self.active[call.id] = _ActiveProbe(
            call_id=call.id,
            tool_name=call.name,
            started_at=datetime.now(UTC),
            started_monotonic=monotonic(),
        )

    def finish(self, call: ToolCall, *, cancelled: bool) -> ProbeExecution:
        active = self.active.pop(call.id)
        finished_at = datetime.now(UTC)
        execution = ProbeExecution(
            call_id=call.id,
            tool_name=call.name,
            started_at=active.started_at,
            finished_at=finished_at,
            elapsed_seconds=monotonic() - active.started_monotonic,
            cancelled=cancelled,
        )
        if self.evidence is not None and self.run_id is not None:
            self.evidence.record(
                self.run_id,
                call.name,
                started_at=active.started_monotonic,
                finished_at=active.started_monotonic + (execution.elapsed_seconds or 0),
                cancelled=cancelled,
            )
        self.completed.append(execution)
        return execution

    def snapshot(self) -> tuple[ProbeExecution, ...]:
        running = tuple(
            ProbeExecution(
                call_id=item.call_id,
                tool_name=item.tool_name,
                started_at=item.started_at,
            )
            for item in self.active.values()
        )
        return (*self.completed, *running)


class _RecordedProbeSource(ProbeEvidenceSource):
    def __init__(self, recorder: _ProbeRecorder) -> None:
        super().__init__(("parallel_probe_a", "parallel_probe_b"))
        self._recorder = recorder
        recorder.evidence = self

    async def revision(
        self, signal: CheckpointSignal, *, cancellation: CancellationToken
    ) -> str:
        self._recorder.run_id = signal.run_id
        return await super().revision(signal, cancellation=cancellation)

    def close_run(self, run_id: str) -> None:
        super().close_run(run_id)
        if self._recorder.run_id == run_id:
            self._recorder.run_id = None


def _evaluation_plan(semantic: bool) -> EvaluationPlan:
    requirements: tuple[EvaluationCriterion, ...] = (
        EvaluationCriterion(
            "probe_a_completed", "Probe A completes", "probe_a", ("probes",)
        ),
        EvaluationCriterion(
            "probe_b_completed", "Probe B completes", "probe_b", ("probes",)
        ),
        EvaluationCriterion(
            "probes_overlapped", "Completed probes overlap", "overlap", ("probes",)
        ),
    )
    if semantic:
        requirements += (
            EvaluationCriterion(
                "probe_summary",
                "The proposed final answer accurately summarizes the completed parallel probes and does not claim unsupported results",
                "summary_quality",
                ("probes", "$completion"),
                semantic=True,
                guard_method="summary_guard",
                completion_only=True,
            ),
        )
    return EvaluationPlan(
        PROBE_GOAL,
        "streamlit-probes.v3-semantic" if semantic else "streamlit-probes.v3",
        requirements,
    )


async def _summary_guard(
    request: VerificationRequest, cancellation: CancellationToken
) -> CheckResult:
    cancellation.raise_if_cancelled()
    value = request.evidence["probes"].value
    passed = isinstance(value, Mapping) and all(
        value.get(key) is True
        for key in ("parallel_probe_a", "parallel_probe_b", "overlapped")
    )
    return CheckResult(
        EvaluationStatus.PASS if passed else EvaluationStatus.FAIL,
        "Host probe evidence confirms completion and overlap"
        if passed
        else "Host probe evidence does not confirm completion and overlap",
        ("probes",),
    )


class DemoJudgeModel(ModelPort):
    """Credential-free stand-in exercising the same structured judge protocol."""

    async def stream(
        self, request: ModelRequest, *, cancellation: CancellationToken
    ) -> AsyncIterator[ModelStreamEvent]:
        cancellation.raise_if_cancelled()
        message = request.messages[-1]
        if not isinstance(message, UserMessage):
            raise ValueError("demo judge requires the structured evaluation request")
        data = json.loads(message.content)
        evidence = data["evidence"]
        candidate = next(
            (item["value"] for item in evidence if isinstance(item["value"], str)), ""
        )
        passed = "parallel_probe_a" in candidate and "parallel_probe_b" in candidate
        content = json.dumps(
            {
                "criterion_id": data["criterion_id"],
                "status": "pass" if passed else "fail",
                "rationale": "Demo summary check: both completed probes are named"
                if passed
                else "Demo summary must name both completed probes",
                "evidence_refs": [item["reference"] for item in evidence],
                "missing_evidence": [],
            }
        )
        yield ModelResponseCompleted(AssistantMessage(content), ModelUsage(40, 20, 60))


class DemoValidationModel(ModelPort):
    """Validate parallel tools, with an explicit feedback-driven cycle scenario."""

    def __init__(self) -> None:
        self._batch = 0

    async def stream(
        self,
        request: ModelRequest,
        *,
        cancellation: CancellationToken,
    ) -> AsyncIterator[ModelStreamEvent]:
        cancellation.raise_if_cancelled()
        current_messages = _current_task_messages(request)
        results = tuple(
            message
            for message in current_messages
            if isinstance(message, ToolResultMessage)
        )
        usage = ModelUsage(input_tokens=12, output_tokens=8, total_tokens=20)
        task = next(
            (
                m.content
                for m in reversed(request.messages)
                if isinstance(m, UserMessage)
            ),
            "",
        )
        completion_demo = task.startswith("Demonstrate completion recovery.")
        was_rejected = any(
            isinstance(message, TransientInstruction)
            and message.source == "completion_audit"
            for message in current_messages
        )
        if completion_demo and not results and not was_rejected:
            yield ModelResponseCompleted(
                AssistantMessage("Parallel validation is complete."), usage
            )
            return
        cycle_demo = task.startswith("Demonstrate trajectory recovery.")
        parallel_requested = any(
            isinstance(message, AssistantMessage) and len(message.tool_calls) == 2
            for message in current_messages
        )
        confirmed = any(
            isinstance(message, TransientInstruction)
            and message.source == "trajectory:cycle_confirmed"
            for message in current_messages
        )
        if not results or (cycle_demo and not parallel_requested):
            self._batch += 1
            batch = self._batch
            calls: tuple[ToolCall, ...] = (
                ToolCall(f"demo-{batch}-a", "parallel_probe_a"),
                ToolCall(f"demo-{batch}-b", "parallel_probe_b"),
            )
            if cycle_demo and not confirmed:
                calls = (calls[len(results) % 2],)
            yield ModelResponseCompleted(
                AssistantMessage(tool_calls=calls),
                usage,
            )
            return

        steering = tuple(
            message.content
            for message in current_messages
            if isinstance(message, TransientInstruction)
            and message.source == "steering"
        )
        labels = ", ".join(message.tool_name for message in results)
        text = f"Parallel validation completed with {labels}."
        if cycle_demo:
            text += " Changed to a parallel batch after confirmed-cycle feedback."
        if steering:
            text += f" Applied steering: {' | '.join(steering)}"
        yield ModelTextDelta(text)
        yield ModelResponseCompleted(AssistantMessage(content=text), usage)


def _current_task_messages(request: ModelRequest) -> Sequence[object]:
    for index in range(len(request.messages) - 1, -1, -1):
        if isinstance(request.messages[index], UserMessage):
            return request.messages[index + 1 :]
    return request.messages


def _validation_tools(
    recorder: _ProbeRecorder,
    delay_seconds: float,
) -> FunctionToolExecutor:
    async def probe(
        call: ToolCall,
        cancellation: CancellationToken,
    ) -> ToolExecutionResult:
        cancellation.raise_if_cancelled()
        recorder.start(call)
        try:
            await asyncio.sleep(delay_seconds)
        except asyncio.CancelledError:
            recorder.finish(call, cancelled=True)
            raise
        execution = recorder.finish(call, cancelled=False)
        return ToolExecutionResult(
            {
                "tool": call.name,
                "started_at": execution.started_at.isoformat(),
                "finished_at": execution.finished_at.isoformat()
                if execution.finished_at is not None
                else None,
                "elapsed_seconds": execution.elapsed_seconds,
            }
        )

    schema: JsonObject = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    return FunctionToolExecutor(
        (
            FunctionTool(
                ToolDefinition(
                    name="parallel_probe_a",
                    description="Wait briefly and return timing data for probe A.",
                    input_schema=schema,
                ),
                probe,
            ),
            FunctionTool(
                ToolDefinition(
                    name="parallel_probe_b",
                    description="Wait briefly and return timing data for probe B.",
                    input_schema=schema,
                ),
                probe,
            ),
        )
    )


class StreamlitRuntimeController:
    """Keep one AgentHarness on a dedicated, long-lived asyncio loop."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        model_factory: ModelFactory = DemoValidationModel,
        judge_factory: ModelFactory | None = None,
        planner_factory: ModelFactory | None = None,
        command_timeout: float = 5.0,
    ) -> None:
        if command_timeout <= 0:
            raise ValueError("command_timeout must be greater than zero")
        self.config = config
        self._model_factory = model_factory
        if (
            config.semantic_review
            and judge_factory is None
            and model_factory is not DemoValidationModel
        ):
            raise ValueError(
                "semantic review requires an explicit judge_factory with a custom Actor"
            )
        if config.dynamic_planning and planner_factory is None:
            raise ValueError("dynamic planning requires an explicit planner_factory")
        self._planner_factory = planner_factory
        self._judge_factory = judge_factory or DemoJudgeModel
        self._command_timeout = command_timeout
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._harness: AgentHarness | None = None
        self._store: JsonlSessionStore | None = None
        self._recorder: _ProbeRecorder | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._follow_up_tasks: set[asyncio.Task[None]] = set()
        self._latest_outcome: RunOutcome | None = None
        self._controls: list[ControlReceipt] = []
        self._last_error: str | None = None
        self._startup_error: BaseException | None = None
        self._trajectory_updates: list[TrajectoryUpdate] = []
        self._evaluation_reports: list[EvaluationReport] = []
        self._trajectory_pipeline: _InspectableTrajectoryPipeline | None = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"ejagent-{config.agent_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=command_timeout):
            raise TimeoutError("EJAgent runtime did not start in time")
        if self._startup_error is not None:
            raise RuntimeError(
                "EJAgent runtime failed to start"
            ) from self._startup_error

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def start_run(self, task: str) -> None:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        self._call(self._start_run(task.strip()))

    def cancel(self, reason: str = "Cancelled from Streamlit") -> bool:
        return self._call(self._cancel(reason))

    def steer(self, content: str) -> ControlReceipt:
        if not isinstance(content, str) or not content.strip():
            raise ValueError("steering content must not be empty")
        return self._call(self._steer(content.strip()))

    def follow_up(self, task: str) -> ControlReceipt:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("follow-up task must not be empty")
        return self._call(self._follow_up(task.strip()))

    def snapshot(self) -> RuntimeSnapshot:
        return self._call(self._snapshot())

    def close(self) -> None:
        if self._closed.is_set():
            return
        loop = self._loop
        try:
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
                future.result(timeout=max(self._command_timeout, 10.0))
        finally:
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=max(self._command_timeout, 10.0))
            self._closed.set()

    def __enter__(self) -> StreamlitRuntimeController:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._initialize())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
        else:
            self._ready.set()
            loop.run_forever()
        finally:
            if self._harness is not None:
                try:
                    loop.run_until_complete(self._harness.shutdown())
                except BaseException as exc:
                    self._last_error = f"{type(exc).__name__}: {exc}"
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._closed.set()

    async def _initialize(self) -> None:
        recorder = _ProbeRecorder()
        store = JsonlSessionStore(self.config.store_root)
        monitor = None
        if self.config.trajectory_enabled:
            source = _RecordedProbeSource(recorder)
            journal = JsonlEvaluationJournal(
                self.config.store_root
                / "evaluations"
                / (hashlib.sha256(self.config.agent_id.encode()).hexdigest() + ".jsonl")
            )

            def report_sink(report: EvaluationReport) -> None:
                if (
                    not self._evaluation_reports
                    or self._evaluation_reports[-1].run_id != report.run_id
                ):
                    self._evaluation_reports.clear()
                    self._trajectory_updates.clear()
                    if self._trajectory_pipeline is not None:
                        self._trajectory_pipeline.deliveries.clear()
                self._evaluation_reports.append(report)
                journal(report)

            judge = (
                ModelJudge(
                    self._judge_factory(),
                    limits=JudgeLimits(
                        max_requests=self.config.judge_max_requests,
                        max_tokens=self.config.judge_max_tokens,
                        timeout_seconds=self.config.judge_timeout_seconds,
                    ),
                )
                if self.config.semantic_review
                else None
            )
            evaluator = GoalEvaluator(
                sources={"probes": source},
                verifiers={
                    "probe_a": boolean_field("parallel_probe_a"),
                    "probe_b": boolean_field("parallel_probe_b"),
                    "overlap": boolean_field("overlapped"),
                    "summary_guard": _summary_guard,
                },
                semantic_judge=judge,
                report_sink=report_sink,
            )
            monitor = EvaluationMonitor(
                evaluator, update_sink=self._trajectory_updates.append
            )
            self._trajectory_pipeline = _InspectableTrajectoryPipeline(
                monitor.context_pipeline()
            )
        planner = None
        if self.config.dynamic_planning:
            assert self._planner_factory is not None
            template = _evaluation_plan(self.config.semantic_review)
            planner = ModelTaskPlanner(
                self._planner_factory(),
                capabilities=tuple(
                    VerificationCapability(item.criterion_id, item)
                    for item in template.requirements
                ),
                environment={
                    "scope": "Only this Run probe completion, overlap and optional probe summary can be verified. Other user goals are unsupported. Summary review requires both probes to finish and overlap."
                },
            )
        harness = AgentHarness(
            agent_id=self.config.agent_id,
            model=self._model_factory(),
            planner=planner,
            tools=_validation_tools(recorder, self.config.probe_delay_seconds),
            trajectory=monitor,
            completion_policy=CompletionPolicy(
                CompletionMode.ENFORCE
                if self.config.completion_enforced
                else CompletionMode.OBSERVE,
                self.config.completion_max_retries,
            ),
            context=self._trajectory_pipeline,
            initial_messages=(
                SystemMessage(
                    "You are validating EJAgent. When asked to verify parallel tools, "
                    "call parallel_probe_a and parallel_probe_b in one response."
                ),
            ),
            store=store,
            limits=self.config.limits,
            configuration_revision=(
                "streamlit-example-v2-trajectory"
                if self.config.trajectory_enabled
                else "streamlit-example-v2"
            ),
        )
        await harness.start()
        self._recorder = recorder
        self._store = store
        self._harness = harness

    async def _start_run(self, task: str) -> None:
        harness = self._require_harness()
        if harness.status is HarnessStatus.RUNNING:
            raise RuntimeError("a Run is already active; use follow-up instead")
        if self._run_task is not None and not self._run_task.done():
            raise RuntimeError("a Run submission is already pending")
        self._last_error = None
        self._run_task = asyncio.create_task(
            self._capture_run(
                harness.run(
                    task,
                    evaluation_plan=_evaluation_plan(self.config.semantic_review)
                    if self.config.trajectory_enabled
                    and not self.config.dynamic_planning
                    else None,
                )
            )
        )

    async def _capture_run(self, run: Coroutine[object, object, RunOutcome]) -> None:
        try:
            self._latest_outcome = await run
        except BaseException as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"

    async def _cancel(self, reason: str) -> bool:
        return self._require_harness().cancel(reason)

    async def _steer(self, content: str) -> ControlReceipt:
        receipt = self._require_harness().steer(content)
        self._record_control(receipt)
        return receipt

    async def _follow_up(self, task: str) -> ControlReceipt:
        handle = self._require_harness().follow_up(
            task,
            evaluation_plan=_evaluation_plan(self.config.semantic_review)
            if self.config.trajectory_enabled and not self.config.dynamic_planning
            else None,
        )
        self._record_control(handle.receipt)
        if handle.accepted:
            watcher = asyncio.create_task(self._capture_follow_up(handle.wait()))
            self._follow_up_tasks.add(watcher)
            watcher.add_done_callback(self._follow_up_tasks.discard)
        return handle.receipt

    async def _capture_follow_up(
        self,
        outcome: Coroutine[object, object, RunOutcome],
    ) -> None:
        try:
            self._latest_outcome = await outcome
        except BaseException as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
        else:
            self._last_error = None

    async def _snapshot(self) -> RuntimeSnapshot:
        harness = self._require_harness()
        store = self._require_store()
        recorder = self._require_recorder()
        return RuntimeSnapshot(
            agent_id=harness.agent_id,
            status=harness.status,
            revision=harness.revision,
            messages=harness.messages,
            last_result=harness.last_result,
            latest_outcome=self._latest_outcome,
            audits=await store.load_audit(harness.agent_id),
            pending_follow_ups=harness.pending_follow_up_count,
            controls=tuple(self._controls),
            probes=recorder.snapshot(),
            last_error=self._last_error,
            trajectory_updates=tuple(self._trajectory_updates),
            evaluation_reports=tuple(self._evaluation_reports),
            task_definition=harness.last_task_definition,
            execution_plan=harness.last_execution_plan,
            trajectory_contexts=(
                tuple(self._trajectory_pipeline.deliveries)
                if self._trajectory_pipeline is not None
                else ()
            ),
        )

    async def _shutdown(self) -> None:
        harness = self._require_harness()
        await harness.shutdown()
        tasks = tuple(self._follow_up_tasks)
        if self._run_task is not None:
            tasks = (*tasks, self._run_task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _call(self, coroutine: Coroutine[object, object, _ResultT]) -> _ResultT:
        if self._closed.is_set():
            coroutine.close()
            raise RuntimeError("EJAgent runtime is closed")
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise RuntimeError("EJAgent runtime is not running")
        future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        try:
            return future.result(timeout=self._command_timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError("EJAgent runtime command timed out") from exc

    def _record_control(self, receipt: ControlReceipt) -> None:
        self._controls.append(receipt)
        del self._controls[:-20]

    def _require_harness(self) -> AgentHarness:
        if self._harness is None:
            raise RuntimeError("AgentHarness is not initialized")
        return self._harness

    def _require_store(self) -> JsonlSessionStore:
        if self._store is None:
            raise RuntimeError("JsonlSessionStore is not initialized")
        return self._store

    def _require_recorder(self) -> _ProbeRecorder:
        if self._recorder is None:
            raise RuntimeError("probe recorder is not initialized")
        return self._recorder


def message_to_data(message: ConversationMessage) -> dict[str, object]:
    """Convert a committed message into values Streamlit can render."""

    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content}
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.content}
    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": thaw_json_value(call.arguments),
                }
                for call in message.tool_calls
            ],
        }
    assert isinstance(message, ToolResultMessage)
    return {
        "role": "tool",
        "tool_call_id": message.tool_call_id,
        "tool_name": message.tool_name,
        "result": thaw_json_value(message.result),
        "is_error": message.is_error,
    }
