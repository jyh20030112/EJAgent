from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import TracebackType
from typing import Self
from uuid import uuid4

from ejagent.contracts.audit import RunAudit
from ejagent.contracts.context import ContextPipeline
from ejagent.contracts.control import (
    CancellationSource,
    ControlKind,
    ControlReceipt,
    ControlStatus,
    RunCancelledError,
    SteeringInput,
)
from ejagent.contracts.conversation import ConversationSnapshot
from ejagent.contracts.evaluation import (
    CompletionMode,
    CompletionPolicy,
    EvaluationPlan,
)
from ejagent.contracts.json import JsonValue
from ejagent.contracts.lifecycle import ManagedResource
from ejagent.contracts.messages import ConversationMessage
from ejagent.contracts.model import ModelPort
from ejagent.contracts.observer import RunObserver
from ejagent.contracts.planning import (
    ExecutionPlan,
    PlanningError,
    PlanningRequest,
    PlanningResult,
    TaskDefinition,
    TaskPlanner,
)
from ejagent.contracts.runs import (
    AuditRecord,
    FailureCode,
    RunDelta,
    RunFailure,
    RunIntent,
    RunLimits,
    RunOutcome,
    RunPhase,
    RunResult,
    RunSpec,
    RunStatus,
    StopReason,
)
from ejagent.contracts.session import (
    SessionCommit,
    SessionSnapshot,
    SessionStore,
    SessionStoreError,
)
from ejagent.contracts.tools import ToolExecutor
from ejagent.harness._control import (
    FollowUpDiscardedError,
    FollowUpHandle,
    _QueuedFollowUp,
    _RunControls,
)
from ejagent.harness._planning import _PlanningSession, append_planning_audit
from ejagent.kernel import RuntimeKernel, TrajectoryMonitor

RunIdFactory = Callable[[], str]
Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


class HarnessStatus(StrEnum):
    """Observable lifecycle phase of one AgentHarness."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    CLOSED = "closed"


class HarnessClosedError(RuntimeError):
    """A Run was requested after Harness shutdown began."""


class SessionStoreProtocolError(RuntimeError):
    """A SessionStore returned a snapshot that violates its contract."""


class AgentHarness:
    """Own one agent's resources, Conversation, and atomic Run commits."""

    def __init__(
        self,
        *,
        agent_id: str,
        model: ModelPort,
        tools: ToolExecutor,
        context: ContextPipeline | None = None,
        planner: TaskPlanner | None = None,
        trajectory: TrajectoryMonitor | None = None,
        completion_policy: CompletionPolicy | None = None,
        initial_messages: Iterable[ConversationMessage] = (),
        store: SessionStore | None = None,
        observers: Iterable[RunObserver] = (),
        resources: Iterable[object] = (),
        limits: RunLimits | None = None,
        configuration_revision: str = "default",
        run_id_factory: RunIdFactory | None = None,
        clock: Clock | None = None,
        steering_capacity: int = 16,
        follow_up_capacity: int = 16,
    ) -> None:
        if not isinstance(agent_id, str):
            raise TypeError("agent_id must be a string")
        agent_id = agent_id.strip()
        if not agent_id:
            raise ValueError("agent_id must not be empty")
        if not isinstance(configuration_revision, str):
            raise TypeError("configuration_revision must be a string")
        if not configuration_revision.strip():
            raise ValueError("configuration_revision must not be empty")
        if limits is not None and not isinstance(limits, RunLimits):
            raise TypeError("limits must be RunLimits or None")
        if run_id_factory is not None and not callable(run_id_factory):
            raise TypeError("run_id_factory must be callable or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        self._validate_capacity(steering_capacity, "steering_capacity")
        self._validate_capacity(follow_up_capacity, "follow_up_capacity")

        initial = SessionSnapshot(
            agent_id=agent_id,
            conversation=ConversationSnapshot(messages=tuple(initial_messages)),
        )
        self._completion_policy = completion_policy or CompletionPolicy()
        if not isinstance(self._completion_policy, CompletionPolicy):
            raise TypeError("completion_policy must be CompletionPolicy")
        if (
            self._completion_policy.mode is CompletionMode.ENFORCE
            and trajectory is None
        ):
            raise ValueError("completion enforcement requires a trajectory monitor")
        if planner is not None and trajectory is None:
            raise ValueError("task planning requires a trajectory monitor")
        if planner is not None and any(
            t.name == "update_plan" for t in tools.definitions
        ):
            raise ValueError("update_plan is reserved when task planning is enabled")
        self._planner = planner
        self._trajectory = trajectory
        self._last_task_definition: TaskDefinition | None = None
        self._last_execution_plan: ExecutionPlan | None = None
        self._active_planning: _PlanningSession | None = None
        self._agent_id = agent_id
        self._model = model
        self._tools = tools
        self._context = context
        self._observers = tuple(observers)
        self._store = store
        self._snapshot = initial
        self._limits = limits or RunLimits()
        self._configuration_revision = configuration_revision
        self._run_id_factory = run_id_factory or (lambda: str(uuid4()))
        self._clock = clock or _utc_now
        self._steering_capacity = steering_capacity
        self._follow_up_capacity = follow_up_capacity
        self._kernel = RuntimeKernel(
            model=model,
            tools=tools,
            context=context,
            trajectory=trajectory,
            clock=self._clock,
        )
        self._resources = self._managed_resources(
            (
                self._store,
                self._model,
                self._tools,
                self._context,
                *self._observers,
                *getattr(trajectory, "resources", ()),
                *getattr(planner, "resources", ()),
                planner,
                *resources,
            )
        )
        self._started_resources: tuple[ManagedResource, ...] = ()
        self._status = HarnessStatus.NEW
        self._closing = False
        self._active_cancellation: CancellationSource | None = None
        self._active_controls: _RunControls | None = None
        self._follow_ups: deque[_QueuedFollowUp] = deque()
        self._outstanding_follow_ups = 0
        self._follow_up_worker: asyncio.Task[None] | None = None
        self._observer_tasks: set[asyncio.Task[None]] = set()
        self._lifecycle_lock = asyncio.Lock()
        self._run_lock = asyncio.Lock()

    @property
    def last_task_definition(self) -> TaskDefinition | None:
        """Latest prepared task; durable copies live in Run audit records."""
        return self._last_task_definition

    @property
    def last_execution_plan(self) -> ExecutionPlan | None:
        """Latest Run execution plan, including updates before a failed outcome."""
        return (
            self._active_planning.plan
            if self._active_planning is not None
            else self._last_execution_plan
        )

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def status(self) -> HarnessStatus:
        return self._status

    @property
    def revision(self) -> int:
        return self._snapshot.revision

    @property
    def messages(self) -> tuple[ConversationMessage, ...]:
        return self._snapshot.messages

    @property
    def last_result(self) -> RunResult | None:
        return self._snapshot.last_result

    @property
    def snapshot(self) -> SessionSnapshot:
        return self._snapshot

    @property
    def pending_follow_up_count(self) -> int:
        return self._outstanding_follow_ups

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.shutdown()

    async def start(self) -> None:
        """Start owned resources transactionally and restore Conversation state."""

        async with self._lifecycle_lock:
            if self._status in (HarnessStatus.READY, HarnessStatus.RUNNING):
                return
            if self._closing or self._status is HarnessStatus.CLOSED:
                raise HarnessClosedError(f"agent harness {self._agent_id!r} is closed")

            self._status = HarnessStatus.STARTING
            started: list[ManagedResource] = []
            try:
                for resource in self._resources:
                    await resource.start()
                    started.append(resource)
                if self._store is not None:
                    loaded = await self._store.load(self._agent_id)
                    if loaded is not None:
                        self._validate_loaded_snapshot(loaded)
                        self._snapshot = loaded
            except BaseException:
                await self._rollback_start(started)
                self._status = HarnessStatus.NEW
                raise

            self._started_resources = tuple(started)
            self._status = HarnessStatus.READY

    async def run(
        self,
        task: str,
        *,
        limits: RunLimits | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        evaluation_plan: EvaluationPlan | None = None,
    ) -> RunOutcome:
        """Run one task after earlier calls, then atomically commit on success."""

        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must not be empty")
        return await self._execute(
            intent=RunIntent.TASK,
            task=task,
            limits=limits,
            metadata=metadata,
            evaluation_plan=evaluation_plan,
        )

    async def continue_run(
        self,
        *,
        limits: RunLimits | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        evaluation_plan: EvaluationPlan | None = None,
    ) -> RunOutcome:
        """Continue committed Conversation without appending a user message."""

        return await self._execute(
            intent=RunIntent.CONTINUE,
            task=None,
            limits=limits,
            metadata=metadata,
            evaluation_plan=evaluation_plan,
        )

    def cancel(self, reason: str | None = None) -> bool:
        """Request cooperative cancellation of the active Run, if any."""

        source = self._active_cancellation
        return source.cancel(reason) if source is not None else False

    def steer(self, content: str) -> ControlReceipt:
        """Queue one transient instruction for the next model-call safe point."""

        if not isinstance(content, str) or not content.strip():
            raise ValueError("steering content must not be empty")
        input_id = uuid4().hex
        if self._closing or self._status is HarnessStatus.CLOSED:
            status = ControlStatus.CLOSED
        elif self._status is not HarnessStatus.RUNNING:
            status = ControlStatus.NOT_RUNNING
        elif self._active_controls is None or self._active_controls.closed:
            status = ControlStatus.TOO_LATE
        elif self._active_controls.offer(
            SteeringInput(input_id=input_id, content=content.strip())
        ):
            status = ControlStatus.ACCEPTED
        else:
            status = ControlStatus.QUEUE_FULL
        return ControlReceipt(input_id, ControlKind.STEERING, status)

    def follow_up(
        self, task: str, *, evaluation_plan: EvaluationPlan | None = None
    ) -> FollowUpHandle:
        """Submit an independent FIFO Run to follow the active Run chain."""

        if not isinstance(task, str) or not task.strip():
            raise ValueError("follow-up task must not be empty")
        if evaluation_plan is not None and not isinstance(
            evaluation_plan, EvaluationPlan
        ):
            raise TypeError("evaluation_plan must be EvaluationPlan or None")
        input_id = uuid4().hex
        if self._closing or self._status is HarnessStatus.CLOSED:
            status = ControlStatus.CLOSED
        elif self._status is not HarnessStatus.RUNNING:
            status = ControlStatus.NOT_RUNNING
        elif self._outstanding_follow_ups >= self._follow_up_capacity:
            status = ControlStatus.QUEUE_FULL
        else:
            status = ControlStatus.ACCEPTED
        receipt = ControlReceipt(input_id, ControlKind.FOLLOW_UP, status)
        handle = FollowUpHandle(receipt)
        if receipt.accepted:
            self._follow_ups.append(
                _QueuedFollowUp(task.strip(), handle, evaluation_plan)
            )
            self._outstanding_follow_ups += 1
            if self._follow_up_worker is None:
                self._follow_up_worker = asyncio.create_task(self._run_follow_ups())
        return handle

    async def shutdown(self) -> None:
        """Stop accepting Runs, cancel active work, and release resources."""

        async with self._lifecycle_lock:
            if self._status is HarnessStatus.CLOSED:
                return
            self._closing = True
            self._status = HarnessStatus.STOPPING
            self._discard_pending_follow_ups("AgentHarness is shutting down")
            self.cancel("AgentHarness is shutting down")

            async with self._run_lock:
                await self._flush_observers()
                failures: list[BaseException] = []
                for resource in reversed(self._started_resources):
                    try:
                        await resource.shutdown()
                    except BaseException as exc:
                        failures.append(exc)
                self._started_resources = ()
                self._status = HarnessStatus.CLOSED

            worker = self._follow_up_worker
            if worker is not None and worker is not asyncio.current_task():
                await asyncio.gather(worker, return_exceptions=True)

            if failures:
                raise BaseExceptionGroup(
                    "one or more AgentHarness resources failed to shut down",
                    failures,
                )

    async def _execute(
        self,
        *,
        intent: RunIntent,
        task: str | None,
        limits: RunLimits | None,
        metadata: Mapping[str, JsonValue] | None,
        evaluation_plan: EvaluationPlan | None,
    ) -> RunOutcome:
        if evaluation_plan is not None and not isinstance(
            evaluation_plan, EvaluationPlan
        ):
            raise TypeError("evaluation_plan must be EvaluationPlan or None")
        await self.start()
        async with self._run_lock:
            if self._closing or self._status is HarnessStatus.CLOSED:
                raise HarnessClosedError(f"agent harness {self._agent_id!r} is closed")

            base = self._snapshot
            run_id = self._run_id_factory()
            cancellation = CancellationSource()
            controls = _RunControls(self._steering_capacity)
            self._active_cancellation = cancellation
            self._active_controls = controls
            self._status = HarnessStatus.RUNNING
            planning: _PlanningSession | None = None
            preparation_records: list[AuditRecord] = []
            self._last_task_definition = None
            self._last_execution_plan = None
            try:
                try:
                    if (
                        self._planner is not None
                        and task is not None
                        and evaluation_plan is None
                    ):
                        preparation_records.append(
                            AuditRecord(run_id, 1, "planning_started", self._clock())
                        )
                        prepared = await cancellation.token.run(
                            self._planner.plan(
                                PlanningRequest(
                                    run_id,
                                    task,
                                    base.messages,
                                    tuple(self._tools.definitions),
                                ),
                                cancellation=cancellation.token,
                            )
                        )
                        if not isinstance(prepared, PlanningResult) or not isinstance(
                            prepared.definition, TaskDefinition
                        ):
                            raise PlanningError(
                                "planner must return a PlanningResult with TaskDefinition"
                            )
                        preparation_records.append(
                            AuditRecord(
                                run_id,
                                2,
                                "planning_usage",
                                self._clock(),
                                {
                                    "model_requests": prepared.requests,
                                    "usage": prepared.usage.to_dict()
                                    if prepared.usage
                                    else None,
                                    "accounting": "separate_from_actor_run_usage",
                                },
                            )
                        )
                        definition = prepared.definition
                        if (
                            definition.execution_plan.version != 1
                            or definition.execution_plan.based_on_checkpoint is not None
                        ):
                            raise PlanningError(
                                "initial execution plan must be version 1 without a checkpoint"
                            )
                        evaluation_plan = definition.evaluation_plan
                        assert self._trajectory is not None
                        validate = getattr(self._trajectory, "validate_plan", None)
                        if validate is not None:
                            validate(evaluation_plan)
                        planning = _PlanningSession(
                            run_id=run_id,
                            definition=definition,
                            tools=self._tools,
                            context=self._context,
                            trajectory=self._trajectory,
                            clock=self._clock,
                        )
                        self._active_planning = planning
                        self._last_task_definition = definition
                        self._last_execution_plan = definition.execution_plan
                    spec = RunSpec(
                        run_id=run_id,
                        base_revision=base.revision,
                        intent=intent,
                        task=task,
                        messages=base.messages,
                        limits=limits or self._limits,
                        configuration_revision=self._configuration_revision,
                        metadata=metadata or {},
                        evaluation_plan=evaluation_plan,
                        completion_policy=self._completion_policy,
                    )
                except (PlanningError, RunCancelledError, ValueError) as exc:
                    if not preparation_records:
                        raise
                    cancelled = isinstance(exc, RunCancelledError)
                    if not any(r.kind == "planning_usage" for r in preparation_records):
                        preparation_records.append(
                            AuditRecord(
                                run_id,
                                len(preparation_records) + 1,
                                "planning_usage",
                                self._clock(),
                                {
                                    "model_requests": exc.requests
                                    if isinstance(exc, PlanningError)
                                    else None,
                                    "usage": exc.usage.to_dict()
                                    if isinstance(exc, PlanningError) and exc.usage
                                    else None,
                                    "accounting": "separate_from_actor_run_usage",
                                    "incomplete": True,
                                },
                            )
                        )
                    failure = RunFailure(
                        RunPhase.PREPARATION,
                        FailureCode.CANCELLED
                        if cancelled
                        else FailureCode.POLICY_REJECTED,
                        str(exc),
                        cause=exc,
                    )
                    preparation_records.append(
                        AuditRecord(
                            run_id,
                            len(preparation_records) + 1,
                            "planning_failed",
                            self._clock(),
                            {"reason": str(exc)},
                        )
                    )
                    outcome = RunOutcome(
                        result=RunResult(
                            run_id,
                            RunStatus.CANCELLED if cancelled else RunStatus.FAILED,
                            StopReason.EXTERNAL_ABORT
                            if cancelled
                            else StopReason.BEHAVIOR_STOP,
                            0,
                        ),
                        delta=RunDelta(base.revision),
                        failure=None if cancelled else failure,
                    )
                else:
                    kernel = (
                        self._kernel
                        if planning is None
                        else RuntimeKernel(
                            model=self._model,
                            tools=planning,
                            context=planning,
                            trajectory=planning,
                            clock=self._clock,
                        )
                    )
                    outcome = await kernel.run(
                        spec, cancellation=cancellation.token, controls=controls
                    )
                if planning is not None:
                    self._last_execution_plan = planning.plan
                outcome = append_planning_audit(
                    outcome,
                    (
                        *preparation_records,
                        *(planning.records if planning is not None else ()),
                    ),
                )
                discarded = controls.close()
                self._active_controls = None
                outcome = self._append_discarded_steering(outcome, discarded)
                outcome = await self._commit(base, outcome)
                self._dispatch_observers(base, outcome)
                return outcome
            finally:
                self._active_planning = None
                controls.close()
                self._active_cancellation = None
                if self._active_controls is controls:
                    self._active_controls = None
                if not self._closing:
                    self._status = HarnessStatus.READY

    async def _run_follow_ups(self) -> None:
        try:
            while self._follow_ups:
                queued = self._follow_ups.popleft()
                try:
                    if self._closing:
                        queued.handle._fail(
                            FollowUpDiscardedError("AgentHarness is shutting down")
                        )
                        continue
                    try:
                        outcome = await self._execute(
                            intent=RunIntent.TASK,
                            task=queued.task,
                            evaluation_plan=queued.evaluation_plan,
                            limits=None,
                            metadata={
                                "control_input_id": queued.handle.receipt.input_id
                            },
                        )
                    except HarnessClosedError:
                        queued.handle._fail(
                            FollowUpDiscardedError("AgentHarness is shutting down")
                        )
                    except BaseException as exc:
                        queued.handle._fail(exc)
                    else:
                        queued.handle._resolve(outcome)
                finally:
                    self._outstanding_follow_ups -= 1
        finally:
            self._follow_up_worker = None

    def _append_discarded_steering(
        self,
        outcome: RunOutcome,
        discarded: tuple[SteeringInput, ...],
    ) -> RunOutcome:
        if not discarded:
            return outcome
        records = list(outcome.audit_records)
        for item in discarded:
            records.append(
                AuditRecord(
                    run_id=outcome.result.run_id,
                    sequence=len(records) + 1,
                    kind="steering_discarded",
                    occurred_at=self._clock(),
                    payload={
                        "input_id": item.input_id,
                        "content": item.content,
                        "reason": "run_finished",
                    },
                )
            )
        return RunOutcome(
            result=outcome.result,
            delta=outcome.delta,
            audit_records=tuple(records),
            failure=outcome.failure,
        )

    def _dispatch_observers(
        self,
        base: SessionSnapshot,
        outcome: RunOutcome,
    ) -> None:
        if not self._observers:
            return
        audit = SessionCommit(
            agent_id=self._agent_id,
            base=base.conversation,
            outcome=outcome,
        ).audit
        for observer in self._observers:
            task = asyncio.create_task(self._notify_observer(observer, audit))
            self._observer_tasks.add(task)
            task.add_done_callback(self._observer_done)

    def _observer_done(self, task: asyncio.Task[None]) -> None:
        self._observer_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def _flush_observers(self) -> None:
        if self._observer_tasks:
            await asyncio.gather(*tuple(self._observer_tasks), return_exceptions=True)

    @staticmethod
    async def _notify_observer(observer: RunObserver, audit: RunAudit) -> None:
        await observer.observe(audit)

    def _discard_pending_follow_ups(self, reason: str) -> None:
        while self._follow_ups:
            queued = self._follow_ups.popleft()
            queued.handle._fail(FollowUpDiscardedError(reason))
            self._outstanding_follow_ups -= 1

    async def _commit(
        self,
        base: SessionSnapshot,
        outcome: RunOutcome,
    ) -> RunOutcome:
        commit = SessionCommit(
            agent_id=self._agent_id,
            base=base.conversation,
            outcome=outcome,
        )
        if self._store is None:
            self._snapshot = SessionSnapshot(
                agent_id=self._agent_id,
                conversation=commit.resulting_conversation,
                last_result=(
                    outcome.result if commit.advances_revision else base.last_result
                ),
            )
            return outcome
        try:
            snapshot = await self._store.commit(commit)
        except SessionStoreError as exc:
            return self._persistence_failure(outcome, exc)

        self._validate_committed_snapshot(snapshot, commit, base)
        self._snapshot = snapshot
        return outcome

    def _persistence_failure(
        self,
        outcome: RunOutcome,
        error: SessionStoreError,
    ) -> RunOutcome:
        failure = RunFailure(
            phase=RunPhase.COMMIT,
            code=FailureCode.PERSISTENCE_FAILED,
            message=str(error) or type(error).__name__,
            retryable=True,
            cause=error,
        )
        result = RunResult(
            run_id=outcome.result.run_id,
            status=RunStatus.FAILED,
            stop_reason=StopReason.PERSISTENCE_FAILED,
            turns=outcome.result.turns,
            usage=outcome.result.usage,
        )
        audit = (
            *outcome.audit_records,
            AuditRecord(
                run_id=outcome.result.run_id,
                sequence=len(outcome.audit_records) + 1,
                kind="commit_failed",
                occurred_at=self._clock(),
                payload={"error": str(error) or type(error).__name__},
            ),
        )
        return RunOutcome(
            result=result,
            delta=outcome.delta,
            audit_records=audit,
            failure=failure,
        )

    def _validate_loaded_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, SessionSnapshot):
            raise SessionStoreProtocolError(
                "SessionStore.load() must return SessionSnapshot or None"
            )
        if snapshot.agent_id != self._agent_id:
            raise SessionStoreProtocolError(
                f"SessionStore loaded agent {snapshot.agent_id!r} for "
                f"{self._agent_id!r}"
            )

    def _validate_committed_snapshot(
        self,
        snapshot: object,
        commit: SessionCommit,
        base: SessionSnapshot,
    ) -> None:
        if not isinstance(snapshot, SessionSnapshot):
            raise SessionStoreProtocolError(
                "SessionStore.commit() must return SessionSnapshot"
            )
        expected_result = (
            commit.outcome.result if commit.advances_revision else base.last_result
        )
        if (
            snapshot.agent_id != self._agent_id
            or snapshot.revision != commit.resulting_revision
            or snapshot.conversation != commit.resulting_conversation
            or snapshot.last_result != expected_result
        ):
            raise SessionStoreProtocolError(
                "SessionStore.commit() returned a snapshot inconsistent "
                "with the proposed commit"
            )

    async def _rollback_start(self, started: list[ManagedResource]) -> None:
        for resource in reversed(started):
            try:
                await resource.shutdown()
            except BaseException:
                pass

    @staticmethod
    def _managed_resources(
        resources: Iterable[object],
    ) -> tuple[ManagedResource, ...]:
        managed: list[ManagedResource] = []
        seen: set[int] = set()
        for resource in resources:
            identity = id(resource)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(resource, ManagedResource):
                managed.append(resource)
        return tuple(managed)

    @staticmethod
    def _validate_capacity(value: object, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_name} must be an integer")
        if value <= 0:
            raise ValueError(f"{field_name} must be greater than zero")
