from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import streamlit as st
from streamlit_runtime import (
    COMPLETION_DEMO_TASK,
    TRAJECTORY_DEMO_TASK,
    DemoJudgeModel,
    DemoValidationModel,
    ProbeExecution,
    RuntimeConfig,
    RuntimeSnapshot,
    StreamlitRuntimeController,
)

from ejagent.contracts import (
    AssistantMessage,
    ModelPort,
    RunAudit,
    RunFailure,
    RunResult,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
    thaw_json_value,
)
from ejagent.harness import HarnessStatus
from ejagent.providers import ModelConfig, OpenAIModelPort

_CONTROLLER_KEY = "ejagent_runtime_controller"
_NOTICE_KEY = "ejagent_runtime_notice"
_DEMO_MODE = "Demo validation (no API key)"
_PROVIDER_MODE = "OpenAI-compatible provider"


def _controller() -> StreamlitRuntimeController | None:
    value = st.session_state.get(_CONTROLLER_KEY)
    return value if isinstance(value, StreamlitRuntimeController) else None


def _start_controller(config: RuntimeConfig, mode: str) -> None:
    existing = _controller()
    if existing is not None:
        existing.close()
    model_factory: Callable[[], ModelPort]
    if mode == _DEMO_MODE:
        model_factory = DemoValidationModel
    else:

        def model_factory() -> ModelPort:
            return OpenAIModelPort(ModelConfig.from_env())

    st.session_state[_CONTROLLER_KEY] = StreamlitRuntimeController(
        config,
        model_factory=model_factory,
        planner_factory=model_factory if config.dynamic_planning else None,
        judge_factory=(DemoJudgeModel if mode == _DEMO_MODE else model_factory)
        if config.semantic_review
        else None,
    )
    st.session_state[_NOTICE_KEY] = (
        f"Runtime started for {config.agent_id!r}; revision restored from JSONL."
    )


def _stop_controller() -> None:
    controller = _controller()
    if controller is not None:
        controller.close()
    st.session_state.pop(_CONTROLLER_KEY, None)
    st.session_state[_NOTICE_KEY] = "Runtime stopped and resources released."


def _render_sidebar() -> None:
    controller = _controller()
    active = controller is not None and not controller.closed
    with st.sidebar:
        st.header("Runtime configuration")
        mode = st.selectbox(
            "Model mode",
            (_DEMO_MODE, _PROVIDER_MODE),
            disabled=active,
        )
        agent_id = st.text_input(
            "Agent ID",
            value="streamlit-validation",
            disabled=active,
        )
        store_root = st.text_input(
            "JSONL directory",
            value=".ejagent-sessions",
            disabled=active,
        )
        max_turns = st.number_input(
            "Maximum turns",
            min_value=1,
            max_value=100,
            value=20,
            disabled=active,
        )
        token_budget_enabled = st.checkbox("Limit tokens", disabled=active)
        max_tokens = st.number_input(
            "Maximum tokens",
            min_value=1,
            value=10_000,
            disabled=active or not token_budget_enabled,
        )
        probe_delay = st.slider(
            "Probe delay (seconds)",
            min_value=0.25,
            max_value=5.0,
            value=1.5,
            step=0.25,
            disabled=active,
        )
        trajectory_enabled = st.checkbox(
            "Trajectory feedback", value=True, disabled=active
        )
        st.caption(
            "Fixed planning evaluates probe A, probe B, and their overlap for each Run. "
            "Other chat goals are outside this evaluation."
        )

        dynamic_planning = st.checkbox(
            "Dynamic task planning",
            value=False,
            disabled=active or not trajectory_enabled or mode == _DEMO_MODE,
            help="Use a separate model request to derive goals, acceptance criteria and an execution plan from your query. Only registered probe capabilities are available.",
        )
        semantic_review = st.checkbox(
            "Semantic completion review",
            value=False,
            help="Fixed plans include summary review. Dynamic planning exposes it as an optional capability, selected when applicable to the query.",
            disabled=active or not trajectory_enabled,
        )
        completion_enforced = st.checkbox(
            "Require completion approval",
            value=False,
            disabled=active or not trajectory_enabled,
        )
        completion_max_retries = st.number_input(
            "Completion retries",
            min_value=0,
            value=2,
            disabled=active or not completion_enforced or not trajectory_enabled,
        )
        judge_max_requests = st.number_input(
            "Judge request budget per Run",
            min_value=1,
            value=8,
            disabled=active or not semantic_review or not trajectory_enabled,
        )
        judge_max_tokens = st.number_input(
            "Judge token budget per Run",
            min_value=1,
            value=16_384,
            disabled=active or not semantic_review or not trajectory_enabled,
        )
        st.caption(
            "Semantic review uses a separate judge request and budget. Demo mode uses a deterministic stand-in. Rejected completions retry within this Run when approval is required."
        )

        if mode == _PROVIDER_MODE:
            configured_model = os.getenv("CHAT_MODEL") or "not configured"
            st.caption(f"CHAT_MODEL: `{configured_model}`")
            st.caption("Credentials are read from `.env`; secrets are never shown.")

        start_col, stop_col = st.columns(2)
        if start_col.button("Start", type="primary", disabled=active):
            try:
                _start_controller(
                    RuntimeConfig(
                        agent_id=agent_id,
                        store_root=Path(store_root),
                        max_turns=int(max_turns),
                        max_tokens=int(max_tokens) if token_budget_enabled else None,
                        probe_delay_seconds=float(probe_delay),
                        trajectory_enabled=trajectory_enabled,
                        dynamic_planning=bool(
                            dynamic_planning
                            and trajectory_enabled
                            and mode == _PROVIDER_MODE
                        ),
                        semantic_review=bool(semantic_review and trajectory_enabled),
                        completion_enforced=bool(
                            completion_enforced and trajectory_enabled
                        ),
                        completion_max_retries=int(completion_max_retries),
                        judge_max_requests=int(judge_max_requests),
                        judge_max_tokens=int(judge_max_tokens),
                    ),
                    mode,
                )
            except Exception as exc:
                st.error(f"Unable to start runtime: {type(exc).__name__}: {exc}")
            else:
                st.rerun()
        if stop_col.button("Stop", disabled=not active):
            _stop_controller()
            st.rerun()

        st.divider()
        st.caption(
            "Changing the Agent ID starts a new logical session. Reusing it restores "
            "the committed Conversation and Audit from JSONL."
        )


def _render_message(message: object) -> None:
    if isinstance(message, SystemMessage):
        with st.expander("System instruction", expanded=False):
            st.write(message.content)
    elif isinstance(message, UserMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AssistantMessage):
        with st.chat_message("assistant"):
            if message.content:
                st.write(message.content)
            if message.tool_calls:
                st.caption("Tool calls")
                st.json(
                    [
                        {
                            "id": call.id,
                            "name": call.name,
                            "arguments": thaw_json_value(call.arguments),
                        }
                        for call in message.tool_calls
                    ]
                )
    elif isinstance(message, ToolResultMessage):
        with st.chat_message("assistant", avatar="🛠️"):
            state = "error" if message.is_error else "completed"
            st.caption(f"{message.tool_name} · {state}")
            st.json(thaw_json_value(message.result))


def _render_chat(
    controller: StreamlitRuntimeController,
    snapshot: RuntimeSnapshot,
) -> None:
    for message in snapshot.messages:
        _render_message(message)

    if snapshot.status is HarnessStatus.RUNNING:
        st.info("A Run is active. Use Steering, Follow-up, or Cancel below.")
    prompt = st.chat_input(
        "Give the agent a task",
        disabled=snapshot.status is HarnessStatus.RUNNING,
    )
    if prompt:
        try:
            controller.start_run(prompt)
        except Exception as exc:
            st.error(f"Run submission failed: {type(exc).__name__}: {exc}")
        else:
            st.rerun()


def _render_controls(
    controller: StreamlitRuntimeController,
    snapshot: RuntimeSnapshot,
) -> None:
    st.subheader("Live controls")
    if snapshot.status is not HarnessStatus.RUNNING:
        st.caption("Start a Run to enable Cancel, Steering, and Follow-up.")
    if st.button(
        "Run parallel validation",
        type="primary",
        disabled=snapshot.status is HarnessStatus.RUNNING,
    ):
        controller.start_run(
            "Call parallel_probe_a and parallel_probe_b together, then report "
            "whether they overlapped."
        )
        st.rerun()

    if st.button(
        "Run trajectory recovery",
        disabled=(
            snapshot.status is HarnessStatus.RUNNING
            or not controller.config.trajectory_enabled
        ),
    ):
        controller.start_run(TRAJECTORY_DEMO_TASK)
        st.rerun()
    st.caption(
        "Recovery demo: repeat sequential probes, then use confirmed-cycle feedback "
        "to switch to a parallel batch. Demo mode takes 8 turns; allow at least 160 "
        "demo tokens. Provider behavior and token usage can vary."
    )

    if st.button(
        "Run completion recovery",
        disabled=snapshot.status is HarnessStatus.RUNNING
        or not controller.config.completion_enforced,
    ):
        controller.start_run(COMPLETION_DEMO_TASK)
        st.rerun()
    st.caption(
        "Completion recovery first proposes an unsupported claim, then uses audit feedback to run both probes. Requires completion approval; demo mode takes 3 turns."
    )

    if st.button(
        "Cancel active Run",
        disabled=snapshot.status is not HarnessStatus.RUNNING,
    ):
        accepted = controller.cancel()
        st.session_state[_NOTICE_KEY] = (
            "Cancellation requested." if accepted else "No cancellable Run was active."
        )
        st.rerun()

    with st.form("steering_form", clear_on_submit=True):
        steering = st.text_input("Steering instruction")
        submitted = st.form_submit_button(
            "Send steering",
            disabled=snapshot.status is not HarnessStatus.RUNNING,
        )
        if submitted:
            try:
                receipt = controller.steer(steering)
            except Exception as exc:
                st.error(f"Steering failed: {type(exc).__name__}: {exc}")
            else:
                st.session_state[_NOTICE_KEY] = (
                    f"Steering {receipt.status.value}: {receipt.input_id}"
                )

    with st.form("follow_up_form", clear_on_submit=True):
        follow_up = st.text_input("Follow-up task")
        submitted = st.form_submit_button(
            "Queue follow-up",
            disabled=snapshot.status is not HarnessStatus.RUNNING,
        )
        if submitted:
            try:
                receipt = controller.follow_up(follow_up)
            except Exception as exc:
                st.error(f"Follow-up failed: {type(exc).__name__}: {exc}")
            else:
                st.session_state[_NOTICE_KEY] = (
                    f"Follow-up {receipt.status.value}: {receipt.input_id}"
                )

    if snapshot.controls:
        st.caption("Recent admission receipts")
        st.dataframe(
            [
                {
                    "input_id": receipt.input_id,
                    "kind": receipt.kind.value,
                    "status": receipt.status.value,
                }
                for receipt in reversed(snapshot.controls)
            ],
            hide_index=True,
            use_container_width=True,
        )


def _probe_row(probe: ProbeExecution) -> dict[str, object]:
    return {
        "call_id": probe.call_id,
        "tool": probe.tool_name,
        "started_at": probe.started_at.isoformat(timespec="milliseconds"),
        "finished_at": probe.finished_at.isoformat(timespec="milliseconds")
        if probe.finished_at is not None
        else "running",
        "elapsed_seconds": round(probe.elapsed_seconds, 3)
        if probe.elapsed_seconds is not None
        else None,
        "cancelled": probe.cancelled,
    }


def _render_probes(probes: tuple[ProbeExecution, ...]) -> None:
    if not probes:
        st.info("Run the parallel validation to capture tool timing.")
        return
    st.dataframe(
        [_probe_row(probe) for probe in reversed(probes)],
        hide_index=True,
        use_container_width=True,
    )
    completed = [
        probe
        for probe in probes
        if probe.finished_at is not None and not probe.cancelled
    ]
    if len(completed) < 2:
        return
    first, second = completed[-2:]
    assert first.finished_at is not None
    assert second.finished_at is not None
    latest_start = max(first.started_at, second.started_at)
    earliest_finish = min(first.finished_at, second.finished_at)
    wall_time = max(first.finished_at, second.finished_at) - min(
        first.started_at,
        second.started_at,
    )
    if latest_start < earliest_finish:
        st.success(
            f"The latest tool pair overlapped. Batch wall time: "
            f"{wall_time.total_seconds():.3f}s."
        )
    else:
        st.error("The latest tool pair did not overlap.")


def _result_data(
    result: RunResult,
    failure: RunFailure | None,
) -> dict[str, object]:
    return {
        "run_id": result.run_id,
        "status": result.status.value,
        "stop_reason": result.stop_reason.value,
        "turns": result.turns,
        "output": result.output,
        "usage": result.usage.to_dict(),
        "failure": {
            "phase": failure.phase.value,
            "code": failure.code.value,
            "message": failure.message,
            "retryable": failure.retryable,
        }
        if failure is not None
        else None,
    }


def _audit_records(audit: RunAudit) -> list[dict[str, object]]:
    return [
        {
            "sequence": record.sequence,
            "kind": record.kind,
            "occurred_at": record.occurred_at.isoformat(timespec="milliseconds"),
            "payload": thaw_json_value(record.payload),
        }
        for record in audit.records
    ]


def _render_audits(audits: tuple[RunAudit, ...]) -> None:
    if not audits:
        st.info("No durable Runs have been recorded for this Agent ID.")
        return
    for audit in reversed(audits):
        label = (
            f"{audit.result.status.value} · {audit.run_id} · "
            f"revision {audit.base_revision} → {audit.resulting_revision}"
        )
        with st.expander(label):
            st.json(_result_data(audit.result, audit.failure))
            st.dataframe(
                _audit_records(audit),
                hide_index=True,
                use_container_width=True,
            )


def _render_trajectory(
    controller: StreamlitRuntimeController, snapshot: RuntimeSnapshot
) -> None:
    st.subheader("Probe trajectory")
    if not controller.config.trajectory_enabled:
        st.info("Trajectory feedback is disabled. Stop the runtime to enable it.")
        return
    st.caption(
        "Coverage measures the current Run's dynamically selected acceptance criteria."
        if controller.config.dynamic_planning
        else "Coverage measures three probe requirements: A completes, B completes, "
        "and a completed pair overlaps, plus optional final-answer review. "
        "Details below cover the latest Run in this runtime; Audit retains receipts."
    )
    st.caption(
        "Completion policy: "
        + (
            "approval required; bounded retries in the same Run"
            if controller.config.completion_enforced
            else "observe only"
        )
    )
    if controller.config.dynamic_planning:
        st.caption(
            "Dynamic planning selects probe acceptance conditions from the current query. Planner costs are recorded separately in Audit."
        )
        if snapshot.task_definition is not None:
            with st.expander("Task definition and acceptance conditions"):
                st.json(thaw_json_value(snapshot.task_definition.to_dict()))
        if snapshot.execution_plan is not None:
            with st.expander("Current execution plan", expanded=True):
                st.json(thaw_json_value(snapshot.execution_plan.to_dict()))
    updates = snapshot.trajectory_updates
    if not updates:
        st.info(
            "Run parallel validation or trajectory recovery to collect checkpoints."
        )
        return
    latest = updates[-1]
    if (
        snapshot.evaluation_reports
        and snapshot.evaluation_reports[-1].checkpoint_id != latest.checkpoint_id
    ):
        st.info(
            "The latest evaluation was interrupted. Coverage below belongs to the previous accepted checkpoint; evaluation details retain the interrupted work and costs."
        )
    progress = latest.assessment.progress[-1]
    st.caption(f"Run: {latest.signal.run_id}")
    columns = st.columns(3)
    columns[0].metric(
        "Requirement coverage", f"{progress.current_requirement_coverage:.0%}"
    )
    columns[1].metric("Best coverage", f"{progress.best_requirement_coverage:.0%}")
    columns[2].metric("Assessment", latest.verdict)
    st.dataframe(
        [
            {
                "Checkpoint": item.checkpoint_id,
                "Turn": item.signal.turn,
                "Trigger": item.signal.trigger.value,
                "Coverage": item.assessment.progress[-1].current_requirement_coverage,
                "Progress": item.assessment.progress[-1].status.value,
                "Assessment": item.verdict,
                "Event": item.context_event.kind.value,
                "Completion allowed (advice)": item.completion_allowed,
                "Actions": item.signal.cumulative_cost.actor_actions,
            }
            for item in updates
        ],
        hide_index=True,
        width="stretch",
    )
    st.write("Current verified requirements")
    st.json(dict(latest.checkpoint.requirements))
    reports = snapshot.evaluation_reports
    if reports:
        report = reports[-1]
        st.subheader("Evaluation details")
        st.dataframe(
            [
                {
                    "Criterion": item.criterion_id,
                    "Method": item.method,
                    "Status": item.status.value,
                    "Reason": item.rationale,
                    "Evidence references": list(item.evidence_refs),
                    "Evidence versions": dict(item.evidence_versions),
                    "Missing evidence": list(item.missing_evidence),
                }
                for item in (*report.requirements, *report.constraints)
            ],
            hide_index=True,
            width="stretch",
        )
        columns = st.columns(4)
        columns[0].metric(
            "Actor model requests", latest.signal.cumulative_cost.model_requests
        )
        columns[1].metric(
            "Judge model requests", sum(item.cost.model_requests for item in reports)
        )
        columns[2].metric(
            "Judge reported tokens",
            sum(
                item.cost.model_input_tokens + item.cost.model_output_tokens
                for item in reports
            ),
        )
        columns[3].metric(
            "Judge unreported requests",
            sum(item.cost.model_unreported_requests for item in reports),
        )
        st.dataframe(
            [
                {
                    "Checkpoint": item.checkpoint_id,
                    "Source reads": item.cost.source_reads,
                    "Revision checks": item.cost.revision_checks,
                    "Verifier calls": item.cost.verifier_calls,
                    "Cache hits": item.cost.cache_hits,
                    "Judge requests": item.cost.model_requests,
                    "Evaluation ms": item.cost.elapsed_ms,
                }
                for item in reports
            ],
            hide_index=True,
            width="stretch",
        )
        with st.expander("Evidence and diagnostics"):
            attempts = [
                {
                    "Checkpoint": item.checkpoint_id,
                    "Criterion": attempt.criterion_id,
                    "Request": attempt.request_index,
                    "Format retry": attempt.retry_index,
                    "Format steer": attempt.format_steered,
                    "Outcome": attempt.outcome,
                    "Finish reason": attempt.finish_reason,
                    "Detail": attempt.detail,
                }
                for item in reports
                for attempt in item.judge_attempts
            ]
            if attempts:
                st.dataframe(attempts, hide_index=True, width="stretch")
            st.json(
                {
                    "report_ref": report.report_ref,
                    "evidence": {
                        key: {
                            "reference": report.evidence_ref(key),
                            "source": evidence.location,
                            "revision": evidence.revision,
                            "value": thaw_json_value(evidence.value),
                        }
                        for key, evidence in report.evidence.items()
                    },
                    "diagnostics": dict(report.diagnostics),
                    "invalidated_refs": report.invalidated_refs,
                }
            )
        st.caption(
            "Full evaluation reports and observed evidence are retained under the session folder's evaluations directory. Report references also appear in Audit."
        )
    st.subheader("Trajectory instructions in model context")
    st.caption(
        "These are the actual instructions included in built model contexts. "
        "Suspected cycles are withheld. When completion approval is required, "
        "failed audits feed the next model call in the same Run."
    )
    for delivery in snapshot.trajectory_contexts:
        with st.expander(f"Turn {delivery.turn} · {delivery.instruction.source}"):
            st.json(json.loads(delivery.instruction.content))


def _render_runtime(snapshot: RuntimeSnapshot) -> None:
    columns = st.columns(4)
    columns[0].metric("Status", snapshot.status.value)
    columns[1].metric("Revision", snapshot.revision)
    columns[2].metric("Messages", len(snapshot.messages))
    columns[3].metric("Follow-ups", snapshot.pending_follow_ups)
    if snapshot.last_error:
        st.error(snapshot.last_error)
    if snapshot.latest_outcome is not None:
        st.subheader("Latest attempted Run")
        st.json(
            _result_data(
                snapshot.latest_outcome.result,
                snapshot.latest_outcome.failure,
            )
        )
    elif snapshot.last_result is not None:
        st.subheader("Last restored committed Run")
        result = snapshot.last_result
        st.json(
            {
                "run_id": result.run_id,
                "status": result.status.value,
                "stop_reason": result.stop_reason.value,
                "turns": result.turns,
                "output": result.output,
                "usage": result.usage.to_dict(),
            }
        )


@st.fragment(run_every=0.5)
def _runtime_fragment() -> None:
    controller = _controller()
    if controller is None:
        st.info("Start the runtime from the sidebar. Demo mode needs no API key.")
        return
    try:
        snapshot = controller.snapshot()
    except Exception as exc:
        st.error(f"Runtime snapshot failed: {type(exc).__name__}: {exc}")
        return

    notice = st.session_state.pop(_NOTICE_KEY, None)
    if notice:
        st.toast(notice)

    status_columns = st.columns(3)
    status_columns[0].metric("Harness", snapshot.status.value)
    status_columns[1].metric("Revision", snapshot.revision)
    status_columns[2].metric("Durable Runs", len(snapshot.audits))

    chat_tab, controls_tab, probes_tab, trajectory_tab, runtime_tab, audit_tab = (
        st.tabs(
            ("Chat", "Controls", "Parallel tools", "Trajectory", "Runtime", "Audit")
        )
    )
    with chat_tab:
        _render_chat(controller, snapshot)
    with controls_tab:
        _render_controls(controller, snapshot)
    with probes_tab:
        _render_probes(snapshot.probes)
    with trajectory_tab:
        _render_trajectory(controller, snapshot)
    with runtime_tab:
        _render_runtime(snapshot)
    with audit_tab:
        _render_audits(snapshot.audits)


st.set_page_config(page_title="EJAgent Validation", page_icon="🥚", layout="wide")
st.title("EJAgent Harness Validation")
st.caption(
    "Exercise durable Conversation state, concurrent tools, live controls, "
    "trajectory feedback, Run limits, revisions, and audit records from one page."
)
_render_sidebar()
_runtime_fragment()
