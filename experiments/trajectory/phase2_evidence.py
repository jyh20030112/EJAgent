#!/usr/bin/env python3
"""PROTOTYPE: execute the Phase-2 trajectory Context entry gates."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from ejagent._trajectory import (
    EnvironmentFact,
    FactValidity,
    ProgressSnapshot,
    ShadowTrajectoryAnalyzer,
    TrajectoryCheckpoint,
    TrajectoryContextEvent,
    TrajectoryContextEventKind,
    TrajectoryContextFrame,
    TrajectoryContextProjector,
    TrajectoryVerdict,
)
from ejagent.contracts import RunAudit, RunResult, RunStatus, StopReason

NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)
PHASE1_FS001_SHA256 = "3205fa5ee54867120396e35bc4833586307169623334f0091d637d26935db981"


def _audit(run_id: str) -> RunAudit:
    return RunAudit(
        result=RunResult(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            stop_reason=StopReason.TEXT_RESPONSE,
            turns=1,
            output="fixture complete",
        ),
        base_revision=0,
        resulting_revision=1,
        committed=True,
        records=(),
    )


def _fact(
    *,
    domain: str,
    sequence: int,
    state: str,
    fact_id: str | None = None,
    validity: FactValidity = FactValidity.CURRENT,
    invalidated_at: str | None = None,
    reason: str | None = None,
) -> EnvironmentFact:
    return EnvironmentFact(
        fact_id=fact_id or f"{domain}:state:{sequence}",
        subject=f"{domain}/target",
        predicate="current_state",
        value=state,
        scope=(f"{domain}:goal",),
        source=f"{domain}:deterministic-probe",
        observed_at=NOW + timedelta(seconds=sequence),
        checkpoint_id=f"{domain}:cp{sequence}",
        evidence_ref=f"{domain}://probe/current-state",
        freshness="valid until the target generation changes",
        authority=f"{domain} current State only",
        validity=validity,
        invalidated_at_checkpoint=invalidated_at,
        validity_reason=reason,
    )


def _checkpoint(
    *,
    domain: str,
    sequence: int,
    state: str,
    requirements: Mapping[str, bool | None],
    evidence: tuple[str, ...] = (),
    actions: tuple[str, ...] = (),
    causal_batch_id: str | None = None,
    causally_complete: bool = True,
    unattributed: tuple[str, ...] = (),
    exclusion_reason: str | None = None,
    extra_facts: tuple[EnvironmentFact, ...] = (),
) -> TrajectoryCheckpoint:
    return TrajectoryCheckpoint(
        checkpoint_id=f"{domain}:cp{sequence}",
        projection_version=f"{domain}:v1",
        state_fingerprint=f"fp:{domain}:{state}",
        environment_facts={"state": state},
        requirements=requirements,
        constraints={f"{domain}:safety": True},
        new_evidence=evidence,
        actor_action_count=sequence,
        causal_action_signatures=actions,
        causally_complete=causally_complete,
        facts=(_fact(domain=domain, sequence=sequence, state=state), *extra_facts),
        fact_capture_complete=True,
        causal_batch_id=causal_batch_id,
        unattributed_action_ids=unattributed,
        causal_exclusion_reason=exclusion_reason,
    )


def _routing_cycle() -> tuple[TrajectoryCheckpoint, ...]:
    domain = "deployment-routing"
    return (
        _checkpoint(
            domain=domain,
            sequence=0,
            state="blue",
            requirements={"serve-blue": True, "serve-green": False},
        ),
        _checkpoint(
            domain=domain,
            sequence=1,
            state="green",
            requirements={"serve-blue": False, "serve-green": True},
            actions=("route:green",),
            evidence=("green route is healthy but blue requirement regressed",),
        ),
        _checkpoint(
            domain=domain,
            sequence=2,
            state="blue",
            requirements={"serve-blue": True, "serve-green": False},
            actions=("route:blue",),
        ),
        _checkpoint(
            domain=domain,
            sequence=3,
            state="green",
            requirements={"serve-blue": False, "serve-green": True},
            actions=("route:green",),
        ),
        _checkpoint(
            domain=domain,
            sequence=4,
            state="blue",
            requirements={"serve-blue": True, "serve-green": False},
            actions=("route:blue",),
        ),
    )


def _healthy_scenarios() -> dict[str, tuple[TrajectoryCheckpoint, ...]]:
    return {
        "productive_wait": (
            _checkpoint(
                domain="async-job",
                sequence=0,
                state="10-percent",
                requirements={"job-complete": False},
            ),
            _checkpoint(
                domain="async-job",
                sequence=1,
                state="60-percent",
                requirements={"job-complete": False},
                actions=("wait:poll",),
                evidence=("progress increased to 60 percent",),
            ),
            _checkpoint(
                domain="async-job",
                sequence=2,
                state="complete",
                requirements={"job-complete": True},
                actions=("wait:poll",),
                evidence=("job completed",),
            ),
        ),
        "evidence_gaining_exploration": (
            _checkpoint(
                domain="incident-diagnosis",
                sequence=0,
                state="unresolved",
                requirements={"cause-known": False},
            ),
            _checkpoint(
                domain="incident-diagnosis",
                sequence=1,
                state="unresolved",
                requirements={"cause-known": False},
                actions=("inspect:logs",),
                evidence=("network cause excluded",),
            ),
            _checkpoint(
                domain="incident-diagnosis",
                sequence=2,
                state="unresolved",
                requirements={"cause-known": False},
                actions=("inspect:metrics",),
                evidence=("storage boundary isolated",),
            ),
        ),
        "legitimate_retry": (
            _checkpoint(
                domain="rate-limited-api",
                sequence=0,
                state="rate-limited",
                requirements={"request-complete": False},
            ),
            _checkpoint(
                domain="rate-limited-api",
                sequence=1,
                state="rate-limited",
                requirements={"request-complete": False},
                actions=("retry:request",),
                evidence=("retry-after window decreased",),
            ),
            _checkpoint(
                domain="rate-limited-api",
                sequence=2,
                state="complete",
                requirements={"request-complete": True},
                actions=("retry:request",),
                evidence=("request accepted after retry window",),
            ),
        ),
        "regress_then_recover": (
            _checkpoint(
                domain="schema-migration",
                sequence=0,
                state="old-reader",
                requirements={"new-writes": False, "old-reads": True},
            ),
            _checkpoint(
                domain="schema-migration",
                sequence=1,
                state="new-writer",
                requirements={"new-writes": True, "old-reads": False},
                actions=("migrate:writer",),
                evidence=("new write path verified; old reader regressed",),
            ),
            _checkpoint(
                domain="schema-migration",
                sequence=2,
                state="compatible",
                requirements={"new-writes": True, "old-reads": True},
                actions=("migrate:reader",),
                evidence=("compatibility verifier passed",),
            ),
        ),
    }


def _concurrent_scenarios() -> dict[str, tuple[TrajectoryCheckpoint, ...]]:
    domain = "replicated-config"
    baseline = _checkpoint(
        domain=domain,
        sequence=0,
        state="diverged",
        requirements={"replicas-converged": False},
    )
    ambiguous = (
        baseline,
        _checkpoint(
            domain=domain,
            sequence=1,
            state="diverged",
            requirements={"replicas-converged": False},
            actions=("write:primary", "write:replica"),
            causal_batch_id="ambiguous-batch-1",
            causally_complete=False,
            unattributed=("call-primary", "call-replica"),
            exclusion_reason="concurrent completions were not joined to one State Delta",
        ),
    )
    attributed = (
        baseline,
        _checkpoint(
            domain=domain,
            sequence=1,
            state="diverged",
            requirements={"replicas-converged": False},
            actions=("write:primary", "write:replica"),
            causal_batch_id="attributed-batch-1",
        ),
        _checkpoint(
            domain=domain,
            sequence=2,
            state="diverged",
            requirements={"replicas-converged": False},
            actions=("write:primary", "write:replica"),
            causal_batch_id="attributed-batch-2",
        ),
    )
    return {"ambiguous": ambiguous, "attributed": attributed}


def _projection_matrix(
    checkpoint: TrajectoryCheckpoint,
    progress: ProgressSnapshot,
) -> dict[str, object]:
    projector = TrajectoryContextProjector()
    expected_feedback = {
        TrajectoryContextEventKind.FACTS_UPDATED: False,
        TrajectoryContextEventKind.PROGRESS_EVALUATED: False,
        TrajectoryContextEventKind.CYCLE_SUSPECTED: False,
        TrajectoryContextEventKind.CYCLE_CONFIRMED: True,
        TrajectoryContextEventKind.CONSTRAINT_VIOLATED: True,
        TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED: True,
        TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED: True,
    }
    old_fact = _fact(
        domain="deployment-routing",
        sequence=3,
        state="green",
        fact_id="deployment-routing:old-green",
        validity=FactValidity.INVALIDATED,
        invalidated_at=checkpoint.checkpoint_id,
        reason="route generation changed",
    )
    projection_checkpoint = TrajectoryCheckpoint(
        checkpoint_id=checkpoint.checkpoint_id,
        projection_version=checkpoint.projection_version,
        state_fingerprint=checkpoint.state_fingerprint,
        environment_facts=checkpoint.environment_facts,
        requirements=checkpoint.requirements,
        constraints=checkpoint.constraints,
        new_evidence=checkpoint.new_evidence,
        actor_action_count=checkpoint.actor_action_count,
        causal_action_signatures=checkpoint.causal_action_signatures,
        causally_complete=checkpoint.causally_complete,
        facts=(*checkpoint.facts, old_fact),
        fact_capture_complete=True,
    )
    results: dict[str, object] = {}
    for kind, expected in expected_feedback.items():
        arguments: dict[str, tuple[str, ...]] = {}
        event_checkpoint = projection_checkpoint
        event_progress = progress
        if kind is TrajectoryContextEventKind.CYCLE_CONFIRMED:
            arguments["causal_actions"] = ("route:green", "route:blue")
        elif kind is TrajectoryContextEventKind.CONSTRAINT_VIOLATED:
            arguments["affected_items"] = ("deployment-routing:safety",)
            event_checkpoint = replace(
                projection_checkpoint,
                constraints={"deployment-routing:safety": False},
            )
            event_progress = replace(
                progress,
                constraints={"deployment-routing:safety": False},
            )
        elif kind is TrajectoryContextEventKind.EXTERNAL_STATE_CHANGED:
            arguments["invalidated_fact_ids"] = (old_fact.fact_id,)
        elif kind is TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED:
            arguments["missing_evidence"] = ("production routing probe",)
        event = TrajectoryContextEvent(
            event_id=f"phase2:{kind.value}",
            kind=kind,
            evidence_refs=("phase2://evidence",),
            **arguments,
        )
        projected = projector.project(
            TrajectoryContextFrame(
                run_id="deployment-routing",
                turn=8,
                goal="Serve both routing requirements without violating availability.",
                checkpoint=event_checkpoint,
                progress=event_progress,
                event=event,
                current_plan="Toggle the global route.",
                refuted_hypotheses=("One global route can satisfy both requirements",),
            )
        )
        content = projected.instruction.content
        state = json.loads(content)["trajectory_context"]
        results[kind.value] = {
            "state_visible": state["checkpoint"] == checkpoint.checkpoint_id,
            "expected_feedback_visible": expected,
            "feedback_visible": state["feedback"] is not None,
            "fingerprint_hidden": checkpoint.state_fingerprint not in content,
            "completion_continues_same_run": (
                kind is not TrajectoryContextEventKind.COMPLETION_AUDIT_FAILED
                or "Continue this Run" in content
            ),
        }
    return results


def run_phase2_evidence() -> dict[str, object]:
    analyzer = ShadowTrajectoryAnalyzer(max_period=3)
    routing = _routing_cycle()
    routing_report = analyzer.analyze(_audit("deployment-routing"), routing)
    healthy_reports = {
        name: analyzer.analyze(_audit(name), checkpoints)
        for name, checkpoints in _healthy_scenarios().items()
    }
    concurrent_reports = {
        name: analyzer.analyze(_audit(f"concurrent-{name}"), checkpoints)
        for name, checkpoints in _concurrent_scenarios().items()
    }
    stale_checkpoint = TrajectoryCheckpoint(
        checkpoint_id="freshness:cp0",
        projection_version="freshness:v1",
        state_fingerprint="unknown",
        environment_facts={"state": "unknown"},
        requirements={"state-known": False},
        constraints={},
        facts=(
            _fact(
                domain="freshness",
                sequence=0,
                state="unknown",
                validity=FactValidity.UNKNOWN,
                reason="source freshness could not be evaluated",
            ),
        ),
        fact_capture_complete=True,
    )
    stale_report = analyzer.analyze(_audit("freshness"), (stale_checkpoint,))
    projection = _projection_matrix(routing[-1], routing_report.progress[-1])
    gates = {
        "cross_domain_failure_detected": (
            routing_report.verdict is TrajectoryVerdict.NON_PROGRESS_CYCLE
        ),
        "productive_wait_preserved": (
            healthy_reports["productive_wait"].verdict is TrajectoryVerdict.NO_CYCLE
        ),
        "exploration_preserved": (
            healthy_reports["evidence_gaining_exploration"].verdict
            is TrajectoryVerdict.NO_CYCLE
        ),
        "legitimate_retry_preserved": (
            healthy_reports["legitimate_retry"].verdict is TrajectoryVerdict.NO_CYCLE
        ),
        "regress_then_recover_preserved": (
            healthy_reports["regress_then_recover"].verdict
            is TrajectoryVerdict.NO_CYCLE
        ),
        "ambiguous_concurrent_batch_excluded": (
            concurrent_reports["ambiguous"].verdict
            is TrajectoryVerdict.CAUSALLY_AMBIGUOUS
        ),
        "attributed_concurrent_batch_assessed": (
            concurrent_reports["attributed"].verdict
            is TrajectoryVerdict.NON_PROGRESS_CYCLE
        ),
        "unknown_freshness_fails_closed": (
            stale_report.verdict is TrajectoryVerdict.INSUFFICIENT_EVIDENCE
        ),
        "event_context_matrix_matches_policy": all(
            bool(item["state_visible"])
            and bool(item["feedback_visible"])
            is bool(item["expected_feedback_visible"])
            and bool(item["fingerprint_hidden"])
            and bool(item["completion_continues_same_run"])
            for item in projection.values()
            if isinstance(item, Mapping)
        ),
    }
    return {
        "experiment": "trajectory-phase2-entry-evidence-v2",
        "phase1_fs001_artifact_sha256": PHASE1_FS001_SHA256,
        "domains_observed": [
            "authentication-policy",
            "deployment-routing",
            "async-job",
            "incident-diagnosis",
            "rate-limited-api",
            "schema-migration",
            "replicated-config",
        ],
        "routing_cycle": {
            "verdict": routing_report.verdict.value,
            "period": routing_report.period,
            "candidate_checkpoint_ids": list(routing_report.candidate_checkpoint_ids),
            "diagnostics": list(routing_report.diagnostics),
        },
        "healthy_controls": {
            name: {
                "verdict": report.verdict.value,
                "diagnostics": list(report.diagnostics),
            }
            for name, report in healthy_reports.items()
        },
        "concurrent_batches": {
            name: {
                "verdict": report.verdict.value,
                "diagnostics": list(report.diagnostics),
            }
            for name, report in concurrent_reports.items()
        },
        "freshness_control": {
            "verdict": stale_report.verdict.value,
            "diagnostics": list(stale_report.diagnostics),
        },
        "event_context_matrix": projection,
        "completion_audit_policy": "continue_current_run",
        "gates": gates,
        "all_gates_passed": all(gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    report = run_phase2_evidence()
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
