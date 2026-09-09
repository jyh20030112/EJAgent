"""Optional append-only reports including resolvable, bounded evidence content."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from ejagent.contracts.json import thaw_json_value
from ejagent.evaluation.types import EvaluationReport, ItemEvaluation


class JsonlEvaluationJournal:
    """Synchronous report sink; retain or remove this log under host policy.

    Each line is self-contained. Evidence references resolve within that report,
    including source revisions and the observed content; source paths need not
    remain available. This does not alter the SessionStore schema.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = Lock()

    def __call__(self, report: EvaluationReport) -> None:
        def item_payload(item: ItemEvaluation) -> dict[str, object]:
            return {
                "criterion_id": item.criterion_id,
                "method": item.method,
                "status": item.status.value,
                "rationale": item.rationale,
                "evidence_refs": item.evidence_refs,
                "evidence_versions": dict(item.evidence_versions),
                "missing_evidence": item.missing_evidence,
            }

        payload = {
            "schema": "ejagent.evaluation.v1",
            "report_ref": report.report_ref,
            "run_id": report.run_id,
            "checkpoint_id": report.checkpoint_id,
            "plan": asdict(report.plan) if report.plan else None,
            "status": "evaluated" if report.plan else "not_evaluated",
            "requirements": [item_payload(item) for item in report.requirements],
            "constraints": [item_payload(item) for item in report.constraints],
            "evidence": {
                key: {
                    "evidence_ref": report.evidence_ref(key),
                    "revision": item.revision,
                    "value": thaw_json_value(item.value),
                    "location": item.location,
                    "observed_at": item.observed_at.isoformat(),
                }
                for key, item in report.evidence.items()
            },
            "diagnostics": dict(report.diagnostics),
            "invalidated_refs": report.invalidated_refs,
            "new_evidence": report.new_evidence,
            "fact_capture_complete": report.fact_capture_complete,
            "cost": asdict(report.cost),
            "judge_attempts": [asdict(attempt) for attempt in report.judge_attempts],
        }
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line)
