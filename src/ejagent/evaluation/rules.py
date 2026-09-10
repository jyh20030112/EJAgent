"""Small deterministic rules; hosts can register any async read-only verifier."""

from __future__ import annotations

import json
from collections.abc import Mapping

from ejagent.contracts.control import CancellationToken
from ejagent.evaluation.types import (
    CheckResult,
    EvaluationStatus,
    VerificationRequest,
    Verifier,
)


def _result(request: VerificationRequest, passed: bool, rationale: str) -> CheckResult:
    return CheckResult(
        EvaluationStatus.PASS if passed else EvaluationStatus.FAIL,
        rationale,
        request.criterion.evidence_keys,
    )


async def file_exists(
    request: VerificationRequest, cancellation: CancellationToken
) -> CheckResult:
    cancellation.raise_if_cancelled()
    values = [item.value for item in request.evidence.values()]
    if any(
        not isinstance(value, Mapping) or not isinstance(value.get("exists"), bool)
        for value in values
    ):
        return CheckResult(
            EvaluationStatus.UNKNOWN,
            "file existence has no boolean evidence",
            missing_evidence=request.criterion.evidence_keys,
        )
    passed = all(
        isinstance(value, Mapping) and value.get("exists") is True for value in values
    )
    return _result(
        request,
        passed,
        "required artifact exists" if passed else "required artifact is absent",
    )


def json_fields(*fields: str) -> Verifier:
    """Require a JSON object containing the configured top-level fields."""

    async def verify(
        request: VerificationRequest, cancellation: CancellationToken
    ) -> CheckResult:
        cancellation.raise_if_cancelled()
        for item in request.evidence.values():
            value = item.value
            if not isinstance(value, Mapping) or not isinstance(
                value.get("exists"), bool
            ):
                return CheckResult(
                    EvaluationStatus.UNKNOWN,
                    "file existence is unavailable",
                    missing_evidence=request.criterion.evidence_keys,
                )
            if value.get("exists") is False:
                return _result(request, False, "required JSON artifact is absent")
            text = value.get("text")
            if not isinstance(text, str):
                return CheckResult(
                    EvaluationStatus.UNKNOWN,
                    "artifact text is unavailable",
                    missing_evidence=request.criterion.evidence_keys,
                )
            try:
                parsed = json.loads(text)
            except ValueError:
                return _result(request, False, "artifact is not valid JSON")
            if not isinstance(parsed, dict) or any(key not in parsed for key in fields):
                return _result(request, False, "artifact lacks required JSON fields")
        return _result(request, True, "required JSON fields are present")

    return verify


def boolean_field(field: str) -> Verifier:
    """Check one proposition; disagreeing independent sources remain conflict."""

    async def verify(
        request: VerificationRequest, cancellation: CancellationToken
    ) -> CheckResult:
        cancellation.raise_if_cancelled()
        values = [
            item.value.get(field) if isinstance(item.value, Mapping) else None
            for item in request.evidence.values()
        ]
        if any(not isinstance(value, bool) for value in values):
            return CheckResult(
                EvaluationStatus.UNKNOWN,
                f"{field} has no boolean evidence",
                missing_evidence=request.criterion.evidence_keys,
            )
        if any(value != values[0] for value in values):
            return CheckResult(
                EvaluationStatus.CONFLICT,
                f"sources disagree on {field}",
                request.criterion.evidence_keys,
            )
        return _result(request, values[0] is True, f"observed {field}={values[0]}")

    return verify


async def command_succeeded(
    request: VerificationRequest, cancellation: CancellationToken
) -> CheckResult:
    """Verify exit status only; the host command defines the scope of this check."""
    cancellation.raise_if_cancelled()
    values = [item.value for item in request.evidence.values()]
    if not values or any(
        not isinstance(value, Mapping) or type(value.get("exit_code")) is not int
        for value in values
    ):
        return CheckResult(
            EvaluationStatus.UNKNOWN,
            "command exit status unavailable",
            missing_evidence=request.criterion.evidence_keys,
        )
    passed = all(
        isinstance(value, Mapping) and value.get("exit_code") == 0 for value in values
    )
    details = "\n".join(
        str(value.get("stderr", ""))[-2048:]
        for value in values
        if isinstance(value, Mapping)
    )
    return _result(
        request,
        passed,
        "host verification command succeeded"
        if passed
        else "host verification command failed: " + details,
    )
