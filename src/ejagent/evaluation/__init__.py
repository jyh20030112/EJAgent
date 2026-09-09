"""Deterministic, evidence-based task evaluation for the Agent Harness."""

from ejagent.contracts.evaluation import (
    CompletionMode,
    CompletionPolicy,
    EvaluationCriterion,
    EvaluationPlan,
)
from ejagent.evaluation.engine import GoalEvaluator
from ejagent.evaluation.journal import JsonlEvaluationJournal
from ejagent.evaluation.judge import JudgeLimits, JudgeUsage, ModelJudge
from ejagent.evaluation.rules import boolean_field, file_exists, json_fields
from ejagent.evaluation.sources import FileEvidenceSource, ProbeEvidenceSource
from ejagent.evaluation.trajectory import EvaluationMonitor, EvaluationReceipt
from ejagent.evaluation.types import (
    CheckResult,
    EvaluationCost,
    EvaluationProtocolError,
    EvaluationReport,
    EvaluationStatus,
    EvidenceSnapshot,
    EvidenceSource,
    EvidenceUnavailable,
    ItemEvaluation,
    JudgeAttempt,
    VerificationRequest,
    Verifier,
)

__all__ = [
    "CompletionMode",
    "CompletionPolicy",
    "JudgeLimits",
    "JudgeUsage",
    "JudgeAttempt",
    "ModelJudge",
    "CheckResult",
    "EvaluationCost",
    "EvaluationCriterion",
    "EvaluationMonitor",
    "EvaluationPlan",
    "EvaluationProtocolError",
    "EvaluationReceipt",
    "EvaluationReport",
    "EvaluationStatus",
    "EvidenceSnapshot",
    "EvidenceSource",
    "EvidenceUnavailable",
    "FileEvidenceSource",
    "GoalEvaluator",
    "ItemEvaluation",
    "JsonlEvaluationJournal",
    "ProbeEvidenceSource",
    "VerificationRequest",
    "Verifier",
    "boolean_field",
    "file_exists",
    "json_fields",
]
