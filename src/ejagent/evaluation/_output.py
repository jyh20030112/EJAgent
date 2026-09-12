"""Judge wire schema; current criterion and evidence binding stay in ModelJudge."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ejagent._structured_output import OutputModel

Text = Annotated[str, Field(min_length=1, max_length=2048, pattern=r"\S")]


class JudgeOutput(OutputModel):
    criterion_id: Annotated[str, Field(min_length=1, pattern=r"\S")]
    status: Literal["pass", "fail", "unknown", "conflict"]
    rationale: Text
    evidence_refs: list[Text]
    missing_evidence: list[Text]

    @model_validator(mode="after")
    def consistent_verdict(self) -> Self:
        if self.status != "unknown" and not self.evidence_refs:
            raise ValueError("judgment has no evidence")
        if self.status in ("pass", "fail") and self.missing_evidence:
            raise ValueError("known judgment also claims missing evidence")
        return self
