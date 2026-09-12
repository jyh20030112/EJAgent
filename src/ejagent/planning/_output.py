"""Planner wire schema; capability binding remains in ModelTaskPlanner."""

from typing import Annotated, Literal

from pydantic import Field

from ejagent._structured_output import OutputModel

Text = Annotated[str, Field(min_length=1, max_length=4096, pattern=r"\S")]
Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"\S")]
TaskText = Annotated[str, Field(min_length=1, max_length=16_384, pattern=r"\S")]


class CriterionOutput(OutputModel):
    id: Identifier
    capability: Identifier
    description: Text


class StepOutput(OutputModel):
    id: Identifier
    description: Text
    requirement_ids: Annotated[list[Identifier], Field(min_length=1)]
    status: Literal["pending"]


class PlannerOutput(OutputModel):
    task: TaskText
    goal: TaskText
    criteria: Annotated[list[CriterionOutput], Field(min_length=1, max_length=64)]
    steps: Annotated[list[StepOutput], Field(min_length=1, max_length=64)]
    unknowns: Annotated[list[Text], Field(max_length=32)]
    unsupported: Annotated[list[Text], Field(max_length=32)]
