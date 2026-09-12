# Structured Model Output and Recovery

Planner and Judge responses pass through the internal `ejagent._structured_output`
module before business binding. Stable task, plan, and evaluation contracts remain
dataclasses; Pydantic models define the model-facing JSON format.

## One definition for prompts and validation

`planning._output.PlannerOutput` and `evaluation._output.JudgeOutput` inherit
strict configuration with extra fields forbidden. Required fields have no implicit
defaults. Nested models use the same configuration. JSON Schema in each role's
system prompt is generated from that same model, using Pydantic's
[schema generation](https://docs.pydantic.dev/latest/concepts/json_schema/).
Validation uses [strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)
to reject incorrect types rather than silently convert them.

Provider `response_format` remains configurable and defaults to JSON object mode.
Supplying a provider-specific schema does not replace local validation.

## Recovery flow

1. The role checks response completion, size, and its usage budget.
2. `OutputRecovery` unwraps a single complete Markdown JSON fence if present.
3. JSON parsing checks syntax; Pydantic checks the decoded object.
4. Success proceeds to existing capability, plan, or evidence binding.
5. Failure supplies a temporary format steer and diagnostic data before another
   request, while retry, timeout, token, and prompt limits allow it.

Diagnostics include a syntax location or up to eight field errors, bounded to
2,048 characters, plus at most 2,048 characters of the previous response. Error
data goes into a separate `UserMessage`, never into the fixed steer instruction.
The original task or evidence payload is preserved. These recovery messages do
not enter Actor context or committed conversation history.

Valid fenced JSON is accepted immediately. Only a generic no-Markdown reminder
survives for the next request: within a Judge Run, or on the Planner instance.
Raw response excerpts never cross evaluations/preparations. Plain validated JSON
clears the reminder.

## Budgets and failure boundaries

Both roles default to one format retry. Planner attempts share one preparation
timeout/token budget; Judge attempts consume its existing Run budget. Missing
usage prevents unaccounted retries. Exhausted Planner recovery raises
`PlanningError`; Judge returns `unknown` with failed-attempt diagnostics. Valid
negative or unknown judgments do not trigger format retries.

Business binding remains in each role. Planner binding failures stop preparation;
Judge preserves its existing bounded retry behavior for invalid evidence or
criterion references. This change adds no heuristic JSON repair, environment
investigation, or automatic adjustment of limits after truncation.
