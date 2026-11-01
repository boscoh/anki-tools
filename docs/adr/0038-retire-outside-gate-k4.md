# Retire the outside gate K4

Core gate K4 failed when the outside set of the core cards was more than 1.3
times the target count, or when the rare outside words were more than 0.1 of the
outside set. The two ratios were chosen numbers, set in `CoreProfile` as
`outside_set_ratio` and `outside_rare_ratio`.

Status: accepted.

## Decision

The core flow no longer has gate K4. The code removes `gate_outside`, its `Gate`
entry, and the two `CoreProfile` fields. The id K4 is not reused. The ids K1, K2,
K5 to K14, K16, and K18 to K22 keep their numbers (ADR 0027).

The manifest counts `outside` and `outside_rare` stay. The core report still
shows the Outside lemmas and Rare outside lines. The flow step K4, the
selection, is a different thing and keeps its name.

## Reason

The outside set comes from the selection algorithm. The cost terms of the
selection already limit it: a new outside word costs `lambda_new`, and a rare
word costs more. A pass value that is a chosen ratio does not need to exist. This
is the same decision as ADR 0033 for G3 and G4, and ADR 0035 for G14.

Nothing in the selection reads the two ratios. Only the gate read them. A change
of the input deck or of the length window moved the outside count, and the gate
then failed with no fault in the cull (ADR 0034).

## Consequences

The core `--check` runs 18 gates: K1, K2, K5 to K14, K16, and K18 to K22. A deck
with a larger outside set now passes. The reader sees the counts in the report.

ADR 0007 names the two ratios. Its status line points to this ADR.

## Rejected

A higher ratio or a ratio for each language. Both only move the chosen number.
