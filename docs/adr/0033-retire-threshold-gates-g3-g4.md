# Retire the threshold gates G3 and G4

Cull gate G3 failed when fewer than 0.98 of the targets had one card. Cull gate
G4 failed when fewer than 0.95 of the targets had two cards. The profile held
both values as `target_coverage_gate` and `double_coverage_gate`.

Status: accepted.

## Decision

The cull no longer has gates G3 and G4. The code removes `gate_coverage_once`,
`gate_coverage_twice`, their `Gate` entries, and the two profile fields.

- The ids G3 and G4 are retired. No later gate reuses them. The ids G5 to G18
  keep their numbers (ADR 0027).
- The coverage CSV and the "Targets with 2" line of the cull report stay. Only
  the pass or fail test goes away.
- G2 (use cap), G14 (Swadesh count), and core K2 stay. Their pass values come
  from a rule or a count, not from a chosen share.

## Reason

No data set the values 0.98 and 0.95. A gate whose pass value is a chosen
threshold does not find a bug. After ADR 0030 the French cull gave 0.867 for
G4. The gate failed, but the deck was not wrong: the sense test removed a
lemma from some cards, and a valid candidate card was cut by `max_cards` or by
the use cap. A failed value showed no fault to fix.

## Consequences

The cull `--check` runs 16 gates: G1, G2, and G5 to G18. A deck with a low
coverage share now passes. The reader sees the share in the report and in the
coverage CSV.

ADR 0007 names the two profile fields. This ADR replaces that part of ADR 0007.

## Rejected

A lower threshold. Any new value is as arbitrary as the old value.

A renumbered table. A renumbered gate breaks the references in the specs, the
issues, and the reports (ADR 0027).
