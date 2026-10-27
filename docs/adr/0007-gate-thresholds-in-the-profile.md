# Put the gate thresholds in the profile

The cull gates read two coverage shares from module constants of
`ranking/cullgates.py`: `TARGET_COVERAGE_GATE = 0.98` for gate G3 and
`DOUBLE_COVERAGE_GATE = 0.95` for gate G4. The core gate K4 read two ratios
from its own body: `1.3 * targets` for the outside set and `0.1 * outside` for
the rare outside words. The shared flow applied the same four numbers to every
language.

Status: accepted

## Decision

The four numbers become profile fields, and a gate reads them through its file
reader. `CullProfile` holds `target_coverage_gate` and
`double_coverage_gate`; `CoreProfile` holds `outside_set_ratio` and
`outside_rare_ratio`, with the old values as defaults.

A language profile writes a threshold only when the value differs from the
default. The defaults `0.98`, `0.95`, `1.3`, and `0.1` are the shared values.

The gate messages do not change. G3 and G4 format the profile value the same
way as the constant, so the four tables stay byte-identical.

## Consequences

A gate threshold is a property of the language, not of the shared flow. The
next calibration of one value, for example the cull coverage pair of
`anki-ann4`, changes one profile field and no gate body.

This ADR records the mechanism only. It does not judge the values. Each value
needs its own calibration decision.

## Rejected

A gate module per language. The gate bodies stay shared, and only their input
values move.

Keeping the constants and reading them from the flow. The flow would then hold
a value that only the gates read, and the language package would not own it.
