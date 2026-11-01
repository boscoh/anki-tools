# Retire the identity, kept-order, and template gates

Cull gates G10, G11, and G15 and core gates K9, K10, and K14 shared three
functions: `gate_identity`, `gate_kept_order`, and `gate_templates`.

Status: accepted.

## Decision

The cull and the core no longer have these gates. The code removes the three
shared functions, the two wrappers `gate_order` and `gate_card_templates`, the
six `Gate` entries, `CARD_CLASSES`, and the `rank_rows` reader of `GateFiles`.
The ids are not reused (ADR 0027).

The cull `--check` runs 10 gates: G1, G2, G5, G6, G8, G9, G12, G13, G16, G17.
The core `--check` runs 13 gates: K1, K2, K5, K6, K8, K11 to K13, K16, K18 to
K21.

## Reason

One build step writes both sides of each comparison:

- `assemble.py` writes the deck title to the manifest and to the package.
  G10 and K9 compare the two copies.
- The flow writes the kept rows, the CSV, and the `kept_order` hash from one
  list. G11 and K10 compare them again.
- `deck_templates` writes the templates, and G15 and K14 compare the package
  with `deck_templates`.

A fault in one of these steps shows in the unit tests of the build. The gates
find no fault that a test does not find. G16 and K21 still fail on a wrong note
type, wrong fields, or a second card template.

## Consequences

- The manifest still holds `kept_order`. No gate reads it.
- The build still writes `rank.csv`. No gate reads it.
- A wrong front template or wrong CSS no longer fails `--check`. The unit tests
  of the deck writer cover them.
- Pack gates G10, G11, and G15 are different checks and stay.
