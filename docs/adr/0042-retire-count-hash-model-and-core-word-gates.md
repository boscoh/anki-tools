# Retire the count, hash, note type, and core word gates

Cull gates G8, G13, G16, and G17 and core gates K8, K12, K21, and K19 shared four
functions: `gate_counts`, `gate_manifest_hashes`, `gate_output_model`, and
`gate_core_word`.

Status: accepted.

## Decision

The cull and the core no longer have these gates. The code removes the four
shared functions, `output_model_problems`, `core_word_problems`, the cull and
core wrappers, the core bold-target helpers, and the eight `Gate` entries. The
ids are not reused (ADR 0027).

The cull `--check` runs 6 gates: G1, G2, G5, G6, G9, and G12. The core
`--check` runs 9 gates: K1, K2, K5, K6, K11, K13, K16, K18, and K20.

## Reason

- One build step writes both sides of the count and hash comparisons. The deck,
  the CSV files, and the manifest come from one run.
- `output_fields` builds the note type, so `gate_output_model` checks the writer
  against itself.
- A fault in these steps shows in the unit tests of the build.

## Consequences

A CSV file that someone edits after the build now passes `--check`. A wrong
gloss, a missing core word, or a missing bold now passes `--check`. The build
still drops a kept card that has no core word (ADR 0036).

Cull G9 `gate_media` still compares the media set with the card audio.
