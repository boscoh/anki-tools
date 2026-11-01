# Retire the unchecked, tail, and unranked gates

Core gates K16 `gate_unchecked_sense`, K18 `gate_tail`, and K20
`gate_unranked` are retired.

Status: accepted.

## Decision

The core no longer has these gates. The code removes the three functions, the
three `Gate` entries, and the manifest params `max_unchecked_sense` and
`max_unranked`. The ids are not reused (ADR 0027).

The core `--check` runs 5 gates: K2, K5, K6, K11, and K13.

The manifest still holds the counts `unchecked` and `unranked`. The build
report still shows `Unchecked`. A reader can see them. No gate limits them.

## Reason

- K16 and K20 compared a count with a limit of 0. The build wrote the limit to
  the manifest, so the limit was a constant of the producer and not a rule of
  the reviewer.
- K16 and K20 test the input data (a sense row, a frequency rank), not the
  analysis of the output deck.
- K18 read `new_lemmas` from the CSV. The build wrote that value. The gate did
  not count the lemmas again, so it tested the writer (ADR 0043).

## Consequences

- A tier F target with no sense row, or with no frequency rank, no longer fails
  `--check`. The count shows in the manifest.
- A tail card with no new content lemma no longer fails `--check`.
- Old manifests in `apkg/` keep the two params. No code reads them.
- The specs `specs/core-word-small-deck.md` and
  `specs/core-deck-all-languages.md` still describe the three gates. They are
  design records, and this ADR replaces them.
