# Gates fail on analysis, not on the writer

A review of the cull and core gates found three gates that tested the writer
and not the analysis, and two gates that could fail on a valid result.

Status: accepted.

## Decision

- Cull `G9` `gate_media` is retired.
- Core `K1` `gate_card_count` is retired.
- Cull `G12` no longer tests the status of a coverage row. It keeps the row
  count and the duplicate-word test.
- Core `K2` `gate_covered_targets` now fails on a target that a card row
  offers and that has a coverage count of 0.
- Cull `G1` `gate_card_cap` now accepts a short deck only when no passed card
  has the reason `max_cards`.

The ids are not reused (ADR 0027).

The cull `--check` runs 5 gates: G1, G2, G5, G6, and G12. The core `--check`
runs 8 gates: K2, K5, K6, K11, K13, K16, K18, and K20.

## Reason

- `G9` compares the audio field of each card with the media list of the same
  package. One writer makes both sides (ADR 0041).
- `K1` compares three counts that come from one list `ordered`.
- `G12` compared the status of a row with `coverage_status(count)`. The writer
  makes the status with the same function.
- Old `K2` counted the rows with `count >= 1` and the rows with a status other
  than `absent` and `absent_sense`. The writer gives every such row
  `count >= 1`, so the gate could not fail. The gate also skipped
  `absent_sense`, the case that its name describes.
- A card row lists only the targets that pass the sense test. A target with
  that evidence and no card is a selection miss. The four core manifests in
  `apkg/` hold no such target, so the new `K2` passes on them.
- Old `G1` accepted a short deck only when the count equalled `counts.passed`.
  `passed` is the count before the `similar` and `use_cap` rejections, so a
  small pool that these rules limit failed the gate. A card with the reason
  `max_cards` was never rejected, so it shows that the selection stopped early.

## Consequences

- A wrong media list in the package no longer fails `--check`. No test checks
  that the media list equals the card audio. This is open work.
- A count that differs between the CSV, the deck, and the manifest no longer
  fails `--check`.
- The shared code is still a limit. `K13`, `G2`, and `G6` use the tokenizer and
  the sense test of the build. A fault in that code passes these gates. An
  independent recount is open work.
- `K18` reads `new_lemmas` from the CSV. It does not count the lemmas again.
  This is open work.
- The limits `max_unchecked_sense` and `max_unranked` stay 0 in the manifest.
  `K16` and `K20` read them from the run that wrote them.
