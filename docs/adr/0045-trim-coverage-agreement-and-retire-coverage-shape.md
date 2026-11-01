# Trim the coverage agreement gate and retire the coverage shape gate

Status: accepted.

## Decision

- Cull `G12` `gate_coverage_shape` is retired. The id is not reused (ADR 0027).
- Core `K11` `gate_coverage_agreement` keeps one test: the coverage count of
  each target equals the use count in the kept cards. The tests of the status
  and the position order are removed.

The cull `--check` runs 4 gates: G1, G2, G5, and G6. The core `--check` runs 5
gates: K2, K5, K6, K11, and K13.

## Reason

- `G12` tested that the coverage CSV has `target_words` rows. `build_targets`
  keeps the first `target_words` candidates, so a short frequency list gives
  fewer rows. That is a fact about the input, not a fault of the analysis. The
  build report shows the target count.
- `G12` tested that the target words are distinct. `build_targets` builds one
  candidate for each lemma, so the test cannot fail.
- The `K11` status test checked that a status is in `STATUSES`. The writer
  sets the status from a fixed list.
- The `K11` position test checked the order of an input list that the writer
  sorts.
- The `K11` count test stays. `result.uses` and `result.credited` are two
  records of the selection, so a fault in the accounting can break it.

## Consequences

- A coverage CSV with fewer rows than `target_words` no longer fails
  `--check`.
- The cull `--check` has no coverage test. G2 is the one count test of the cull
  selection.
