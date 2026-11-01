# Delete core gates K2 and K11 and use constant limits

Status: accepted.

## Decision

- Core `K2` `gate_covered_targets` and core `K11` `gate_coverage_agreement` are
  deleted. The ids are not reused (ADR 0027).
- Cull `G1` keeps the card cap from the manifest params, because the user sets
  it with `--cards`.
- Cull `G2` and `G6` read their limits from the producer constants
  `DEFAULT_MAX_USES` and `SIMILARITY`. The exempt count of `G2` is
  `EXEMPT_TOP`. No gate reads these three values from the manifest params. The
  manifest still writes them.
- Cull `G5` checks only the rows with the source `frequency`. A reserve card
  has its own length window.
- The reader `read_input_deck` strips one pair of wrapping double quotes from
  the English field, and the output note writes the clean value.

The cull `--check` runs 4 gates: G1, G2, G5, and G6. The core `--check` runs 3
gates: K5, K6, and K13.

## Reason

- `K2` almost never fails. A target that a card offers gets a kept card in the
  selection, so the gate tests the selection against itself.
- `K11` compared two records that one build step wrote: the coverage count and
  the use count of the kept rows.
- A gate that reads its limit from the manifest tests the build against the
  value that the build wrote. A wrong build value then passes. A constant gives
  the gate its own limit.
- The cap of `G1` is a user option, so only the manifest knows it.
- Spanish `G5` failed on the reserve card `Tengo barriga cervecera.`. The
  reserve cards come from the Swadesh pack and have their own window.
- 31 of 125 Cantonese cards have English text in double quotes. The quotes are
  noise in the `sense` filter and in the output deck.

## Consequences

- The core gates test the analysis of the cards (K5, K6) and the target form
  (K13). The coverage CSV has no gate.
- A change of `DEFAULT_MAX_USES`, `SIMILARITY`, or `EXEMPT_TOP` moves the
  producer and the gate together.
- `G5` does not test the reserve cards. The pack gates test them.
- A Cantonese rebuild writes English text without the wrapping quotes.
