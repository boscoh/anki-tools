# Drop silent cards in the input reader, and retire C3 to C5

The core flow had three rules of its own after the shared filter F1 to F9:

- C3: no name of the name list in the sentence.
- C4: no loan word of the loan word list in the sentence.
- C5: the audio file of the card is in the media map of the input APKG.

Status: accepted.

## Decision

The core flow no longer has the rules C3, C4, and C5. The ids are not reused.

- `filter_core_cards` applies F1 to F9 only. It takes the cards and the shape
  profile.
- `scan` calls `drop_silent_cards` (`ranking/scan.py`) after `load_input_deck`.
  The function keeps the cards whose audio file is in the media map, and it gives
  the kept cards new indices.
- The name list stays. `build_targets` still uses it to keep a name out of the
  target list.
- The loan word list is gone: `read_loanwords`, `CoreProfile.loan_path`,
  `LOAN_PATH`, and the `vocab/*_loanwords.txt` files.
- The build report shows `Dropped, no audio file`. The report lines
  `Rejected names` and `Rejected loan words` are gone, and so is the gate note
  K17 in the docs.

## Reason

C5 is a fact about the input file, not a rule about a sentence. A card with no
audio file is not a card of the deck. The input reader is the place to drop it,
before the filter, the tokens, and the targets see it. Gate K7 checked the
audio of the output, so K5 did not need to check the media map. ADR 0040 retires
K7. The core K8 gate now checks the media set.

The name and loan word lists were hand lists that a person had to keep in step
with the data. The owner decided that the core deck does not need them.

## Consequences

- A card with a name or a loan word can now enter the core deck.
- A silent card has no row in `core.csv`. Before, it had the reason `C5`.
- The cull flow does not change. Its stage ids C0 to C5 are a separate namespace
  (ADR 0027).

## Rejected

Keep C5 as a filter rule. It mixes a file check with the sentence rules and
needs the media map in the filter signature.
