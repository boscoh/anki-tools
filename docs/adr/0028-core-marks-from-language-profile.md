# Take the core marks from the language profile

The cull took its sentence marks from the language profile, through the shape
guard. The core applied the Latin constants of `ranking/cardfilter.py`:
`FINAL_MARKS`, `QUOTE_MARKS`, `DIALOGUE_DASHES`, and `INTERNAL_MARKS`.
`CoreProfile` held no mark fields. The core also held its own copy of the
`length_window` test in `_filter_reason`.

Status: accepted.

## Decision

`CoreProfile` holds `final_marks`, `quote_marks`, and `internal_marks`. The four
core adapters pass the marks of the language pack profile.

`CardShape`, `filter_reason`, `Marks`, `MARK_RULES`, and `LENGTH_RULES` live in
`ranking/cardfilter.py`. `filter_reason` takes any profile with the three mark
sets. The cull and the core call it with their own marks and their own token
counts. `filter_cards` takes a required `card_rule`, so it has no Latin default.
`filter_core_cards` takes `marks`.

The core has no second `length_window` test. Both flows reach `_passes_length`
in `sentence_pack/pool.py` through `check_sentence_shape`.

The Mandarin and Cantonese `_sentence` adapters still convert the final mark to
ASCII. That conversion is a deck style choice, not a filter rule.

## Consequences

The Latin constants are gone. A new language gives its marks once, in its pack
profile, and the three flows use them.

A full-width mark inside a sentence now counts for the Chinese decks. A Cantonese
card with a full-width `！` inside and an ASCII `!` at the end fails
`one_final_mark`.

The rebuild changed one deck. The card counts before and after:

| Deck      | Input cards | Passed before | Passed after | Kept before | Kept after |
| --------- | ----------- | ------------- | ------------ | ----------- | ---------- |
| French    | 5928        | 3338          | 3338         | 273         | 273        |
| Spanish   | 9263        | 4719          | 4719         | 267         | 267        |
| Mandarin  | 5009        | 2811          | 2811         | 147         | 147        |
| Cantonese | 999         | 923           | 916          | 126         | 126        |

Seven Cantonese cards left the passed set with `one_final_mark`. The card cap had
dropped them before (`max_cards`), so the kept set and the kept order hold. Gates
K1 to K21 pass on the four rebuilt decks.

## Rejected

One shared mark set for every language. The marks differ by language, and the pack
profile already holds them.
