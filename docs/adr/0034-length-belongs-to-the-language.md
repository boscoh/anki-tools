# Length belongs to the language

ADR 0032 kept a shared English word window, `english_bounds = (2, 12)`, next to the
unit window of each language. The stored decks show that the unit window already
bounds the English length: the median unit count and the median English word count
are equal for French, Spanish, Mandarin, and Cantonese. Inside each unit window the
English window removed 1 card for Mandarin, 1 for Cantonese, 32 of 4880 for French,
and 14 of 5983 for Spanish.

Status: accepted. It supersedes the English window of ADR 0032. The rest of ADR 0032
stands.

## Decision

The language module states the sentence length rule alone: `unit_count`,
`min_units`, `max_units`, `fallback_min_units`, `preferred_min_units`, and
`preferred_max_units`. A unit is a word-equivalent: a word, or a Han character.

The rule `english_words` is gone, with `english_bounds` and
`DEFAULT_ENGLISH_BOUNDS`. The pack, the cull, and the core no longer read the length
of the English text. `FactInput` holds `duplicate`, `unknown_words`, `hits`, and
`sense_hits`. `filter_reason` returns the first failed shape rule.

The Mandarin window stays at 4 to 7 Han characters.

The Cantonese window moves from 3 to 12 to 4 to 7 Han characters, the Mandarin
window. Both languages use `unit_count` over Han characters. `PREFERRED_MAX_HAN`
moves from 8 to 7. The input deck of the Cantonese core holds 999 cards. A trial
run of `anki fix yue --core` with each window gives:

| Window  | Cards passed | Targets | Cards kept | Gates failed |
| ------- | ------------ | ------- | ---------- | ------------ |
| 3 to 12 | 889          | 129     | 128        | K6           |
| 3 to 8  | 827          | 124     | 128        | K4, K6       |
| 4 to 8  | 749          | 120     | 120        | K4, K6       |
| 4 to 7  | 655          | 115     | 109        | K6           |

Gate K6 fails on one duplicate card in every run, so it is not an effect of the
window. Gate K4 (outside words) fails at 3 to 8 and 4 to 8 because the target count
falls faster than the outside count. The 4 to 7 window passes K4 and keeps 109
cards, 19 fewer than before.

## Consequences

The pack tests 8 ids, the cull 7, and the core 7 plus C3 to C5.

A pair with a very long or a very short English line now passes when the target
sentence is inside its window. The `english_link`, `sense`, and `unique` rules still
reject a wrong pair or a duplicate.

The Mandarin pack fixture gains one candidate that the English window rejected before.

## Rejected

A ratio test between the unit count and the English word count. No bad pair needs it
yet. Add it when the review of a deck shows a mismatched pair that the other rules
pass.

A shared default window for all languages. The units agree across languages, but the
best bounds do not: Mandarin works at 4 to 7 and French at 4 to 10. A shared default
needs a count of units over the raw input decks first.
