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
