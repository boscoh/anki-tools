# Use one WordNet sense per gloss; add the core coverage floor K23

Status: accepted.

## Decision

1. `scripts/build_english_senses.py` expands only the first WordNet synset of a
   gloss word (the first synset of the right part of speech for a `to` gloss).
   Before, it expanded every synset and so added the words of rare senses. The
   builder also adds the plain inflections of the gloss word, so the gloss word
   and its forms stay in the list. `vocab/english_senses.json` is rebuilt.
   `EXTRA_WORDS`, `FUNCTION_WORDS`, and `GLOSS_FALLBACKS` are not changed.
2. Core gate `K23` (`gate_coverage_floor`) fails when the share of coverage rows
   with at least one kept card is below `coverage_floor`. An absent target counts
   as not covered. Each language sets the floor in its `CoreProfile`
   (ADR 0007). The floor is the current share rounded down to 0.05. It is a
   ratchet: a drop means the selection or the input got worse. The gate reads
   the coverage CSV and the profile. It does not read the manifest.

## Evidence

Words in the list of one gloss (words plus phrases, 207 glosses):

| Measure          | Before | After |
| ---------------- | ------ | ----- |
| Median           | 58     | 13    |
| p90              | 164    | 33    |
| Max              | 420    | 123   |
| Total forms      | 15483  | 3558  |
| `big` words      | 145    | 33    |
| `fish` words     | 34     | 13    |
| `sun` words      | 69     | 11    |

The loosest lists were `to hold` (420), `to see` (370), `stick` (301), `to give`
(290), and `to blow` (284). After the change they are `to vomit` (123), `to die`
(90), `to live` (65), `child` (59), and `dirty` (59).

`big` no longer accepts `bad`, `adult`, or `great`. `fish` no longer accepts
`angle`. `sun` no longer accepts `insolate`. A test holds these pairs and a
size bound of 125 for each list.

French core, rebuilt from the same input, 50 words:

| Measure                | Before | After |
| ---------------------- | ------ | ----- |
| Cards                  | 275    | 276   |
| Targets                | 216    | 216   |
| `absent_sense` targets | 4      | 6     |
| `covered1`             | 212    | 210   |

Spanish core: the cards (273), the targets (236), and `absent_sense` (7) do not
change. In French, `grand` was a target of the cards "a great poet" and "a great
artist". After the change it is not.

Coverage share (rows of the coverage CSV with a kept card) of the committed
decks, and the floors:

| Language  | Covered | Rows | Share | Floor |
| --------- | ------- | ---- | ----- | ----- |
| French    | 212     | 257  | 0.825 | 0.80  |
| Spanish   | 222     | 257  | 0.864 | 0.85  |
| Mandarin  | 117     | 261  | 0.448 | 0.40  |
| Cantonese | 89      | 212  | 0.420 | 0.40  |

The manifest `targets` count leaves out the absent rows, so it is not the
denominator of `K23`.

## Consequences

- A word that the first synset does not hold is not accepted for the gloss.
  Add it to `EXTRA_WORDS` when a real card needs it.
- A lower share needs a new floor in the language core module, and a reason.
- `K5` still fails on the committed French and Mandarin decks, because they are
  older than the filter rules. A rebuild clears it.
