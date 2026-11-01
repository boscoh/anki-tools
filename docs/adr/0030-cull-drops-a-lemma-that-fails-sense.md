# Drop a lemma that fails sense in the cull

The cull took every content lemma of a card as a target of that card. It never
read the English of the card. A card whose English showed another sense of the
lemma still gave gain, a use, and a `card_targets` entry for that lemma, and
`card_cores` wrote a `Core` word and a `CoreGloss` that no check confirmed.

Status: accepted.

## Decision

`select_cards` takes `senses`, the tier F rows of `read_senses` keyed by lemma.
`run_cull` builds the map once and derives the glosses from it. For each card
the function calls `sense_failure(english, row)` on every content lemma that has
a row, and removes each failed lemma from the lemma set of that card. The
removed lemma adds no gain, no use, no similarity, and no `card_targets` entry
for that card.

A lemma with no row is not checked and counts as a pass. The cull runs no
compound test: its targets are tokenizer lemmas and its languages have no
compounds. `hardest_rank` still reads every content lemma, so the card order
does not change.

## Consequences

The sense rows hold 300 target lemmas per language, so the test runs only on
the targets. The card counts before and after, from a rebuild of both decks to
`/tmp` with the same input, profile, and sense file:

| Deck           | Passed | Kept before | Kept after | Frequency cards before | Frequency cards after | Dropped, no core before | Dropped, no core after |
| -------------- | ------ | ----------- | ---------- | ---------------------- | --------------------- | ----------------------- | ---------------------- |
| French cull    | 5578   | 925         | 869        | 731                    | 675                   | 75                      | 131                    |
| Spanish cull   | 7892   | 750         | 693        | 576                    | 519                   | 250                     | 307                    |

The "before" run is the working tree before this change. It differs from the
924 and 747 of ADR 0029 because of other open changes.

The frequency cards of the two decks lose 56 and 57 cards. A selected card that
now has no target still takes a slot, and `card_cores` drops it afterwards. The
dropped-no-core count rises by the same amount.

The kept sets of frequency cards:

| Deck           | Kept before | Kept after | In both | Lost | Gained |
| -------------- | ----------- | ---------- | ------- | ---- | ------ |
| French cull    | 731         | 675        | 477     | 254  | 198    |
| Spanish cull   | 576         | 519        | 445     | 131  | 74     |

The target coverage:

| Deck           | Absent before | Absent after | Partial before | Partial after | Covered before | Covered after |
| -------------- | ------------- | ------------ | -------------- | ------------- | -------------- | ------------- |
| French cull    | 1             | 6            | 11             | 34            | 288            | 260           |
| Spanish cull   | 22            | 36           | 102            | 112           | 176            | 152           |

The targets with the most lost uses are French `ne` (50), `plus` (35), `faire`
(32), and `pouvoir` (19), and Spanish `tener` (18), `no` (11), and `hacer` (11).
The sense words of a function word are narrow: `ne` accepts only `not`, so an
English line with `never`, `no`, or `can't` fails. The result shows the cost of
those rows, and a wider row is a data change, not a code change.

The cull gates and the new sense recheck gate are a separate item (S4).

## Rejected

Reject the whole card when one lemma fails. A card may teach a second target
whose sense holds, and the core flow drops only the target.

Run the compound test in the cull. No cull language has compounds.
