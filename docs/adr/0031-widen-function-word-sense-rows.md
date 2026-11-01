# Widen the sense rows of function words

ADR 0030 made the cull drop a lemma that fails `sense_failure`. The hand-written
rows of the function words were narrow, so the test dropped real uses: `ne`
accepted only `not`, and an English line with `never`, `no`, `nothing`, or
`only` failed.

Status: accepted.

## Decision

The `senses` words of 14 French and 20 Spanish rows are wider. Each added word is
an inflection or a common English paraphrase of the same lemma sense. No code
changed. The rows keep their first sense word, so the card gloss and gate G17
do not change. `false_friends` did not change.

The rows, by kind of addition:

| Kind                         | French rows                                | Spanish rows                          |
| ---------------------------- | ------------------------------------------ | ------------------------------------- |
| Negation words               | `ne`, `pas`, `personne`                    | `no`                                  |
| Comparative and superlative  | `plus`, `bien`                             | `más`, `mucho`                        |
| Light verb paraphrase        | `faire`, `prendre`, `arriver`              | `hacer`, `tener`, `poner`, `dar`      |
| Modal and obligation words   | `pouvoir`, `vouloir`, `devoir`, `savoir`, `falloir`, `tout` | `querer`, `gustar`, `costar` |
| Time and inflection forms    | none                                       | `vez`, `tiempo`, `tanto`, `volver`, `dejar`, `llevar`, `ver`, `saber`, `solo`, `padre` |

Examples: French `ne` gains `never`, `no`, `nothing`, `nobody`, `only` (`ne ...
que`), and `cannot`. `plus` gains `than`, `again`, `anymore`, and the common `-er`
and `-est` forms. `pouvoir` gains `cannot`, `ca` (the token of `can't`), `afford`,
`maybe`, and `allowed`. `vouloir` gains `please` (`veuillez`) and `wo` (the token
of `won't`). Spanish `tener` gains `must`, `need`, `afraid` (`tener miedo`), and
`old` (`tener ... años`). `hacer` gains `ago`, `become`, `hot`, and `cold`.

A drop that stays is not a false positive. These lemmas keep their failures:

| Lemma                  | Why the drop is right                                              |
| ---------------------- | ------------------------------------------------------------------ |
| `taire`                | the lemmatizer reads `tu as` as `taire`; the English has no sense  |
| `voir`                 | the lemmatizer reads `vit` (lives) as `voir`                       |
| `aller`, `rendre`      | `s'en aller` and `se rendre` give another sense of the lemma       |
| `mettre`               | `se mettre à` (start) gives another sense of the lemma             |
| `heure`, `train`       | `neuf heures` and `en train de` have no English word               |
| `ne`, `pas`, `plus`    | the rest have an English line with no shared word, for example `Tom is behind everybody in mathematics` for `pas` |

## Consequences

The counts come from a rebuild of both cull decks to `/tmp` with the same input,
profile, and options as ADR 0030. The failures are the pairs of a passed card
and a row lemma that fail `sense_failure`; there are 12496 French and 8430
Spanish pairs.

| Deck         | Failed pairs before | Failed pairs after | Cards before | Cards after | Frequency cards before | Frequency cards after |
| ------------ | ------------------- | ------------------ | ------------ | ----------- | ---------------------- | --------------------- |
| French cull  | 2823                | 2208               | 869          | 888         | 675                    | 694                   |
| Spanish cull | 1448                | 1114               | 693          | 703         | 519                    | 529                   |

"Before" is the state of ADR 0030. The deck before ADR 0030 held 925 French and
750 Spanish cards.

The failed pairs of the lemmas with the most lost uses:

| Lemma         | Before | After |
| ------------- | ------ | ----- |
| `ne`          | 212    | 62    |
| `plus`        | 180    | 62    |
| `faire`       | 155    | 104   |
| `pas`         | 105    | 46    |
| `pouvoir`     | 66     | 29    |
| `personne`    | 63     | 13    |
| `vouloir`     | 42     | 8     |
| `tener`       | 115    | 67    |
| `hacer`       | 81     | 54    |
| `más`         | 66     | 16    |
| `no`          | 62     | 32    |
| `mucho`       | 47     | 24    |
| `querer`      | 36     | 14    |
| `vez`         | 31     | 6     |

The target coverage after the widening, against ADR 0030:

| Deck         | Absent before | Absent after | Partial before | Partial after | Covered before | Covered after |
| ------------ | ------------- | ------------ | -------------- | ------------- | -------------- | ------------- |
| French cull  | 6             | 5            | 34             | 35            | 260            | 260           |
| Spanish cull | 36            | 34           | 112            | 110           | 152            | 156           |

The gain in cards is small because most failures of a function word are idioms
whose English holds no shared word. A wider row would pass them only by accepting
words of another sense, and the test would stop rejecting.

## Rejected

Add `get` or `be` to a light verb row. These words are in so many English lines
that the test would accept every card of the lemma.
