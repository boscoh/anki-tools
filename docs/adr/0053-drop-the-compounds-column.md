# Drop the compounds column

Status: accepted. Supersedes the `compounds` part of ADR 0049 (step 2) and the
`compounds` remedy in ADR 0047.

The `compounds` column held a hand-kept list of sentence strings per word, and
five code paths read it: `form_compounds`, `compound_spans`, `inside_spans`,
the jieba compound registration, and the French and Spanish `hits` change. A
list that grows by one row per false compound does not scale, and the sense
test already sees the same faults from the English side. The column and its code
are removed. The sense test (`sense_failure`) is the only guard for a compound
that takes another sense.

## Evidence

Six decks were built with the old code and with the column removed. The core
and cull gates pass on all six (K5, K6, K13, K23, K24; G1, G2, G5, G6).

| Deck           | Cards before | after | `covered1` before | after | K23 share after | Floor |
| -------------- | ------------ | ----- | ----------------- | ----- | --------------- | ----- |
| French core    | 277          | 277   | 212               | 212   | 0.825           | 0.80  |
| Spanish core   | 273          | 273   | 229               | 229   | 0.891           | 0.85  |
| Mandarin core  | 146          | 145   | 120               | 119   | 0.456           | 0.40  |
| Cantonese core | 115          | 115   | 94                | 94    | 0.443           | 0.40  |

The French and Spanish core decks keep the same cards and targets. The French
and Spanish cull decks are byte-identical. Mandarin and Cantonese swap one to
two cards each.

The three known cases failed once the column was gone, and two English-side
data fixes close them:

- Cantonese `蟲` in `昆蟲`: the gloss `worm` does not hold `insects`, so the
  sense test rejects `He hates insects.` (`sense`). No data change.
- French `quelques` in `quelque chose`: `vocab/english_senses.json` listed
  `something`, `someone`, and `somebody` as words of the gloss `some`, so card
  676 ("That is something you should not have said.") credited `S19:quelques`.
  `scripts/build_english_senses.py` no longer lists the three words for `some`.
  The rebuilt JSON changes one key, `some`, and drops these three words. With the
  fix the French core deck credits no `quelque chose` card to `quelques`.
- Mandarin `得到` in `听得到`: the tier F row had `senses` = `can`, so
  `Can you hear me?` passed. The row now lists `get`, `obtain`, `receive`,
  `gain`, and their inflections. The sense test rejects `Can you hear me?`. The
  input has only five cards with `得到`, all of the form `听得到` and `办得到`, so
  no card for `得到` is kept. Before the change no card credited it either.

Unit tests pin the three cases through `sense_failure`
(`tests/test_english_sense.py`).

## Consequences

- A compound with another sense is now a fault of the English-side data: fix it
  with a `senses` or `false_friends` entry or a change in
  `scripts/build_english_senses.py`, not with a sentence string.
- The target hit is the plain occurrence of the form. Mandarin `来` inside
  `出来`, and a Cantonese `聽` inside `聽日`, are hits. The sense test stands
  alone to reject them. This is the cost of the change.
- The Mandarin core deck was stale in `apkg/` before this change and is rebuilt
  with it.
