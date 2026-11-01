# Add two sense exceptions, two compound rows, and a Core word check in K13

Status: accepted. The `compounds` part (step 2, its evidence, and the
consequence about `compounds` entries) is superseded by ADR 0053, which removed
the column. The rows no longer exist.

## Decision

1. `EXTRA_WORDS` in `scripts/build_english_senses.py` gets `thin`: `slim`,
   `slimmer`, `slimmest`, and `flow`: `run`, `runs`, `ran`, `running`.
   `vocab/english_senses.json` is rebuilt with the builder.
2. The `compounds` column got two tier S rows, with no code change (ADR 0053
   removed them):
   Cantonese `蟲` blocks `昆蟲`, and Mandarin `得到` blocks `听得到`. The
   tokenizer and `hits` do not change.
3. Core gate `K13` (`gate_target_in_sentence`) also tests the word in the `Core`
   field of each kept card. The sentence must hold that word. The test is the
   test that K13 uses for a target: `profile.hits` for tier S, the lemma set for
   tier F. The tier comes from the coverage row of the word. A card with an
   empty `Core` field is skipped for this part.

## Evidence

Step 1. ADR 0048 narrowed the English table. Two French targets lost their only
card. The old committed deck and a new build differ in these rows:

| Target         | Old card                                   | Verdict      |
| -------------- | ------------------------------------------ | ------------ |
| `S35:mince`    | "My uncle is slim, but my aunt is fat."    | right sense  |
| `S144:couler`  | "Run hot water into the bath."             | right sense  |

`slim` is a word for thin. `faire couler de l'eau` means that water runs, which
is the sense of `to flow`. The first WordNet synset of `thin` and of `flow` does
not hold `slim` or `run`. With the two exceptions, French `covered1` is 212 again
and `absent_sense` is 4 again. The 6 words are the whole change of the table:
3558 forms before, 3565 after.

The same build also moved three targets from two cards to one card (`craindre`,
`tirer`, `juste`). They keep one card and are not `absent_sense`, so this ADR
does not change them.

Step 2. `compounds` is read by `form_compounds` for tier S rows. Mandarin and
Cantonese `form_occurrences` and `form_offsets` skip an occurrence inside a
listed compound. French and Spanish do not call it, so a `quelque chose` row
would need a code change and is not added. A unit test shows that
`昆蟲` gives no hit for `蟲`, `听得到` gives no hit for `得到`, and a plain
`蟲` and `我得到了` still give one hit. Without the two rows the test fails. The
committed Cantonese deck failed K13 on `佢好憎昆蟲` after the row was added. A
rebuild removed the card.

Step 3. Before this change K13 read only `row["targets"]`. In the French core
deck (the review count), 118 of 273 cards show a Core word that is not in `targets`, and 67 tail
cards have no credited target. A swap of the Core word between two cards passed.
`note_values` writes the Core word and the gloss in two fields (`Core` and
`CoreGloss`), so the gate reads the `Core` field and needs no parse. The new
unit test has a pass case, a case with an empty Core field, and a case with a
swapped Core word. Only the swapped case fails. On the committed decks, K13
passes for French, Spanish, and Mandarin, so the check found no card problem.

## Rebuild

All six decks are rebuilt in `apkg/` with the current code and data:

| Deck           | Cards before | after | Targets before | after | `covered1` before | after | `absent_sense` before | after |
| -------------- | ------------ | ----- | -------------- | ----- | ----------------- | ----- | --------------------- | ----- |
| French core    | 276          | 277   | 216            | 216   | 212               | 212   | 4                     | 4     |
| Spanish core   | 267          | 273   | 227            | 236   | 222               | 229   | 5                     | 7     |
| Mandarin core  | 147          | 145   | 140            | 147   | 117               | 119   | 23                    | 28    |
| Cantonese core | 109          | 106   | 116            | 115   | 89                | 87    | 27                    | 28    |

| Deck           | Cards before | after | Targets | `covered` before | after |
| -------------- | ------------ | ----- | ------- | ---------------- | ----- |
| French cull    | 888          | 891   | 300     | 295              | 290   |
| Spanish cull   | 703          | 695   | 300     | 266              | 265   |

All gates pass on the fresh builds: core K5, K6, K13, K23 and cull G1, G2, G5,
G6. The core `coverage_floor` values do not change. The share of Cantonese is
0.410 against a floor of 0.40.

## Consequences

- A word that a real card needs, and that the first synset does not hold, goes
  in `EXTRA_WORDS`. The check is a rebuild and a diff of the coverage CSV.
- (Superseded by ADR 0053.) A new false match in Mandarin or Cantonese needed
  one `compounds` entry. A false match in French or Spanish needed a code
  change.
- K13 now fails when the `Core` field and the sentence disagree.

## Amendment: a tier S `senses` word passes the form (anki-jkeq)

`sense_failure` for a tier S form first passes the English when it holds a
`senses` word of the row. Otherwise a `false_friends` word rejects it, then the
gloss test runs. Before, a `senses` word only cancelled a false-friend hit. A
card that held a false-friend word and no sense word is still `false_friend`.
The cull and the tier F test do not change. Rebuilding the Spanish and French
core decks and the Spanish cull gives byte-identical CSVs; the Cantonese core
deck also gives identical CSVs, because no row has a `senses` word the gloss
misses yet.
