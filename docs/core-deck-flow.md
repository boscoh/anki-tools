# Core Deck Flow

This document shows how `anki fix <language> --core` turns a sentence deck into
a core deck. It shows the stage order, the rule order, and the gate order. The
shared code lives in `anki_tools/ranking/`. The language values live in
`anki_tools/languages/<language>/core.py`.

Related: `docs/vocab-deck-flow.md`, `docs/cull-deck-flow.md`,
`docs/swadesh-sentence-flow.md`, and `specs/core-word-small-deck.md`.

## 1. Stage map

One command runs the stages in order: `anki fix <language> --core`. The
`--check` flag runs the gates on the output and skips K0 to K6.

| Stage | Name    | Input                                     | Output                                      |
| ----- | ------- | ----------------------------------------- | ------------------------------------------- |
| K0    | extract | input APKG                                | one card per note: sentence, English, audio |
| K1    | scan    | K0, frequency list, name list             | tokens with lemma and tag, ranks, names     |
| K2    | targets | K1, Swadesh word table, tier F sense file | core list: tier S, then tier F              |
| K3    | filter  | K1                                        | passed cards, reason per card               |
| K4    | select  | K2, K3, ranks                             | kept cards, quota, uses                     |
| K5    | order   | K4                                        | card order, assigned target per card        |
| K6    | write   | K0, K4, K5                                | APKG and four side files                    |

The gates run after K6. The table holds 4 rows: K5, K6, K13, and K23. The other
gate ids are retired (ADR 0038 to 0044, ADR 0046). K23 is the coverage floor
(ADR 0048). The flow step K4 is the selection.

The `--cards` and `--words` values change the tier F target count and the
card cap. The other `CoreRequest` fields keep their defaults.

The input fields come from the profile: `input_model_name`,
`input_sentence_field`, `input_english_field`, and `input_audio_field`. K0
keeps the lowest card of each note. It cleans the sentence with
`clean_sentence`. It strips one pair of wrapping double quotes from the English
text (`clean_english`), and the output deck writes the clean value.

K1 drops a card whose audio file is not in the media map of the input APKG
(`drop_silent_cards` in `ranking/scan.py`). The kept cards get new indices. The
build report shows the count as `Dropped, no audio file`.

## 2. Flow diagram

```text
K0  input APKG (sentence, English, audio)
        |
        v
K1  scan: cards, tokens, frequency ranks, names
        |
        v
K2  core list (tier S in Swadesh order, then tier F by rank)
        |
        v
K3  filter
        |
        +--> F1 to F9  shared card filter
        |
        v
    target hits (`core_hits`)
        |
        +--> compounds: Mandarin and Cantonese, a compound hides a hit
        |
        +--> sense: sense_failure per target; a failed target leaves the card
        |
        v
K4  selection: seed, depth, completion, top-up, tail
        |
        +--> quota per target, value minus outside cost
        |
        +--> reject a card too close to a kept card
        |
        v
K5  order: target position, example number, new words, units
        |
        v
K6  APKG, .core.csv, .coverage.csv, .manifest.json, .rank.csv
        |
        v
    gates K5, K6, K13, K23
```

## 3. Core list (K2)

### 3.1 Two tiers

| Tier | Source                                            | Order              |
| ---- | ------------------------------------------------- | ------------------ |
| S    | the reviewed Swadesh word table                   | `swadesh_index`    |
| F    | the blended frequency list, after the tier S form | blended rank       |

Tier S comes first. Tier F follows in rank order. A target has a 1-based
`position` in the list. The key is `S<index>:<form>` for tier S and
`F:<lemma>` for tier F.

A tier F lemma that equals a tier S form is not a target. Two Swadesh rows
with the same form give one target when the glosses share a sense word. They
give two targets when the glosses differ. `glosses_share_sense` is the test.

A target that no candidate card holds does not enter the list. The run stops
with an error when the list is empty.

### 3.2 Tier F target filter

These rules are in `build_targets` of `anki_tools/ranking/corelist.py`.

| Rule            | Value                                             |
| --------------- | ------------------------------------------------- |
| Part of speech  | `NOUN`, `VERB`, `ADJ`, or `ADV`                   |
| Deck count      | at least 3 content tokens                         |
| Content share   | at least 0.5 of all occurrences of the lemma      |
| Lemma length    | at least `min_lemma_length`, 2 by default         |
| Name list       | the lemma is not in the name stoplist             |

The tier F targets are the first `--words` of the rest in rank order. The
default is 50.

### 3.3 Match rule

| Tier | Unit    | A card holds the target when                          |
| ---- | ------- | ----------------------------------------------------- |
| S    | a form  | `profile.hits` finds the form, by the pack match rule |
| F    | a lemma | a content token of the card has the lemma             |

Both tiers call `sense_failure` of `anki_tools/sentence_pack/english_sense.py`
with the sense row of the target. A tier S form passes the false-friend words of
its row and the Swadesh gloss. A tier F lemma needs a sense word of its row and
no false-friend word. The pack and the cull call the same function.

A hit that fails the sense test adds to the `sense_failures` count of the
target. The card stays in the filter result, but it gets no credit for that
target.

## 4. Sentence rules (K3)

A card must pass every rule. `filter_cards` applies F1 to F9 through
`filter_core_cards`.

| Key | The rule passes when                                            |
| --- | --------------------------------------------------------------- |
| F1  | the English text has 2 to 12 words                              |
| F2  | the token count is inside the `min_units` to `max_units` window |
| F3  | the text holds no digit, bracket, link, or ellipsis             |
| F4  | the last character is an accepted final mark                    |
| F5  | the text holds exactly one final mark                           |
| F6  | the text holds no quotation mark and no dialogue dash           |
| F7  | the text holds at most one comma, colon, or semicolon           |
| F8  | the text holds no proper noun tag                               |
| F9  | the sentence text is not kept yet                               |

The run stores the first failed rule as the `reason` of the card. The reason
value is one of F1 to F9. A passed card has the empty reason.

## 5. Selection (K4)

### 5.1 Quota

```text
quota(w) = 1                   if share(w) > ubiquitous_share (0.25)
quota(w) = per_word (2)        otherwise
quota(w) = 3                   only for the first third_examples targets
```

`share(w)` is the count of candidates that hold the target, divided by the
count of candidates. A common pronoun gets one example, because a second
example adds no skill.

### 5.2 Value and cost

```text
weight(w) = 1 / log2(position(w) + 1)
demand(u) = 1.0, 0.35, 0.12            for example 1, 2, 3
value(c)  = sum of weight(w) * demand(uses[w])
            over the targets w of c with uses[w] < quota[w]

outside(c) = content lemmas of c that are not core targets
new(c)     = outside(c) minus introduced
cost(c)    = lambda_new * sum(mult(w) for w in new(c))
             + lambda_repeat * |outside(c) - new(c)|
             + lambda_len * max(0, units(c) - simple_units)

mult(w)    = 1.0 if rank(w) <= 1000, 1.8 if rank(w) <= 3000, else 3.0
score(c)   = value(c) - cost(c)
```

A new outside word costs more than a repeated one. A rare new word costs more
than a frequent one. `CoreOptions` holds the lambda values and the rank
limits.

### 5.3 Passes

| Pass       | Quota rule    | Accept rule                             | Goal                       |
| ---------- | ------------- | --------------------------------------- | -------------------------- |
| seed       | 1 per target  | `score > 0`, small cost                 | one example of each target |
| depth      | `quota(w)`    | `score > 0`                             | the full quota             |
| completion | `quota(w)`    | the best card, even at a negative score | fill a short quota         |
| top-up     | 3 for the top | `score > 0`                             | a third example            |
| tail       | none          | `score > 0`, gain is new lemmas         | width from the rest        |

The seed pass uses `lambda_seed` as its new-word cost. The top-up pass runs
only when `CoreOptions.third_examples` is not zero. The tail pass runs only
when `CoreOptions.tail_cards` is not zero.

The loop keeps the card with the largest key:

```text
(round(score, 9), -new word count, -unit count, -input index)
```

A tie goes to fewer new words, then fewer units, then the lower input index.
Two runs give the same output.

### 5.4 Duplicate and budget rules

The selector rejects a card in two cases:

- the lemma Jaccard value against a kept card that shares a lemma is at or
  above `options.similarity`
- the card would push the distinct outside lemma count above
  `options.outside_budget`, when that value is not zero

The reason value is `similar`, `outside_budget`, or `max_cards`.

The run stops before the selection when the quota cannot fit the card budget:

```text
if sum(quotas) > 2.1 * max_cards:
    stop with "the card budget is too small for this quota"
```

## 6. Card order (K5)

`assign_targets` gives each kept card the lowest-position target that is
still free. A card with no free target has no assigned target. The writer
then uses the lowest-position target that the card holds.

`order_cards` sorts the kept cards with this key:

1. the position of the assigned target, or `len(positions) + 1` without one
2. the example number of that target
3. the count of new outside words
4. the unit count
5. the input index

## 7. Files (K6)

| File                   | Content                                                 |
| ---------------------- | ------------------------------------------------------- |
| `<language>_core.apkg` | the deck: the kept cards in the order of K5             |
| `<stem>.core.csv`      | one row per input card: kept, pass, value, cost, reason |
| `<stem>.coverage.csv`  | one row per target: key, word, gloss, quota, count      |
| `<stem>.manifest.json` | params, counts, frequency source, file hashes           |
| `<stem>.rank.csv`      | the intermediate ranking file of `reorder_deck`         |

`identity.default_output(language, "core")` gives `apkg/<language>_core.apkg`.

Every name comes from the output file stem (`anki_tools/ranking/identity.py`):
the deck title, the note type name, and the side-file stems. The deck title
equals the note type name and `manifest.source.model`. The writer
(`anki_tools/apkg/shape.py`) clones the input note type under that name, so
the output holds one note type, one field set, and one card template. The
assembler (`anki_tools/ranking/assemble.py`) writes the side files, the
manifest, and the report.

The writer deletes the notes that no kept card uses, and it deletes the media
files that those notes hold. It deletes the card of a note that the first
template did not make, keeps the lowest `ord`, and sets that `ord` to 0. The
kept card keeps its id, its `due` value, and its scheduling columns.

The field set is `output_fields(profile)`: `Audio`, `Sentence`, `English`, the
`Romanization` field when the language has a romanization, `Core`, and
`CoreGloss`. The writer moves the kept notes to
the new note type, then deletes the input note type and every other note type.
`reorder_deck` applies the order of the rank CSV to the output deck.

The writer writes the assigned target inside a `<b>` tag in each sentence
field. `coreflow.note_values` writes `Core`, `CoreGloss`, and the bold sentence.
assigned target gives the pair, or the held target of the lowest core position.
A kept card that holds no core target is not written. Its CSV row holds
`kept = 0` and `reason = no_core`, and the manifest counts it in
`counts.dropped_no_core`.

The manifest records the output note type in `source.model` and the input note
type in `source.input_model`.

The coverage status of a target is one of these values:

| Status         | Meaning                                                  |
| -------------- | -------------------------------------------------------- |
| `covered`      | the count reaches the quota                              |
| `partial`      | the count is below the quota, and not zero               |
| `absent`       | the count is zero, and the pool holds no sense hit       |
| `absent_sense` | the pool holds a hit, but every hit fails the sense test |
| `unchecked`    | the target has no sense row                              |

## 8. Audit gates

`--check` reads the output files only. The gates fall in these groups:

| Group      | Gates              | Checks                                                              |
| ---------- | ------------------ | ------------------------------------------------------------------- |
| rules      | K5, K6             | F1 to F9, similarity                                                |
| cards      | K13                | target in the text                                                  |
| coverage   | K23                | share of targets with a kept card, floor from the language profile  |

The build report holds the value of K3 and K15, Covered twice, and the count of
cards that K1 drops for a missing audio file. It also holds the count of
unchecked targets. The manifest holds the count of unranked tier F targets. No
gate limits these counts, and a report line does not fail the run.

The `sense` filter is the one sense test (ADR 0039); no gate repeats it.

## 9. File map

| Path                                            | Content                                    |
| ----------------------------------------------- | ------------------------------------------ |
| `anki_tools/ranking/coreflow.py`                | K0 to K6 flow, output field values         |
| `anki_tools/ranking/coregates.py`               | gates K5, K6, K13, K23, 4 rows             |
| `anki_tools/gates.py`                           | gate value, row, runner, reader contract   |
| `anki_tools/ranking/gates.py`                   | report, shared gates                       |
| `anki_tools/ranking/assemble.py`                | deck, side files, manifest, report         |
| `anki_tools/ranking/identity.py`                | deck title, note type, side-file stems     |
| `anki_tools/ranking/deckprofile.py`             | language facts, core view, cull view       |
| `anki_tools/apkg/shape.py`                      | the written package shape                  |
| `anki_tools/ranking/core.py`                    | value, cost, passes, assignment, order     |
| `anki_tools/ranking/corelist.py`                | tier S and tier F target list, sense rows  |
| `anki_tools/ranking/cardfilter.py`              | shared filter rules F1 to F9               |
| `anki_tools/ranking/scan.py`                    | `CoreProfile` and the scan adapters        |
| `anki_tools/ranking/reorder.py`                 | reorder the deck from the rank CSV         |
| `anki_tools/ranking/duplicates.py`              | Jaccard value and the limit test           |
| `anki_tools/ranking/fixrun.py`                  | the shared `fix` route                     |
| `anki_tools/languages/<language>/core.py`       | one `CoreProfile` per language             |
| `anki_tools/cli.py`                             | the `fix <lang> --core` command            |

## 10. Invariants

- The filter keys must equal `FILTER_KEYS` in order. The test
  `tests/test_shared_coreflow.py` checks the rule table.
- A language gives values only. Do not add a language condition to
  `coreflow.py` or to `core.py`.
- A new language needs one `CoreProfile` in `languages/<language>/core.py` and
  one registry entry in `cli.py`. It needs no new command.
- The gates read the output files only. A gate function lives in
  `coregates.py`, and it takes a `CoreFiles` reader.
- The core list order is the Swadesh order first, then the tier F rank order.
  The card order depends on this order.
