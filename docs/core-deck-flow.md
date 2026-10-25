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
| K3    | filter  | K1, name list, loan word list, media map  | passed cards, reason per card               |
| K4    | select  | K2, K3, ranks                             | kept cards, quota, uses                     |
| K5    | order   | K4                                        | card order, assigned target per card        |
| K6    | write   | K0, K4, K5                                | APKG and four side files                    |

The gates K1 to K20 run after K6.

The `--cards`, `--words`, `--examples`, and `--tail` values change
`CoreOptions` and the tier F target count. The other `CoreRequest` fields
keep their defaults.

The input fields come from the profile: `model_name`, `sentence_field`,
`english_field`, and `audio_field`. K0 keeps the lowest card of each note. It
cleans the sentence with `clean_sentence`.

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
        +--> C1 to C5  predicate, clause, name, loan word, audio
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
    gates K1 to K20
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

The tier S hit also needs the English sense. The form must pass
`sense_filter`, if the language has one, and `holds_tier_s_sense`. The tier F
hit needs `sense_match` against the sense row of the target.

A hit that fails the sense test adds to the `sense_failures` count of the
target. The card stays in the filter result, but it gets no credit for that
target.

## 4. Sentence rules (K3)

A card must pass every rule. `filter_cards` applies F1 to F9, and
`filter_core_cards` applies C1 to C5.

### 4.1 Shared card filter

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

### 4.2 Core guards

| Key | The rule passes when                                          |
| --- | ------------------------------------------------------------- |
| C1  | the card has a `VERB`, `AUX`, or `ADJ` token                  |
| C2  | the sentence starts with no orphan conjunction                |
| C3  | no whole word of the name list occurs in the sentence         |
| C4  | no word of the loan word list occurs in the sentence          |
| C5  | the card audio file exists in the media map of the input APKG |

C3 and C4 are whole-word tests. Thus the name `tom` does not reject
`tomber`.

The run stores the first failed rule as the `reason` of the card. The reason
value is one of F1 to F9 or C1 to C5. A passed card has the empty reason.

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
| `<stem>_core.apkg`     | the deck: the kept cards in the order of K5             |
| `<stem>.core.csv`      | one row per input card: kept, pass, value, cost, reason |
| `<stem>.coverage.csv`  | one row per target: key, word, gloss, quota, count      |
| `<stem>.manifest.json` | params, counts, frequency source, file hashes           |
| `<stem>.rank.csv`      | the intermediate ranking file of `reorder_deck`         |

`default_output` gives `<stem>_core.apkg` next to the input file.

The deck name comes from `profile.deck_name`. The note model keeps the name
of the input model. The writer deletes the notes that no kept card uses, and
it deletes the media files that those notes hold. `reorder_deck` applies the
order of the rank CSV to the output deck.

The writer updates the card templates and the CSS of the model. It adds the
core word field and the core gloss field to the model, and it fills them on
each kept card that has a target. It writes the assigned target inside a
`<b>` tag in each sentence field.

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

| Group      | Gates           | Checks                                                |
| ---------- | --------------- | ----------------------------------------------------- |
| size       | K1, K4          | card count, outside set and rare outside words        |
| coverage   | K2, K3, K11     | covered targets, two-card share, coverage CSV         |
| rules      | K5, K6, K17     | F1 to F9 and C1 to C5, similarity, name and loan word |
| files      | K7, K8, K12     | audio files, counts, manifest hashes                  |
| identity   | K9, K10         | deck name, model name, kept order                     |
| cards      | K13, K14, K15   | target in the text, templates and CSS, bold target    |
| sense      | K16, K19, K20   | unchecked target, core gloss, unranked tier F target  |
| shape      | K18             | each tail card brings a new content lemma             |

K3 reports the share of targets with two cards. It does not fail.

K14 checks the templates and the CSS of the language profile. K15 checks that
the bold text is the assigned target. K19 checks that the core gloss is the
gloss of the core word, and that the gloss has no comma.

## 9. File map

| Path                                            | Content                                    |
| ----------------------------------------------- | ------------------------------------------ |
| `anki_tools/ranking/coreflow.py`                | K0 to K6 flow, deck writer, manifest       |
| `anki_tools/ranking/coregates.py`               | gates K1 to K20                            |
| `anki_tools/ranking/core.py`                    | value, cost, passes, assignment, order     |
| `anki_tools/ranking/corelist.py`                | tier S and tier F target list, sense rows  |
| `anki_tools/ranking/cardfilter.py`              | shared filter rules F1 to F9               |
| `anki_tools/ranking/scan.py`                    | `CoreProfile` and the scan adapters        |
| `anki_tools/ranking/reorder.py`                 | reorder the deck from the rank CSV         |
| `anki_tools/ranking/duplicates.py`              | Jaccard value and the limit test           |
| `anki_tools/languages/<language>/core.py`       | one `CoreProfile` per language             |
| `anki_tools/cli.py`                             | `_run_core_command` of `fix <lang> --core` |

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
