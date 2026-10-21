# Swadesh Sentence Pack Flow

This document shows how the `anki swadesh` sentence commands turn the Tatoeba
exports into one Anki deck. It shows the stage order and the guard order. The
code lives in `anki_tools/sentence_pack/`.

## 1. Stage map

| Stage | Command                             | Input             | Output                                    |
| ----- | ----------------------------------- | ----------------- | ----------------------------------------- |
| S0    | `anki swadesh fetch --lang <code>`  | Tatoeba exports   | `data/tatoeba/*.tsv`                      |
| S1    | `anki swadesh sentences <language>` | reviewed word CSV | word table in memory                      |
| S2    | `anki swadesh sentences <language>` | S0 and S1         | candidate pool in memory, pool report     |
| S3    | `anki swadesh sentences <language>` | S2                | card CSV, review CSV, pool report JSON    |
| S4    | `anki swadesh audio <language>`     | S3                | `audio/<language>_sentences/*.mp3`, index |
| S5    | `anki swadesh pack <language>`      | S1, S3, S4        | APKG, manifest, attribution               |
| Audit | `anki swadesh audit <language>`     | S5 files          | gate table G1 to G21                      |

The `sentences` command runs S1, S2, and S3 in one process.

## 2. Flow diagram

```text
S0  Tatoeba TSV files
        |
        v
S1  word table (reviewed forms, one row per concept)
        |
        v
S2  candidate pool
        |
        +--> F1  has an English link
        |
        +--> F2 to F8  shape guards (cheap tests, stop at the first failure)
        |
        +--> measure  predicate, orphan, unknown words, hits, sense hits
        |
        +--> F9 to sense  fact guards (stop at the first failure)
        |
        v
    accepted facts: core tier, and scaffold tier when fallback is on
        |
        v
S3  selection
        |
        +--> assign each sentence to the lowest Swadesh index that needs a card
        |
        +--> reject a sentence that is too close to a kept card
        |
        v
    vocab/<language>_sentences.csv
        |
        v
S4  audio files and the audio index
        |
        v
S5  APKG, manifest, attribution
        |
        v
    audit gates G1 to G21
```

## 3. S2 guard chain

The pool walks the linked sentences in Tatoeba id order. Each sentence goes
through the guards in one fixed order. The order is the `FILTER_KEYS` list in
`anki_tools/sentence_pack/pool.py`.

### 3.1 Shape guards

These guards are cheap. A sentence that fails one of them does not go through
the measure step.

| Key | Name                      | The guard passes when                                 |
| --- | ------------------------- | ----------------------------------------------------- |
| F2  | length window             | the unit count is inside the core or scaffold window  |
| F3  | script                    | the text holds no Latin letter and no digit           |
| F4  | no repeated pattern       | the text is not a repeated pattern                    |
| F5  | final mark                | the last character is an accepted final mark          |
| F6  | one final mark            | the text holds exactly one final mark                 |
| F7  | no quotation mark         | the text holds no quotation mark and no dialogue dash |
| F8  | at most one internal mark | the text holds at most one comma, colon, or semicolon |

F5 and F6 apply only when `final_mark` is true. The scaffold window skips F6
and F8.

### 3.2 Measure step

The pool measures the sentence only after the shape guards pass.

| Value        | Meaning                                                            |
| ------------ | ------------------------------------------------------------------ |
| `predicate`  | the tokenizer gives at least one verb or adjective tag             |
| `orphan`     | the sentence starts with a conjunction and has no main clause      |
| `unknown`    | the token count outside the known word set                         |
| `english`    | the word count of the paired English text                          |
| `hits`       | the forms that appear one time, after the form false-friend filter |
| `sense_hits` | the hits whose English text holds the card sense                   |
| `dedupe`     | the exact dedupe key of the prepared text                          |

### 3.3 Fact guards

| Key       | Name                  | The guard passes when                               |
| --------- | --------------------- | --------------------------------------------------- |
| F9        | predicate             | the sentence has a predicate tag                    |
| F10       | no orphan conjunction | the sentence has a main clause                      |
| F11       | English word count    | the English text has 2 to 12 words                  |
| F12       | unique text           | the dedupe key is not kept yet                      |
| `unknown` | unknown token budget  | the unknown token count is at most `max_unknown`    |
| `sense`   | English sense         | the sentence has no hit, or one hit holds the sense |

### 3.4 Decision

The pool records the first failed guard. It then counts the sentence as a pass
of every guard before that guard. Thus `counts[key]` is the number of
sentences that pass the guard `key`, not the number that fail it. The value
decreases from F1 to `sense`.

```text
shape failure = first failed shape guard, or None
fact failure  = first failed fact guard, or None

record_pass(counts, shape failure or fact failure)

if no guard failed and hits is not empty:
    keep a core sentence
elif the scaffold window is open and hits is not empty
     and no scaffold guard fails:
    keep a scaffold sentence
```

### 3.5 Core window and scaffold window

The core window uses `profile.min_units` to `profile.max_units`. The scaffold
window uses `profile.fallback_min_units` to `profile.max_units`.

| Setting              | Core window | Scaffold window        |
| -------------------- | ----------- | ---------------------- |
| Minimum unit count   | `min_units` | `fallback_min_units`   |
| F6 one final mark    | must pass   | may fail               |
| F8 one internal mark | must pass   | may fail               |
| `sense` guard        | must pass   | skipped                |
| F12 duplicate check  | core seen   | core and scaffold seen |

`SCAFFOLD_SHAPE_FAILURES` holds F6 and F8. `SCAFFOLD_SHAPE_KEYS` holds the
shape guards that the scaffold path still applies. `SCAFFOLD_SKIP_GUARDS`
holds the `sense` guard.

A scaffold sentence becomes a candidate only for a word that has no core
candidate.

## 4. S3 selection guards

The selector walks the forms in Swadesh index order, primary form first.

| Guard              | The selector accepts a sentence when                     |
| ------------------ | -------------------------------------------------------- |
| card cap           | the card count is below `max_cards`                      |
| form quota         | the assigned form has fewer cards than `per_word`        |
| unused sentence    | the Tatoeba id is not used by an earlier card            |
| open form          | the sentence covers a form that still needs a card       |
| duplicate distance | the Jaccard value stays below `profile.similarity_limit` |

The duplicate guard compares only kept cards that share a bucket key. The
selector sorts the candidates with this rank key:

1. unknown word count
2. audio penalty
3. negated mean Zipf value
4. length penalty
5. Tatoeba id

`assign_target` gives the sentence to the lowest covered form that still
needs a card. Thus one sentence never becomes two cards.

## 5. Audit gates

The audit reads the output files only. The gates fall in these groups:

| Group          | Gates              | Checks                                              |
| -------------- | ------------------ | --------------------------------------------------- |
| word table     | G1, G2, G15        | row count, index, label, review status              |
| cards          | G3 to G6, G14, G19 | target, id, match count, similarity, sentence shape |
| audio          | G7, G8, G16        | sentence file, word file, media count               |
| manifest       | G11, G13           | counts, hashes, APKG note count                     |
| coverage       | G10                | core coverage against `coverage_min`                |
| language rules | G20, G21           | false-friend compound, English sense                |
| attribution    | G12                | author names                                        |

A language pack can add a gate. For example, Cantonese adds G20 and G21.

## 6. File map

| Path                                      | Content                                 |
| ----------------------------------------- | --------------------------------------- |
| `anki_tools/sentence_pack/pool.py`        | S2 guards, measure step, pool report    |
| `anki_tools/sentence_pack/select.py`      | S3 rank key, duplicate index, set cover |
| `anki_tools/sentence_pack/audit.py`       | gate table and gate implementations     |
| `anki_tools/sentence_pack/profile.py`     | language values and hook functions      |
| `anki_tools/sentence_pack/languages/*.py` | one module per language                 |

## 7. Guard invariants

- The guard keys must equal `FILTER_KEYS[1:]` in order. The test
  `tests/test_sentence_pack_guards.py` checks this rule.
- A guard table has one entry for each filter key. Do not add a filter to the
  pool loop alone.
- `record_pass` uses the `FILTER_KEYS` index. A new guard must go into
  `FILTER_KEYS` before it can go into a guard table.
- The audit shape checks use the same guard table. The core check applies all
  shape keys. The scaffold check applies `SCAFFOLD_SHAPE_KEYS`.
