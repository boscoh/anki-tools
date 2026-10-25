# Swadesh Sentence Pack Flow

This document shows how the `anki swadesh pack` command turns the Tatoeba
exports into one Anki deck. It shows the stage order and the guard order. The
shared code lives in `anki_tools/sentence_pack/`. The language values live in
`anki_tools/languages/<language>/`.

## 1. Stage map

One command runs the stages in order: `anki swadesh pack <language>`. The
`--stage` value stops the run after that stage. The default value runs every
stage.

| Stage | Stop after          | Input             | Output                                    |
| ----- | ------------------- | ----------------- | ----------------------------------------- |
| S0    | `--stage fetch`     | Tatoeba exports   | `data/tatoeba/*.tsv`                      |
| S1    | `--stage sentences` | reviewed word CSV | word table in memory                      |
| S2    | `--stage sentences` | S0 and S1         | candidate pool in memory, pool report     |
| S3    | `--stage sentences` | S2                | card CSV, review CSV, pool report JSON    |
| S4    | `--stage audio`     | S3                | `audio/<language>_sentences/*.mp3`, index |
| S5    | (default)           | S1, S3, S4        | APKG, manifest, attribution               |

The audit gate runs after S5: `anki swadesh audit <language>`.

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

The pool walks the linked sentences in Tatoeba id order. Then it appends the
manual frame sentences of `vocab/<language>_frames.csv`. A frame carries
Tatoeba id 0 and the author `manual`, and it passes the same guards. Each
sentence goes through the guards in one fixed order. The order is the
`FILTER_KEYS` list in `anki_tools/sentence_pack/pool.py`.

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

The measure step reads the tokens from the token cache of the language. The
pool warms the cache in batches of 512 sentences with one spaCy pipeline pass.
The disk cache at `data/tokens_<language>.sqlite` holds one row per sentence.
Thus a second build does not tokenize the same sentence again.

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
| frame sentence     | a manual frame has id 0, so the unused rule skips it     |
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

## 5. Language guards

Two form-level guards add language data. The shared scan code is in
`anki_tools/sentence_pack/guards.py`. The data is in the language module.

### 5.1 False-friend compound

The `FALSE_FRIENDS` table maps a word-table form to compounds that use the
form in a different sense. An occurrence inside one of those compounds does not
count as a hit.

| Pack      | Example        | Result                        |
| --------- | -------------- | ----------------------------- |
| Mandarin  | 闻 inside 新闻 | the occurrence does not count |
| Mandarin  | 闻 in 我闻花香 | the occurrence counts         |
| Cantonese | 聽 inside 聽日 | the occurrence does not count |
| Cantonese | 聽 in 我聽音樂 | the occurrence counts         |

`compound_spans` and `inside_spans` give the blocked spans and the test.
Mandarin applies the helpers to the Han character stream. Cantonese applies
them to the sentence text.

### 5.2 Form sense filter

The `SENSE_CHECKS` table maps a form to two word sets: the sense words and the
false-friend words. `make_sense_filter` builds the filter. The filter rejects a
form when the paired English holds a false-friend word and no sense word.

| Pack      | Form | The filter rejects when the English holds |
| --------- | ---- | ----------------------------------------- |
| Mandarin  | 抓   | grab, catch, arrest, with no scratch word |
| Cantonese | 揸   | drive, fly, pilot, with no hold word      |

The S2 measure step applies this filter to the hits. A form that the filter
rejects does not enter the hit list.

### 5.3 English gloss sense

`holds_sense` compares the English of a sentence with the Swadesh gloss of the
card word. The S2 `sense` guard rejects a sentence when every hit fails this
test. The guard keeps a sentence that has no hit.

### 5.4 Whole-token match

Cantonese 我 and 我哋 are different concepts. `form_offsets` accepts a form
only when it holds whole tokens. An occurrence inside a longer word counts when
that word is not itself a target form. So 食 counts inside 食飯, and 我 does not
count inside 我哋.

Mandarin uses `form_occurrences` on the Han character stream, because jieba
merges some forms into a longer token. So 水 counts inside 喝水.

### 5.5 Gate numbers

| Pack      | False-friend gate | English sense gate |
| --------- | ----------------- | ------------------ |
| Mandarin  | G17               | G18                |
| Cantonese | G20               | G21                |
| French    | none              | G21                |
| Greek     | none              | G20                |

The shared English sense gate and the shared false-friend gate live in
`anki_tools/sentence_pack/audit.py`. The word table gate, the boundary gate,
the attribution gate, and the other language gates live in the language
module. The gate table of the language lists them in gate number order.

## 6. Audit gates

The audit reads the output files only. The gates fall in these groups:

| Group          | Gates              | Checks                                              |
| -------------- | ------------------ | --------------------------------------------------- |
| word table     | G1, G2, G15        | row count, index, label, review status              |
| cards          | G3 to G6, G14, G19 | target, id, match count, similarity, sentence shape |
| audio          | G7, G8, G16        | sentence file, word file, media count               |
| manifest       | G11, G13           | counts, hashes, APKG note count                     |
| coverage       | G10                | core coverage against `coverage_min`                |
| language rules | per pack           | false-friend compound, English sense                |
| attribution    | G12                | author names                                        |

A language pack can add a gate. For example, Cantonese adds G20 and G21.

## 7. File map

| Path                                         | Content                                  |
| -------------------------------------------- | ---------------------------------------- |
| `anki_tools/sentence_pack/pool.py`           | S2 guards, measure step, pool report     |
| `anki_tools/sentence_pack/select.py`         | S3 rank key, duplicate index, set cover  |
| `anki_tools/sentence_pack/audit.py`          | gate runner and the shared gates         |
| `anki_tools/sentence_pack/guards.py`         | shared span and occurrence helpers       |
| `anki_tools/sentence_pack/english_sense.py`  | English gloss sense check                |
| `anki_tools/sentence_pack/tokens.py`         | token cache and spaCy pipeline           |
| `anki_tools/sentence_pack/profile.py`        | profile dataclass and registry           |
| `anki_tools/sentence_pack/pack.py`           | S5 APKG build and manifest               |
| `anki_tools/languages/<language>/profile.py` | language values, hooks, gates            |
| `anki_tools/languages/mandarin/stages.py`    | Mandarin gate runner                     |

A shared module imports no language package. A language module holds the
values and the gate functions of that language.

## 8. Guard invariants

- The guard keys must equal `FILTER_KEYS[1:]` in order. The test
  `tests/test_sentence_pack_guards.py` checks this rule.
- A guard table has one entry for each filter key. Do not add a filter to the
  pool loop alone.
- `record_pass` uses the `FILTER_KEYS` index. A new guard must go into
  `FILTER_KEYS` before it can go into a guard table.
- The audit shape checks use the same guard table. The core check applies all
  shape keys. The scaffold check applies `SCAFFOLD_SHAPE_KEYS`.
- A shared gate function lives in `audit.py`. A language gate function lives
  in the language module. The language profile gives the `false_friend_spans`
  and `sense_filter` hooks that the shared gates call.
