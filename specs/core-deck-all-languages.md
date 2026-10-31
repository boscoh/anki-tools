# Spec: Core Deck Flow for All Languages

| Field   | Value                                                          |
| ------- | -------------------------------------------------------------- |
| Status  | draft                                                          |
| Date    | 2026-10-24                                                     |
| Owner   | anki-tools                                                     |
| Input   | a sentence deck with a text field, an English field, and audio |
| Related | `specs/core-word-small-deck.md` (French core deck)             |
| Related | `specs/french-1000-sentence-cull.md` (breadth cull)            |
| Issues  | anki-ww9                                                       |

## Contents

1. [Summary](#1-summary)
2. [Target record](#2-target-record)
3. [Scan](#3-scan)
4. [Core list](#4-core-list)
5. [Sense guard](#5-sense-guard)
6. [Selection](#6-selection)
7. [Order and card](#7-order-and-card)
8. [Data requirements](#8-data-requirements)
9. [Interface](#9-interface)
10. [Gates and acceptance](#10-gates-and-acceptance)
11. [Spanish pilot](#11-spanish-pilot)
12. [Risks](#12-risks)
13. [Milestones](#13-milestones)

## 1. Summary

One flow for every language:

```text
scan -> core list with a gloss -> filter -> sense guard -> cover -> order -> write
```

The French core deck is the reference implementation. This spec generalizes
its data contract, not its selection policy. The scan, the target record, and
the sense guard become shared. The cover stays a per-language policy with two
presets: a core quota and a breadth cull.

### 1.1 Non-goals

- This spec does not merge the selection algorithms. The core quota and the
  cull use cap stay separate policies.
- The output note type of section 14 changes every core deck and every cull
  deck. That change is the one exception here. A later change to a published
  deck is a separate decision with a new manifest.
- It adds no per-language CLI flags. The language profile supplies the
  tokenizer, the word table, the sense file, and the frequency source.

## 2. Target record

One record per target. The selection and the card back read the same record,
so the sentence and the gloss cannot disagree.

| Field           | Type           | Meaning                                          |
| --------------- | -------------- | ------------------------------------------------ |
| `key`           | string         | `S<index>:<form>` or `F:<lemma>`                 |
| `tier`          | `S` or `F`     | Swadesh concept, or a frequency word after it    |
| `position`      | int            | 1-based position in the core list                |
| `form`          | string         | Surface form of a tier S target, lemma of tier F |
| `gloss`         | string         | English gloss shown on the card back             |
| `senses`        | set of strings | English words that show the intended meaning     |
| `false_friends` | set of strings | English words that mark another meaning          |
| `checked`       | bool, derived  | True for tier S, or when `senses` is not empty   |

Rules:

- A tier S gloss comes from the reviewed word table, for example
  `vocab/spanish_words.csv`.
- A tier F gloss is the first word of its sense row. Tier F has no reviewed
  gloss of its own, so the sense row supplies both the test and the back text.
- A target without a sense row is `unchecked`. It selects cards with no sense
  test, and gate K16 counts it.

## 3. Scan

The flow starts with one pass over the deck:

1. Extract the sentences with `extract_sentences`.
2. Tokenize with the language tokenizer.
3. Count the content tokens.
4. Blend the frequency ranks.
5. Keep the tokens per card and the ranks per lemma.

| Language | Tokenizer               | Frequency source        | Available |
| -------- | ----------------------- | ----------------------- | --------- |
| fr       | spaCy `fr_core_news_sm` | Tatoeba + `wordfreq`    | yes       |
| es       | spaCy `es_core_news_sm` | `wordfreq`              | yes       |
| zh       | jieba                   | `wordfreq zh`           | yes       |
| yue      | Cifu + jieba            | spoken + written Cifu   | yes       |

The frequency candidate builder is shared: `anki_tools/ranking/corelist.py`
holds `build_targets`. The scan helpers still live in the French package:
`load_input_deck`, `tokenize_deck`, `load_frequency`, and `frequency_ranks`.
The Cantonese core scan lives in its own module. The Spanish and Mandarin rank
modes do not scan at all today.

## 4. Core list

Build the target list in one order:

1. The reviewed Swadesh forms, in table order.
2. The tier F words, in blended frequency order, up to `--words`.

The tier F candidate set is the lemma set of the scan. The external frequency
list only sorts that set; it adds no members. A scan lemma that the list does
not hold gets `UNRANKED_RANK`. Build the full ranked candidate list, drop the
forms that the Swadesh table already holds, then cut to `--words`. Do not cut
before the exclusion.

Apply the same rules as the French core: the content share, the name list, and
the loan word list. The minimum deck count is one more than the example quota,
so the selector keeps one spare candidate.

The manifest counts the tier F targets whose rank is `UNRANKED_RANK`. Gate K20
caps that count, like K16 caps `unchecked`. An unranked word sorts below every
ranked word; its tie-break is the deck count, then the form.

`build_core_targets` and the target record types live in the same shared
module. A language package supplies the word-table triples and the tier F
sense rows.

Invariant: for French, the new builder must produce the current 209 targets
(159 tier S and 50 tier F) with the same positions, keys, glosses, and sense
rows. Compare the list, not only the deck.

## 5. Sense guard

The guard tests the paired English of each candidate card.

- Tier S: `holds_sense(gloss, english)`. The gloss rows live in
  `vocab/english_senses.json`, built by `scripts/build_english_senses.py`. A
  gloss without a row does not reject the card.
- Tier F: `sense_match(english, senses, false_friends)`. An empty `senses`
  accepts the card and marks the target `unchecked`. A `false_friends` word
  rejects the card even when a sense word is present.

Target status after the guard:

| Status         | Meaning                                             |
| -------------- | --------------------------------------------------- |
| `covered`      | two or more cards                                   |
| `partial`      | one card                                            |
| `absent`       | no candidate holds the form                         |
| `absent_sense` | candidates hold the form, every one fails the sense |
| `unchecked`    | no sense row, so no card was tested                 |

Gate K16 accepts at most `max_unchecked_sense` targets, 0 by default. A
language with no sense rows therefore cannot ship a tier F core deck without
raising that limit. Raising it is an explicit decision, recorded in the
manifest.

The guard reads the English translation, so a deck with no usable English
field cannot use it. The English quality sets the ceiling of the test.

### 5.1 Tier F sense rows

A tier F sense row has three columns in the language TSV, for example
`vocab/french_core_senses.tsv`:

```text
word	senses	false_friends
faire	do,does,did,done,doing,make,makes,made,making
pas	not
```

The row does four things:

1. It turns the sense test on. Without it the target is `unchecked`.
2. It filters cards by the English. `Il fait beau` with `The weather is nice`
   fails, because no sense word is present. `Je fais mes devoirs` with
   `I do my homework` passes.
3. It supplies the back gloss. `pas` shows `not` on the back, and the sentence
   must show the same sense in its English.
4. It can reject a false friend. The column exists, and the code applies it,
   but all 51 French rows leave it empty. The mechanism is untested in
   production.

One row covers one intended meaning. A word with several common meanings needs
either several rows or a wider sense list. The French `plus` row lists
`more,most,else,longer,further`, so `no longer` passes, although it is a
different sense. A false friend could reject it.

Tier F senses are single English words. The TSV has no phrase column, so a
multiword sense needs a wider word list or a false friend. Tier S supports
phrases through `vocab/english_senses.json`.

Counts: French has 51 rows for 50 tier F words plus `ne`. A 300-card core deck
in another language needs roughly 50 to 150 rows. A Swadesh-only deck needs
none, because tier S uses the gloss table.

## 6. Selection

The cover keeps a card only when its gain is positive. Each language supplies:

- the quota per target,
- the exempt set and the use cap,
- the duplicate test,
- the order key.

The core preset uses the five passes of `anki_tools/ranking/core.py`: seed,
depth, completion, tail, top-up, with the outside-word cost. The breadth preset
uses the cull cap and the Swadesh reserve. A stratified preset stays the
Spanish mode and is not part of this spec. The M6 decision moves Mandarin to
the word core and retires its stratified preset.

## 7. Order and card

- Order the cards by core position, or by difficulty when the language chooses
  that key.
- The card back shows the sentence, the English, `Core`, and `CoreGloss`. The
  core word is bold.
- Both modes write a note type of their own. The output holds one note type,
  one field set, and one card template (section 14).

## 8. Data requirements

| Language | Word table                    | Sense table                      | Tier F rows | State |
| -------- | ----------------------------- | -------------------------------- | ----------- | ----- |
| fr       | `vocab/english-french.csv`    | `vocab/french_core_senses.tsv`   | 51          | ready |
| es       | `vocab/spanish_words.csv`     | `vocab/spanish_core_senses.tsv`  | 50          | ready |
| zh       | `vocab/english-mandarin.csv`  | `vocab/mandarin_core_senses.tsv` | 50          | ready |
| yue      | `vocab/english-cantonese.csv` | none                             | 0           | ready |

Every language also needs a deck with a usable English field. The French deck
has one. The Spanish deck at `apkg/Spanish 9000 sentences with native audio.apkg`
has one. Check the other decks before a build.

Rule C3 reads the name stoplist of its language: `vocab/french_names.txt`
holds 202 names, `vocab/cantonese_names.tsv` holds 5, and
`vocab/mandarin_names.txt` holds 1. `vocab/mandarin_names.txt` is small
because the Mandarin deck holds almost no personal name: a scan of its 5009
notes found 希德 (Sid) twice and no other. Spanish has no name list yet.

### 8.1 Frequency source

| Language | Primary                | Fallback                    | License                   |
| -------- | ---------------------- | --------------------------- | ------------------------- |
| fr       | wordfreq + Tatoeba     | Lexique 4.00                | Apache-2.0 + CC BY-SA 4.0 |
| es       | FrequencyWords es_full | Leipzig spa_web             | CC BY-SA 4.0 / CC BY 4.0  |
| zh       | wordfreq zh            | FrequencyWords zh_cn, zh_tw | Apache-2.0 + CC BY-SA 4.0 |
| yue      | Cifu + HKCanCor        | HKCanCor alone              | GPL-3.0 / CC BY 4.0       |

Rules:

- No single source covers all four languages. wordfreq, FrequencyWords, and
  Leipzig have no `yue` list.
- Do not use SUBTLEX-ESP. Its license is Creative Commons
  Attribution-NonCommercial-NoDerivatives 3.0: no derivatives and no
  commercial use.
- Do not use the SUBTLEX-CH files. The page states no license.
- Cifu is GPL-3.0. HKCanCor (CC BY 4.0) is the clean fallback for Cantonese.
- CC BY-SA 4.0 is share-alike. The attribution file and the manifest must name
  the source and the license of the derived TSV.

### 8.2 Frequency cache

One TSV per language in `data/`, for example `data/spanish_frequency.tsv`:

| Column  | Meaning                                            |
| ------- | -------------------------------------------------- |
| word    | Surface form, as it appears in the deck            |
| lemma   | Lemma of the form, from the source or a lemmatizer |
| count   | Token count                                        |
| pm      | Count per million tokens                           |
| rank    | 1-based rank of the lemma                          |
| pos     | Part of speech, empty when the source has none     |
| source  | Short source name                                  |
| license | SPDX identifier                                    |

`data/spanish_frequency.tsv` also carries `tatoeba_count`, `tatoeba_pm`,
`written_pm`, and `zipf`. Those four columns are the cull blend of
`specs/french-1000-sentence-cull.md`. `cull_frequency_rows()` maps them onto
the shared `FrequencyRow`. A language adds the four columns when its cull
needs them.

- Record the file sha256 and the source version in the manifest, next to
  `LICENSES`.
- `build_targets` looks up lemmas in the rank table. A form-only source needs
  a lemma column. Without it, unmatched lemmas fall to `UNRANKED_RANK = 100000`.
- Normalize case and accents. For Chinese, pin the script and convert with
  OpenCC.
- Keep the top 50,000 rows, as `data/french_frequency.tsv` does.

## 9. Interface

Keep the reduced surface of `anki fix fr`:

```text
--core --cards --words --output --check
```

The language command selects the tokenizer, word table, frequency source, and
sense file from the profile. No new per-language flags.

## 10. Gates and acceptance

Reuse gates K1 to K21 with the language file paths. A language is accepted when:

- every K gate passes;
- `checked` equals the targets with a sense row plus every tier S target;
- `unchecked` is within the limit;
- `unranked` is within the limit;
- every note holds one `Core` and one `CoreGloss`, and the gloss equals the
  target's first sense (K19, section 14.9);
- the output holds one note type with the output field set (K21, section 14.9);
- the manifest records the kept order and the hashes.

K3 reports the share of coverable targets with two cards and does not gate.
The share measures the input deck, not the build: a target needs a card whose
English holds the gloss, and a small deck cannot supply two. The selection
still tries for two (quota 2) and takes the second card where the pool holds
one. A hard limit on the share punishes a language for its input, and 0.85 has
no source in the data.

## 11. Spanish pilot

Input: `apkg/Spanish 9000 sentences with native audio.apkg`.

| Step | Work                                                             | Acceptance                                    |
| ---- | ---------------------------------------------------------------- | --------------------------------------------- |
| S1   | scan and tokenize the deck; compare lemma counts with the ranker | counts agree with `fix es`                    |
| S2   | Swadesh-only core list, 0 tier F rows, 1 to 2 examples           | K1 to K19 pass with 0 unchecked               |
| S3   | build the frequency cache (8.2), then author 50 tier F rows      | every tier F target is `checked`              |
| S4   | build the 300-card deck                                          | K1 to K19 pass; coverage CSV agrees           |
| S5   | compare with the current `fix es --cards` output                  | report coverage, outside words, sense rejects |

Step S2 proves the flow without any new data. Step S3 proves the sense contract.
Step S5 decides whether the core guarantee beats the current complexity spread.

### 11.1 S5 comparison

A 300-card complexity spread against the 266-card word core:

| Deck              | Cards | Targets 2+ | Tier F 2+ | Outside | Sense rejects |
| ----------------- | ----- | ---------- | --------- | ------- | ------------- |
| word core         | 266   | 198 of 226 | 50 of 50  | 180     | 128           |
| complexity spread | 300   | 67 of 226  | 33 of 50  | 100     | 135           |

The word core covers 2.9 times the targets at the quota, so it beats the
complexity spread. The spread holds fewer outside words and 49 cards that teach
no core target.

### 11.2 Open items

- Resolved: the sentence field is `SpanishSentence`, the English field is
  `EnglishSentence`, and the audio field is `Audio`, on the model
  `Spanish sentences cloze+` (anki21b, zstd media).
- Resolved: `es_core_news_sm` is installed.
- Resolved: the word table is `vocab/spanish_words.csv`; the older
  `vocab/english-spanish.csv` has slashed variant rows that cannot match tokens.
- Resolved: the tier F word list is the first 50 lemmas of `build_targets`
  after the Swadesh exclusion.
- Resolved: the lemmatizer gave two artefacts among the 50 tier F targets. It
  gave `quier` for `quieres`, where `querer` is already a target, and it gave
  `buen` for the `buen` and `buena` forms, where the dictionary lemma is
  `bueno`. `CoreProfile.lemma_overrides` repairs the tokens after
  `tokenize_deck`, and the Spanish profile maps `quier` to `querer` and `buen`
  to `bueno`. `F:quier` merges into `F:querer`. `bueno` is Swadesh row 185, so
  the tier S target `S185:bueno` covers the `buen` and `buena` cards and the
  duplicate tier F row is gone. The tier F row `buen` is removed, and the rows
  `salir` and `gente` enter the 50. The deck holds 265 cards, 226 targets, 0
  unchecked, and 0 unranked.
- Open: `mirar` is a tier F target, so two kept cards can hold one content
  lemma each. The Spanish profile sets `similarity=0.5` to keep gate K6. The
  shared dedup compares target keys, and K6 compares content lemmas.

### 11.3 Cantonese pilot

Input: `apkg/LTL Cantonese Deck Level 1 - Short.apkg`, 999 notes.

The deck is small. It holds two or more usable cards for 70 of its 212
word-table rows:

| Usable cards | Forms |
| ------------ | ----- |
| 3 or more    | 48    |
| 2            | 22    |
| 1            | 38    |
| 0            | 104   |

So two examples per target are a best effort. The selection takes the second
card where the pool holds one and keeps one card where it does not. Gate K3
reports the share and does not fail on it (section 10).

| Run          | Cards | Targets | Two cards          | Other failures     |
| ------------ | ----- | ------- | ------------------ | ------------------ |
| `--words 0`  | 126   | 130     | 60 of 103 (0.583)  | none               |
| `--words 50` | 174   | 180     | 103 of 153 (0.673) | K16 (50), K19 (43) |

The Swadesh-only run passes every gate. The 50 tier F targets need the sense
rows of `vocab/cantonese_core_senses.tsv`, so `DEFAULT_CORE_WORDS` is 0 until
those rows exist (`anki-eiy`).

The 27 `absent_sense` targets and the 752 sense failures share one cause: the
English field does not hold the gloss of the form. Example: `四` occurs one
time, and its card reads `"Don't speak nonsense!"`. A second input with paired
English, the Tatoeba pool of `anki-5z4`, improves that share but is no longer a
prerequisite.

### 11.4 Mandarin pilot

Input: `apkg/Neri's Chinese Sents Part 2.apkg`, 5009 cards on the model
`0 2024 Sentences Deck`.

The adapter is `anki_tools/languages/mandarin/core.py`. The tier S form unit is
the reviewed word-table form, which is one jieba word. The hit rule reads the
form as a character span in the Han text, and the English gloss of the row
decides the card. The character span keeps the forms that jieba merges into a
longer token, for example `水` inside `喝水`. `FALSE_FRIENDS` blocks the spans
where the form takes another sense, so `来` does not hit inside `出来` and
`起来`. That hit rule answers the word-boundary risk of section 12.

The run sets 50 tier F words, two examples per word, no tail cards, and a
similarity limit of 0.65. The sense rows live in
`vocab/mandarin_core_senses.tsv`, and `vocab/mandarin_loanwords.txt` lists the
nine loan words for rule C4.

```text
uv run anki fix zh "apkg/Neri's Chinese Sents Part 2.apkg" --core --output apkg/mandarin_core.apkg
uv run anki fix zh "apkg/Neri's Chinese Sents Part 2.apkg" --core --check --output apkg/mandarin_core.apkg
```

| Measure        | Value                      |
| -------------- | -------------------------- |
| Input cards    | 5009                       |
| Passed scan    | 2812                       |
| Rejected C1    | 1096                       |
| Rejected F1    | 752                        |
| Rejected F2    | 348                        |
| Cards          | 147                        |
| Targets        | 140 (90 tier S, 50 tier F) |
| Covered once   | 117                        |
| Covered twice  | 96 of 117 (0.821)          |
| Full sense     | 98                         |
| Partial sense  | 19                         |
| Absent sense   | 23                         |
| Outside lemmas | 40, rare 1                 |
| Unchecked      | 0                          |
| Sense failures | 1424                       |

The build passes all 20 gates with 147 cards and 140 targets. Gate K3 reports
the two-card share 0.821 and does not gate. The 23 `absent_sense` targets and
the 1424 sense failures share one cause: the English field of the deck does
not hold the gloss of the form. The kept files are `apkg/mandarin_core.apkg`
with its `.core.csv`, `.coverage.csv`, `.manifest.json`, and `.rank.csv`.

Two repairs followed the first build. The templates now read the audio field
`[a]` of the model, because `{{Audio}}` named no field, and the Mandarin
profile sets the `highlight` hook, so the core word is bold. The sense
rows of `一样` and `计划` now start with `same` and `plan`, because the deck
gave those words in that sense.

```text
Gate  Result  Detail
----  ------  --------------------------------------------------------------------------------------------------------------
K1    PASS    manifest 147, CSV 147, APKG 147
K2    PASS    covered 117 of 117
K3    PASS    two cards for 96 of 117 (0.821)
K4    PASS    outside 40 of 140 targets, rare 1
K5    PASS    147 card(s) pass the rules
K6    PASS    no pair reaches the limit
K7    PASS    0 card(s) without audio:
K8    PASS    notes 147, CSV 147, media 147 of 147
K9    PASS    decks ['Mandarin Core Sentences'], models ['0 2024 Sentences Deck']
K10   PASS    147 rows, order matches
K11   PASS    140 rows, issues
K12   PASS    hashes match
K13   PASS    0 bad hit(s):
K14   PASS    1 model(s) match, CSS match
K15   PASS    106 bold card(s), issues
K16   PASS    0 unchecked, limit 0
K17   PASS    name hits 0, loan hits 0
K18   PASS    0 tail card(s), 0 without a new lemma
K19   PASS    147 core word(s), 0 bad:
K20   PASS    0 unranked, limit 0

all 20 gates passed
```

## 12. Risks

- Data authoring is the long pole. The Spanish pilot needs 50 rows, and a full
  deck needs more.
- The English translation limits the sense guard. A weak English field rejects
  good cards or passes bad ones.
- Resolved for Mandarin (section 11.4): the character-span hit rule gives the
  word-level sense test a boundary. Another script needs the same check before
  the sense test is meaningful.
- Adding the guard to the existing cull changes its deck. Measure before the
  change, and record a new manifest.
- The false-friend column is empty in French. Spanish sets one row: `solo`
  rejects `only` and `just`, so the target keeps the `alone` sense. The
  mechanism has no test beyond that row.
- A shared flow must not become a dispatcher. The policy stays a per-language
  preset, not a branch inside the scan.
- A small input deck limits the two-card share. K3 reports the share and does
  not gate; measure the usable cards of the word table before a language
  starts.

## 13. Milestones

| Milestone | Work                                                       |
| --------- | ---------------------------------------------------------- |
| M1        | accept this spec                                           |
| M2        | shared target record and scan with language hooks          |
| M3        | French core list unchanged on the new record               |
| M4        | Spanish Swadesh-only deck, gates pass                      |
| M5        | Spanish tier F sense rows and the 300-card deck            |
| M6        | Mandarin: word core; complexity spread retired             |
| M7        | Cantonese: Swadesh-only core, best-effort examples        |
| M8        | output note type for both modes (section 14)               |

## 14. Output note type

Both modes write a note type of their own. The output package holds one note
type, one field set, and one card template. The source note type stays in the
input package only. Section 7 names this result. This section is the design.

The four inputs use four note types and four field sets: `French Sentences`,
`Spanish sentences cloze+`, `0 2024 Sentences Deck`, and
`LTL Note Type - Cantonese`. The Mandarin input has 19 fields and the card
reads three of them.

### 14.1 Two identities, one seam

The input identity locates the source notes. The output identity names and
shapes the result. One value must not do both jobs.

The output identity has one source of truth: the output file stem of the run.
`anki_tools/ranking/identity.py` is the only place that builds these names:

| Artifact | Value |
| --- | --- |
| Deck title | output file stem |
| Note type name | output file stem |
| `manifest.source.model` and `manifest.source.deck` | output file stem |
| Side-file stems: `.core.csv`, `.cull.csv`, `.coverage.csv`, `.manifest.json`, `.rank.csv` | output file stem |

`DeckIdentity(output_path)` gives `stem`, `title`, `model_name`, and `side()`,
plus `core_sides()` and `cull_sides()`. `identity.default_output(language,
mode)` gives the default file of a mode: `apkg/french_core.apkg`,
`apkg/french_1000.apkg`. ADR 0015 records
the decision.

The input identity stays in the profile: `input_model_name`,
`input_sentence_field`, `input_english_field`, and `input_audio_field`. The
profile holds no output name, so the title and the note type cannot drift.

| Profile field | Change |
| --- | --- |
| `model_name` | rename to `input_model_name` |
| `sentence_field` | rename to `input_sentence_field` |
| `english_field` | rename to `input_english_field` |
| `audio_field` | rename to `input_audio_field` |
| `output_model_name` | delete: the identity module names the note type |
| `deck_name`, `default_deck_name` | delete: the identity module names the deck |
| `core_field`, `core_gloss_field` | delete: the constant `OUTPUT_CORE_FIELD` |
| `romanization_source` | add, `str | None = None` |

`anki_tools/ranking/deckprofile.py` `LanguageDeckProfile` holds the language
facts once, and gives a core mode view and a cull mode view. A mode view adds
the values that only that mode states. ADR 0004 keeps its decision: each mode
supplies its own values.

The output field names are not profile values. They live in
`anki_tools/ranking/coreflow.py`, next to `TARGET_SEPARATOR`:

```python
OUTPUT_FIELDS = ("Audio", "Sentence", "English", "Core", "CoreGloss")
OUTPUT_AUDIO_FIELD = "Audio"
OUTPUT_SENTENCE_FIELD = "Sentence"
OUTPUT_ENGLISH_FIELD = "English"
OUTPUT_ROMANIZATION_FIELD = "Romanization"
OUTPUT_CORE_FIELD = "Core"
OUTPUT_GLOSS_FIELD = "CoreGloss"
```

- `output_fields(profile)` gives the three shared names, then `Romanization`
  when `profile.romanization_source` is not None, then `Core` and `CoreGloss`.
- A language cannot rename `Core` or `Romanization`, so a gate cannot lose its
  field. A language supplies only the name of the input field that holds the
  romanization.

Readers after the change:

| Function | Reads |
| --- | --- |
| `coreflow.read_input_deck` | the four `input_*` names |
| `coreflow.output_values` | `input_audio_field`, `input_english_field`, `romanization_source`, and the `OUTPUT_*` names |
| `coreflow.note_values` | `input_model_name`, `OUTPUT_CORE_FIELD`, `OUTPUT_GLOSS_FIELD`, `OUTPUT_SENTENCE_FIELD`, and `highlight` |
| `assemble.write_run` | `DeckIdentity` and the `OutputShape` of the caller |
| `coregates.gate_bold` (K15) | `OUTPUT_SENTENCE_FIELD` |
| `gates.gate_core_word` (K19, G17) | `OUTPUT_CORE_FIELD`, `OUTPUT_GLOSS_FIELD` |
| `cullflow.run_cull` | the four `input_*` names and `sense_path` |
| `gates.gate_identity` (K9, G10) | `manifest.params.name` and `manifest.source.model` |
| `mandarin.core.finalize_output`, `cantonese.core.finalize_output` | `OUTPUT_SENTENCE_FIELD` and the only note type |

`anki_tools/apkg/shape.py` states the written package shape behind
`OutputShape` and `write_package`: one deck, one standard note type with the
stated fields, and one card per note. The `anki2` and `anki21b` adapters sit
behind the seam. The `anki2` note type holds the output deck link, and the
`anki21b` note type has no such field. Both set the standard kind.

`anki_tools/ranking/assemble.py` holds the write order, the manifest schema,
and the report layout once. `DeckRun` states the selected cards, the complete
output field values, the two side tables, the manifest values, and the report
lines. The core selection and the cull selection are the two adapters.

`coreflow.note_values(input_path, cards, kept, profile, core_values, bold=...)`
gives the complete field values of every kept note: the audio, the sentence
with the bold core word, the English, the optional romanization, the core
word, and the gloss. One pass writes them, so `fill_core_fields` and
`fill_bold` go away.

### 14.2 The output field set

Every mode and every language uses one of two sets. The value rule is the same
in both.

| Position | Field | Value | Reader |
| --- | --- | --- | --- |
| 1 | `Audio` | the input audio field text, `[sound:<file>]`, verbatim | `front.html` and `back.html` of the mode |
| 2 | `Sentence` | `InputCard.sentence`, the clean plain text; `fill_bold` writes the core word in bold | `back.html`, K15, G5, G17 |
| 3 | `English` | the input English field text, verbatim; the pack English for a reserve card | `back.html` |
| 4 | `Romanization` | the input field named by `profile.romanization_source`, verbatim; absent when the source is None; Cantonese writes the jyutping of `Sentence` instead | `back.html` of every mode |
| 5 | `Core` | the target form of the card (14.7) | `back.html`, K19, G17 |
| 6 | `CoreGloss` | the English gloss of `Core` (14.7) | `back.html`, K19, G17 |

| Mode | Language | Fields |
| --- | --- | --- |
| core | fr, es | `Audio, Sentence, English, Core, CoreGloss` |
| core | zh, yue | `Audio, Sentence, English, Romanization, Core, CoreGloss` |
| cull | fr, es | `Audio, Sentence, English, Core, CoreGloss` |

Rules:

- `Romanization` is the one optional field. Mandarin sets
  `romanization_source = "Sentence (Latin)"` and Cantonese sets
  `romanization_source = "Jyutping"`. The flow adds the field and copies the
  input value. A language with no romanization leaves the source None, so the
  field and the card line are absent. The LTL deck labels Yale as `Jyutping`,
  so `cantonese.core.finalize_output` writes the jyutping of the sentence over
  the copied value and reports the count.
- `Eng Audio` is not an output field, because no template reads it. The
  `Eng Audio` branch of `cantonese.core.finalize_output` and its media delete
  go away with the field.
- `freq ranks`, `Difficulty`, `NumberOfSpanishWords`, `ClozeWord`,
  `Sentence (Annotated)`, `Explanation 1`, and the other 16 Mandarin fields are
  not output fields. Section 14.8 gives the reason for `freq ranks`.
- No written note holds an empty `Audio`, `Sentence`, `English`, `Core`, or
  `CoreGloss`. A `Romanization` field can be empty. Every back template
  keeps the `{{#Romanization}}` block only when the field exists. The flow strips
  the block for a language with no romanization.

The templates of `templates/<lang>_core/` and `templates/<lang>_cull/` read the
output names:

| Template token today | Token after the change |
| --- | --- |
| `{{sound}}` | `{{Audio}}` |
| `{{French}}` | `{{Sentence}}` |
| `{{Back}}` | `{{English}}` |
| `{{SpanishSentence}}` | `{{Sentence}}` |
| `{{EnglishSentence}}` | `{{English}}` |
| `{{Sentence (Translation)}}` | `{{English}}` |
| `{{[a]}}` | `{{Audio}}` |
| `{{Front}}` (Cantonese) | `{{Sentence}}` |
| `{{Audio}}`, `{{Sentence}}`, `{{#Romanization}}…{{/Romanization}}` | no change |
| `{{Core}}`, `{{CoreGloss}}` | no change |

The core front holds `{{Audio}}` only. The cull front holds `{{Audio}}` and the
hint line. Both backs hold one unconditional line:

```html
<div class="target">{{Core}} · {{CoreGloss}}</div>
```

The `{{#Core}}` condition goes away. Every note holds both fields (K19, G17),
so a conditional would only hide a fault. The `french_cull` and `spanish_cull`
backs gain this line.

The CSS comes from `profile.card_css()`, the value of `templates/shared/card.css`
plus the language layer, as today. K14 and G15 check it.
The romanization field is one shared rule, not one rule per language. A new
language sets `romanization_source` to the input field that holds the reading.
The flow adds `Romanization` to the field set. When the source is None, the
flow removes the block from the template, because Anki rejects a template that
names a missing field. When the field exists, every back shows the line.


### 14.3 The deck title and the note type name

The rule: the deck title and the note type name are the title case of the
output file stem. The stem names the output file and the side files, and it is
the source of truth (section 14.1). ADR 0015 replaces the naming rows of ADR
0001 and ADR 0013. ADR 0019 replaces the title row of ADR 0015.

| Mode | Default output | Deck title | Note type |
| --- | --- | --- | --- |
| core | `apkg/french_core.apkg` | `French Core` | `French Core` |
| cull | `apkg/french_1000.apkg` | `French 1000` | `French 1000` |
- `--output <path>` sets the stem for every one of these names.
- The core and cull flows create their note type, so its name is the deck
  title (ADR 0013).
- The name holds the name of no input model and no note count.
- The manifest `source.deck` and `source.model` hold the same value. Gate K9 and
  gate G10 compare the output with those manifest values.

### 14.4 How the output note type is built

Two routes were possible:

| Route | Keeps | Loses |
| --- | --- | --- |
| (a) a new model in the input package, with the kept notes moved to it | the note ids, the guids, the tags, the card ids, the `due` values, every scheduling column, the `revlog`, the media map and its ids, and the input format | nothing that the user sees |
| (b) a package from scratch with `AnkiPackage.create` | the field values only | the note ids, the card ids, every scheduling column, the `revlog`, and the media ids; a cull would reset the schedule of the user deck |

The flow uses route (a). The cull removes cards from a deck that the user
studies, so the kept cards must keep their card rows and their `due` values. A
package from scratch cannot keep a card id, and genanki writes new card rows.

`anki_tools/apkg/shape.py` `write_package` does these steps, in this order, in
one package:

1. Delete the cards of every note that the run does not keep
   (`delete_cards(doomed, cleanup_audio=False)`). The media files stay until
   step 8.
2. `package.prune_extra_cards(input_model_id)`: for each note, delete every card
   except the one with the lowest `ord`, and set `ord = 0` on the card that
   stays. The French input holds two cards per note (11,856 cards for 5,928
   notes), so this step is necessary. The kept card keeps its `id`, its `due`,
   and every scheduling column.
3. `rename_deck(package, deck_name)`.
4. `new_id = adapter.note_type(package, input_model_id, shape, deck_id)`: the
   `anki2` adapter passes the deck link, and the `anki21b` adapter does not.
   Both set the standard kind. The note type takes `shape.model_name`,
   `shape.fields`, `shape.templates`, and `shape.css`.
5. `sort_index = output_fields(profile).index(OUTPUT_SENTENCE_FIELD)`. For each
   kept note, `package.update_note(note_id, output_values(profile, names, raw,
   card.sentence), new_id, sort_index)`. `names` are the input field names and
   `raw` the input field values of that note. The card `ord` of the kept card
   is already 0 from step 2, so the card belongs to the single template.
6. `package.delete_model(model_id)` for the input model and for every other
   model of the package.
7. Insert the reserve notes into `new_id` when the run holds reserve cards. The
   reserve media files enter the package with `add_media_file`, and that call
   does not write the map.
8. `package.prune_media(package.used_media())`: delete every media file that no
   note references and write the media map one time.
9. `package.save(output_path)`.

`coreflow.output_values(profile, input_names, input_values, sentence)` gives the
values in the `output_fields(profile)` order: `Audio` from `input_audio_field`,
`Sentence` from `sentence`, `English` from `input_english_field`, each extra
field from the input field of the same name, and `""` for `Core` and
`CoreGloss`. `coreflow.note_values` then writes the core word, the gloss, and
the bold sentence, so the values that `write_package` writes are complete.

`anki_tools/ranking/assemble.py` `write_run` writes the deck, the side files, the
manifest, and the report in this order:

| Mode | Steps after the selection |
| --- | --- |
| core | `write_rank_csv`, `write_package` (to `<output>.tmp.apkg`), `reorder_deck`, the card CSV, the coverage CSV, the manifest, the report |
| cull | `write_package`, the cull CSV, the coverage CSV, the manifest, the report |
The cull does not call `reorder_deck` today, and it must not: `reorder_deck`
writes `due` values from the rank CSV, and it reads only `type = 0` cards.
The cull keeps the input `due` values.

### 14.5 The AnkiPackage methods

`anki_tools/apkg/package.py` gains four methods and one changed signature.
No new dependency and no new deck builder.

`clone_model(source_model_id, name, fields, templates, css) -> str`

- Copies the source note type: the `col.models` entry for `anki2`, the
  `notetypes` config blob for `anki21b`.
- Gives the copy a new id in `random.randrange(1 << 30, 1 << 31)`, the range of
  `create`, retried while the id is in use.
- Sets the name, the ordered field names, one template per `(qfmt, afmt)`
  named `Card 1`, `Card 2`, and so on, and the CSS. For `anki2`, build each
  template dict from the first template dict of the source, with the new
  `name`, `ord`, `qfmt`, and `afmt`.
- Field config: copy the config of the first field row of the source
  (`anki21b`), as the current `add_model_field` does. For `anki2`, build each
  field dict from the first field dict of the source, with the new `name` and
  `ord`, as the current `add_model_field` does.
- `anki2`: copy the model dict of the source, so `latexPre`, `latexPost`,
  `tags`, and `vers` stay valid; then set `sortf = 1`, the `Sentence` field,
  and `req = [[0, "any", [0]]]`.
- `anki21b`: keep the notetype config of the source and replace the CSS with
  `_build_notetype_config`. Set the sort index, protobuf field 1, to the index
  of `Sentence`. The source values differ: Spanish holds 10, and Mandarin and
  Cantonese hold no field 1, which reads as 0. A kept source value points at
  `Audio` or past the last field of the new model.
- Add no note and no card.
- `raises NotImplementedError` for the `anki21` format.
- `returns` the new model id as a string.

`delete_model(model_id) -> None`

- `anki21b`: delete the `notetypes` row and the `fields` and `templates` rows of
  that id.
- `anki2`: delete the `col.models` entry.
- `raises ValueError` when one note still holds that `mid`.
- `raises NotImplementedError` for the `anki21` format.

`update_note(note_id, field_values, model_id=None, sort_index=0) -> None`

- Write the whole `flds` string with `\x1f` between the values.
- Set `mid` to `model_id` when it is not None.
- Set `mod`, `usn = -1`, `sfld` to `field_values[sort_index]` without the HTML
  tags, and `csum` to the Anki field checksum of `sfld`: `0` for an empty
  value, else the first 8 hex digits of the SHA1 digest of the stripped
  value, read as a base-16 integer.
- `raises ValueError` when the value count differs from the field count of the
  note type.
- One commit.

`prune_extra_cards(model_id) -> int`

- For each note that holds `mid = model_id`, find the card with the lowest
  `ord`, delete the other cards of that note, and set `ord = 0` on the card
  that stays.
- Deletes a card row only. It keeps the note, its media, and the scheduling
  columns of the card that stays.
- Works on `anki2`, `anki21`, and `anki21b`, because the `cards` table is the
  same in all three.
- `returns` the count of deleted cards. One commit.

`add_note_and_card` gains a `model_id: str` argument. It sets
`note.mid = int(model_id)` and `csum` from `sfld`. Today it clones the `mid`
of the first note row, and that value is wrong once the package holds a second
note type.

Format limits:

- `clone_model`, `delete_model`, and `update_note` read and write `anki2` and
  `anki21b`. An `anki21` package raises `NotImplementedError`, the limit that
  `add_model_field` has today. No `anki fix` input uses `anki21`.
  `prune_extra_cards` works on all three formats.
- The output keeps the format of the input. `anki21b` in gives `anki21b` out,
  with the zstd collection and the zstd media. `anki2` in gives `anki2` out.
  An `anki21b` media map holds no key, so `anki_tools/apkg/media.py` renumbers
  the kept media files to `0..n-1` and writes the protobuf entries in that
  order. A note field holds the media file name, not the archive name.
- `add_model_field` loses its last caller with `_write_core_model`. Delete it
  and its test. `remove_model_template` and `_trim_templates` lose their
  callers too, so delete them together.

### 14.6 The unused note types

Step 6 of 14.4 removes every note type except the new one. The French input
carries `French Sentences` and `Basic`. The Mandarin input carries
`0 2024 Sentences Deck` and `TEXT CARD`. After the run the output holds one
note type.

The order is necessary: a note type goes only after the notes that hold it are
gone or moved. Step 1 deletes the notes that the run does not keep, and step 5
moves the notes that it keeps, so step 6 can delete the source.

### 14.7 The core word and the gloss

`Core` holds the form of a target that the card teaches. `CoreGloss` holds the
English gloss of that form. Every written note holds both.

Core mode reads `CoreResult`:

1. the assigned target of the card, `result.assigned[index]`;
2. else the held target with the lowest position, `min(candidate.targets,
   key=position)`;
3. else the card holds no core word.

The gloss is `CoreTarget.gloss`: the word-table gloss of a tier S target, or
the first sense of the sense row of a tier F target.

Cull mode reads `SelectionResult.card_targets[index]`, the target lemmas of
the card in rank order. This value exists today, so the rule adds no selection
logic.

1. Take the first lemma of `card_targets[index]` that has a row in the sense
   file `profile.sense_path` (`read_senses`, section 5.1). That lemma is the
   `Core` word.
2. Its `CoreGloss` is the first sense of that row (`SenseWords.first`).
3. When no lemma of the card has a sense row, `Core` is the first lemma and
   `CoreGloss` is empty.
4. A reserve card takes `Core = ReserveCard.target` and
   `CoreGloss = ReserveCard.gloss` from the reserve pack, written by
   `reserve_fields` at insert time.

`coreflow.fill_core_fields(output_path, cards, values)` writes the pairs.
`values` maps an input index to a `(word, gloss)` pair. It counts the notes
that it fills.

A kept card with no core word leaves the deck:

- `run_core` and `run_cull` compute the pair of every kept card before the
  write.
- A kept card with an empty word is not written. Its CSV row holds `kept = 0`
  and `reason = no_core`, and the manifest counts it in
  `counts.dropped_no_core`.
- `write_rank_csv`, `kept_order_hash`, `write_cull_csv`, and `counts.cards` use
  the reduced list.
- The French core run drops 27 cards of 300 today, because a tail card brings
  new lemmas and teaches no core target. The rule makes the 273 cards that
  stay compliant.

Reason for one rule and not a per-language tail switch: it covers four
languages and both modes, and it keeps the 67 tail cards that do hold a
target.

Data requirement: the sense file must hold one row for every target of the
cull list, 300 rows at the default `--target-words`. G17 fails until that
file is complete.

### 14.8 The frequency rank

Decision: the output has no frequency field.

- The source and the scale of the input `freq ranks` value are unknown
  (`specs/french-1000-sentence-cull.md`, open question 5).
- The four inputs disagree. French holds `freq ranks`, Spanish holds
  `Difficulty` and `NumberOfSpanishWords`, and the Mandarin and Cantonese
  inputs hold no rank field. A card value with no one meaning is noise.
- The rank is a selection value. It stays in the `rank` column of the coverage
  CSV and in the rank CSV, where the build and the review read it.

A later build that wants the rank on the card adds one field to
`output_fields` and one line to the back template of that language.

### 14.9 Gates

The gate numbers stay stable. `anki_tools/ranking/gates.py` holds the runner,
the report, and the four shared gates. The core file reader and the cull file
reader are the two adapters. The report is
`anki_tools.sentence_pack.audit.print_audit`, so the two modes print one table.

| Gate | Mode | Check |
| --- | --- | --- |
| K9 `gate_identity` | core | the deck title is `manifest.params.name`, the only note type name is `manifest.source.model`, and the two are equal |
| K14 `gate_templates` | core | the templates equal `profile.templates` and the CSS equals `profile.card_css()`, on the only note type |
| K15 `gate_bold` | core | every kept note holds a bold core word in the `Sentence` field |
| K19 `gate_core_word` (renamed from `gate_gloss`) | core | every note holds a non-empty `Core`, a non-empty `CoreGloss`, and a bold core word; the gloss holds no comma, and the gloss is a coverage gloss of that word |
| K21 `gate_output_model` (new) | core | the package holds one note type named `manifest.source.model`; its field names equal `output_fields(profile)` in order; it holds one card template; every note holds one card with `ord = 0` |
| G10 `gate_identity` | cull | as K9 |
| G15 `gate_card_templates` | cull | the templates of `manifest.source.model` and the CSS classes, as today, on the only note type |
| G16 `gate_output_model` (new) | cull | as K21 |
| G17 `gate_core_word` (new) | cull | as K19, with the coverage CSV of the cull as the gloss source |

- K19 keeps its number, so an older gate table still maps. Its name changes
  because it now counts a missing core word, not only a wrong gloss.
- `CullFiles.output()` gains `fields`, and `write_coverage_csv` gains a
  `gloss` column, so G17 can compare a note gloss with a coverage gloss:
  `COVERAGE_COLUMNS = ("word", "rank", "count", "status", "gloss")`.
- `write_coverage_csv(targets, selection, path, glosses: Mapping[str, str])`
  reads the gloss of a lemma from the sense file.
- `coreflow.coverage_glosses(rows)` gives the word-to-gloss map of a coverage
  CSV. K19 and G17 both read it, so a gloss is spelled one way. It replaces
  `coregates._target_glosses`.

### 14.10 Tests

Add one small test for each new seam. Do not download a file.

| File | Test | Behaviour |
| --- | --- | --- |
| `tests/test_apkg_package.py` | `test_clone_model_copies_the_shape_and_takes_a_new_id` | a clone of an `anki2` model and of an `anki21b` model holds the new name, the new fields, one template, and the CSS; the source keeps its own fields |
| `tests/test_apkg_package.py` | `test_delete_model_refuses_a_note_that_still_uses_it` | `ValueError`, and the model is still there |
| `tests/test_apkg_package.py` | `test_update_note_writes_the_fields_and_the_model` | `flds`, `mid`, `sfld`, and `csum` change in one call |
| `tests/test_apkg_package.py` | `test_prune_extra_cards_keeps_one_card_per_note` | a note with two cards keeps the lowest `ord`, that card has `ord = 0`, and the returned count is 1 |
| `tests/test_core_deck.py` | `test_the_run_writes_one_note_type_with_the_output_fields` | the tiny deck gives one note type named after the output stem, the five fields in order, one template, and one card per note |
| `tests/test_core_deck.py` | `test_every_kept_note_holds_a_core_word_and_a_gloss` | K19 accepts the tiny deck and rejects a note with an empty `Core` |
| `tests/test_core_deck.py` | `test_every_kept_note_holds_a_bold_core_word` | every kept note holds `<b>` in the `Sentence` field |
| `tests/test_core_deck.py` | `test_a_kept_card_without_a_core_word_leaves_the_deck` | the card is not written, the CSV row holds `no_core`, and the manifest counts it |
| `tests/test_core_deck.py` | `test_the_kept_card_keeps_its_note_id_and_due` | the note id and the card `due` of the input row survive |
| `tests/test_core_deck.py` | `test_gate_output_model_fails_on_a_spare_field` | K21 reports the extra field name |
| `tests/test_cull_shared.py` | `test_the_cull_writes_the_output_note_type` | one note type named after the output stem, with the five fields, one card per note, and the input note type gone |
| `tests/test_cull_shared.py` | `test_the_cull_fills_the_core_word_and_the_gloss` | a frequency card takes its lemma and the first sense; a reserve card takes its pack target and gloss |
| `tests/test_cull_shared.py` | `test_the_cull_keeps_the_due_of_a_kept_card` | a `due` of 42 on a kept card stays 42, and the dropped card is gone |
| `tests/test_identity.py` | `test_the_title_and_the_note_type_take_the_output_stem` | the stem gives the deck title and the note type name |
| `tests/test_assemble.py` | `test_the_assembler_writes_one_report_and_one_identity` | both modes write one deck, two side CSVs, one manifest, and one report |
| `tests/test_output_shape.py` | `test_the_shape_holds_on_an_anki2_fixture`, `test_the_shape_holds_on_an_anki21b_fixture` | one deck, one standard note type with the stated fields, and one card per note, in both formats |
| `tests/test_gates.py` | `test_the_runner_reports_a_gate_that_cannot_read_a_file`, `test_the_report_prints_one_table` | the runner keeps the table after an error, and the report prints one table |
| `tests/test_language_deck_profile.py` | `test_the_two_mode_views_agree_on_the_language_facts` | the core view and the cull view of French and Spanish agree on the input names, the romanization, the sense file, and the CSS |
| `tests/test_cantonese_core.py` | `test_the_jyutping_field_survives_the_new_model` | `romanization_source = "Jyutping"` puts `Romanization` at position 4 with the input value |
| `tests/test_mandarin_core.py` | `test_the_templates_read_the_output_audio_field` | no template token names `[a]` |
| `tests/test_mandarin_core.py` | `test_output_values_copies_the_pinyin_from_the_latin_field` | `Romanization` comes from `Sentence (Latin)` |

The tests that must change:

| File | Test or group | Change |
| --- | --- | --- |
| `tests/test_core_deck.py` | `TestCardDesign` | the field tokens become `{{Audio}}`, `{{Sentence}}`, `{{English}}`, `{{Core}}`, and `{{CoreGloss}}` |
| `tests/test_core_deck.py` | `test_the_model_gets_the_core_field_and_templates` | assert the one output note type, its field set, and the `Core` values |
| `tests/test_core_deck.py` | `test_the_gloss_gate_*` | the gate name, and the two failures: a missing gloss and a wrong gloss |
| `tests/test_core_deck.py` | `test_adding_the_field_twice_changes_nothing` | delete with `add_model_field` |
| `tests/test_cull_shared.py` | `_profile()` | the new field names, `sense_path`, `romanization_source`; no `output_model_name` or `default_deck_name` |
| `tests/test_cull_shared.py` | `test_a_second_profile_drives_every_output_value` | assert one note type, the five fields, and the `Core` values |
| `tests/test_cull_shared.py` | `test_coverage_csv_statuses` | the extra `gloss` column |
| `tests/test_french_cull.py` | the deck-writer calls, the G10 and G15 tests | the output stem as the deck and note type name; the new G16 and G17 |
| `tests/test_apkg_package.py` | the `add_model_field` tests | delete with the method |
| `tests/test_spanish_cull.py` | `test_the_reserve_fields_fit_the_spanish_sentence_model` | five values in the output order |
| `tests/test_mandarin_core.py` | `test_templates_name_the_audio_field` | `{{Audio}}`, not `{{[a]}}` |

### 14.11 Migration and risk

- The five committed decks change. The default output names are now
  `apkg/french_core.apkg`, `apkg/mandarin_core.apkg`,
  `apkg/cantonese_core.apkg`, `apkg/spanish_core.apkg`, and
  `apkg/french_1000.apkg`. Rebuild each deck in one run and commit the deck
  with its side files and its manifest. The Spanish cull has no committed deck,
  so its rebuild needs no commit.
- The side-file stems equal the output file stem, so a deck that changed name
  also changed the stem of its `.core.csv`, `.cull.csv`, `.coverage.csv`,
  `.manifest.json`, and `.rank.csv` files. Delete the stale side files of a
  deck that the run does not rebuild.
- The side files change in three ways: `manifest.source.model` holds the output
  name and `manifest.source.input_model` the input name, the cull coverage CSV
  gains the `gloss` column, and each manifest gains `counts.dropped_no_core`.
- The French core deck falls from 300 cards to 273, because 27 tail cards hold
  no core target (14.7). Replace the numbers of section 14.3 of
  `specs/core-word-small-deck.md` with the measured values.
- Anki matches a deck by title and a note type by name. The deck and the note
  type both change name to the title case of the stem (ADR 0019), so a user who
  imports the new deck beside the old deck gets duplicate cards. The release
  note must say: to keep the review history, rename the old deck to the new
  title in Anki before the import, then delete the old note type in Tools >
  Manage Note Types. Otherwise delete the old deck and its note type, or import
  into a fresh profile.
- A note keeps its guid, so Anki matches the note and changes its note type on
  import. A cull import does not delete the notes that the cull drops. That
  limit holds today.
- A sense file that does not cover the cull target list fails G17. That is a
  data task, and the gate message names the row count.
- `docs/core-deck-flow.md` and `docs/cull-deck-flow.md` name the input model
  and the two added fields. The implementer updates the profile field list,
  the writer paragraph, and the gate table of each document.
