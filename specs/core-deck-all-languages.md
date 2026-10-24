# Spec: Core Deck Flow for All Languages

| Field   | Value                                                          |
| ------- | -------------------------------------------------------------- |
| Status  | draft                                                          |
| Date    | 2026-10-24                                                     |
| Owner   | anki-tools                                                     |
| Input   | a sentence deck with a text field, an English field, and audio |
| Related | `specs/core-word-small-deck.md` (French core deck)             |
| Related | `specs/french-1000-sentence-cull.md` (breadth cull)            |
| Related | `specs/yue-sentence-selection.md` (Cantonese cover)            |
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
- It does not change the current cull or core decks. Any change to a published
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
| zh       | jieba                   | `wordfreq` + characters | yes       |
| yue      | Cifu + jieba            | spoken + written Cifu   | yes       |

The frequency candidate builder is shared: `anki_tools/ranking/corelist.py`
holds `build_targets`. The scan helpers still live in the French package:
`load_input_deck`, `tokenize_deck`, `load_frequency`, and `frequency_ranks`.
The Cantonese selection scans inside its own module. The Spanish and Mandarin
rank modes do not scan at all today.

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
  assigned target is bold.
- A core deck changes the note model. A breadth deck keeps the input model.
  That stays a mode choice.

## 8. Data requirements

| Language | Word table                    | Sense table                     | Tier F rows | State      |
| -------- | ----------------------------- | ------------------------------- | ----------- | ---------- |
| fr       | `vocab/english-french.csv`    | `vocab/french_core_senses.tsv`  | 51          | ready      |
| es       | `vocab/spanish_words.csv`     | `vocab/spanish_core_senses.tsv` | 50          | ready      |
| zh       | `vocab/english-mandarin.csv`  | none                            | 0           | needs rows |
| yue      | `vocab/english-cantonese.csv` | none                            | 0           | needs rows |

Every language also needs a deck with a usable English field. The French deck
has one. The Spanish deck at `apkg/Spanish 9000 sentences with native audio.apkg`
has one. Check the other decks before a build.

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
--core --cards --words --examples --tail --output --check
```

The language command selects the tokenizer, word table, frequency source, and
sense file from the profile. No new per-language flags.

## 10. Gates and acceptance

Reuse gates K1 to K20 with the language file paths. A language is accepted when:

- every K gate passes;
- `checked` equals the targets with a sense row plus every tier S target;
- `unchecked` is within the limit;
- `unranked` is within the limit;
- every `CoreGloss` equals the target's first sense;
- the manifest records the kept order and the hashes.

## 11. Spanish pilot

Input: `apkg/Spanish 9000 sentences with native audio.apkg`.

| Step | Work                                                             | Acceptance                                    |
| ---- | ---------------------------------------------------------------- | --------------------------------------------- |
| S1   | scan and tokenize the deck; compare lemma counts with the ranker | counts agree with `fix es`                    |
| S2   | Swadesh-only core list, 0 tier F rows, 1 to 2 examples           | K1 to K19 pass with 0 unchecked               |
| S3   | build the frequency cache (8.2), then author 50 tier F rows      | every tier F target is `checked`              |
| S4   | build the 300-card deck                                          | K1 to K19 pass; coverage CSV agrees           |
| S5   | compare with the current `fix es --max-sentences` output         | report coverage, outside words, sense rejects |

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

## 12. Risks

- Data authoring is the long pole. The Spanish pilot needs 50 rows, and a full
  deck needs more.
- The English translation limits the sense guard. A weak English field rejects
  good cards or passes bad ones.
- Mandarin needs a word boundary and a segmentation check before a word-level
  sense test is meaningful.
- Adding the guard to the existing cull changes its deck. Measure before the
  change, and record a new manifest.
- The false-friend column is empty in French. Spanish sets one row: `solo`
  rejects `only` and `just`, so the target keeps the `alone` sense. The
  mechanism has no test beyond that row.
- A shared flow must not become a dispatcher. The policy stays a per-language
  preset, not a branch inside the scan.

## 13. Milestones

| Milestone | Work                                                       |
| --------- | ---------------------------------------------------------- |
| M1        | accept this spec                                           |
| M2        | shared target record and scan with language hooks          |
| M3        | French core list unchanged on the new record               |
| M4        | Spanish Swadesh-only deck, gates pass                      |
| M5        | Spanish tier F sense rows and the 300-card deck            |
| M6        | Mandarin: word core; complexity spread retired             |
| M7        | Cantonese core deck on the same contract                   |
