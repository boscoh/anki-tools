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
  `vocab/english-spanish.csv`.
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

The scan is already shared in part: the French cull and core reuse
`load_input_deck`, `tokenize_deck`, `load_frequency`, and `frequency_ranks`.
The Cantonese selection scans inside its own module. The Spanish and Mandarin
rank modes do not scan at all today.

## 4. Core list

Build the target list in one order:

1. The reviewed Swadesh forms, in table order.
2. The tier F words, in blended frequency order, up to `--words`.

Apply the same rules as the French core: the content share, the minimum deck
count, the name list, and the loan word list. Cut to the requested count.

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
uses the cull cap and the Swadesh reserve. A stratified preset for Mandarin and
Spanish remains a separate policy and is not part of this spec.

## 7. Order and card

- Order the cards by core position, or by difficulty when the language chooses
  that key.
- The card back shows the sentence, the English, `Core`, and `CoreGloss`. The
  assigned target is bold.
- A core deck changes the note model. A breadth deck keeps the input model.
  That stays a mode choice.

## 8. Data requirements

| Language | Word table                    | Sense table                    | Tier F rows | State      |
| -------- | ----------------------------- | ------------------------------ | ----------- | ---------- |
| fr       | `vocab/english-french.csv`    | `vocab/french_core_senses.tsv` | 51          | ready      |
| es       | `vocab/english-spanish.csv`   | none                           | 0           | needs rows |
| zh       | `vocab/english-mandarin.csv`  | none                           | 0           | needs rows |
| yue      | `vocab/english-cantonese.csv` | none                           | 0           | needs rows |

Every language also needs a deck with a usable English field. The French deck
has one. The Spanish deck at `apkg/Spanish 9000 sentences with native audio.apkg`
has one. Check the other decks before a build.

## 9. Interface

Keep the reduced surface of `anki fix fr`:

```text
--core --cards --words --examples --tail --output --check
```

The language command selects the tokenizer, word table, frequency source, and
sense file from the profile. No new per-language flags.

## 10. Gates and acceptance

Reuse gates K1 to K19 with the language file paths. A language is accepted when:

- every K gate passes;
- `checked` equals the targets with a sense row plus every tier S target;
- `unchecked` is within the limit;
- every `CoreGloss` equals the target's first sense;
- the manifest records the kept order and the hashes.

## 11. Spanish pilot

Input: `apkg/Spanish 9000 sentences with native audio.apkg`.

| Step | Work                                                             | Acceptance                                    |
| ---- | ---------------------------------------------------------------- | --------------------------------------------- |
| S1   | scan and tokenize the deck; compare lemma counts with the ranker | counts agree with `fix es`                    |
| S2   | Swadesh-only core list, 0 tier F rows, 1 to 2 examples           | K1 to K19 pass with 0 unchecked               |
| S3   | author 50 tier F rows                                            | every tier F target is `checked`              |
| S4   | build the 300-card deck                                          | K1 to K19 pass; coverage CSV agrees           |
| S5   | compare with the current `fix es --max-sentences` output         | report coverage, outside words, sense rejects |

Step S2 proves the flow without any new data. Step S3 proves the sense contract.
Step S5 decides whether the core guarantee beats the current complexity spread.

Open items:

- the English field name of the Spanish deck;
- the audio field and the media check;
- the `es_core_news_sm` model install;
- the tier F word list for Spanish (blended frequency, content share, names).

## 12. Risks

- Data authoring is the long pole. The Spanish pilot needs 50 rows, and a full
  deck needs more.
- The English translation limits the sense guard. A weak English field rejects
  good cards or passes bad ones.
- Mandarin needs a word boundary and a segmentation check before a word-level
  sense test is meaningful.
- Adding the guard to the existing cull changes its deck. Measure before the
  change, and record a new manifest.
- The false-friend column is empty in French. The mechanism needs a test with
  real data.
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
| M6        | decision on Mandarin: word core or keep complexity spread  |
| M7        | Cantonese core deck on the same contract                   |
