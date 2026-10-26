# Spec: Cantonese Sentence Selection with Spoken Frequency

| Field   | Value                                                                     |
| ------- | ------------------------------------------------------------------------- |
| Status  | M0 to M5 done; M6 (Tatoeba pool) open                                     |
| Date    | 2026-10-19                                                                |
| Owner   | anki-tools                                                                |
| Issues  | epic `anki-g40`; milestones M0 to M6 (section 13); `anki-afn`, `anki-58i` |
| Related | `specs/mandarin-swadesh-sentence-pack.md`                                 |

## Contents

1. [Summary](#1-summary)
2. [Goals and scope](#2-goals-and-scope)
3. [Expected results](#3-expected-results)
4. [Pipeline](#4-pipeline)
5. [Frequency list](#5-frequency-list)
6. [Tokenizer](#6-tokenizer)
7. [Sentence filter](#7-sentence-filter)
8. [Selection](#8-selection)
9. [Interface](#9-interface)
10. [Data sources and licenses](#10-data-sources-and-licenses)
11. [Quality gates and tests](#11-quality-gates-and-tests)
12. [Risks and open questions](#12-risks-and-open-questions)
13. [Milestones](#13-milestones)
14. [Appendix: evidence](#14-appendix-evidence)

## 1. Summary

`anki fix yue deck.apkg` ranks and reorders a Cantonese deck. It keeps every
card, and its frequency data comes from written news text. This spec makes it
keep fewer cards, with the widest vocabulary, weighted to spoken Cantonese.

### 1.1 Three changes

| Change                | What it does                                                                                  | Section |
| --------------------- | --------------------------------------------------------------------------------------------- | ------- |
| Spoken frequency list | Blends Cifu spoken counts with written counts; built on demand                                | 5       |
| Sentence filter       | Removes fragments, mixed script, repeats, names, and hard sentences                                  | 7       |
| Diversity selection   | Greedy cover of the most frequent words; a limit on uses of one word; easy-to-hard card order | 8       |

### 1.2 Result at a glance

The defaults include the word-use cap (8.5). Each cell shows the new value,
then the value of the current ranking at the same card count.

| Input                           | Sentences in | Cards | Words         | `spoken_mass` |
| ------------------------------- | ------------ | ----- | ------------- | ------------- |
| LTL deck                        | 999          | 641   | 948 / 764     | 66.8 / 64.0   |
| LTL deck, cap 300               | 999          | 300   | 688 / 410     | 66.1 / 57.4   |
| Tatoeba Cantonese pool          | 20,853       | 1,173 | 1,974 / 1,172 | 79.6 / 67.6   |
| Tatoeba Cantonese pool, cap 500 | 20,853       | 500   | 1,330 / 572   | 77.2 / 60.5   |

`spoken_mass` is the share of spoken Cantonese tokens that the kept words
cover (2.2). Section 3 has the full tables.

### 1.3 Key decisions

| Decision                | Choice                                                                    | Evidence     |
| ----------------------- | ------------------------------------------------------------------------- | ------------ |
| Spoken frequency source | Cifu `SpokenAdult`, blended with written counts at 0.7                    | 5.2, 14.1    |
| Tokenizer               | jieba with a Cifu dictionary                                              | 6, 14.2      |
| Sentence length         | 3 to 12 Han characters                                                    | 7, 14.3      |
| Target words            | The 1500 most frequent words that the pool contains                       | 8.1          |
| Selection method        | Greedy cover, 2 cards per word, weight `1/log2(rank+1)`                   | 8.2, 14.5    |
| Uses of one word        | A word outside the 100 most frequent words is in at most 8 kept sentences | 8.5, 14.6    |
| Card order              | Easiest hardest-word first                                                | 8.4          |
| Compatibility           | Rank mode is the default and reproduces today's output                    | 9.2          |

### 1.4 Open blockers

- Cifu is usable for local builds. Its license for a public build is not
  clear (open question 1, M0).

## 2. Goals and scope

### 2.1 Goals

1. Keep as much vocabulary diversity as the input allows, with the fewest
   cards.
2. Weight that diversity to the most frequent words of spoken Cantonese, not
   written Cantonese.
3. Keep each sentence short, complete, and unlike the other kept sentences.
4. Do not use one word in too many sentences. A word outside the 100 most
   frequent words appears in at most 8 kept sentences (8.5).

### 2.2 Measures

| Measure                        | Meaning                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `cards`                        | Sentences kept                                                                                           |
| `words`                        | Distinct words in the kept sentences                                                                     |
| `top500`, `top1000`, `top1500` | How many of the 500, 1000, or 1500 most frequent words of the blend list (5.2) are in the kept sentences |
| `spoken_mass`                  | Share of the 306,339 Cifu spoken tokens that belong to a word in the kept sentences                      |
| `max_uses`                     | The most kept sentences that one word outside the 100 most frequent words appears in (8.5)               |

`spoken_mass` estimates the coverage of running speech. A learner who knows
every kept word recognizes this share of spoken tokens. Cifu gives the
tokenizer, the rank list, and the test, so the absolute values are high. Use
`spoken_mass` to compare methods. Do not use it as a claim about real speech.

### 2.3 In scope

- A spoken frequency list, built from public corpora and cached locally.
- A Cantonese-aware word tokenizer.
- New options for `anki fix yue`, and a report of what the filter removed.
- The blend of the frequency list, which runs at the first `anki fix yue`.
- The Tatoeba Cantonese export as a second input (M6). Section 3.4 gives the
  numbers.

### 2.4 Out of scope

- Audio, TTS, and new card templates.
- The command and card fields for the Tatoeba pool (a later spec).
- Changes to `fix zh`, `fix fr`, or `fix es`.
- Swadesh targets, which the Mandarin pack uses. This spec uses the most
  frequent words (open question 4).

## 3. Expected results

All numbers come from a prototype of sections 5 to 8 (3.5). The test inputs
are:

- **LTL deck:** `apkg/LTL Cantonese Deck Level 1 - Short.apkg`, 999 sentences.
- **Tatoeba pool:** the Cantonese export of 2026-10-17, 20,853 sentences.

The default settings are:

- `--per-word 2`, `--top-words 1500`, and no `--max-sentences`.
- `--max-uses 8`, with the 100 most frequent words exempt (8.5).
- `--known-top 2000`, `--max-unknown 2`, and 3 to 12 Han characters.
- Similarity 0.67.

### 3.1 Cards at the defaults

| Input                  | Sentences in | Pass Y1 to Y8 | Cards kept | Words | top500 | top1000 | top1500 | `spoken_mass` |
| ---------------------- | ------------ | ------------- | ---------- | ----- | ------ | ------- | ------- | ------------- |
| LTL deck               | 999          | 858           | 641        | 948   | 223    | 325     | 385     | 66.8%         |
| Tatoeba Cantonese pool | 20,853       | 4,256         | 1,173      | 1,974 | 356    | 580     | 782     | 79.6%         |

The implementation reproduces these values on 2026-10-19: 641 cards, 948
words, top500 223, top1000 325, top1500 385, `spoken_mass` 66.8. The card count
is 6 below 647, inside the 5 percent of gate G11. Before rule Y8 the run gave
650 cards and 962 words. With `--max-sentences 300`: 688 words, 217 top500
words, `spoken_mass` 66.1 (G10).

What the selection removes, against the passed pool:

| Input   | Cards removed        | Words lost     | Top 1500 words lost | `spoken_mass` lost |
| ------- | -------------------- | -------------- | ------------------- | ------------------ |
| LTL     | 217 of 858 (25%)     | 35 of 983      | 1                   | 0.0 points         |
| Tatoeba | 3,083 of 4,256 (72%) | 1,504 of 3,478 | 9                   | 0.9 points         |

Most lost words are rare, outside the target set.

What the rules of section 7 remove, before the selection runs:

| Input   | Effect of rules Y1 to Y8                                                  |
| ------- | ------------------------------------------------------------------------- |
| LTL     | 13 of 236 top-500 words are lost; `spoken_mass` falls from 67.7% to 66.8% |
| Tatoeba | top500 falls from 457 to 357; `spoken_mass` falls from 95.8% to 80.5%     |

Rule Y1 causes most of the Tatoeba loss. It rejects 15,445 of
20,853 sentences, because only 26.3% have 12 Han characters or fewer.

What the word-use cap changes, against the same run without it (14.6):

| Input             | Cards | Words | `spoken_mass` | `max_uses` |
| ----------------- | ----- | ----- | ------------- | ---------- |
| LTL, no cap       | 674   | 970   | 66.8%         | 22         |
| LTL, cap of 8     | 641   | 948   | 66.8%         | 8          |
| Tatoeba, no cap   | 1,307 | 2,058 | 79.9%         | 37         |
| Tatoeba, cap of 8 | 1,173 | 1,974 | 79.6%         | 8          |

### 3.2 Cards for each setting

| Input   | `--top-words` | `--per-word 1` | `--per-word 2` | `--per-word 3` |
| ------- | ------------- | -------------- | -------------- | -------------- |
| LTL     | 500           | 295            | 409            | 473            |
| LTL     | 1000 or more  | 536            | 647            | 689            |
| Tatoeba | 500           | 263            | 458            | 617            |
| Tatoeba | 1000          | 552            | 857            | 1,081          |
| Tatoeba | 1500          | 795            | 1,173          | 1,410          |
| Tatoeba | 2000          | 1,039          | 1,419          | 1,647          |

- These LTL cells come from the prototype before rule Y8. Y8 removes 14
  sentences, so the implementation keeps 641 cards at the defaults (3.2, 3.1).
- The LTL rows for 1000 or more are equal, because the passed deck has only
  973 ranked words.
- At `--per-word 1`, each card carries 1.82 (LTL) or 1.89 (Tatoeba) target words.
- A second card for each word is often not possible. 327 of the 973 LTL
  target words and 897 of the 1,500 Tatoeba target words reach 2 cards.

### 3.3 Cards with a cap, against the current ranking

`--max-sentences` stops the selection early. The current ranking is the first
N sentences of the passed pool, in the order of the current ranker. Each cell
shows the new value, then the current value.

| Input   | Cards | Words         | top500    | top1000   | `spoken_mass` |
| ------- | ----- | ------------- | --------- | --------- | ------------- |
| LTL     | 300   | 688 / 410     | 217 / 147 | 306 / 195 | 66.1 / 57.4   |
| LTL     | 500   | 887 / 618     | 223 / 185 | 325 / 257 | 66.8 / 62.5   |
| LTL     | 641   | 948 / 764     | 223 / 199 | 325 / 284 | 66.8 / 64.0   |
| Tatoeba | 300   | 977 / 357     | 316 / 138 | 476 / 179 | 74.8 / 54.8   |
| Tatoeba | 500   | 1,330 / 572   | 335 / 178 | 527 / 244 | 77.2 / 60.5   |
| Tatoeba | 1,000 | 1,894 / 1,026 | 356 / 237 | 580 / 355 | 79.6 / 66.5   |
| Tatoeba | 1,173 | 1,974 / 1,172 | 356 / 247 | 580 / 374 | 79.6 / 67.6   |

For the LTL deck, the current order is the real `rank.csv` order. For
Tatoeba, it uses the present frequency score (Apple Daily) and complexity
score, without the similarity penalty. It is an approximation.

How to read the table:

- At 300 cards, the LTL selection keeps 68% more distinct words and 8.7 points more
  `spoken_mass`.
- At 500 cards, the Tatoeba selection keeps 2.3 times the distinct words and 16.7
  points more `spoken_mass`.
- The curve flattens. Tatoeba gains 2.4 points from 500 to 1,000 cards, and
  0.0 points from 1,000 to 1,173 cards.
- A cap of 500 gives 97% of the final Tatoeba `spoken_mass` (77.2 of 79.6)
  with 43% of the cards.
- The LTL deck needs no cap. 500 cards give 66.8%, the same as the full
  result.

### 3.4 The Tatoeba Cantonese pool

| Measure                                   | Value                                       |
| ----------------------------------------- | ------------------------------------------- |
| Cantonese sentences                       | 20,853                                      |
| Sentences with 12 Han characters or fewer | 26.3%                                       |
| Sentences that pass Y1 to Y7              | 4,256                                       |
| Sentences with an English link            | 7,038                                       |
| Passed sentences with an English link     | 3,644                                       |
| Sentences with native audio               | 1,784                                       |
| Audio license (1,781 of 1,784 rows)       | CC BY 4.0                                   |
| Default cards with an English link        | 995 of 1,173                                |
| Default cards with native audio           | 349 of 1,173                                |
| Cards when the pool needs an English link | 1,154 (`spoken_mass` 79.0%, 403 with audio) |

Effect of a longer limit (`--max-han`) on the Tatoeba run:

| `--max-han` | Pass pool | Cards | Words | `spoken_mass` |
| ----------- | --------- | ----- | ----- | ------------- |
| 12          | 4,256     | 1,173 | 1,974 | 79.6%         |
| 14          | 4,595     | 1,152 | 2,025 | 80.1%         |
| 16          | 4,760     | 1,151 | 2,037 | 80.3%         |

A limit of 16 gains 0.7 points and gives longer sentences. Keep 12.

### 3.5 How the numbers were made

A prototype of sections 5 to 8 made all numbers in this section. The
prototype is not in the repository. It used:

- The Cifu file and the Apple Daily file of 2026-10-19.
- The tokenizer of section 6, and the blend of 5.2 with `w = 0.7`.
- The override `你好` = `多謝`, and rules Y1 to Y7.
- The word-use cap of 8.5, with a limit of 8 and the top 100 words exempt.
- Tatoeba data from `downloads.tatoeba.org/exports/per_language/yue/`.

A count can differ by one from a later table because of a tie in the sort.
Gates G10, G11, and G12 and milestone M4 turn these numbers into tests.

Rule Y8 came after the prototype. The LTL cells in 3.1, 3.3, 14.3, and 14.4
include its effect. The LTL cells in 3.2, 14.5, and 14.6 predate it.

## 4. Pipeline

| Stage | Name           | Input              | Output                                        |
| ----- | -------------- | ------------------ | --------------------------------------------- |
| Y0    | Fetch          | network            | cached Cifu and Apple Daily files             |
| Y1    | Frequency list | Y0, override files | a frequency list in memory                     |
| Y2    | Tokenize       | Y1, deck sentences | tokens per sentence                           |
| Y3    | Filter         | Y2                 | passed and rejected sentences, reasons        |
| Y4    | Select         | Y3, Y1             | kept sentences in learning order              |
| Y5    | Write          | Y4, original deck  | `_reordered.apkg`, `.rank.csv`, `.select.csv` |

- `anki fix yue` runs Y0 to Y5. It builds the frequency list when no TSV exists.
- Each stage gives the same output for the same input.

## 5. Frequency list

### 5.1 Row fields

The list is built in memory and no file is written. It has one row per word,
sorted by `rank`.

| Column         | Type   | Example | Meaning                                          |
| -------------- | ------ | ------- | ------------------------------------------------ |
| `word`         | string | 係      | Traditional form, after the variant map (5.3)    |
| `jyutping`     | string | hai6    | Cifu reading; empty when the word is not in Cifu |
| `spoken_count` | int    | 15318   | Cifu `SpokenAdult`                               |
| `spoken_pm`    | float  | 50003.4 | Spoken count per million tokens                  |
| `written_pm`   | float  | 12164.2 | Mean of the Cifu and Apple Daily written values  |
| `blend_pm`     | float  | 38651.6 | Value from 5.2                                   |
| `rank`         | int    | 1       | Rank by `blend_pm`, 1 is most frequent           |
| `source`       | string | cifu+ad | `cifu+ad`, `cifu`, `ad`, or `override`           |

### 5.2 Blend

```text
written_pm = mean of (Cifu written count per million, Apple Daily count per million)
             over the sources that have the word
blend_pm   = w * spoken_pm + (1 - w) * written_pm      if spoken_count >= 5
blend_pm   = written_pm                                 otherwise
```

- `w` is `SPOKEN_WEIGHT`, 0.7. The module sets it; no option changes it.
- The spoken sample is small, so a count below 5 uses the written term only.

The blend fixes most spoken words (14.1). It does not fix `你好`, because
HKCanCor writes it as two tokens. The override file (5.4) covers this case.

The blend changes the order of the old list. On the LTL deck, 603 of 998
sentences move by more than 100 places (M2).

### 5.3 Variant map

The written lists use news-standard characters. Spoken text uses common
variants (`anki-58i`).

- File: `vocab/cantonese_variants.tsv`, columns `variant` and `standard`.
- Use: apply the map to a token for the frequency lookup only. Never change
  the card text.
- Direction: pick the form that has the larger count in Cifu. Record the
  result in the file (M1).

| Variant | Standard | Meaning |
| ------- | -------- | ------- |
| 咁      | 噉       | so      |
| 枱      | 檯       | table   |

### 5.4 Overrides

- File: `vocab/cantonese_frequency_overrides.tsv`, columns `word`, `rank_like`,
  and `reason`.
  `reason`.
- Effect: a row gives the word the blend value of the word in `rank_like`.
- Example row: `你好`, `多謝`, `HKCanCor splits the greeting`.
- Start with the greetings `你好`, `晚安`, and `拜拜`. Review each row by hand.
- A word from this file has `source=override`.

## 6. Tokenizer

Build one `jieba.Tokenizer` instance for Cantonese. Do not change the global
jieba dictionary, because the Mandarin code uses it.

| Setting    | Value                                                                    |
| ---------- | ------------------------------------------------------------------------ |
| Dictionary | Every Cifu word with a Han character and a count above 0                 |
| Weight     | `spoken_count * 40 + written_count * 2` (a spoken count counts 20 times) |
| Overrides  | Words from 5.4, at the same weight as the top word                       |
| Gate       | G2: 92% of Han tokens on the LTL deck have a spoken count                |

Change the weights only when the G2 result improves. The test is in 14.2:
this tokenizer scores 92.3%, against 91.1% for jieba default and 85.4% for
PyCantonese.

## 7. Sentence filter

A sentence passes when it passes every rule. Apply the rules in order. Store
the first failed rule as the `reason`.

| Rule | Condition                                                         | Change from Mandarin F-rule        |
| ---- | ----------------------------------------------------------------- | ---------------------------------- |
| Y1   | 3 to 12 Han characters                                            | F2 was 4 to 10                     |
| Y2   | No Latin letter and no digit                                      | F3, same                           |
| Y3   | Not a repeated pattern (`is_repeated_zh_sentence`)                | F4, same                           |
| Y4   | Exactly 1 final mark: `。`, `？`, `！`, `.`, `?`, `!`, at the end | F5 and F6 together                 |
| Y5   | No quotation mark                                                 | F7, same                           |
| Y6   | At most 1 internal comma, colon, or semicolon                     | F8, same                           |
| Y7   | At most `--max-unknown` tokens outside the known set              | New, from 8.2 of the Mandarin spec |
| Y8   | No foreign name from `vocab/cantonese_names.tsv`                  | New                                |

The known set is the first `--known-top` words (default 2000) of the Y1 list.
A token is unknown when it has no rank, or a rank above `--known-top`.
Punctuation tokens do not count.

Rule Y8 reads `vocab/cantonese_names.tsv`. The file has a header and one name
line, in the `name` column. A name matches when it is a substring of the
sentence, because the tokenizer splits a name such as 蕾拉 into two tokens.
The first rows are 蕾拉, 森美, 康納達, and 蘇嗎. Add a row when a deck holds
another foreign name. Latin names and English words do not need a row: rule
Y2 rejects them.

Differences from the Mandarin rules:

| Difference                       | Reason                                                                                              |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| Y1 is wider                      | Cantonese sentences are short. 3 character sentences such as 你先。 are good cards.                 |
| No predicate rule (F9)           | `jieba.posseg` has no Cantonese tagger. Sentence-final particles (呀, 喎, 嘅) cause false failures. |
| No orphan-conjunction rule (F10) | The test did not check it. Revisit after M2.                                                        |

Final-mark warning: if more than 20 percent of a deck fails Y4, print a
warning that names `--no-final-mark`. That option drops Y4 for the run.

Measured effect on the LTL deck: 858 of 999 sentences pass (14.3).

## 8. Selection

### 8.1 Target words

The target set is the `--top-words` most frequent words (default 1500) among
the words that appear as a token in at least one passed sentence.

- The order is the Y1 rank.
- A word with no Y1 rank is not a target.
- The coverage report marks a top-`--top-words` word of the Y1 list as
  `absent` when no passed sentence uses it.

### 8.2 Greedy cover

1. Give each target word a weight: `1 / log2(rank + 1)`.
2. Set the count `n[w] = 0` for each target word.
3. For each remaining sentence, compute the gain: the sum of
   `weight[w] / (n[w] + 1)` over target words `w` in the sentence with
   `n[w] < --per-word`.
4. Keep the sentence with the highest gain. Break a tie by lower complexity,
   then fewer Han characters, then lower original order.
5. Add 1 to `n[w]` for each target word in the kept sentence.
6. Remove any sentence that is too similar to a kept sentence (8.3).
7. Remove any sentence that holds a word at its `--max-uses` limit (8.5).
8. Stop when `--max-sentences` is reached, or when every gain is 0.

Greedy cover is not always the smallest set. The test in 3.1 shows that it
keeps nearly all top 1500 words that the passed pool holds: 1 of 392 are lost
for the LTL deck, and 9 of 791 for Tatoeba.

The weight `1/log2(rank+1)` keeps both `words` and `spoken_mass` high. The
test of four weights is in 14.5.

### 8.3 Similarity

Two sentences are duplicates in either case:

- They have the same unique-character count, and their character Jaccard
  value is at or above 0.67, the value of `SelectionOptions.similarity_threshold`.
- They have the same token set.

Reuse `char_similarity_zh` and `char_similarity_zh_same_len` from
`anki_tools/languages/mandarin/characters.py`.

### 8.4 Learning order

Sort the kept sentences so that each card uses words that earlier cards
introduced. The sort keys, in order:

1. `hardest_rank`: the rank of the least frequent token. Ascending. A token
   with no rank counts as rank 1,000,000, so a sentence of only unranked words
   comes last, not first.
2. Complexity score from `complexity_score_zh`. Ascending.
3. Han length. Ascending.
4. Original order. Ascending.

This order replaces the combined score order of `rank_sentences_yue`. The
selection always uses this order: it has no flag, and `SelectRequest.order`
defaults to `coverage`.

### 8.5 Word-use cap

`--per-word` sets how many cards each target word should get. `--max-uses`
sets the most that one word can get.

- A word outside the 100 most frequent words (by Y1 rank) appears in at most
  `--max-uses` kept sentences. The default is 8.
- A sentence counts once for a word, even when the word occurs twice in it.
- Skip a sentence that holds a word at its limit. The sentence stays out,
  because counts only grow.
- The 100 most frequent words are exempt. They are normal speech: in the
  Tatoeba selection, 我 is in 33% of the cards, 佢 in 22%, and 你 in 19%
  (14.6). A limit on them would remove most sentences.
- `--max-uses 0` turns the limit off. The value must be at least `--per-word`.
- The size 100 is a constant in the code. It is not an option.

Cost: the cap skips sentences that still add new words. At the defaults it
removes 5% of the LTL cards and 10% of the Tatoeba cards, and `spoken_mass`
falls by 0.0 and 0.3 points (3.1). With a card cap, the loss is 0.1 point or less
(14.6).

## 9. Interface

### 9.1 Frequency list

There is no frequency command. The first `anki fix yue` builds the frequency
list and runs gates G1, G3, and G4 on the blend.

### 9.2 Options of `anki fix yue`

The common flags stay on the command. The selection tuning values live in
`SelectRequest`, and the core tuning values live in `CoreOptions`. The module
`anki_tools/languages/cantonese/select_pipeline.py` holds `SelectRequest`,
`run_select`, `check_select`, and `default_output`.

| Option            | Default | Meaning                                                                    |
| ----------------- | ------- | -------------------------------------------------------------------------- |
| `--output`        | none    | Output file, else `<input>_reordered.apkg`                                 |
| `--text-field`    | none    | Sentence field of rank mode; else inferred                                |
| `--keep-filtered` | off     | Keep the similar cards (rank mode only)                                    |
| `--select`        | off     | Run the coverage selection; the default is rank mode                       |
| `--core`          | off     | Build the core deck                                                        |
| `--cards`         | none    | Card cap of the chosen mode; sets `SelectRequest.max_sentences`            |
| `--words`         | 0       | Core mode: tier F words after the Swadesh set                              |
| `--examples`      | 2       | Core mode: examples per content target                                     |
| `--tail`          | 0       | Core mode: cards after the core passes                                     |
| `--check`         | off     | Run the gates of core mode on the output only                              |

These values are not flags. The code reads them from `SelectRequest`, and
`--cards` is the only flag that sets one.

| Field                  | Default    | Rule                                                              |
| ---------------------- | ---------- | ----------------------------------------------------------------- |
| `max_sentences`        | none       | Stop Y4 at this count. Without it, Y4 stops at 0 gain (`--cards`) |
| `per_word`             | 2          | Cards wanted for each target word                                 |
| `top_words`            | 1500       | Size of the target set                                            |
| `order`                | `coverage` | `coverage` (8.4); the CLI has no `--order` flag                   |
| `min_han`              | 3          | Minimum Han characters (Y1)                                       |
| `max_han`              | 12         | Maximum Han characters (Y1)                                       |
| `max_unknown`          | 2          | Unknown tokens allowed in a sentence (Y7)                         |
| `known_top`            | 2000       | Size of the known set (Y7)                                        |
| `similarity_threshold` | 0.67       | Character Jaccard threshold (8.3)                                 |
| `final_mark`           | true       | Apply rule Y4; false skips it                                     |

`--order` no longer exists, and it never affected rank mode.

### 9.3 Output files

For `anki fix yue "deck.apkg"`:

| File                  | Content                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| `deck_reordered.apkg` | The kept cards, in the learning order                                       |
| `deck.rank.csv`       | The existing columns plus `hardest_rank` and `kept`                         |
| `deck.select.csv`     | One row per input sentence: `sentence`, `kept`, `reason`, `targets`, `gain` |
| `deck.coverage.csv`   | One row per target word: `word`, `rank`, `count`, `status`                  |

Values:

| Field    | Values                                                                                        |
| -------- | --------------------------------------------------------------------------------------------- |
| `count`  | Kept sentences that contain the word                                                          |
| `status` | `covered`, `partial` (count below `--per-word`), or `absent`                                  |
| `reason` | `Y1` to `Y8`, `similar`, `no_gain`, `use_cap` (all from Y4), or `cap` (`--max-sentences` hit) |

A run prints a summary:

- Sentences in, rejected by rule, removed by the select stage (Y4), and kept.
- The percent of the top 500 and the top 1500 words that reach `--per-word`.
- The most uses of one word outside the top 100 (`max_uses`).

### 9.4 Code changes

- `rank_sentences_yue` uses `--freq`. The default is `blend`. The weights
  stay: complexity 0.3, frequency 0.5, similarity penalty 0.2.
- `Sentence` gets `hardest_rank: int` (default 0) and `reject_reason: str`
  (default empty).
- `load_frequency_data_yue` gets a `source` parameter: `written`, `spoken`,
  or `blend`. `written` keeps today's output.
- The blend changes the order of the existing output. M2 reports the changed
  sentences before the default changes (`anki-afn`).

## 10. Data sources and licenses

| Data                  | Where                                                                       | Cache path                                     |
| --------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Apple Daily frequency | `raw.githubusercontent.com/chaaklau/appledaily-frequency/.../word-freq.tsv` | `~/.cache/anki-tools/appledaily-word-freq.tsv` |
| Cifu lexicon          | `github.com/gwinterstein/Cifu`, `Lexicon/Cifu-v1.txt`                       | `~/.cache/anki-tools/cifu-v1.tsv`              |

Rules:

- Download both files to the cache. Do not commit them.
- The Cifu repository has a GPL-3.0 `LICENSE` file. The data comes from
  corpora with other licenses. Do not copy Cifu into the repository. Check
  the license status before any public release (M0).
- Do not write the generated list to a file. The blend is built in memory on
  each run.
- `YUE_FREQUENCY_PATH` keeps its meaning for the Apple Daily file. Add
  `CIFU_PATH` for the Cifu file.
- Add to `ATTRIBUTION.md`: Cifu (Lai and Winterstein 2020), HKCanCor (Luke
  and Wong 2015, CC BY 4.0), and the Apple Daily list (CC BY 4.0).

## 11. Quality gates and tests

### 11.1 Gates

The build of the frequency list and the unit tests run these checks. A failure
stops the run with a `RuntimeError`. Lower a value only with a new measurement.

| Gate | Check                                                                                                                           | From |
| ---- | ------------------------------------------------------------------------------------------------------------------------------- | ---- |
| G1   | the frequency list has at least 40,000 rows, and its ranks are 1 to N with no gap                                              | M1   |
| G2   | On the LTL deck, at least 92 percent of Han tokens have a spoken count (measured 92.3)                                          | M2   |
| G3   | 係, 嘅, 唔, 我, 你 are in the first 20 ranks                                                                                    | M1   |
| G4   | 你好, 早晨, 唔該 are in the first 3000 ranks (overrides apply)                                                                  | M1   |
| G5   | No kept sentence fails Y1 to Y8                                                                                                 | M3   |
| G6   | No two kept sentences are duplicates by 8.3                                                                                     | M4   |
| G7   | Every kept sentence has a row in `.select.csv`, and the row counts match the input                                              | M3   |
| G8   | rank mode with `--freq written` gives the same ranking CSV as the old version                                                   | M5   |
| G9   | The kept cards have a non-decreasing `hardest_rank` when `--order coverage`                                                     | M4   |
| G10  | LTL deck, `--select --cards 300`: at least 650 words, 210 top500 words, and `spoken_mass` 65.0 or more (measured 688, 217, 66.1) | M4   |
| G11  | LTL deck, defaults: 641 cards, within 5 percent (measured 641)                                                                  | M4   |
| G12  | No word outside the 100 most frequent words is in more than `max_uses` kept sentences (8.5)                                     | M4   |

G2 comes from 14.2. G10 and G11 come from 3.1 and 3.3.

G8 was checked on 2026-10-19: rank mode with `--freq written` gives the same
rank, note id, sentence, and score for all 998 LTL rows as the code before
this spec.

### 11.2 Tests

Add `tests/test_yue_selection.py`. Use small fixture files. Do not download in
tests. Patch the download function, as the existing yue tests do.

| Area           | Test                                                                                                                                                              |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Blend          | Counts below 5 use the written term; the weight changes the order                                                                                                 |
| Cifu reader    | Skips the header; reads the empty count cells                                                                                                                     |
| Tokenizer      | The instance does not change the global jieba result                                                                                                              |
| Variant map    | Lookup only; card text unchanged                                                                                                                                  |
| Rules Y1 to Y8 | A small fixture with one failing sentence for each rule                                                                                                           |
| Greedy cover   | A rare word wins over a repeat; the `--per-word` limit stops the gain                                                                                             |
| Similarity     | An anagram pair and a same-token pair                                                                                                                             |
| Word-use cap   | A word outside the top 100 stops at `--max-uses` sentences; a top-100 word does not; a word twice in one sentence counts once; `--max-uses 0` turns the limit off |
| Learning order | Non-decreasing `hardest_rank`                                                                                                                                     |
| Regression     | The `--no-select` result equals the stored ranking CSV                                                                                                            |

## 12. Risks and open questions

### 12.1 Risks

| Risk                                                                                               | Effect                                               | Mitigation                                                                        |
| -------------------------------------------------------------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------- |
| Spoken sample is small (306k tokens)                                                               | Wrong rank below about 2000                          | Use spoken only for count of 5 or more; blend                                     |
| Cifu is biased to the HKCanCor 1997-98 vocabulary                                                  | Old words rank high                                  | Written term at weight 0.3; overrides                                             |
| Cifu license is unclear for redistribution                                                         | Cannot ship a derived list                           | Cache only; `data/` ignored; open question 1                                      |
| Tokenizer splits words wrongly (我仲讀)                                                            | Wrong target match                                   | Cifu dictionary; G2; fix by hand in overrides                                     |
| The blend changes the order of the current output                                                  | Old decks look different                             | `--no-select`, `--freq written`, M2 report                                        |
| Small input deck limits diversity                                                                  | Selection gains little                               | Report coverage; Tatoeba pool (3.4, M6)                                           |
| Strict rules drop natural short sentences                                                          | Fewer cards                                          | Y1 starts at 3 characters; `.select.csv` shows reasons                            |
| Variant map goes the wrong way                                                                     | Wrong frequency for 咁 or 枱                         | Pick the larger Cifu count; review in M1                                          |
| `spoken_mass` is circular: Cifu gives the tokenizer, the rank, and the test                        | Values are too high                                  | Use it to compare methods only; G10 compares with the current ranker              |
| Tatoeba Cantonese sentences are long (26.3% have 12 characters or fewer)                           | Rule Y1 rejects 74% of them                          | Keep 12; a limit of 16 gains 0.7 points (3.4)                                     |
| The word-use cap drops sentences that add new words (5 to 10 percent of the cards at the defaults) | Fewer cards; `spoken_mass` falls by up to 0.3 points | `--max-uses 0` turns it off; with a card cap the loss is 0.1 point or less (14.6) |
| The 100 most frequent words are exempt and appear in many cards (我 in 33%)                        | Cards look alike                                     | Normal speech; the size 100 is a code constant                                    |

### 12.2 Open questions

| #   | Question                                                                                                 | Evidence                                                                      | Decide at |
| --- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | --------- |
| 1   | Is Cifu usable in a public build? Ask the authors, or use HKCanCor alone                                 | GPL-3.0 repository file; data from CC BY 4.0 HKCanCor and other corpora       | M0        |
| 2   | Is a spoken weight of 0.7 right?                                                                         | Compare 0.5, 0.7, and 0.9 on hand-marked sentences                            | M2        |
| 3   | Add a Cantonese part-of-speech check to replace F9?                                                      | No Cantonese tagger in `jieba.posseg`                                         | M3        |
| 4   | Add a `--targets swadesh` mode (`vocab/english-cantonese.csv`)?                                          | The Mandarin pack uses Swadesh targets                                        | later     |
| 5   | Add a stop rule for small gain, such as `--min-gain`?                                                    | The last 173 Tatoeba cards add 0.0 points (3.3)                               | M4        |
| 6   | Should `sqrt(blend_pm)` replace the log-rank weight?                                                     | +0.6 to 0.7 points, -37 to 50 words (14.5)                                    | M4        |
| 7   | Is 8 the right value for `--max-uses`?                                                                   | Limits 5 and 12 remove 18% and 5% of the Tatoeba cards at the defaults (14.6) | M4        |
| 8   | Should a soft penalty replace the hard cap? A hard cap skips a sentence even when it adds many new words | The cap costs up to 0.3 points at the defaults (3.1)                          | M4        |

## 13. Milestones

| Milestone | Issue      | Task                                                                    | Output                                  |
| --------- | ---------- | ----------------------------------------------------------------------- | --------------------------------------- |
| M0        | `anki-4uh` | Fetch Cifu to the cache; check the license; add `CIFU_PATH`             | cached file, license note               |
| M1        | `anki-vfx` | Reader, blend, variant map, overrides, build on demand                  | frequency list; G1, G3, G4              |
| M2        | `anki-d1o` | Tokenizer, `--freq` option, change report for `anki-afn`                | report of changed sentences; G2         |
| M3        | `anki-8ox` | Rules Y1 to Y8, `.select.csv`                                           | G5, G7                                  |
| M4        | `anki-7y5` | Greedy cover, similarity, word-use cap, learning order, `.coverage.csv` | G6, G9, G10, G11, G12; options complete |
| M5        | `anki-0ne` | README, `ATTRIBUTION.md`, tests, regression gate G8                     | release check                           |
| M6        | `anki-5z4` | Tatoeba Cantonese pool: download, English links, command spec           | numbers of 3.4 reproduced; second spec  |

Order: M0 to M5 run in sequence. M2 comes before the default changes. It
reports each sentence whose rank moves by more than 100 places, so the user
can check the result.

Acceptance:

| Milestone | Condition                                                                                                                                                                                                 |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M4        | LTL deck, defaults: all 223 top-500 words of the passed set, at most 680 cards, `spoken_mass` 66.8%, no word over its limit (G10, G11, G12). The user marks 5 percent of the kept cards as good sentences |
| M6        | Tatoeba pool: 500 cards give `spoken_mass` of 77.0% or more; the default run gives about 1,200 cards (3.1, 3.3)                                                                                           |

## 14. Appendix: evidence

All numbers come from a test run on 2026-10-19. The test deck is the LTL
deck: 999 sentences, 1000 rows in the rank CSV.

### 14.1 Frequency sources

| Source                            | Genre           | Size                                      | License                                     | Verdict                            |
| --------------------------------- | --------------- | ----------------------------------------- | ------------------------------------------- | ---------------------------------- |
| Apple Daily list (in use today)   | Written news    | 493.9M tokens, 275k distinct words        | CC BY 4.0                                   | Keep as the written term           |
| Cifu `SpokenAdult` column         | Spoken adult    | 306,339 tokens, 11,970 words with a count | repo file is GPL-3.0; source corpora differ | Use as the spoken term             |
| Cifu `Written` column             | Written         | 2.36M tokens                              | as above                                    | Use for the written term           |
| HKCanCor                          | Spoken, 1997-98 | about 230k words                          | CC BY 4.0                                   | Already inside Cifu; no direct use |
| cantfreq (HKCanCor, HKCAC, IARPA) | Spoken          | 3 corpora                                 | no license file                             | Do not use                         |
| UD Cantonese-HK                   | Film subtitles  | small                                     | see repo                                    | Later, for validation only         |

Cifu (Lai and Winterstein 2020) has 51,798 entries. Each entry has a Jyutping
reading and counts for four genres. Its `SpokenAdult` column comes from
HKCanCor and CantoMap. The spoken sample gives a good rank for about the top
2000 words only.

Spoken and written text rank words in a different order. The top spoken words
are 係 呢 嘅 我 你 佢 唔 就 個 咁. The written list ranks the particles low.

Blend test with `w = 0.7`:

| Word | Written-only rank | Blend rank |
| ---- | ----------------- | ---------- |
| 早晨 | 20387             | 871        |
| 唔該 | not measured      | 592        |
| 多謝 | not measured      | 402        |
| 點解 | not measured      | 125        |
| 你好 | 18180             | 28344      |

The written-only ranks of 早晨 and 你好 come from `anki-afn`.

### 14.2 Tokenizer test on the LTL deck

| Measure                            | jieba default | jieba + Cifu words | PyCantonese `segment` |
| ---------------------------------- | ------------- | ------------------ | --------------------- |
| Tokens (Han only)                  | 4626          | 4303               | 4094                  |
| Distinct words                     | 1169          | 1133               | 1112                  |
| Tokens with a spoken count         | 91.1%         | 92.3%              | 85.4%                 |
| Distinct words with a spoken count | 70.5%         | 75.9%              | 61.7%                 |
| Tokens with any Cifu count         | 95.8%         | 98.4%              | 88.3%                 |

- A merged dictionary with the Apple Daily words was worse (82.1% spoken
  tokens), because it adds long written-only words.
- PyCantonese adds a 40 MB dependency and scores lower.
- The jieba default splits Cantonese wrongly. Example: 我仲讀 stays as 我 and
  仲讀. The Cifu dictionary does not fix every case, so G2 guards the result.

### 14.3 Sentence rules on the LTL deck

The table shows the first failed rule for each sentence.

| Result                   | Count |
| ------------------------ | ----- |
| Pass                     | 858   |
| Y1 length not 3 to 12    | 30    |
| Y2 Latin letter or digit | 50    |
| Y3 repeated pattern      | 1     |
| Y4 not one final mark    | 14    |
| Y7 more than 2 unknown   | 32    |
| Y8 foreign name          | 14    |
| Total                    | 999   |

- Example of a Y2 failure: `送Tom走。` has rank 3 in the current output. The
  current ranker does not remove it.
- Han length of the deck: 4 to 8 characters for 814 of 999 sentences.

### 14.4 Vocabulary reach of the LTL deck

| Measure                | All 999 sentences | The 858 that pass |
| ---------------------- | ----------------- | ----------------- |
| Words                  | 1133              | 983               |
| Top 500 blended words  | 236               | 223               |
| Top 1000 blended words | 345               | 326               |
| Top 1500 blended words | 414               | 386               |
| `spoken_mass`          | 67.7%             | 66.8%             |

The LTL deck is small, so it limits the result. A selection step cannot add
words that the deck does not have. It still removes repeat sentences, and it
puts the deck in a learning order. A larger set gives more diversity (3.1,
3.4).

### 14.5 Test of four weight functions (no word-use cap)

Values are `words`, top1000, and `spoken_mass`, with a cap of 300 or 500 cards.

| Weight           | LTL 300        | Tatoeba 300    | Tatoeba 500      |
| ---------------- | -------------- | -------------- | ---------------- |
| `1/log2(rank+1)` | 694, 306, 65.9 | 977, 476, 74.8 | 1,329, 528, 77.2 |
| `blend_pm`       | 582, 328, 66.2 | 841, 483, 75.2 | 1,237, 572, 77.8 |
| `sqrt(blend_pm)` | 628, 323, 66.3 | 927, 483, 75.4 | 1,292, 562, 77.9 |
| 1 (no weight)    | 718, 287, 64.5 | 989, 444, 72.3 | 1,334, 513, 76.3 |

- `blend_pm` gains at most 0.6 points of `spoken_mass`, and it loses 90 to
  140 words at 300 cards.
- The weight 1 loses up to 2.5 points.
- `sqrt(blend_pm)` gains 0.6 to 0.7 points and loses 37 to 50 words (open
  question 6).
- Decision: use `1/log2(rank+1)`, which keeps both measures high.

### 14.6 Test of the word-use cap

How often one word repeats in the selection without a cap (681 LTL cards,
1,307 Tatoeba cards). Each row is a band of the Y1 rank.

| Rank band            | LTL words | LTL most uses | LTL words over 5 uses | Tatoeba words | Tatoeba most uses | Tatoeba words over 5 uses |
| -------------------- | --------- | ------------- | --------------------- | ------------- | ----------------- | ------------------------- |
| 1 to 100             | 77        | 216 (32%)     | 40                    | 94            | 426 (33%)         | 69                        |
| 101 to 500           | 147       | 22 (3%)       | 17                    | 263           | 37 (3%)           | 62                        |
| 501 to 2000          | 202       | 6 (1%)        | 2                     | 588           | 13 (1%)           | 19                        |
| Over 2000 or no rank | 558       | 7 (1%)        | 1                     | 1,113         | 11 (1%)           | 5                         |

- The most used words are 我 (33% of the Tatoeba cards), 佢 (22%), and 你
  (19%). They are normal speech, so the rule exempts the top 100.
- The real repeats are in ranks 101 to 500: one word is in up to 37 Tatoeba
  cards, and 62 words are in more than 5.

Test of the limit and of the exempt size. `max_uses` is the largest count of
one non-exempt word.

| Run              | Limit, exempt | Cards | Words | top1000 | `spoken_mass` | `max_uses` | Words over 5 uses |
| ---------------- | ------------- | ----- | ----- | ------- | ------------- | ---------- | ----------------- |
| LTL defaults     | none          | 681   | 984   | 328     | 66.8%         | 22         | 20                |
| LTL defaults     | 5, 100        | 610   | 926   | 326     | 66.8%         | 5          | 0                 |
| LTL defaults     | 8, 100        | 647   | 961   | 327     | 66.8%         | 8          | 22                |
| LTL defaults     | 12, 100       | 666   | 974   | 328     | 66.8%         | 12         | 22                |
| LTL defaults     | 8, 200        | 669   | 977   | 328     | 66.8%         | 8          | 12                |
| LTL defaults     | 8, 50         | 605   | 933   | 320     | 66.0%         | 8          | 23                |
| LTL cap 300      | none          | 300   | 694   | 306     | 65.9%         | 9          | 2                 |
| LTL cap 300      | 5, 100        | 300   | 692   | 306     | 66.0%         | 5          | 0                 |
| LTL cap 300      | 8, 100        | 300   | 694   | 307     | 66.0%         | 8          | 2                 |
| LTL cap 300      | 12, 100       | 300   | 694   | 306     | 65.9%         | 9          | 2                 |
| LTL cap 300      | 8, 200        | 300   | 694   | 306     | 65.9%         | 4          | 0                 |
| LTL cap 300      | 8, 50         | 300   | 694   | 301     | 65.3%         | 8          | 5                 |
| Tatoeba defaults | none          | 1,307 | 2,058 | 583     | 79.9%         | 37         | 86                |
| Tatoeba defaults | 5, 100        | 1,076 | 1,879 | 572     | 79.2%         | 5          | 0                 |
| Tatoeba defaults | 8, 100        | 1,173 | 1,974 | 580     | 79.6%         | 8          | 72                |
| Tatoeba defaults | 12, 100       | 1,241 | 2,021 | 583     | 79.8%         | 12         | 78                |
| Tatoeba defaults | 8, 200        | 1,252 | 2,027 | 582     | 79.8%         | 8          | 46                |
| Tatoeba defaults | 8, 50         | 1,072 | 1,890 | 575     | 79.1%         | 8          | 93                |
| Tatoeba cap 500  | none          | 500   | 1,329 | 528     | 77.2%         | 10         | 15                |
| Tatoeba cap 500  | 5, 100        | 500   | 1,319 | 524     | 77.1%         | 5          | 0                 |
| Tatoeba cap 500  | 8, 100        | 500   | 1,330 | 527     | 77.2%         | 8          | 15                |
| Tatoeba cap 500  | 12, 100       | 500   | 1,329 | 528     | 77.2%         | 10         | 15                |
| Tatoeba cap 500  | 8, 200        | 500   | 1,329 | 528     | 77.2%         | 8          | 3                 |
| Tatoeba cap 500  | 8, 50         | 500   | 1,328 | 523     | 77.0%         | 8          | 31                |

- The LTL cells in the table above come from the prototype before rule Y8.
  With Y8, the defaults give 674 cards without a cap and 641 with a cap of 8.
- A limit of 8 removes 5% of the LTL cards and 10% of the Tatoeba cards at
  the defaults. `spoken_mass` falls by 0.0 and 0.3 points.
- With a card cap, the limit is almost free: 0.1 point or less. At Tatoeba
  500, the word count stays at 1,330.
- A limit of 5 is too strict: it removes 18% of the Tatoeba cards and 0.7
  points. A limit of 12 rarely applies: it removes 5%.
- An exempt size of 50 costs more (Tatoeba defaults: 1,072 cards, 79.1%). A
  size of 200 leaves more repeats (46 words over 5 uses, against 72).
- Decision: a limit of 8, and an exempt size of 100.

Tested and dropped: caps on the shape of a sentence. These were the opening
word, the ending particle, the form (statement, question, or exclamation),
and the frame. The opening, ending, and frame caps gave no gain in `top1000`
or `spoken_mass`. The frame cap lost 45 words on the LTL defaults (984 to
939). Only the form cap helped, and only with a card cap: Tatoeba 500
`top1000` rose from 528 to 548. This spec keeps none of them.
