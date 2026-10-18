# Spec: Cantonese Sentence Selection with Spoken Frequency

| Field   | Value                                                                     |
| ------- | ------------------------------------------------------------------------- |
| Status  | draft                                                                     |
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

| Change                | What it does                                                                                      | Section |
| --------------------- | ------------------------------------------------------------------------------------------------- | ------- |
| Spoken frequency list | Blends Cifu spoken counts with written counts; `anki freq yue`                                    | 5       |
| Sentence filter       | Removes fragments, mixed script, repeats, and hard sentences                                      | 7       |
| Diversity selection   | Greedy cover of the most frequent words; caps on repeated sentence types; easy-to-hard card order | 8       |

### 1.2 Result at a glance

The defaults include the type caps (8.5). Each cell shows the new value, then
the value of the current ranking at the same card count.

| Input                           | Sentences in | Cards | Types         | `spoken_mass` |
| ------------------------------- | ------------ | ----- | ------------- | ------------- |
| LTL deck                        | 999          | 613   | 933 / 730     | 66.8 / 63.6   |
| LTL deck, cap 300               | 999          | 300   | 671 / 410     | 65.9 / 57.4   |
| Tatoeba Cantonese pool          | 20,853       | 1,236 | 2,020 / 1,237 | 79.8 / 69.3   |
| Tatoeba Cantonese pool, cap 500 | 20,853       | 500   | 1,318 / 572   | 77.6 / 60.5   |

`spoken_mass` is the share of spoken Cantonese tokens that the kept words
cover (2.2). Section 3 has the full tables.

### 1.3 Key decisions

| Decision                | Choice                                                                           | Evidence     |
| ----------------------- | -------------------------------------------------------------------------------- | ------------ |
| Spoken frequency source | Cifu `SpokenAdult`, blended with written counts at 0.7                           | 5.2, 14.1    |
| Tokenizer               | jieba with a Cifu dictionary                                                     | 6, 14.2      |
| Sentence length         | 3 to 12 Han characters                                                           | 7, 14.3      |
| Target words            | The 1500 most frequent words that the pool contains                              | 8.1          |
| Selection method        | Greedy cover, 2 cards per word, weight `1/log2(rank+1)`                          | 8.2, 14.5    |
| Repeated sentence types | Caps on one opening word (20%), ending particle (8%), form (75%), and frame (3%) | 8.5, 14.6    |
| Card order              | Easiest hardest-word first                                                       | 8.4          |
| Compatibility           | `--no-select` reproduces today's output                                          | 9.2, gate G8 |

### 1.4 Open blockers

- Cifu is usable for local builds. Its license for a public build is not
  clear (open question 1, M0).
- The new default frequency changes today's ranking. M2 must report the
  changed sentences before the default changes (`anki-afn`).

## 2. Goals and scope

### 2.1 Goals

1. Keep as much vocabulary diversity as the input allows, with the fewest
   cards.
2. Weight that diversity to the most frequent words of spoken Cantonese, not
   written Cantonese.
3. Keep each sentence short, complete, and unlike the other kept sentences.
4. Do not keep too many sentences of one type. A type is the opening word,
   the ending particle, the form, or the frame of a sentence (8.5).

### 2.2 Measures

| Measure                        | Meaning                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `cards`                        | Sentences kept                                                                                           |
| `types`                        | Distinct words in the kept sentences                                                                     |
| `top500`, `top1000`, `top1500` | How many of the 500, 1000, or 1500 most frequent words of the blend list (5.2) are in the kept sentences |
| `spoken_mass`                  | Share of the 306,339 Cifu spoken tokens that belong to a word in the kept sentences                      |
| Largest type share             | For each type key (8.5), the share of the kept cards that the most common value holds                    |

`spoken_mass` estimates the coverage of running speech. A learner who knows
every kept word recognizes this share of spoken tokens. Cifu gives the
tokenizer, the rank list, and the test, so the absolute values are high. Use
`spoken_mass` to compare methods. Do not use it as a claim about real speech.

### 2.3 In scope

- A spoken frequency list, built from public corpora and cached locally.
- A Cantonese-aware word tokenizer.
- New options for `anki fix yue`, and a report of what the filter removed.
- The command `anki freq yue`, which writes the frequency list.
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
- `--known-top 2000`, `--max-unknown 2`, and 3 to 12 Han characters.
- Similarity 0.67.
- The type caps of 8.5: opening 20%, ending 8%, form 75%, frame 3%, floor 5.

### 3.1 Cards at the defaults

| Input                  | Sentences in | Pass Y1 to Y7 | Cards kept | Types | top500 | top1000 | top1500 | `spoken_mass` |
| ---------------------- | ------------ | ------------- | ---------- | ----- | ------ | ------- | ------- | ------------- |
| LTL deck               | 999          | 872           | 613        | 933   | 224    | 326     | 390     | 66.8%         |
| Tatoeba Cantonese pool | 20,853       | 4,256         | 1,236      | 2,020 | 357    | 583     | 791     | 79.8%         |

What the selection removes, against the passed pool:

| Input   | Cards removed        | Types lost     | Top 1500 words lost | `spoken_mass` lost |
| ------- | -------------------- | -------------- | ------------------- | ------------------ |
| LTL     | 259 of 872 (30%)     | 66 of 999      | 2                   | 0.0 points         |
| Tatoeba | 3,020 of 4,256 (71%) | 1,458 of 3,478 | 0                   | 0.7 points         |

Most lost types are rare words, outside the target set.

What the rules of section 7 remove, before the selection runs:

| Input   | Effect of rules Y1 to Y7                                                  |
| ------- | ------------------------------------------------------------------------- |
| LTL     | 12 of 236 top-500 words are lost; `spoken_mass` falls from 67.7% to 66.8% |
| Tatoeba | top500 falls from 457 to 357; `spoken_mass` falls from 95.8% to 80.5%     |

Rule Y1 causes most of the Tatoeba loss. It rejects 15,445 of 20,853
sentences, because only 26.3% have 12 Han characters or fewer.

What the type caps change, against the same run without them (14.6):

| Input            | Cards | Types | `spoken_mass` | Largest opening share | Largest frame share |
| ---------------- | ----- | ----- | ------------- | --------------------- | ------------------- |
| LTL, no caps     | 681   | 984   | 66.8%         | 24%                   | 9%                  |
| LTL, caps        | 613   | 933   | 66.8%         | 22%                   | 3%                  |
| Tatoeba, no caps | 1,307 | 2,058 | 79.9%         | 23%                   | 6%                  |
| Tatoeba, caps    | 1,236 | 2,020 | 79.8%         | 21%                   | 3%                  |

### 3.2 Cards for each setting

| Input   | `--top-words` | `--per-word 1` | `--per-word 2` | `--per-word 3` |
| ------- | ------------- | -------------- | -------------- | -------------- |
| LTL     | 500           | 283            | 396            | 459            |
| LTL     | 1000 or more  | 500            | 613            | 658            |
| Tatoeba | 500           | 260            | 449            | 607            |
| Tatoeba | 1000          | 531            | 869            | 1,131          |
| Tatoeba | 1500          | 772            | 1,236          | 1,574          |
| Tatoeba | 2000          | 1,065          | 1,601          | 1,964          |

- The LTL rows for 1000 or more are equal, because the passed deck has only
  973 ranked words.
- At `--per-word 1`, each card carries about 1.9 target words.
- A second card for each word is often not possible. 303 of the 973 LTL
  target words and 935 of the 1500 Tatoeba target words reach 2 cards.

### 3.3 Cards with a cap, against the current ranking

`--max-sentences` stops the selection early. The current ranking is the first
N sentences of the passed pool, in the order of the current ranker. Each cell
shows the new value, then the current value.

| Input   | Cards | Types         | top500    | top1000   | `spoken_mass` |
| ------- | ----- | ------------- | --------- | --------- | ------------- |
| LTL     | 300   | 671 / 410     | 216 / 147 | 303 / 195 | 65.9 / 57.4   |
| LTL     | 500   | 866 / 618     | 223 / 185 | 323 / 257 | 66.7 / 62.5   |
| LTL     | 613   | 933 / 730     | 224 / 196 | 326 / 279 | 66.8 / 63.6   |
| Tatoeba | 300   | 972 / 357     | 314 / 138 | 463 / 179 | 74.4 / 54.8   |
| Tatoeba | 500   | 1,318 / 572   | 345 / 178 | 547 / 244 | 77.6 / 60.5   |
| Tatoeba | 1,000 | 1,895 / 1,026 | 357 / 237 | 583 / 355 | 79.7 / 66.5   |
| Tatoeba | 1,236 | 2,020 / 1,237 | 357 / 255 | 583 / 384 | 79.8 / 69.3   |

For the LTL deck, the current order is the real `rank.csv` order. For
Tatoeba, it uses the present frequency score (Apple Daily) and complexity
score, without the similarity penalty. It is an approximation.

How to read the table:

- At 300 cards, the LTL selection keeps 64% more types and 8.5 points more
  `spoken_mass`.
- At 500 cards, the Tatoeba selection keeps 2.3 times the types and 17.1
  points more `spoken_mass`.
- The curve flattens. Tatoeba gains 2.1 points from 500 to 1,000 cards, and
  0.1 point from 1,000 to 1,236 cards.
- A cap of 500 gives 97% of the final Tatoeba `spoken_mass` (77.6 of 79.8)
  with 40% of the cards.
- The LTL deck needs no cap. 500 cards give 66.7%, against 66.8% for the full
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
| Default cards with an English link        | 1,052 of 1,236                              |
| Default cards with native audio           | 366 of 1,236                                |
| Cards when the pool needs an English link | 1,209 (`spoken_mass` 79.2%, 409 with audio) |

Effect of a longer limit (`--max-han`) on the Tatoeba run:

| `--max-han` | Pass pool | Cards | Types | `spoken_mass` |
| ----------- | --------- | ----- | ----- | ------------- |
| 12          | 4,256     | 1,236 | 2,020 | 79.8%         |
| 14          | 4,595     | 1,222 | 2,065 | 80.3%         |
| 16          | 4,760     | 1,222 | 2,109 | 80.6%         |

A limit of 16 gains 0.8 points and gives longer sentences. Keep 12.

### 3.5 How the numbers were made

A prototype of sections 5 to 8 made all numbers in this section. The
prototype is not in the repository. It used:

- The Cifu file and the Apple Daily file of 2026-10-19.
- The tokenizer of section 6, and the blend of 5.2 with `w = 0.7`.
- The override `你好` = `多謝`, and rules Y1 to Y7.
- The type caps of 8.5, with the default shares.
- Tatoeba data from `downloads.tatoeba.org/exports/per_language/yue/`.

A count can differ by one from a later table because of a tie in the sort.
Gates G10, G11, and G12 and milestone M4 turn these numbers into tests.

## 4. Pipeline

| Stage | Name           | Input              | Output                                        |
| ----- | -------------- | ------------------ | --------------------------------------------- |
| Y0    | Fetch          | network            | cached Cifu and Apple Daily files             |
| Y1    | Frequency list | Y0, override files | `data/yue_frequency.tsv`                      |
| Y2    | Tokenize       | Y1, deck sentences | tokens per sentence                           |
| Y3    | Filter         | Y2                 | passed and rejected sentences, reasons        |
| Y4    | Select         | Y3, Y1             | kept sentences in learning order              |
| Y5    | Write          | Y4, original deck  | `_reordered.apkg`, `.rank.csv`, `.select.csv` |

- `anki freq yue` runs Y0 and Y1.
- `anki fix yue` runs Y2 to Y5.
- Each stage gives the same output for the same input.

## 5. Frequency list

### 5.1 Output file

`data/yue_frequency.tsv` has one row per word, sorted by `rank`.

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

- `w` is `--spoken-weight`, default 0.7.
- `--freq` selects the term: `blend` (default), `spoken`, or `written`.
- `--freq written` gives today's behavior (Apple Daily only).
- The spoken sample is small, so a count below 5 uses the written term only.

The blend fixes most spoken words (14.1). It does not fix `你好`, because
HKCanCor writes it as two tokens. The override file (5.4) covers this case.

### 5.3 Variant map

The written lists use news-standard characters. Spoken text uses common
variants (`anki-58i`).

- File: `vocab/yue_variants.tsv`, columns `variant` and `standard`.
- Use: apply the map to a token for the frequency lookup only. Never change
  the card text.
- Direction: pick the form that has the larger count in Cifu. Record the
  result in the file (M1).

| Variant | Standard | Meaning |
| ------- | -------- | ------- |
| 咁      | 噉       | so      |
| 枱      | 檯       | table   |

### 5.4 Overrides

- File: `vocab/yue_frequency_overrides.tsv`, columns `word`, `rank_like`, and
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

The known set is the first `--known-top` words (default 2000) of the Y1 list.
A token is unknown when it has no rank, or a rank above `--known-top`.
Punctuation tokens do not count.

Differences from the Mandarin rules:

| Difference                       | Reason                                                                                              |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| Y1 is wider                      | Cantonese sentences are short. 3 character sentences such as 你先。 are good cards.                 |
| No predicate rule (F9)           | `jieba.posseg` has no Cantonese tagger. Sentence-final particles (呀, 喎, 嘅) cause false failures. |
| No orphan-conjunction rule (F10) | The test did not check it. Revisit after M2.                                                        |

Final-mark warning: if more than 20 percent of a deck fails Y4, print a
warning that names `--no-final-mark`. That option drops Y4 for the run.

Measured effect on the LTL deck: 872 of 999 sentences pass (14.3).

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
7. Remove any sentence whose type has reached its cap (8.5).
8. Stop when `--max-sentences` is reached, or when every gain is 0.

Greedy cover is not always the smallest set. The test in 3.1 shows that it
keeps nearly all top 1500 words that the passed pool holds: 2 of 392 are lost
for the LTL deck, and none for Tatoeba.

The weight `1/log2(rank+1)` keeps both types and `spoken_mass` high. The
test of four weights is in 14.5.

### 8.3 Similarity

Two sentences are duplicates in either case:

- They have the same unique-character count, and their character Jaccard
  value is at or above `--similarity-threshold` (default 0.67, as in `fix zh`).
- They have the same token set.

Reuse `char_similarity_zh` and `_char_similarity_zh_same_len` from the
existing code.

### 8.4 Learning order

Sort the kept sentences so that each card uses words that earlier cards
introduced. The sort keys, in order:

1. `hardest_rank`: the rank of the least frequent known token. Ascending.
2. Complexity score from `complexity_score_zh`. Ascending.
3. Han length. Ascending.
4. Original order. Ascending.

This order replaces the combined score order of `rank_sentences_yue`. Use
`--order score` to keep the old order.

### 8.5 Type caps

A sentence has four type values. The selection limits how many kept
sentences share one value.

| Type key  | Value of a sentence                                                                              | Example     | Default share |
| --------- | ------------------------------------------------------------------------------------------------ | ----------- | ------------- |
| `opening` | First Han token                                                                                  | 我          | 20%           |
| `ending`  | Last Han token                                                                                   | 呀          | 8%            |
| `form`    | Class of the final mark: statement (`。`, `.`), question (`？`, `?`), or exclamation (`！`, `!`) | 。          | 75%           |
| `frame`   | Tokens with a Y1 rank of 60 or better stay. Each run of other tokens becomes `_`                 | `佢 _ 咗 _` | 3%            |

Rules:

- The limit for one type value is `max(--type-floor, ceil(share * P))`.
- `P` is the planned card count: `--max-sentences` when it is set, else the
  card count of the same run without type caps.
- Skip a sentence when any of its four values has reached its limit. The
  sentence stays out, because counts only grow.
- The floor (default 5) lets a rare type keep at least 5 cards.
- Set a share to 1 to turn off one key. `--no-type-cap` turns off all four.

The limits apply to counts of `P`. The caps remove cards, so a final share
can be a few points above its default share. Example: statements are 79% of
the 1,236 Tatoeba cards, and the limit is 75% of `P` = 1,307. A loop that
tightens the caps until the count stays fixed would shrink the Tatoeba deck
from 1,307 to 1,118 cards. This spec does not use such a loop.

Why these keys, and why no cap on single words: 14.6.

## 9. Interface

### 9.1 New command

```bash
anki freq yue                        # Y0, Y1: write data/yue_frequency.tsv
anki freq yue --spoken-weight 0.8
anki freq yue --refresh              # download the sources again
anki freq yue --check                # run gates G1, G3, G4
```

### 9.2 Options of `anki fix yue`

The existing options stay. These options are new.

Selection:

| Option            | Default  | Meaning                                               |
| ----------------- | -------- | ----------------------------------------------------- |
| `--max-sentences` | none     | Stop Y4 at this count. Without it, Y4 stops at 0 gain |
| `--per-word`      | 2        | Cards wanted for each target word                     |
| `--top-words`     | 1500     | Size of the target set                                |
| `--order`         | coverage | `coverage` (8.4) or `score` (today's order)           |
| `--no-select`     | off      | Run only the ranking, as today                        |

Type caps (8.5):

| Option                | Default | Meaning                                   |
| --------------------- | ------- | ----------------------------------------- |
| `--max-opening-share` | 0.20    | Cap for one opening word                  |
| `--max-ending-share`  | 0.08    | Cap for one ending particle               |
| `--max-form-share`    | 0.75    | Cap for one sentence form                 |
| `--max-frame-share`   | 0.03    | Cap for one sentence frame                |
| `--type-floor`        | 5       | Cards that any type value can always keep |
| `--no-type-cap`       | off     | Turn off all four caps                    |

Filter:

| Option                   | Default | Meaning                                   |
| ------------------------ | ------- | ----------------------------------------- |
| `--min-han`              | 3       | Minimum Han characters (Y1)               |
| `--max-han`              | 12      | Maximum Han characters (Y1)               |
| `--max-unknown`          | 2       | Unknown tokens allowed in a sentence (Y7) |
| `--known-top`            | 2000    | Size of the known set (Y7)                |
| `--similarity-threshold` | 0.67    | Character Jaccard threshold (8.3)         |
| `--no-final-mark`        | off     | Skip rule Y4                              |

Frequency:

| Option            | Default | Meaning                         |
| ----------------- | ------- | ------------------------------- |
| `--freq`          | blend   | `blend`, `spoken`, or `written` |
| `--spoken-weight` | 0.7     | Value of `w` in 5.2             |

`--freq written --order score --no-select` gives a byte-identical result to
the current version (gate G8).

### 9.3 Output files

For `anki fix yue "deck.apkg"`:

| File                  | Content                                                                                                           |
| --------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `deck_reordered.apkg` | The kept cards, in the learning order                                                                             |
| `deck.rank.csv`       | The existing columns plus `hardest_rank` and `kept`                                                               |
| `deck.select.csv`     | One row per input sentence: `sentence`, `kept`, `reason`, `targets`, `gain`, `opening`, `ending`, `form`, `frame` |
| `deck.coverage.csv`   | One row per target word: `word`, `rank`, `count`, `status`                                                        |

Values:

| Field    | Values                                                                                         |
| -------- | ---------------------------------------------------------------------------------------------- |
| `status` | `covered`, `partial` (count below `--per-word`), or `absent`                                   |
| `reason` | `Y1` to `Y7`, `similar`, `no_gain`, `type_cap` (all from Y4), or `cap` (`--max-sentences` hit) |

A run prints a summary:

- Sentences in, rejected by rule, removed by Y4, and kept.
- The percent of the top 500 and the top 1500 words that reach `--per-word`.
- The largest share of each type key, with its limit.

### 9.4 Code changes

- `rank_sentences_yue` uses `--freq`. The default is `blend`. The weights
  stay: complexity 0.3, frequency 0.5, similarity penalty 0.2.
- `Sentence` gets `hardest_rank: int` (default 0), `reject_reason: str`
  (default empty), and `type_keys: dict[str, str]` (default empty).
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
- Write the generated list to `data/yue_frequency.tsv`. The `data/` folder is
  in `.gitignore`.
- `YUE_FREQUENCY_PATH` keeps its meaning for the Apple Daily file. Add
  `CIFU_PATH` for the Cifu file.
- Add to `ATTRIBUTION.md`: Cifu (Lai and Winterstein 2020), HKCanCor (Luke
  and Wong 2015, CC BY 4.0), and the Apple Daily list (CC BY 4.0).

## 11. Quality gates and tests

### 11.1 Gates

`anki freq yue --check` and the unit tests run these checks. A failure stops
the run with a non-zero exit code. Lower a value only with a new measurement.

| Gate | Check                                                                                                                           | From |
| ---- | ------------------------------------------------------------------------------------------------------------------------------- | ---- |
| G1   | `data/yue_frequency.tsv` has at least 40,000 rows, and its ranks are 1 to N with no gap                                         | M1   |
| G2   | On the LTL deck, at least 92 percent of Han tokens have a spoken count (measured 92.3)                                          | M2   |
| G3   | 係, 嘅, 唔, 我, 你 are in the first 20 ranks                                                                                    | M1   |
| G4   | 你好, 早晨, 唔該 are in the first 3000 ranks (overrides apply)                                                                  | M1   |
| G5   | No kept sentence fails Y1 to Y7                                                                                                 | M3   |
| G6   | No two kept sentences are duplicates by 8.3                                                                                     | M4   |
| G7   | Every kept sentence has a row in `.select.csv`, and the row counts match the input                                              | M3   |
| G8   | `--no-select --freq written --order score` gives the same ranking CSV as the current version                                    | M5   |
| G9   | The kept cards have a non-decreasing `hardest_rank` when `--order coverage`                                                     | M4   |
| G10  | LTL deck, `--max-sentences 300`: at least 640 types, 210 top500 words, and `spoken_mass` 65.0 or more (measured 671, 216, 65.9) | M4   |
| G11  | LTL deck, defaults: 613 cards, within 5 percent (measured 613)                                                                  | M4   |
| G12  | No type value has more kept cards than `max(--type-floor, ceil(share * P))` (8.5)                                               | M4   |

G2 comes from 14.2. G10 and G11 come from 3.1 and 3.3.

### 11.2 Tests

Add `tests/test_yue_selection.py`. Use small fixture files. Do not download in
tests. Patch the download function, as the existing yue tests do.

| Area           | Test                                                                                                                                     |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Blend          | Counts below 5 use the written term; the weight changes the order                                                                        |
| Cifu reader    | Skips the header; reads the empty count cells                                                                                            |
| Tokenizer      | The instance does not change the global jieba result                                                                                     |
| Variant map    | Lookup only; card text unchanged                                                                                                         |
| Rules Y1 to Y7 | A small fixture with one failing sentence for each rule                                                                                  |
| Greedy cover   | A rare word wins over a repeat; the `--per-word` limit stops the gain                                                                    |
| Similarity     | An anagram pair and a same-token pair                                                                                                    |
| Type keys      | `opening`, `ending`, `form`, and `frame` for a fixture sentence                                                                          |
| Type caps      | A cap keeps at most `max(floor, ceil(share * P))` sentences of one value; the floor lets a rare value through; share 1 turns the key off |
| Learning order | Non-decreasing `hardest_rank`                                                                                                            |
| Regression     | The `--no-select` result equals the stored ranking CSV                                                                                   |

## 12. Risks and open questions

### 12.1 Risks

| Risk                                                                         | Effect                                                 | Mitigation                                                                                                           |
| ---------------------------------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Spoken sample is small (306k tokens)                                         | Wrong rank below about 2000                            | Use spoken only for count of 5 or more; blend                                                                        |
| Cifu is biased to the HKCanCor 1997-98 vocabulary                            | Old words rank high                                    | Written term at weight 0.3; overrides                                                                                |
| Cifu license is unclear for redistribution                                   | Cannot ship a derived list                             | Cache only; `data/` ignored; open question 1                                                                         |
| Tokenizer splits words wrongly (我仲讀)                                      | Wrong target match                                     | Cifu dictionary; G2; fix by hand in overrides                                                                        |
| The blend changes the order of the current output                            | Old decks look different                               | `--no-select`, `--freq written`, M2 report                                                                           |
| Small input deck limits diversity                                            | Selection gains little                                 | Report coverage; Tatoeba pool (3.4, M6)                                                                              |
| Strict rules drop natural short sentences                                    | Fewer cards                                            | Y1 starts at 3 characters; `.select.csv` shows reasons                                                               |
| Variant map goes the wrong way                                               | Wrong frequency for 咁 or 枱                           | Pick the larger Cifu count; review in M1                                                                             |
| `spoken_mass` is circular: Cifu gives the tokenizer, the rank, and the test  | Values are too high                                    | Use it to compare methods only; G10 compares with the current ranker                                                 |
| Tatoeba Cantonese sentences are long (26.3% have 12 characters or fewer)     | Rule Y1 rejects 74% of them                            | Keep 12; a limit of 16 gains 0.7 points (3.4)                                                                        |
| Type caps drop good sentences (5 to 10 percent of the cards at the defaults) | Fewer cards                                            | Each cap has an option; `.select.csv` shows `type_cap`; the vocabulary loss is under 1 point of `spoken_mass` (14.6) |
| The four type keys are surface forms only                                    | Two sentences with one meaning can have different keys | Open question 8                                                                                                      |

### 12.2 Open questions

| #   | Question                                                                 | Evidence                                                                  | Decide at |
| --- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------- | --------- |
| 1   | Is Cifu usable in a public build? Ask the authors, or use HKCanCor alone | GPL-3.0 repository file; data from CC BY 4.0 HKCanCor and other corpora   | M0        |
| 2   | Is a spoken weight of 0.7 right?                                         | Compare 0.5, 0.7, and 0.9 on hand-marked sentences                        | M2        |
| 3   | Add a Cantonese part-of-speech check to replace F9?                      | No Cantonese tagger in `jieba.posseg`                                     | M3        |
| 4   | Add a `--targets swadesh` mode (`vocab/english-cantonese.csv`)?          | The Mandarin pack uses Swadesh targets                                    | later     |
| 5   | Add a stop rule for small gain, such as `--min-gain`?                    | The last 236 Tatoeba cards add 0.1 point (3.3)                            | M4        |
| 6   | Should `sqrt(blend_pm)` replace the log-rank weight?                     | +0.6 to 0.7 points, -37 to 50 types (14.5)                                | M4        |
| 7   | Are the share defaults (20, 8, 75, and 3 percent) right?                 | Shares 15, 6, 70, and 2 percent remove 19 percent of the LTL cards (14.6) | M4        |
| 8   | Is a meaning type needed, such as greeting, number, or place?            | The four keys are surface forms only                                      | M4        |

## 13. Milestones

| Milestone | Issue      | Task                                                                 | Output                                  |
| --------- | ---------- | -------------------------------------------------------------------- | --------------------------------------- |
| M0        | `anki-4uh` | Fetch Cifu to the cache; check the license; add `CIFU_PATH`          | cached file, license note               |
| M1        | `anki-vfx` | Reader, blend, variant map, overrides, `anki freq yue`               | `data/yue_frequency.tsv`; G1, G3, G4    |
| M2        | `anki-d1o` | Tokenizer, `--freq` option, change report for `anki-afn`             | report of changed sentences; G2         |
| M3        | `anki-8ox` | Rules Y1 to Y7, `.select.csv`                                        | G5, G7                                  |
| M4        | `anki-7y5` | Greedy cover, similarity, type caps, learning order, `.coverage.csv` | G6, G9, G10, G11, G12; options complete |
| M5        | `anki-0ne` | README, `ATTRIBUTION.md`, tests, regression gate G8                  | release check                           |
| M6        | `anki-5z4` | Tatoeba Cantonese pool: download, English links, command spec        | numbers of 3.4 reproduced; second spec  |

Order: M0 to M5 run in sequence. M2 comes before the default changes. It
reports each sentence whose rank moves by more than 100 places, so the user
can check the result.

Acceptance:

| Milestone | Condition                                                                                                                                                                                                 |
| --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M4        | LTL deck, defaults: all 224 top-500 words of the passed set, at most 650 cards, `spoken_mass` 66.8%, no type over its limit (G10, G11, G12). The user marks 5 percent of the kept cards as good sentences |
| M6        | Tatoeba pool: 500 cards give `spoken_mass` of 77.0% or more; the default run gives about 1,200 cards (3.1, 3.3)                                                                                           |

## 14. Appendix: evidence

All numbers come from a test run on 2026-10-19. The test deck is the LTL
deck: 999 sentences, 1000 rows in the rank CSV.

### 14.1 Frequency sources

| Source                            | Genre           | Size                                      | License                                     | Verdict                            |
| --------------------------------- | --------------- | ----------------------------------------- | ------------------------------------------- | ---------------------------------- |
| Apple Daily list (in use today)   | Written news    | 493.9M tokens, 275k types                 | CC BY 4.0                                   | Keep as the written term           |
| Cifu `SpokenAdult` column         | Spoken adult    | 306,339 tokens, 11,970 types with a count | repo file is GPL-3.0; source corpora differ | Use as the spoken term             |
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

| Measure                    | jieba default | jieba + Cifu words | PyCantonese `segment` |
| -------------------------- | ------------- | ------------------ | --------------------- |
| Tokens (Han only)          | 4626          | 4303               | 4094                  |
| Types                      | 1169          | 1133               | 1112                  |
| Tokens with a spoken count | 91.1%         | 92.3%              | 85.4%                 |
| Types with a spoken count  | 70.5%         | 75.9%              | 61.7%                 |
| Tokens with any Cifu count | 95.8%         | 98.4%              | 88.3%                 |

- A merged dictionary with the Apple Daily words was worse (82.1% spoken
  tokens), because it adds long written-only words.
- PyCantonese adds a 40 MB dependency and scores lower.
- The jieba default splits Cantonese wrongly. Example: 我仲讀 stays as 我 and
  仲讀. The Cifu dictionary does not fix every case, so G2 guards the result.

### 14.3 Sentence rules on the LTL deck

The table shows the first failed rule for each sentence.

| Result                   | Count |
| ------------------------ | ----- |
| Pass                     | 872   |
| Y1 length not 3 to 12    | 31    |
| Y2 Latin letter or digit | 50    |
| Y4 not one final mark    | 14    |
| Y7 more than 2 unknown   | 32    |
| Total                    | 999   |

- Example of a Y2 failure: `送Tom走。` has rank 3 in the current output. The
  current ranker does not remove it.
- Han length of the deck: 4 to 8 characters for 814 of 999 sentences.

### 14.4 Vocabulary reach of the LTL deck

| Measure                | All 999 sentences | The 872 that pass |
| ---------------------- | ----------------- | ----------------- |
| Types                  | 1133              | 999               |
| Top 500 blended words  | 236               | 224               |
| Top 1000 blended words | 345               | 328               |
| Top 1500 blended words | 414               | 392               |
| `spoken_mass`          | 67.7%             | 66.8%             |

The LTL deck is small, so it limits the result. A selection step cannot add
words that the deck does not have. It still removes repeat sentences, and it
puts the deck in a learning order. A larger set gives more diversity (3.1,
3.4).

### 14.5 Test of four weight functions (no type caps)

Values are types, top1000, and `spoken_mass`, with a cap of 300 or 500 cards.

| Weight           | LTL 300        | Tatoeba 300    | Tatoeba 500      |
| ---------------- | -------------- | -------------- | ---------------- |
| `1/log2(rank+1)` | 694, 306, 65.9 | 977, 476, 74.8 | 1,329, 528, 77.2 |
| `blend_pm`       | 582, 328, 66.2 | 841, 483, 75.2 | 1,237, 572, 77.8 |
| `sqrt(blend_pm)` | 628, 323, 66.3 | 927, 483, 75.4 | 1,292, 562, 77.9 |
| 1 (no weight)    | 718, 287, 64.5 | 989, 444, 72.3 | 1,334, 513, 76.3 |

- `blend_pm` gains at most 0.6 points of `spoken_mass`, and it loses 90 to
  140 types at 300 cards.
- The weight 1 loses up to 2.5 points.
- `sqrt(blend_pm)` gains 0.6 to 0.7 points and loses 37 to 50 types (open
  question 6).
- Decision: use `1/log2(rank+1)`, which keeps both measures high.

### 14.6 Test of type caps

Largest value of each type key, as a share of the cards. The selection
column is the defaults without type caps (681 LTL cards, 1,307 Tatoeba
cards).

| Type key            | LTL pool | LTL selection | Tatoeba pool | Tatoeba selection |
| ------------------- | -------- | ------------- | ------------ | ----------------- |
| `opening`           | 我 25%   | 我 24%        | 我 24%       | 我 23%            |
| `ending`            | 呀 10%   | 呀 9%         | 呀 9%        | 呀 6%             |
| `form` (statements) | 71%      | 73%           | 75%          | 80%               |
| `frame`             | `_` 8%   | `_` 9%        | `_` 6%       | `_` 6%            |

- 我, 你, and 佢 open about half of the passed Tatoeba sentences (24%, 15%,
  and 14%).
- The selection makes questions rarer. They are 21% of the Tatoeba pool and
  15% of the selection without caps. With caps they are 16%.
- `_` is a sentence with no top-60 word. It is the largest frame in every run.

Test of three cap strengths. Each row lists the shares for opening, ending,
form, and frame. The planned size `P` is 681 and 1,307 for the default runs.

| Run              | Caps       | Cards | Types | top1000 | `spoken_mass` | Statements | Largest opening | Largest frame |
| ---------------- | ---------- | ----- | ----- | ------- | ------------- | ---------- | --------------- | ------------- |
| LTL defaults     | none       | 681   | 984   | 328     | 66.8%         | 73%        | 24%             | 9%            |
| LTL defaults     | 20/8/75/3  | 613   | 933   | 326     | 66.8%         | 73%        | 22%             | 3%            |
| LTL defaults     | 15/6/70/2  | 552   | 875   | 323     | 66.7%         | 73%        | 19%             | 3%            |
| LTL defaults     | 25/10/80/5 | 656   | 960   | 327     | 66.8%         | 73%        | 25%             | 5%            |
| LTL cap 300      | none       | 300   | 694   | 306     | 65.9%         | 77%        | 24%             | 11%           |
| LTL cap 300      | 20/8/75/3  | 300   | 671   | 303     | 65.9%         | 75%        | 20%             | 3%            |
| LTL cap 300      | 15/6/70/2  | 300   | 651   | 305     | 65.9%         | 70%        | 15%             | 2%            |
| LTL cap 300      | 25/10/80/5 | 300   | 687   | 302     | 66.0%         | 76%        | 25%             | 5%            |
| Tatoeba defaults | none       | 1,307 | 2,058 | 583     | 79.9%         | 80%        | 23%             | 6%            |
| Tatoeba defaults | 20/8/75/3  | 1,236 | 2,020 | 583     | 79.8%         | 79%        | 21%             | 3%            |
| Tatoeba defaults | 15/6/70/2  | 1,172 | 1,971 | 580     | 79.7%         | 78%        | 17%             | 2%            |
| Tatoeba defaults | 25/10/80/5 | 1,303 | 2,058 | 583     | 79.9%         | 80%        | 23%             | 5%            |
| Tatoeba cap 500  | none       | 500   | 1,329 | 528     | 77.2%         | 83%        | 21%             | 5%            |
| Tatoeba cap 500  | 20/8/75/3  | 500   | 1,318 | 547     | 77.6%         | 75%        | 20%             | 3%            |
| Tatoeba cap 500  | 15/6/70/2  | 500   | 1,304 | 534     | 77.3%         | 70%        | 15%             | 2%            |
| Tatoeba cap 500  | 25/10/80/5 | 500   | 1,325 | 535     | 77.5%         | 80%        | 20%             | 5%            |

- The default caps (20/8/75/3) remove 10% of the LTL cards and 5% of the
  Tatoeba cards at the defaults. `spoken_mass` falls by 0.0 and 0.1 points.
- With a card cap, the default caps cost 23 of 694 types (LTL 300) and 11 of
  1,329 types (Tatoeba 500). At Tatoeba 500, `top1000` rises from 528 to 547
  and `spoken_mass` rises from 77.2% to 77.6%, because the caps force more
  questions and exclamations into the set.
- The strict caps (15/6/70/2) remove 19% of the LTL cards at the defaults.
- The loose caps (25/10/80/5) almost never bind.
- Decision: use 20/8/75/3.

Repeats of single words. A content word is a word with a rank worse than 60.
No content word is in more than 5% of the cards. The largest is 嚟: 5% of the
LTL cards and 4% of the Tatoeba cards. The greedy gain `weight / (n + 1)`
already limits repeats, so the spec has no cap on single words.
