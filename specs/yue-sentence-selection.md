# Spec: Cantonese Sentence Selection with Spoken Frequency

Status: draft
Date: 2026-10-19
Revision: 2026-10-19. Added 1.1 to 1.5: goals, card counts, and the numbers.
Owner: anki-tools
Related: `specs/mandarin-swadesh-sentence-pack.md`, beads `anki-afn`, `anki-58i`

## 1. Purpose

`anki fix yue deck.apkg` ranks and reorders a Cantonese deck. It does not
remove any card, and its frequency term comes from written news text. This
spec adds three things:

1. A spoken word frequency list for Cantonese.
2. A filter for good sentences, based on the proper-sentence rules of the
   Mandarin pack.
3. A selection step that keeps a diverse vocabulary, weighted to the most
   frequent spoken words.

The goal is the same as the Mandarin pack: a small deck for listening
comprehension, with short natural sentences and no near-duplicates.

### 1.1 Goals

1. Keep as much vocabulary diversity as the input allows, with the fewest
   cards.
2. Weight that diversity to the most frequent words of spoken Cantonese, not
   written Cantonese.
3. Keep each sentence short, complete, and unlike the other kept sentences.

The spec measures diversity with these numbers:

| Measure                        | Meaning                                                                                                  |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `cards`                        | Sentences kept                                                                                           |
| `types`                        | Distinct words in the kept sentences                                                                     |
| `top500`, `top1000`, `top1500` | How many of the 500, 1000, or 1500 most frequent words of the blend list (6.2) are in the kept sentences |
| `spoken_mass`                  | Share of the 306,339 Cifu spoken tokens that belong to a word in the kept sentences                      |

`spoken_mass` estimates the coverage of running speech. A learner who knows
every kept word recognizes this share of spoken tokens. The tokenizer and the
rank list both come from Cifu, so the absolute values are high. Use
`spoken_mass` to compare methods. Do not use it as a claim about real speech.

### 1.2 Expected number of cards

The test used two inputs. The settings are the defaults of section 11:
`--per-word 2`, `--top-words 1500`, `--known-top 2000`, `--max-unknown 2`,
3 to 12 Han characters, similarity 0.67, and no `--max-sentences`.

| Input                  | Sentences in | Pass Y1 to Y7 | Cards kept | Types | top500 | top1000 | top1500 | spoken_mass |
| ---------------------- | ------------ | ------------- | ---------- | ----- | ------ | ------- | ------- | ----------- |
| LTL deck (`.apkg`)     | 999          | 871           | 679        | 982   | 224    | 328     | 392     | 66.8%       |
| Tatoeba Cantonese pool | 20,853       | 4,254         | 1,307      | 2,058 | 357    | 583     | 791     | 79.9%       |

The selection removes cards that add no new target word. It keeps the
frequent vocabulary:

| Input   | Cards removed from the passed pool | Types lost     | Top 1500 words lost | `spoken_mass` lost |
| ------- | ---------------------------------- | -------------- | ------------------- | ------------------ |
| LTL     | 192 of 871 (22%)                   | 16 of 998      | 0                   | 0.0 points         |
| Tatoeba | 2,947 of 4,254 (69%)               | 1,418 of 3,476 | 0                   | 0.6 points         |

The types that the Tatoeba run loses are rare words, outside the target set.

The rules of section 7 also cost vocabulary before the selection runs:

- LTL deck: 12 of the 236 top-500 words appear only in rejected sentences.
  `spoken_mass` falls from 67.7% to 66.8%.
- Tatoeba pool: `spoken_mass` falls from 95.8% to 80.5%, and top500 falls
  from 457 to 357. Rule Y1 causes most of the loss: it rejects 15,445 of
  20,853 sentences, because only 26.3% of them have 12 Han characters or
  fewer.

### 1.3 Cards for each setting

| Input   | `--top-words` | `--per-word 1` | `--per-word 2` | `--per-word 3` |
| ------- | ------------- | -------------- | -------------- | -------------- |
| LTL     | 500           | 296            | 415            | 486            |
| LTL     | 1000 or more  | 559            | 679            | 734            |
| Tatoeba | 500           | 263            | 459            | 622            |
| Tatoeba | 1000          | 554            | 896            | 1,173          |
| Tatoeba | 1500          | 829            | 1,307          | 1,645          |
| Tatoeba | 2000          | 1,133          | 1,686          | 2,053          |

The LTL rows for 1000 or more are equal, because the passed deck has only 969
ranked words. At `--per-word 1`, each card carries 1.7 to 1.8 target words.
A second card for each word is often not possible: 341 of the 969 LTL target
words and 997 of the 1500 Tatoeba target words reach 2 cards.

### 1.4 Cards with a cap, against the current ranking

`--max-sentences` stops the selection early. The table compares the new
selection with the first N sentences of the passed pool in the order of the
current ranker. Each cell shows the new value, then the current value.

| Input   | Cards | Types         | top500    | top1000   | `spoken_mass` |
| ------- | ----- | ------------- | --------- | --------- | ------------- |
| LTL     | 300   | 693 / 410     | 216 / 147 | 306 / 195 | 66.0 / 57.4   |
| LTL     | 500   | 893 / 618     | 224 / 185 | 328 / 257 | 66.8 / 62.5   |
| LTL     | 679   | 982 / 803     | 224 / 202 | 328 / 290 | 66.8 / 65.0   |
| Tatoeba | 300   | 977 / 357     | 316 / 138 | 476 / 179 | 74.8 / 54.8   |
| Tatoeba | 500   | 1,329 / 572   | 336 / 178 | 528 / 244 | 77.2 / 60.5   |
| Tatoeba | 1,000 | 1,919 / 1,026 | 357 / 237 | 583 / 355 | 79.8 / 66.5   |
| Tatoeba | 1,307 | 2,058 / 1,306 | 357 / 260 | 583 / 393 | 79.9 / 69.9   |

For the LTL deck, the current order is the real `rank.csv` order. For
Tatoeba, the current order uses the present frequency score (Apple Daily) and
complexity score, without the similarity penalty. It is an approximation.

How to read the table:

- At 300 cards, the LTL selection keeps 69% more types and 8.6 points more
  `spoken_mass`.
- At 500 cards, the Tatoeba selection keeps 2.3 times the types and 16.7
  points more `spoken_mass`.
- The curve flattens. Tatoeba gains 2.6 points from 500 to 1,000 cards, and
  0.1 point from 1,000 to 1,307 cards. A cap of 500 gives 97% of the final
  `spoken_mass` (77.2 of 79.9) with 38% of the cards.
- The LTL deck needs no cap. 500 cards already reach the 66.8% of the full
  result.

### 1.5 The Tatoeba Cantonese pool

The export of 2026-10-17 has these counts:

| Measure                                   | Value                                       |
| ----------------------------------------- | ------------------------------------------- |
| Cantonese sentences                       | 20,853                                      |
| Sentences with 12 Han characters or fewer | 26.3%                                       |
| Sentences that pass Y1 to Y7              | 4,254                                       |
| Sentences with an English link            | 7,038                                       |
| Passed sentences with an English link     | 3,644                                       |
| Sentences with native audio               | 1,784                                       |
| Audio license (1,781 of 1,784 rows)       | CC BY 4.0                                   |
| Default cards with an English link        | 1,113 of 1,307                              |
| Default cards with native audio           | 402 of 1,307                                |
| Cards when the pool needs an English link | 1,291 (`spoken_mass` 79.3%, 455 with audio) |

Effect of a longer limit (`--max-han`) on the Tatoeba run:

| `--max-han` | Pass pool | Cards | Types | `spoken_mass` |
| ----------- | --------- | ----- | ----- | ------------- |
| 12          | 4,254     | 1,307 | 2,058 | 79.9%         |
| 14          | 4,593     | 1,297 | 2,109 | 80.4%         |
| 16          | 4,758     | 1,287 | 2,138 | 80.6%         |

A limit of 16 gains 0.7 points and gives longer sentences. Keep 12.

### 1.6 How the numbers were made

A prototype of sections 6 to 8 made all numbers in 1.2 to 1.5. The prototype
is not in the repository. It used the Cifu file and the Apple Daily file of
2026-10-19, the Cifu tokenizer (6.3), the blend of 6.2 with `w = 0.7`, the
override `你好` = `多謝`, and rules Y1 to Y7. Tatoeba data came from
`downloads.tatoeba.org/exports/per_language/yue/`. A count can differ by one
from a later table because of a tie in the sort. Gate G10 and milestone M4
turn these numbers into tests.

## 2. Scope

In scope:

- A spoken frequency list, built from public corpora and cached locally.
- A Cantonese-aware word tokenizer.
- New options for `anki fix yue`.
- A selection report that shows what the filter removed, and why.
- A new `anki freq yue` command that writes the frequency list.
- The Tatoeba Cantonese export as a second input (M6). Section 1.5 gives
  the numbers. Its command is in a later spec.

Out of scope:

- Audio, TTS, and new card templates.
- The command and the card fields for the Tatoeba pool (a later spec).
- Changes to `fix zh`, `fix fr`, or `fix es`.
- Swadesh targets. The Mandarin pack uses them. This spec uses the most
  frequent words instead (6.1).

## 3. Research findings

All numbers come from a test run on 2026-10-19. The test deck is
`apkg/LTL Cantonese Deck Level 1 - Short.apkg`: 999 sentences, 1000 rows in
the rank CSV.

### 3.1 Frequency sources

| Source                            | Genre           | Size                                      | License                                     | Verdict                            |
| --------------------------------- | --------------- | ----------------------------------------- | ------------------------------------------- | ---------------------------------- |
| Apple Daily list (in use today)   | Written news    | 493.9M tokens, 275k types                 | CC BY 4.0                                   | Keep as the written term           |
| Cifu `SpokenAdult` column         | Spoken adult    | 306,339 tokens, 11,970 types with a count | repo file is GPL-3.0; source corpora differ | Use as the spoken term             |
| Cifu `Written` column             | Written         | 2.36M tokens                              | as above                                    | Use for the written term           |
| HKCanCor                          | Spoken, 1997-98 | about 230k words                          | CC BY 4.0                                   | Already inside Cifu; no direct use |
| cantfreq (HKCanCor, HKCAC, IARPA) | Spoken          | 3 corpora                                 | no license file                             | Do not use                         |
| UD Cantonese-HK                   | Film subtitles  | small                                     | see repo                                    | Later, for validation only         |

Cifu (Lai and Winterstein 2020) has 51,798 entries. Each entry has a
Jyutping reading and counts for four genres. Its `SpokenAdult` column comes
from HKCanCor and CantoMap. The spoken sample is small. It gives a good rank
for about the top 2000 words only. Words below a count of 5 need the written
term (6.2).

Spoken and written text rank words in a different order. The top spoken words
are 係 呢 嘅 我 你 佢 唔 就 個 咁. The written list ranks the particles low.
Examples from `anki-afn`: 早晨 has rank 20387 and 你好 has rank 18180 in
the written list.

### 3.2 Coverage on the LTL deck

| Measure                    | jieba default | jieba + Cifu words | PyCantonese `segment` |
| -------------------------- | ------------- | ------------------ | --------------------- |
| Tokens (Han only)          | 4626          | 4303               | 4094                  |
| Types                      | 1169          | 1133               | 1112                  |
| Tokens with a spoken count | 91.1%         | 92.3%              | 85.4%                 |
| Types with a spoken count  | 70.5%         | 75.9%              | 61.7%                 |
| Tokens with any Cifu count | 95.8%         | 98.4%              | 88.3%                 |

Result: use jieba with a Cifu dictionary (6.3). A merged dictionary with the
Apple Daily words was worse (82.1% spoken tokens), because it adds long
written-only words. PyCantonese adds a 40 MB dependency and scores lower.

The jieba default splits Cantonese wrongly. Example: 我仲讀 stays as 我 and
仲讀. The Cifu dictionary does not fix every case, so the tokenizer has a
gate (13, G2).

### 3.3 Effect of the proper-sentence rules on the LTL deck

The rules are in section 7. This table shows the first failed rule for each
sentence.

| Result                   | Count |
| ------------------------ | ----- |
| Pass                     | 871   |
| Y1 length not 3 to 12    | 31    |
| Y2 Latin letter or digit | 50    |
| Y4 not one final mark    | 15    |
| Y7 more than 2 unknown   | 32    |
| Total                    | 999   |

Example of a Y2 failure: `送Tom走。` has rank 3 in the current output. The
current ranker does not remove it.

Han length of the deck: 4 to 8 characters for 814 of 999 sentences.

### 3.4 Vocabulary reach of the deck

| Measure                | All 999 sentences | The 871 that pass |
| ---------------------- | ----------------- | ----------------- |
| Types                  | 1133              | 998               |
| Top 500 blended words  | 236               | 224               |
| Top 1000 blended words | 345               | 328               |
| Top 1500 blended words | 414               | 392               |
| `spoken_mass`          | 67.7%             | 66.8%             |

The LTL deck is small, so it limits the result. A selection step cannot add
words that the deck does not have. It still removes repeat sentences, and it
puts the deck in a learning order. A larger Cantonese set gives more
diversity (1.2 and 1.5).

## 4. Data sources and licenses

| Data                  | Where                                                                       | Cache path                                     |
| --------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Apple Daily frequency | `raw.githubusercontent.com/chaaklau/appledaily-frequency/.../word-freq.tsv` | `~/.cache/anki-tools/appledaily-word-freq.tsv` |
| Cifu lexicon          | `github.com/gwinterstein/Cifu`, `Lexicon/Cifu-v1.txt`                       | `~/.cache/anki-tools/cifu-v1.tsv`              |

Rules:

- Download both files to the cache, in the same way as the Apple Daily file
  now. Do not commit them.
- The Cifu repository has a GPL-3.0 `LICENSE` file. The data comes from
  corpora with other licenses. Do not copy Cifu into the repository. Check
  the license status before any public release (milestone M0).
- Write the generated list to `data/yue_frequency.tsv`. The `data/` folder
  is in `.gitignore`.
- Add these lines to `ATTRIBUTION.md`: Cifu (Lai and Winterstein 2020),
  HKCanCor (Luke and Wong 2015, CC BY 4.0), Apple Daily list (CC BY 4.0).
- `YUE_FREQUENCY_PATH` keeps its meaning for the Apple Daily file. Add
  `CIFU_PATH` for the Cifu file.

## 5. Pipeline

| Stage | Name           | Input              | Output                                        |
| ----- | -------------- | ------------------ | --------------------------------------------- |
| Y0    | Fetch          | network            | cached Cifu and Apple Daily files             |
| Y1    | Frequency list | Y0, override files | `data/yue_frequency.tsv`                      |
| Y2    | Tokenize       | Y1, deck sentences | tokens per sentence                           |
| Y3    | Filter         | Y2                 | passed and rejected sentences, reasons        |
| Y4    | Select         | Y3, Y1             | kept sentences in learning order              |
| Y5    | Write          | Y4, original deck  | `_reordered.apkg`, `.rank.csv`, `.select.csv` |

Y1 is also a stand-alone command (`anki freq yue`). Y2 to Y5 run inside
`anki fix yue`. Each stage gives the same output for the same input.

## 6. Y1: Frequency list

### 6.1 Output

`data/yue_frequency.tsv` has one row per word. The columns are:

| Column         | Type   | Example | Meaning                                          |
| -------------- | ------ | ------- | ------------------------------------------------ |
| `word`         | string | 係      | Traditional form, after the variant map (6.4)    |
| `jyutping`     | string | hai6    | Cifu reading; empty when the word is not in Cifu |
| `spoken_count` | int    | 15318   | Cifu `SpokenAdult`                               |
| `spoken_pm`    | float  | 50003.4 | Spoken count per million tokens                  |
| `written_pm`   | float  | 12164.2 | Mean of the Cifu and Apple Daily written values  |
| `blend_pm`     | float  | 38651.6 | Value from 6.2                                   |
| `rank`         | int    | 1       | Rank by `blend_pm`, 1 is most frequent           |
| `source`       | string | cifu+ad | `cifu+ad`, `cifu`, `ad`, or `override`           |

The file is sorted by `rank`. A person can read it, and it can load in
any tool.

### 6.2 Blend

```text
written_pm = mean of (Cifu written count per million, Apple Daily count per million)
             over the sources that have the word
blend_pm   = w * spoken_pm + (1 - w) * written_pm      if spoken_count >= 5
blend_pm   = written_pm                                 otherwise
```

`w` is `--spoken-weight`, default 0.7.

Test result with `w = 0.7`:

| Word | Written-only rank | Blend rank |
| ---- | ----------------- | ---------- |
| 早晨 | 20387             | 871        |
| 唔該 | not measured      | 592        |
| 多謝 | not measured      | 402        |
| 點解 | not measured      | 125        |
| 你好 | 18180             | 28344      |

You can see that the blend fixes most spoken words, but it does not fix
你好. HKCanCor writes 你好 as two tokens 你 and 好, so Cifu has no entry for
it. Add an override file (6.5).

`--freq` selects the term: `blend` (default), `spoken`, or `written`.
`written` gives the current behavior (Apple Daily only).

### 6.3 Tokenizer

Build one `jieba.Tokenizer` instance for Cantonese. Do not change the global
jieba dictionary, because the Mandarin code uses it.

- Dictionary: every Cifu word that has a Han character and a count above 0.
- Weight: `spoken_count * 40 + written_count * 2`. A spoken count has 20
  times the weight of a written count. The test in 3.2 used these weights.
- Word list from overrides (6.5) is added at the same weight as the top
  word.
- A test on the LTL deck decides the weights (13, G2). Change the weights
  only when the gate result improves.

### 6.4 Variant map

The written lists use news-standard characters. Spoken text uses common
variants. This is beads `anki-58i`. Add `vocab/yue_variants.tsv` with two
columns: `variant` and `standard`. Apply it to a token only for the
frequency lookup, never to the card text. Start with:

| Variant | Standard | Meaning |
| ------- | -------- | ------- |
| 咁      | 噉       | so      |
| 枱      | 檯       | table   |

The direction is open. Pick the form that has the larger count in Cifu, and
record the result in the file (M1).

### 6.5 Overrides

`vocab/yue_frequency_overrides.tsv` has three columns: `word`, `rank_like`,
and `reason`. A row gives the word the blend value of the word named in
`rank_like`. Example row: `你好`, `多謝`, `HKCanCor splits the greeting`.
Use it for words that the corpora split or miss. Start with the greetings
你好, 晚安, and 拜拜. Review each row by hand.
A word in this file has `source=override`.

## 7. Y3: Filter good sentences

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

Differences from the Mandarin rules, and why:

- Y1 is wider. Cantonese sentences are short, and 3 character sentences
  such as 你先。 are good cards. 814 of 999 LTL sentences have 4 to 8
  characters.
- There is no predicate rule (F9). `jieba.posseg` has no Cantonese tagger.
  Many Cantonese sentences end in a particle (呀, 喎, 嘅), and the tags give
  false failures. Revisit after M2.
- There is no orphan-conjunction rule (F10). The test did not check it.
  Revisit after M2.

The known set is the first `--known-top` words (default 2000) of the Y1 list.
A token is unknown when it has no Y1 rank, or a rank above `--known-top`.
Punctuation tokens do not count.

A deck may have sentences with no final mark. If more than 20 percent of the
deck fail Y4, print a warning and tell the user to use `--no-final-mark`
(11, options). That option drops Y4 for the run.

## 8. Y4: Selection for vocabulary diversity

### 8.1 Target words

The target set is the `--top-words` most frequent words (default 1500) among
the words that appear as a token in at least one passed sentence. The order
is the Y1 rank. A word with no Y1 rank is not a target. The coverage report
lists each top-`--top-words` word of the Y1 list that no passed sentence uses
as `absent`.

### 8.2 Greedy cover

Use greedy cover, as in the Mandarin pack. The loop is:

1. Give each target word a weight: `1 / log2(rank + 1)`. This favors the
   most frequent words, with a smooth fall.
2. Set the count `n[w] = 0` for each target word.
3. For each remaining sentence, compute the gain:
   `sum over target words w in the sentence, with n[w] < --per-word, of weight[w] / (n[w] + 1)`.
4. Keep the sentence with the highest gain. Break a tie by lower
   complexity, then by fewer Han characters, then by lower original order.
5. Add 1 to `n[w]` for each target word in the kept sentence.
6. Remove any sentence that is too similar to a kept sentence (8.3).
7. Stop when `--max-sentences` is reached, or when every gain is 0.

Defaults: `--per-word 2`, `--max-sentences` none. With no `--max-sentences`,
the loop stops when every gain is 0. Greedy cover is not always the smallest
set, but the test in 1.2 shows that the result keeps all top 1500 words that
the passed pool holds.

Test of four weight functions for step 1, with a cap of 300 or 500 cards
(types, top1000, `spoken_mass`):

| Weight           | LTL 300        | Tatoeba 300    | Tatoeba 500      |
| ---------------- | -------------- | -------------- | ---------------- |
| `1/log2(rank+1)` | 694, 306, 65.9 | 977, 476, 74.8 | 1,329, 528, 77.2 |
| `blend_pm`       | 582, 328, 66.2 | 841, 483, 75.2 | 1,237, 572, 77.8 |
| `sqrt(blend_pm)` | 628, 323, 66.3 | 927, 483, 75.4 | 1,292, 562, 77.9 |
| 1 (no weight)    | 718, 287, 64.5 | 989, 444, 72.3 | 1,334, 513, 76.3 |

The `blend_pm` weight gains at most 0.6 points of `spoken_mass`, and it loses
90 to 140 types at 300 cards. The weight 1 loses up to 2.5 points. Use
`1/log2(rank+1)`, which keeps both measures high.

### 8.3 Similarity

Reuse `char_similarity_zh` and `_char_similarity_zh_same_len` from the
existing code. Default threshold is 0.67, the same as `fix zh`. The test is
the same: two sentences with the same unique-character count and a Jaccard
value at or above the threshold are duplicates. Add a word-level test as
well: two sentences with the same token set are duplicates.

### 8.4 Learning order

Sort the kept sentences so that each card uses words that earlier cards
introduced:

1. The rank of the least frequent known token of the sentence, ascending.
   This is the `hardest_rank`.
2. Complexity score from `complexity_score_zh`, ascending.
3. Han length, ascending.
4. Original order, ascending.

The output order replaces the combined score order of `rank_sentences_yue`.
Use `--order score` to keep the old order.

## 9. Changes to `rank_sentences_yue`

- The frequency term uses `--freq`. `blend` is the default. The weights
  stay: complexity 0.3, frequency 0.5, similarity penalty 0.2.
- `Sentence` gets two fields: `hardest_rank: int` and `reject_reason: str`.
  The default is 0 and the empty string.
- `load_frequency_data_yue` gets a `source` parameter with the values
  `written`, `spoken`, and `blend`. `written` keeps today's output.
- Change in default behavior: the blend changes the order of the existing
  output. Per `anki-afn`, report the changed sentences before the default
  changes. M2 produces this report.

## 10. Output files

For `anki fix yue "deck.apkg"`:

| File                  | Content                                                                     |
| --------------------- | --------------------------------------------------------------------------- |
| `deck_reordered.apkg` | The kept cards, in the learning order                                       |
| `deck.rank.csv`       | The existing columns plus `hardest_rank` and `kept`                         |
| `deck.select.csv`     | One row per input sentence: `sentence`, `kept`, `reason`, `targets`, `gain` |
| `deck.coverage.csv`   | One row per target word: `word`, `rank`, `count`, `status`                  |

`status` is `covered`, `partial` (count below `--per-word`), or `absent`.
`reason` is a rule name (`Y1` to `Y7`), `similar`, `no_gain`, or `cap`.
`similar` and `no_gain` come from Y4. `cap` means `--max-sentences` stopped
the loop.

A run prints a summary: sentences in, rejected by rule, removed by Y4,
kept, and the percent of the top 500 and the top 1500 words that reach
`--per-word`.

## 11. CLI

New command:

```bash
anki freq yue                        # Y0, Y1: write data/yue_frequency.tsv
anki freq yue --spoken-weight 0.8
anki freq yue --refresh              # download the sources again
```

New options for `anki fix yue`. The existing options stay.

| Option                   | Default  | Meaning                                               |
| ------------------------ | -------- | ----------------------------------------------------- |
| `--max-sentences`        | none     | Stop Y4 at this count. Without it, Y4 stops at 0 gain |
| `--per-word`             | 2        | Cards wanted for each target word                     |
| `--top-words`            | 1500     | Size of the target set                                |
| `--known-top`            | 2000     | Size of the known set for Y7                          |
| `--max-unknown`          | 2        | Unknown tokens allowed in a sentence                  |
| `--min-han`              | 3        | Minimum Han characters                                |
| `--max-han`              | 12       | Maximum Han characters                                |
| `--similarity-threshold` | 0.67     | Character Jaccard threshold                           |
| `--freq`                 | blend    | `blend`, `spoken`, or `written`                       |
| `--spoken-weight`        | 0.7      | Value of `w` in 6.2                                   |
| `--order`                | coverage | `coverage` (8.4) or `score` (today's order)           |
| `--no-select`            | off      | Run only the ranking, as today                        |
| `--no-final-mark`        | off      | Skip rule Y4                                          |

`--no-select` gives the current behavior and keeps old runs repeatable. Pass
`--freq written --order score --no-select` to get a byte-identical result.

## 12. Open questions

1. Tatoeba Cantonese as a candidate pool is in scope (M6). Section 1.5
   gives the numbers. It needs a download step, English links, and a command
   like `anki swadesh sentences`. Write that command in a second spec. The
   pool uses traditional characters already, but check variant characters.
2. Is Cifu usable in a public build? The repository file is GPL-3.0. The
   data comes from corpora with other terms. Ask the authors, or build the
   spoken list from HKCanCor alone (CC BY 4.0) with PyCantonese.
3. Is a spoken weight of 0.7 right? M2 compares 0.5, 0.7, and 0.9 on
   hand-marked sentences.
4. Add a Cantonese part-of-speech check to replace F9?
5. Should the Swadesh Cantonese list (`vocab/english-cantonese.csv`) add a
   `--targets swadesh` mode, to match the Mandarin pack?
6. Add a stop rule for small gain, such as `--min-gain`? The Tatoeba run
   adds 0.1 point of `spoken_mass` for the last 307 cards (1.4).
7. Should `sqrt(blend_pm)` replace the log-rank weight? It gains 0.6 to 0.7
   points at 300 and 500 cards, and loses 37 to 50 types (8.2).

## 13. Quality gates

`anki freq yue --check` and the unit tests run these checks. A failure stops
the run with a non-zero exit code.

| Gate | Check                                                                                                                           |
| ---- | ------------------------------------------------------------------------------------------------------------------------------- |
| G1   | `data/yue_frequency.tsv` has at least 40,000 rows, and its ranks are 1 to N with no gap                                         |
| G2   | On the LTL deck, at least 92 percent of Han tokens have a spoken count (measured 92.3)                                          |
| G3   | 係, 嘅, 唔, 我, 你 are in the first 20 ranks                                                                                    |
| G4   | 你好, 早晨, 唔該 are in the first 3000 ranks (overrides apply)                                                                  |
| G5   | No kept sentence fails Y1 to Y7                                                                                                 |
| G6   | No two kept sentences are duplicates by 8.3                                                                                     |
| G7   | Every kept sentence has a row in `.select.csv`, and the row counts match the input                                              |
| G8   | `--no-select --freq written --order score` gives the same ranking CSV as the current version                                    |
| G9   | The kept cards have a non-decreasing `hardest_rank` when `--order coverage`                                                     |
| G10  | LTL deck, `--max-sentences 300`: at least 650 types, 210 top500 words, and `spoken_mass` 65.0 or more (measured 693, 216, 66.0) |
| G11  | LTL deck, defaults: 679 cards, within 5 percent (measured 679)                                                                  |

G2 comes from section 3.2. G10 and G11 come from sections 1.2 and 1.4.
Lower a value only with a new measurement.

## 14. Tests

Add `tests/test_yue_selection.py`:

- Blend: counts below 5 use the written term; the weight changes the order.
- Cifu reader: skips the header, reads the empty count cells.
- Tokenizer: the instance does not change the global jieba result.
- Variant map: lookup only, card text unchanged.
- Rules Y1 to Y7 on a small fixture, one failing sentence for each rule.
- Greedy cover on a small fixture: a rare word wins over a repeat, and the
  `per-word` limit stops the gain.
- Similarity: an anagram pair and a same-token pair.
- Learning order: non-decreasing `hardest_rank`.
- Regression: `--no-select` result equals the stored ranking CSV.

Use small fixture files. Do not download in tests. Patch the download
function, as the existing yue tests do.

## 15. Risks

| Risk                                                                        | Effect                       | Mitigation                                                           |
| --------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------- |
| Spoken sample is small (306k tokens)                                        | Wrong rank below about 2000  | Use spoken only for count of 5 or more; blend                        |
| Cifu is biased to the HKCanCor 1997-98 vocabulary                           | Old words rank high          | Written term at weight 0.3; overrides                                |
| Cifu license is unclear for redistribution                                  | Cannot ship a derived list   | Cache only; `data/` ignored; question 2                              |
| Tokenizer splits words wrongly (我仲讀)                                     | Wrong target match           | Cifu dictionary; G2; fix by hand in overrides                        |
| The blend changes the order of the current output                           | Old decks look different     | `--no-select`, `--freq written`, M2 report                           |
| Small input deck limits diversity                                           | Selection gains little       | Report coverage; question 1                                          |
| Strict rules drop natural short sentences                                   | Fewer cards                  | Y1 starts at 3 characters; `.select.csv` shows reasons               |
| Variant map goes the wrong way                                              | Wrong frequency for 咁 or 枱 | Pick the larger Cifu count; review in M1                             |
| `spoken_mass` is circular: Cifu gives the tokenizer, the rank, and the test | Values are too high          | Use it to compare methods only; G10 compares with the current ranker |
| Tatoeba Cantonese sentences are long (26.3% have 12 characters or fewer)    | Rule Y1 rejects 74% of them  | Keep 12; a limit of 16 gains 0.7 points (1.5)                        |

## 16. Milestones

| Milestone | Task                                                          | Output                                 |
| --------- | ------------------------------------------------------------- | -------------------------------------- |
| M0        | Fetch Cifu to the cache; check the license; add `CIFU_PATH`   | cached file, license note              |
| M1        | Reader, blend, variant map, overrides, `anki freq yue`        | `data/yue_frequency.tsv`, G1, G3, G4   |
| M2        | Tokenizer, `--freq` option, change report for `anki-afn`      | report of changed sentences, G2        |
| M3        | Rules Y1 to Y7, `.select.csv`                                 | G5, G7                                 |
| M4        | Greedy cover, similarity, learning order, `.coverage.csv`     | G6, G9; `fix yue` options complete     |
| M5        | README, `ATTRIBUTION.md`, tests, regression gate G8           | release check                          |
| M6        | Tatoeba Cantonese pool: download, English links, command spec | numbers of 1.5 reproduced, second spec |

M2 comes before the default changes. The report shows each sentence whose
rank moves by more than 100 places, so the user can check the result.

Acceptance for M4: on the LTL deck with the defaults, the kept set has all
224 top-500 words of the passed set, at most 700 cards, and a `spoken_mass`
of 66.8% (G10, G11). The user marks 5 percent of the kept cards as good
sentences.

Acceptance for M6: on the Tatoeba pool, 500 cards give a `spoken_mass` of
77.0% or more, and the default run gives about 1,300 cards (1.2, 1.4).
