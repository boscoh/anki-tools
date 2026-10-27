# Spec: Cantonese Frequency List

| Field   | Value                                        |
| ------- | -------------------------------------------- |
| Status  | Done                                         |
| Date    | 2026-10-19                                   |
| Owner   | anki-tools                                   |
| Related | `specs/cantonese-swadesh-sentence-pack.md`   |

## 1. Summary

The Cantonese core mode needs a word frequency list that ranks
spoken words well. The Apple Daily list comes from written news text, and it
ranks the spoken particles low. This spec blends that list with the Cifu
spoken counts.

`anki_tools/languages/cantonese/frequency.py` holds the code. The list is built
in memory on each run, so no file is written.

## 2. Frequency list

### 2.1 Row fields

The list has one row per word, sorted by `rank`.

| Column         | Type   | Example | Meaning                                          |
| -------------- | ------ | ------- | ------------------------------------------------ |
| `word`         | string | 係      | Traditional form, after the variant map (2.3)    |
| `jyutping`     | string | hai6    | Cifu reading; empty when the word is not in Cifu |
| `spoken_count` | int    | 15318   | Cifu `SpokenAdult`                               |
| `spoken_pm`    | float  | 50003.4 | Spoken count per million tokens                  |
| `written_pm`   | float  | 12164.2 | Mean of the Cifu and Apple Daily written values  |
| `blend_pm`     | float  | 38651.6 | Value from 2.2                                   |
| `rank`         | int    | 1       | Rank by `blend_pm`, 1 is most frequent           |
| `source`       | string | cifu+ad | `cifu+ad`, `cifu`, `ad`, or `override`           |

### 2.2 Blend

```text
written_pm = mean of (Cifu written count per million, Apple Daily count per million)
             over the sources that have the word
blend_pm   = w * spoken_pm + (1 - w) * written_pm      if spoken_count >= 5
blend_pm   = written_pm                                 otherwise
```

- `w` is `SPOKEN_WEIGHT`, 0.7. The module sets it; no option changes it.
- The spoken sample is small, so a count below 5 uses the written term only.
- Words with equal `blend_pm` sort by `written_pm`, then by the word.
- A word must hold one Han character.

### 2.3 Variant map

The written lists use news-standard characters. Spoken text uses common
variants.

- File: `vocab/cantonese_variants.tsv`, columns `variant` and `standard`.
- Use: apply the map to a token for the frequency lookup only. Never change
  the card text.
- The counts of a variant add to the standard form.

| Variant | Standard | Meaning |
| ------- | -------- | ------- |
| 咁      | 噉       | so      |
| 枱      | 檯       | table   |

### 2.4 Overrides

- File: `vocab/cantonese_frequency_overrides.tsv`, columns `word`,
  `rank_like`, and `reason`.
- Effect: a row gives the word the blend value of the word in `rank_like`.
- Example row: `你好`, `多謝`, `HKCanCor splits the greeting`.
- A word from this file has `source=override`. A word with no row in the blend
  joins the list.

## 3. Tokenizer

Build one `jieba.Tokenizer` instance for Cantonese. Do not change the global
jieba dictionary, because the Mandarin code uses it.

| Setting    | Value                                                                    |
| ---------- | ------------------------------------------------------------------------ |
| Dictionary | Every Cifu word with a Han character and a count above 0                 |
| Weight     | `spoken_count * 40 + written_count * 2` (a spoken count counts 20 times) |
| Overrides  | Words from 2.4, at the same weight as the top word                       |

`spoken_token_coverage` measures the share of Han tokens that have a spoken
Cifu count (gate G2). The Cifu dictionary gives 92.3% on the LTL deck, against
91.1% for the jieba default and 85.4% for PyCantonese.

## 4. Data sources and licenses

| Data                  | Where                                                                       | Cache path                                     |
| --------------------- | --------------------------------------------------------------------------- | ---------------------------------------------- |
| Apple Daily frequency | `raw.githubusercontent.com/chaaklau/appledaily-frequency/.../word-freq.tsv` | `~/.cache/anki-tools/appledaily-word-freq.tsv` |
| Cifu lexicon          | `github.com/gwinterstein/Cifu`, `Lexicon/Cifu-v1.txt`                       | `~/.cache/anki-tools/cifu-v1.tsv`              |

Rules:

- Download both files to the cache. Do not commit them.
- The Cifu repository has a GPL-3.0 `LICENSE` file. The data comes from
  corpora with other licenses. Do not copy Cifu into the repository. Check
  the license status before any public release.
- Do not write the generated list to a file. The blend is built in memory on
  each run.
- `YUE_FREQUENCY_PATH` sets the Apple Daily file. `CIFU_PATH` sets the Cifu
  file.
- `ATTRIBUTION.md` names Cifu (Lai and Winterstein 2020), HKCanCor (Luke and
  Wong 2015, CC BY 4.0), and the Apple Daily list (CC BY 4.0).

## 5. Gates

`check_frequency` runs three gates on each build:

| Gate | Test                                                             |
| ---- | ---------------------------------------------------------------- |
| G1   | At least 40,000 rows, and the ranks are contiguous from 1        |
| G3   | `係`, `嘅`, `唔`, `我`, and `你` are in the top 20                |
| G4   | `你好`, `早晨`, and `唔該` are in the top 3000                    |

A failed gate stops the run.
