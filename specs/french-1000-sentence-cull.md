# Spec: French 1000-Card Cull from the Audio Deck

| Field   | Value                                                          |
| ------- | -------------------------------------------------------------- |
| Status  | implemented                                                       |
| Date    | 2026-10-20                                                     |
| Owner   | anki-tools                                                     |
| Input   | `apkg/6000_French_Sentences_w_Audio.apkg`                      |
| Related | `specs/yue-sentence-selection.md` (selection shape)            |
| Related | `specs/french-swadesh-sentence-pack.md` (French tokenizer)     |
| Issues  | epic `anki-ckn`; milestones D1 to D5 (section 13)              |

## 1. Summary

`apkg/6000_French_Sentences_w_Audio.apkg` holds 5,928 sentences with one audio
file each. This spec culls the deck to 1,000 cards. The kept cards give a wide
and diverse vocabulary, weighted to the most frequent spoken French words. No
non-exempt word appears in more than 4 cards.

The current order gives 1,272 distinct content words in 1,000 cards. The
selection of this spec gives 2,081, and it covers 299 of the 300 target words.
The count of the most-used non-exempt word falls from 131 cards to 4.

### 1.1 Result at a glance

| Measure                    | Current order | This spec |
| -------------------------- | ------------- | --------- |
| Cards                      | 1,000         | 1,000     |
| Distinct content words     | 1,272         | 2,081     |
| Top 300 words covered      | 282           | 299       |
| Top 300 words with 2 cards | 256           | 289       |
| Most cards of one non-exempt word | 131    | 4         |
| Most cards of one exempt word     | 105    | 172       |
| `spoken_mass`              | 0.473         | 0.543     |

`spoken_mass` is the share of the Tatoeba French tokens that the kept words
cover (section 3.2). Use it to compare methods. Do not use it as a claim about
real speech.

### 1.2 Key decisions

| Decision             | Choice                                                       | Evidence |
| -------------------- | ------------------------------------------------------------ | -------- |
| Frequency source     | A blend of `wordfreq` fr and Tatoeba fr counts at 0.5        | 4        |
| Tokenizer            | spaCy `fr_core_news_sm` lemma and part of speech             | 5        |
| Target words         | 300 content lemmas, the option range is 200 to 500           | 5, 9     |
| Selection            | Greedy cover, weight `1/log2(rank+1)`                        | 8, 9     |
| Cards per word       | At most 4, with the top 100 words exempt                     | 8.4, 9.3 |
| Similarity           | Lemma Jaccard at 0.67 or more is a duplicate                 | 8.5      |
| Audio                | Reuse the input MP3 files, no new TTS                        | 10       |
| Compatibility        | The output keeps the input model, fields, and deck order option | 10    |

## 2. Input analysis

The input deck is a Tatoeba French sentence list with synthesized audio. The
numbers come from a read of the APKG on 2026-10-20.

| Measure                          | Value          |
| -------------------------------- | -------------- |
| Cards and notes                  | 5,928          |
| Note model                       | `French Sentences` |
| Fields                           | `French`, `sound`, `Back`, `freq ranks` |
| Media files                      | 5,928          |
| Cards with a sound file          | 5,928          |
| Cards with an English `Back`     | 5,928          |
| Mean words per sentence          | 8.86           |
| Sentences with 3 to 12 words     | 5,766 (97.3%)  |
| Exact duplicate texts            | 0              |

Length of the input sentences:

| Words   | Sentences |
| ------- | --------- |
| 4       | 3         |
| 5       | 25        |
| 6       | 218       |
| 7       | 878       |
| 8       | 1,400     |
| 9       | 1,387     |
| 10      | 969       |
| 11      | 601       |
| 12      | 285       |
| 13      | 120       |
| 14      | 36        |
| 15      | 5         |
| 16      | 1         |

The `freq ranks` field holds about 9 numbers per sentence, with a maximum of
5,999. The source of the list is not known. The selection ignores the field and
computes its own frequency list (open question 5).

The input audio is TTS. The card text, the audio, and the English translation
come from Tatoeba and the deck author. Check the license before a public
release (open question 6).

## 3. Goals and scope

### 3.1 Goals

1. Keep 1,000 cards from the 5,928.
2. Cover 200 to 500 of the most frequent spoken French words. The default is
   300.
3. Keep the most cards that the input allows. The current order gives 1,272
   content words. Aim for 1,900 or more.
4. Do not use one non-exempt word in more than 4 cards.
5. Keep each sentence short, complete, and unlike the other kept sentences.
6. Reuse the input audio and the input note model.

### 3.2 Measures

| Measure             | Meaning                                                                |
| ------------------- | ---------------------------------------------------------------------- |
| `cards`             | Sentences kept                                                         |
| `content`           | Distinct content lemmas in the kept cards                              |
| `top200`, `top300`, `top500` | Target words that a kept card contains                       |
| `covered2`          | Target words that at least two kept cards contain                      |
| `max_uses`          | The most kept cards that hold one non-exempt content lemma             |
| `max_exempt`        | The same value for an exempt word                                      |
| `spoken_mass`       | Share of the Tatoeba French content tokens that the kept lemmas cover  |

### 3.3 In scope

- One cull mode of `anki fix fr` for a French sentence deck with audio.
- A frequency list from public corpora, cached locally.
- A filter, a selection, an APKG writer, a report, and gates.
- Options for the card count, the target count, the word cap, and the order.

### 3.4 Out of scope

- New audio or TTS.
- New card templates or a new note model.
- Changes to the French Swadesh pack.
- A general cull mode for other languages (a later spec).

## 4. Frequency analysis

`wordfreq` has a French list but no separate spoken list. Tatoeba French is a
large set of short everyday sentences. The spec blends the two terms:

```text
tatoeba_pm = Tatoeba lemma count / Tatoeba content tokens * 1e6
written_pm = 10 ** (wordfreq Zipf value - 3)
blend_pm   = w * tatoeba_pm + (1 - w) * written_pm   if count >= 5
blend_pm   = written_pm                              otherwise
```

- `w` is `--spoken-weight`, default 0.5.
- `--spoken-weight 0` gives the `wordfreq` list alone.
- `--spoken-weight 1` gives the Tatoeba list alone.

The Tatoeba input is `data/tatoeba/fra_sentences.tsv`: 728,898 sentences. The
lemma counts need one spaCy pass over the file. Cache the result in
`data/french_frequency.tsv`.

### 4.1 Effect of the blend

The blend changes the target list. The overlap of the top 500 content lemmas
against the `wordfreq` list:

| `--spoken-weight` | Overlap with `wordfreq` top 500 |
| ----------------- | ------------------------------- |
| 0.3               | 0.840                           |
| 0.5               | 0.740                           |
| 0.7               | 0.672                           |

Words that enter the top 500 at weight 0.5 and not at 0 (a sample): vouloir,
taire, aimer, falloir, connaître, appeler, chien, commencer, oublier, écrire,
sentir, bel, tomber, boire, sembler, dormir, rencontrer, mourir, chat.

The everyday words rise. A pure `wordfreq` list gives a target list with fewer
conversation words and more written words.

### 4.2 Top of the list

The first 20 content lemmas at weight 0.5:

| Rank | Lemma     | Deck count | Tatoeba count |
| ---- | --------- | ---------- | ------------- |
| 1    | pas       | 754        | 105,320       |
| 2    | ne        | 661        | 79,139        |
| 3    | faire     | 336        | 43,078        |
| 4    | plus      | 316        | 23,591        |
| 5    | tout      | 274        | 33,315        |
| 6    | pouvoir   | 272        | 32,793        |
| 7    | aller     | 248        | 29,543        |
| 8    | dire      | 136        | 21,362        |
| 9    | bien      | 89         | 11,612        |
| 10   | vouloir   | 196        | 22,146        |
| 11   | voir      | 141        | 14,646        |
| 12   | devoir    | 246        | 18,499        |
| 13   | même      | 59         | 7,296         |
| 14   | très      | 96         | 10,938        |
| 15   | fait      | 27         | 3,496         |
| 16   | savoir    | 77         | 15,770        |
| 17   | taire     | 71         | 15,504        |
| 18   | temps     | 113        | 9,374         |
| 19   | aimer     | 138        | 13,956        |
| 20   | bon       | 102        | 9,301         |

`pas` and `ne` are negation markers. They are in the top 100 and exempt from
the card cap. `taire` is a lemmatizer result of `tais-toi` and similar forms.
It stays because the word is common in the deck.

## 5. Target words

A target is a content lemma that appears in the deck:

| Rule                        | Value                                             |
| --------------------------- | ------------------------------------------------- |
| Part of speech              | `NOUN`, `VERB`, `ADJ`, or `ADV`                   |
| Minimum deck count          | 3                                                 |
| Minimum content share       | 0.5 of all occurrences of the lemma               |
| Minimum length              | 2 characters                                      |
| Name list                   | `vocab/french_names.txt`, one name per line       |

The content-share rule removes function words that a tagger sometimes marks as
content: être, avoir, un, une, que, est, tu, lui, son, si, te, ça, ton, où,
cet, ta. The name list removes jean, marie, mary, tom, and similar lemmas.

The filters leave about 1,684 content lemmas. The target list is the first
`--target-words` of them in the blend order. The default is 300, and the
accepted range is 200 to 500.

The target count sets the coverage goal and the reports. It does not limit the
selection. The gain uses every ranked content lemma, so the 1,000 cards cover
more than the target list (section 9.4).

## 6. Pipeline

| Stage | Name       | Input                        | Output                              |
| ----- | ---------- | ---------------------------- | ----------------------------------- |
| C0    | Extract    | input APKG                   | notes, sentences, audio, English    |
| C1    | Frequency  | Tatoeba fr, `wordfreq`       | `data/french_frequency.tsv`         |
| C2    | Tokenize   | C0, spaCy                    | lemmas and parts of speech per card |
| C3    | Filter     | C2                           | passed and rejected cards, reasons  |
| C4    | Select     | C1, C2, C3                   | kept cards in learning order        |
| C5    | Write      | C0, C4                       | `.apkg`, `.cull.csv`, `.coverage.csv`, manifest |

`anki fix fr --select` runs C0 to C5. A second run with the same input gives
the same output. The audio cache is read only.

## 7. Filter

Apply the rules in order. Store the first failed rule as the `reason`.

| Rule | Condition                                                              | Rejected |
| ---- | ---------------------------------------------------------------------- | -------- |
| F1   | An English `Back` exists with 2 to 12 words                            | 163      |
| F2   | 3 to 12 words                                                          | 115      |
| F3   | No digit, bracket, URL, or ellipsis                                    | 1        |
| F4   | One final mark: `.`, `?`, or `!` at the end                            | 0        |
| F5   | Exactly one final mark                                                 | 25       |
| F6   | No quotation mark and no dialogue dash                                 | 0        |
| F7   | At most one internal comma, colon, or semicolon                        | 45       |
| F8   | No proper name (a `PROPN` token)                                       | 621      |
| F9   | The sentence text is unique                                            | 0        |

Result: 4,958 of 5,928 cards pass (83.6%).

- F8 removes the most cards. A name is noise in a vocabulary deck.
- F9 removes no card. The input has no exact duplicate.
- An optional unknown-word budget is not in the default set. A budget of 3
  content lemmas outside the top 3,000 removes 2 cards and changes no measured
  value.

## 8. Selection

### 8.1 Candidate set

Each passed card has a set of content lemmas. The selector walks the candidates
and keeps the best gain. The gain of a card counts only its uncapped words.
Section 8.4 then removes a card that holds a capped word.

### 8.2 Gain

```text
gain(card) = sum over content lemmas w in card of
             0.1 * weight(w)                         if w is exempt
             weight(w) / (uses[w] + 1)               if w is not exempt and uses[w] < cap
             0                                       otherwise
weight(w)  = 1 / log2(rank(w) + 1)
```

An unranked lemma gets the weight of rank 100,000. A tie breaks on fewer
words, then the input order.

### 8.3 Stop rule

Stop when the card count reaches `--max-cards`, or when every candidate has a
gain of 0. With the defaults, the selection reaches 1,000 cards.

### 8.4 Word cap

| Setting          | Value                                                   |
| ---------------- | ------------------------------------------------------- |
| `--max-uses`     | 4                                                       |
| `--exempt-top`   | 100 content lemmas by rank                              |
| Exempt meaning   | A word that can appear in more than 4 cards             |
| Cap meaning      | A non-exempt word appears in at most 4 kept cards       |

A sentence that holds a non-exempt word at its cap is skipped. The top 100
words are normal speech and are exempt, as in `fix yue`. The input cannot
reach 1,000 cards with no exemption (section 9.3).

### 8.5 Similarity

Two sentences are duplicates when the Jaccard value of their content-lemma sets
is 0.67 or more. Use the lemma sets as buckets, so a new card compares only
with kept cards that share a lemma. Reuse `latin_word_similarity` where it fits.

### 8.6 Order

`--order coverage` is the default. Sort the kept cards by these keys:

1. `hardest_rank`: the rank of the least frequent content lemma. Ascending.
   An unranked lemma counts as rank 1,000,000.
2. Word count. Ascending.
3. Input order. Ascending.

`--order input` keeps the input card order.

## 9. Expected results

The 9.1 default row and the appendix come from the built deck on 2026-10-20.
Tables 9.2 to 9.5 come from a prototype of sections 4 to 8. The input is the
5,928-card deck. The defaults are: 1,000 cards, 300 targets, cap 4, exempt top
100, spoken weight 0.5, 3 to 12 words.

### 9.1 This spec against the current order

| Measure                    | Current order | This spec |
| -------------------------- | ------------- | --------- |
| Cards                      | 1,000         | 1,000     |
| Distinct content words     | 1,272         | 2,081     |
| Top 200 covered            | 195           | 199       |
| Top 300 covered            | 282           | 299       |
| Top 500 covered            | 444           | 498       |
| Top 300 with 2 cards       | 256           | 289       |
| Most cards of one non-exempt word | 131    | 4         |
| Most cards of one exempt word     | 105    | 172       |
| `spoken_mass`              | 0.473         | 0.543     |

The current order is the order of `6000_French_Sentences_w_Audio.rank.csv`,
after the filters. The two sets share 133 cards.

The most-used words in the current order: avoir 131, pas 105, ne 85, faire 53,
plus 50, devoir 50, tout 39, pouvoir 35, vouloir 33, aller 30.

The most-used words in this spec: pas 190, ne 181, pouvoir 70, faire 69, plus
61, devoir 52, tout 46, vouloir 36, aller 32.

### 9.2 Cap sweep

Target 300, all other defaults.

| `--max-uses` | Content | Covered | Covered 2+ | `spoken_mass` |
| ------------ | ------- | ------- | ---------- | ------------- |
| 2            | 1,759   | 300     | 282        | 0.517         |
| 3            | 1,951   | 298     | 285        | 0.535         |
| 4            | 2,004   | 299     | 289        | 0.540         |
| 5            | 2,020   | 299     | 290        | 0.542         |
| 6            | 2,028   | 299     | 290        | 0.542         |
| 8            | 2,044   | 299     | 291        | 0.543         |

The cap costs little after 4. A cap of 4 keeps the strict rule with a content
loss of 40 words against a cap of 8. A cap of 2 loses 245 words. The cap-4 row
reads 2,004 from the prototype; the built default reads 2,081 because the
filter order differs.

### 9.3 Exemption sweep

Target 300, cap 4.

| `--exempt-top` | Cards | Content | Covered | Covered 2+ |
| -------------- | ----- | ------- | ------- | ---------- |
| 0              | 675   | 1,490   | 295     | 271        |
| 20             | 995   | 1,804   | 298     | 289        |
| 30             | 1,000 | 1,868   | 300     | 291        |
| 50             | 1,000 | 1,911   | 300     | 290        |
| 100            | 1,000 | 2,004   | 299     | 289        |

A hard cap on every word cannot reach 1,000 cards. The exempt set is necessary.
The default of 100 follows `fix yue`.

### 9.4 Target sweep

Cap 4, all other defaults.

| `--target-words` | Covered | Covered 2+ | Content |
| ---------------- | ------- | ---------- | ------- |
| 200              | 199     | 195        | 2,004   |
| 300              | 299     | 289        | 2,004   |
| 500              | 498     | 463        | 2,004   |
| 800              | 791     | 661        | 2,004   |

The selected set does not change, because the gain uses every ranked lemma.
The target count changes the coverage claim, not the cards.

### 9.5 Frequency weight sweep

Target 300, cap 4.

| `--spoken-weight` | Content | Covered | Covered 2+ | `spoken_mass` |
| ----------------- | ------- | ------- | ---------- | ------------- |
| 0.0               | 1,903   | 296     | 262        | 0.532         |
| 0.3               | 1,995   | 297     | 283        | 0.538         |
| 0.5               | 2,004   | 299     | 289        | 0.540         |
| 0.7               | 2,008   | 299     | 291        | 0.541         |

The written-only list covers fewer targets in the same 1,000 cards. Weight 0.5
is the default. Weight 0.7 is a small gain; it is an option. The built default
at weight 0.5 reads 2,081 content lemmas and 0.543.

### 9.6 Words without a full card set

One target word has no card: `an` (year). It is the only gap of the 300.

Ten targets have one card: été, alors, français, sens, nouvelle, occuper,
ensemble, enfin, effet, trouve. A second card is a target for a later
parameter change, not a gate failure.

## 10. Interface

### 10.1 Command

```bash
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --select \
    --max-cards 1000 \
    --target-words 300 \
    --max-uses 4 \
    --exempt-top 100 \
    --spoken-weight 0.5 \
    --output apkg/french_1000_sentences.apkg
```

### 10.2 Options

| Option            | Default                    | Meaning                                      |
| ----------------- | -------------------------- | -------------------------------------------- |
| `--max-cards`     | 1000                       | Card cap                                     |
| `--target-words`  | 300                        | Target count, 200 to 500                     |
| `--max-uses`      | 4                          | Most cards for one non-exempt word           |
| `--exempt-top`    | 100                        | Content lemmas exempt from the cap           |
| `--spoken-weight` | 0.5                        | `w` in the blend                             |
| `--min-words`     | 3                          | Minimum word count of a card                 |
| `--max-words`     | 12                         | Maximum word count of a card                 |
| `--similarity`    | 0.67                       | Lemma Jaccard limit                          |
| `--order`         | coverage                   | `coverage` or `input`                        |
| `--name`          | French 1000 Core Sentences | Output deck name                             |
| `--output`        | from the input name        | Output APKG                                  |
| `--select`        | off                        | Run the cull instead of the rank and reorder |

### 10.3 Output files

| File                                 | Content                                                    |
| ------------------------------------ | ---------------------------------------------------------- |
| `apkg/french_1000_sentences.apkg`    | The culled deck with the input model, fields, and audio    |
| `apkg/french_1000_sentences.cull.csv`     | One row per input card: `kept`, `reason`, `targets`, `gain` |
| `apkg/french_1000_sentences.coverage.csv` | One row per target: `word`, `rank`, `count`, `status`       |
| `apkg/french_1000_sentences.manifest.json` | Counts, parameters, frequency source, and hashes        |

The `.cull.csv` has these columns: `source`, `sentence`, `english`, `audio`,
`kept`, `reason`, `targets`, `gain`, `hardest_rank`. The `source` is `swadesh`
or `frequency`. The reserve rows come first.

The `.coverage.csv` has these columns: `word`, `rank`, `count`, `status`. The
`status` is `covered`, `partial` (one card), or `absent`.

### 10.4 Code changes

- Add `anki_tools/languages/french/cull.py` with the stages C1 to C5.
- Add the `--select` option to `anki fix fr` in `cli.py`. Rank mode stays the
  default. The CLI builds a `CullRequest` and calls `run_cull` or `check_cull`.
- Add `anki_tools/ranking/cover.py` for the greedy cover that both language
  selectors use.
- Add `anki_tools/ranking/pipeline.py` for the shared extract, rank, write, and
  reorder stages.
- Reuse `AnkiPackage` for the input and the output.
- Reuse the French spaCy model and `TOKENIZER` from the French pack.

## 11. Quality gates and tests

### 11.1 Gates

`anki fix fr --select --check` runs the gates on the output. A failure stops
the command with a non-zero exit code.

| Gate | Check                                                                  |
| ---- | ---------------------------------------------------------------------- |
| G1   | The card count is `--max-cards` or the pool has no more cards           |
| G2   | No non-exempt content lemma is in more than `--max-uses` kept cards     |
| G3   | Target coverage is at least 0.98 of `--target-words` (measured 0.997)   |
| G4   | At least 0.95 of the targets have 2 cards (measured 0.963)              |
| G5   | Every kept card passes the filter rules. The reserve cards skip the name rule F8, because the Swadesh pack already validated them. |
| G6   | No kept card pair reaches the lemma Jaccard limit                       |
| G7   | Every kept card has its input audio file, and the file is a valid MP3   |
| G8   | The APKG note count equals the cull CSV `kept` count and the manifest   |
| G9   | The APKG holds exactly the media files of the kept cards                |
| G10  | The output deck name and the note model are the requested values        |
| G11  | A second run gives the same kept set and the same order                 |
| G12  | The coverage CSV has one row per target word                            |
| G13  | The manifest hashes match the cull CSV and the coverage CSV             |
| G14  | The merged deck covers 206 of 207 Swadesh concepts, with `intestins` as the exception |
| G15  | The deck model holds the Swadesh-style front and back templates and CSS |

### 11.2 Tests

Add `tests/test_french_cull.py`. Use small fixture APKG and TSV files. Do not
download a file and do not run a full Tatoeba pass.

| Area           | Test                                                                 |
| -------------- | -------------------------------------------------------------------- |
| Frequency      | A count below 5 uses the written term; the weight changes the order    |
| Lemma filter   | A function word with a low content share is not a target               |
| Name filter    | A name in the list is not a target                                     |
| Filter rules   | One failing card for each rule F1 to F9                               |
| Gain           | A rare word wins over a repeated word                                  |
| Cap            | A non-exempt word stops at 4 cards; an exempt word does not            |
| Cap skip       | A card with a capped word can stay when it adds an uncapped word       |
| Similarity     | Two cards with a lemma Jaccard of 0.67 are duplicates                 |
| Order          | `hardest_rank` does not decrease                                      |
| Determinism    | Two runs give the same card set and order                              |
| APKG           | The output holds the kept notes and only their media files             |

## 12. Risks and open questions

### 12.1 Risks

| Risk                                                       | Effect                            | Mitigation                                            |
| ---------------------------------------------------------- | --------------------------------- | ----------------------------------------------------- |
| Tatoeba is not a speech corpus                             | The target list is not truly spoken | Blend at 0.5; `--spoken-weight`; open question 1      |
| `wordfreq` mixes written and subtitle data                 | Written words rise                | The Tatoeba term; the blend                           |
| Lemmatizer artifacts (`as`, `taire`, `fait`)               | A target is a verb form, not a lemma | Keep the word; review the target list; open question 2 |
| The exempt top 100 puts negation markers in 17% of the cards | The cards look alike            | `--exempt-top`; open question 3                       |
| The F8 name filter removes 632 cards                       | Fewer candidates                  | 4,958 cards remain                                    |
| The target count does not control the selection            | The covered set exceeds the goal  | The coverage reports; section 9.4                     |
| The input audio license is not in the APKG                 | A public release is blocked       | Check the source before release; open question 6      |
| The input `freq ranks` source is unknown                   | A missed cross-check              | Ignore the field; open question 5                     |

### 12.2 Open questions

| #   | Question                                                             | Decide at |
| --- | -------------------------------------------------------------------- | --------- |
| 1   | Is SUBTLEX-FR a usable spoken list, and is its license public?        | D2        |
| 2   | Should the target list use a manual review file for lemmatizer noise? | D3        |
| 3   | Is the exempt top 30 better than 100 for the card variety?            | D4        |
| 4   | Should the output add a `Target` field and bold the target?           | D4        |
| 5   | What is the source of the input `freq ranks` field?                   | D2        |
| 6   | What is the license of the input audio and sentence text?             | D5        |

## 13. Milestones

| Milestone | Issue      | Task                                                  | Output                                   |
| --------- | ---------- | ----------------------------------------------------- | ---------------------------------------- |
| D1        | `anki-3jm` | This spec and the analysis                            | `specs/french-1000-sentence-cull.md`     |
| D2        | `anki-mvo` | APKG extract, frequency list, Tatoeba cache            | `data/french_frequency.tsv`              |
| D3        | `anki-3ux` | Filter, gain, cap, similarity, order                  | kept card list                           |
| D4        | `anki-mq7` | APKG writer, cull CSV, coverage CSV, manifest, gates  | `apkg/french_1000_sentences.apkg`        |
| D5        | `anki-7n2` | Tests, README, license check                          | release check                            |
| D6        | `anki-c5g` | Swadesh reserve, merged writer, gate G14              | 206 / 207 Swadesh, 1,000 cards           |

## 14. Appendix: evidence

### 14.1 Filter counts

| Rule | Result                          | Cards |
| ---- | ------------------------------- | ----- |
| F1   | English words outside 2 to 12   | 163   |
| F2   | Word count outside 3 to 12      | 115   |
| F3   | Digit, bracket, URL, ellipsis   | 1     |
| F5   | More than one final mark        | 25    |
| F7   | More than one internal mark     | 45    |
| F8   | Proper name                     | 621   |
| -    | Pass                            | 4,958 |

F4, F6, and F9 remove no card. An optional unknown-word budget removes 2 cards
and is not in the default set.

### 14.2 First cards of the default run

| # | Sentence                                                        | English                          |
| - | --------------------------------------------------------------- | -------------------------------- |
| 1 | Je fus capable de réussir grâce à votre conseil. | I was able to succeed because of your advice. |
| 2 | J'ai réfléchi à propos de quel pauvre type j'ai été. | I thought about what a jerk I had been. |
| 3 | Je suis surpris qu'ils n'aient eu quoi que ce soit à faire. | I'm surprised they didn't have anything to do. |
| 4 | J'ignore si je serai en mesure d'assister à la réunion de demain. | I don't know whether I'll be able to attend tomorrow's meeting. |
| 5 | Vu d'avion, cette île a l'air très belle. | Seen from the plane, the island looks very beautiful. |
| 6 | J'ai comme la vague idée que boracasli utilise un deuxième compte. | I have a hunch boracasli is using a second account. |
| 7 | Le jeu auquel il joue n'a pas d'importance, il gagne toujours. | It doesn't matter what game he plays, he always wins. |
| 8 | Le téléphone ne fonctionne pas actuellement. | The phone doesn't work right now. |
| 9 | Cette réponse n'est pas nécessairement fausse. | This answer may not necessarily be wrong. |
| 10 | Ne t'inquiète pas d'une chose aussi bête. | Don't worry about such a silly thing. |

The table is the built `--order coverage` order.

### 14.3 How the numbers were made

The numbers of tables 9.2 to 9.5 come from a prototype that is not in the
repository. It used:

- The APKG of 2026-10-20, `apkg/6000_French_Sentences_w_Audio.apkg`.
- spaCy `fr_core_news_sm` with the parser and the named-entity model disabled.
- The Tatoeba `fra_sentences.tsv` of 2026-10-20, 728,898 sentences.
- `wordfreq` 3.x, French list.
- The blend of section 4, the filter of section 7, and the selection of
  section 8.
- The name stoplist of section 5.

A count can differ by one from a later table because of a tie in the sort.

## 16. Swadesh reserve

The frequency pass covers 162 of the 207 Swadesh concepts. The input deck does
not hold 37 of the missing 45. Selection cannot add a word that the input does
not have. This section adds a reserved Swadesh set to the cull.

### 16.1 Source

The French Swadesh pack `apkg/swadesh_french_sentences.apkg` holds one card per
concept and one audio file per card.

| Measure                              | Value       |
| ------------------------------------ | ----------- |
| Swadesh concepts in the pack         | 206 of 207  |
| One card per concept                 | 206         |
| Exact duplicate of an input sentence | 3           |
| Near duplicate of an input sentence  | 4           |
| Duplicate reserve card               | 1           |
| Reserve cards after duplicate check  | 198         |
| Reserve cards with audio             | 198         |
| Concept with no source sentence      | `intestins` |

The pack is read with `AnkiPackage`. The note model is
`French Swadesh Sentence`. The fields are `Audio`, `Sentence`, `English`,
`Target`, `Gloss`, `WordAudio`, and `Source`. The audio file comes from the
pack media, so no external audio directory is needed.

### 16.2 Merge into the input model

The output keeps the input model `French Sentences` but replaces its card
templates and CSS with the Swadesh style of section 16.7. Each reserve card
becomes one note of that model:

| Input field  | Reserve value                                             |
| ------------ | --------------------------------------------------------- |
| `French`     | The sentence with the target token in `<b>` tags          |
| `sound`      | `[sound:<file>]` of the pack audio                        |
| `Back`       | `English — target (gloss)`                                |
| `freq ranks` | Empty                                                     |

The note id and card id come after the highest id of the input package. The
guid is a UUID. The tags include `swadesh`.

### 16.3 Selection

1. Load the input deck and the reserve cards.
2. Drop a reserve card whose sentence is in the input deck.
3. Drop a reserve card that reaches the lemma Jaccard limit 0.67 with an input
   card or with an earlier reserve card. The input deck is the source of the
   frequency cards, and two reserve cards must not duplicate each other.
4. Add the reserve lemma sets to the `uses` of the selector.
5. Keep every reserve card. The frequency pass fills `max_cards` minus the
   reserve count.
6. The reserve cards are exempt from the cap. The cap applies to the frequency
   cards only. A common word can appear in many reserve cards, because the
   reserve holds one card per concept. The report gives the two maximums
   separately.

### 16.4 Option

| Option            | Default                              | Meaning                     |
| ----------------- | ------------------------------------ | --------------------------- |
| `--reserve-pack`  | `apkg/swadesh_french_sentences.apkg` | Reserve source              |
| `--no-reserve`    | off                                  | Run the frequency cull only |

### 16.5 Gate G14

`G14`: the output holds at least `swadesh_total - len(exceptions)` Swadesh
concepts. A concept is covered when its form, or the lemma of its head word,
appears in at least one kept card. The card does not need to be assigned to
that concept. Example: the reserve card "Un nuage flottait dans le ciel." is
the card for `flotter`, and it covers `nuage`. The manifest holds
`counts.swadesh_covered`, `counts.swadesh_total`, and `swadesh_exceptions`. The
default exception list is `["intestins"]`, because no source sentence exists.

### 16.6 Expected result

The measured merge of 198 reserve cards and 802 frequency cards:

| Measure                    | Frequency only | With reserve |
| -------------------------- | -------------- | ------------ |
| Swadesh concepts covered   | 162 / 207      | 206 / 207    |
| Top 300 frequency words    | 299            | 297          |
| Top 300 with 2 cards       | 289            | 287          |
| Distinct content lemmas    | 2,081          | 1,981        |
| Cards                      | 1,000          | 1,000        |
| Most cards of one non-exempt frequency word | 4 | 4        |
| Most cards of one non-exempt reserve word   | - | 34       |

The reserve and the frequency cards together cover 206 of 207 concepts. The 7
reserve cards that the duplicate check drops are covered by frequency cards.
`intestins` is the only gap.

### 16.7 Card templates

The output replaces the input templates with the Swadesh style, so the two
decks look alike. The model keeps its two card directions.

Card 1 (comprehension):

- Front: the audio and the hint `Tap to show the sentence`.
- Back: the audio, the French sentence, and the `Back` field.

Card 2 (production):

- Front: the `Back` field and the hint `Tap to show the French`.
- Back: the `Back` field, the French sentence, and the audio.

The CSS is the base `templates/card.css` plus the French pack classes
(`.front-audio`, `.sentence`, `.english`, `.hint`). The `Back` field of a
reserve card already holds `English — target (gloss)`, so the target and the
gloss appear with the English text. Gate G15 checks the templates and the CSS.
