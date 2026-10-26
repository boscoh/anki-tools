# Spec: Cantonese Swadesh Sentence Pack with Audio

| Field   | Value                                                          |
| ------- | -------------------------------------------------------------- |
| Status  | implemented                                                    |
| Date    | 2026-10-20                                                     |
| Owner   | anki-tools                                                     |
| Depends | `specs/mandarin-swadesh-sentence-pack.md` (pipeline shape)     |
| Related | `specs/cantonese-frequency-list.md` (frequency list)           |
| Issues  | epic `anki-ana`; milestones C1 to C6 (section 14)              |

## 1. Purpose

Build a listening-comprehension deck for Cantonese. The deck has two layers:

1. The reviewed Swadesh word list: 207 concepts, 207 forms.
2. Useful sentences that contain those words, with audio on each card.

The existing `anki swadesh build cantonese` command makes one isolated-word
card per Swadesh entry. This spec adds the sentence layer on top of it, as the
Mandarin, Greek, and French packs do.

Cantonese adds three problems to the shared pipeline:

- The sentence script is Traditional Chinese, and Tatoeba holds both scripts.
- Cantonese has no part-of-speech tagger, so the predicate and orphan rules
  cannot run.
- `wordfreq` has no Cantonese word list, so the ranking uses the Chinese list.

## 2. Scope

In scope:

- A reviewed word list, `vocab/english-cantonese.csv`, with a review file.
- A sentence corpus from Tatoeba `yue` with English translations.
- Audio for each sentence and for each target word.
- A build command that writes one `.apkg` file.
- A manifest with provenance and licenses.
- Quality gates G1 to G21.

Out of scope:

- Changes to the existing word deck and its templates.
- The `anki fix yue` rank and core modes.
- Grammar explanations or spaced-repetition tuning.
- A Cantonese frequency file. The pack builds the Cifu and Apple Daily blend
  in memory and writes no frequency file.

## 3. Data

All numbers below come from the Tatoeba export of 2026-10-20 and the
`Module:Swadesh/data/yue` wikitext.

### 3.1 Corpus

| Measure                                       | Value          |
| --------------------------------------------- | -------------- |
| Tatoeba Cantonese (`yue`) sentences           | 20,853         |
| Sentences with 12 Han characters or fewer     | 26.3%          |
| Cantonese to English translation links        | 7,314          |
| Linked source sentences with a text           | 7,038          |
| Linked sentences with 12 Han characters or fewer | 66.5%       |
| Tatoeba Cantonese sentences with native audio | 1,784          |
| Native audio rows with the CC BY 4.0 license  | 1,781          |
| Native audio rows with no license             | 3              |

The linked set is larger than the whole set suggests, because short sentences
are the ones with translations.

### 3.2 Word list

| Measure                                  | Value |
| ---------------------------------------- | ----- |
| Swadesh concepts                         | 207   |
| Module rows in `Module:Swadesh/data/yue` | 207   |
| Module terms                             | 263   |
| Module rows with more than one term      | 47    |
| Reviewed forms in `english-cantonese.csv` | 207  |
| Unique reviewed forms                    | 207   |

The reviewed CSV keeps one form per concept. Section 5 records the compare
with the module terms.

### 3.3 Sentence pool

| Stage                        | Sentences |
| ---------------------------- | --------- |
| Linked sentences             | 7,038     |
| Pass F1 to F12 and the unknown budget, and hold a target | 3,431 |
| Core candidates              | 3,430     |
| Scaffold candidates          | 1         |

The F2 window removes the most: 2,423 sentences fail F2. The unknown-token
budget removes 566 more.

## 4. Pipeline

The pack uses the shared `anki_tools.sentence_pack` stages. The language
module is `anki_tools/sentence_pack/languages/cantonese.py`.

| Stage | Name       | Input                   | Output                                       |
| ----- | ---------- | ----------------------- | -------------------------------------------- |
| S0    | Fetch      | network                 | `data/tatoeba/yue_*.tsv`, `eng_sentences.tsv` |
| S1    | Word table | `vocab/english-cantonese.csv` | `WordForm` list                        |
| S2    | Pool       | S0, S1                  | candidates and `data/cantonese_pool_report.json` |
| S3    | Select     | S2, S1                  | `vocab/cantonese_sentences.csv`              |
| S4    | Audio      | S3                      | `audio/cantonese_sentences/`                 |
| S5    | Pack       | S3, S4                  | `apkg/swadesh_cantonese_sentences.apkg`      |

`anki swadesh pack cantonese --stage fetch` runs S0. The other stages are in
section 10.
Every stage gives the same output for the same input, except the audio cache,
which skips a file that already passes validation.

## 5. Word table

### 5.1 File shape

`vocab/english-cantonese.csv` has three columns, as before:

```text
english,cantonese,transliteration
```

`transliteration` holds the jyutping reading with tone numbers. The loader
adds the Swadesh index from the row order and computes the `wordfreq` Zipf
value. No other file is needed to build the pack.

### 5.2 Script

The reviewed forms use Traditional characters. `prepare_text` applies NFC,
collapses spaces, and converts a simplified Tatoeba row to Traditional with
OpenCC `s2t`. The card text is Traditional, and the `text_original` column
keeps the source text.

### 5.3 Review decisions

The reviewed CSV had three problems: 32 forms that were not module terms, one
wrong character, and one wrong verb. The script
`scripts/build_cantonese_word_table.py` holds the decisions and rewrites
`vocab/cantonese_word_review.csv`.

A form changed when the module term has more linked sentences and keeps the
sense. A form stayed when the module term is less precise. The review file
marks each row `fixed` or `keep`, so gate G15 stops a build with an `open` row.
The corrected forms also enter the vocabulary deck of
`anki swadesh build cantonese`. Its audio cache prunes the old files on the
next run.

| Index | English    | Old form | New form | Reason                                 |
| ----- | ---------- | -------- | -------- | -------------------------------------- |
| 8     | that       | 嗰個     | 嗰       | module term, 481 linked sentences      |
| 42    | mother     | 媽咪     | 阿媽     | module term, 74 linked sentences       |
| 43    | father     | 爸爸     | 阿爸     | module term, 50 linked sentences       |
| 48    | louse      | 蝨乸     | 蝨       | module term replaces the compound      |
| 62    | skin       | 皮膚     | 皮       | module term, 41 linked sentences       |
| 66    | fat (noun) | 肥       | 脂肪     | the noun sense; 肥 is the adjective    |
| 69    | tail       | 尾巴     | 尾       | module term, 32 linked sentences       |
| 86    | guts       | 腸肚     | 腸       | module term, 25 linked sentences       |
| 96    | to spit    | 吐痰     | 吐       | module term; 吐痰 has no candidate     |
| 97    | to vomit   | 嘔吐     | 嘔       | module term; 嘔吐 has no candidate     |
| 101   | to see     | 睇見     | 睇       | the Cantonese verb; 睇見 has no candidate |
| 102   | to hear    | 聽到     | 聽       | module term, 227 linked sentences      |
| 105   | to smell   | 聞到     | 聞       | module term, 17 linked sentences       |
| 106   | to fear    | 害怕     | 驚       | module term, 61 linked sentences       |
| 114   | to cut     | 割       | 切       | module term, 42 linked sentences       |
| 119   | to swim    | 遊       | 游水     | the swim verb; 遊 means to roam        |
| 123   | to lie down | 瞓低    | 瞓       | module term, 76 linked sentences       |
| 127   | to fall    | 跌倒     | 跌       | module term; 跌倒 has no candidate     |
| 129   | to hold    | 揸住     | 揸       | module term, 31 linked sentences       |
| 138   | to sew     | 聯       | 縫       | 聯 means to unite, not to sew (`anki-5kb`) |
| 139   | to count   | 數數     | 數       | module term; 數數 has no candidate     |
| 162   | sky        | 天空     | 天       | module term, 203 linked sentences      |
| 169   | to burn    | 燃燒     | 燒       | module term, 31 linked sentences       |
| 193   | smooth     | 平滑     | 滑       | module term; 平滑 has no candidate     |
| 196   | correct    | 正確     | 啱       | module term, 97 linked sentences       |

These eight forms stayed, and the review file records the reason:

| Index | English | Form | Reason                                          |
| ----- | ------- | ---- | ----------------------------------------------- |
| 7     | this    | 呢個 | unambiguous; 呢 is also a question particle     |
| 88    | back    | 背   | 34 linked sentences; 背脊 has 1                 |
| 108   | to live | 生存 | keeps the sense live; 生 also means birth       |
| 111   | to fight | 打交 | keeps the sense fight; 爭 means to contend     |
| 116   | to stab | 刺   | keeps the sense stab; 插 means insert           |
| 130   | to squeeze | 搾 | the squeeze verb; 挾 means to carry under the arm |
| 136   | to throw | 掟  | the throw verb; 2 candidates                    |
| 178   | day     | 日   | the basic noun; 日頭 has 2 candidates           |

### 5.4 Frequency value

`wordfreq` has no `yue` list. The loader converts a form to simplified
characters and reads the `zh` Zipf value. The value orders the candidates of
one form and marks a rare form `low_zipf` in the review CSV. It is not a claim
about spoken Cantonese frequency.

## 6. Tokenizer

The pack needs a word tokenizer for three jobs: the unknown-token count, the
Zipf words, and the target match.

| Setting    | Value                                                             |
| ---------- | ----------------------------------------------------------------- |
| Library    | `jieba` with a `jieba.Tokenizer` instance                         |
| Dictionary | Every Cifu word with a Han character and a weight above 0         |
| Weight     | `spoken_count * 40 + written_count * 2` (from the Cifu blend)    |
| Extras     | The reviewed forms and the frequency override words               |
| Cache      | `~/.cache/anki-tools/cifu-jieba-dict.txt`                         |

The build builds the tokenizer once and does not change the global jieba
dictionary. The Cifu lexicon is cached and never committed. The first build
downloads it into the cache.

### 6.1 Whole-token match

Cantonese 我 and 我哋 are separate Swadesh concepts. A substring match would
give the card for 我 a sentence that holds 我哋. The pack therefore matches a
form only on token boundaries.

An occurrence inside a longer word counts only when that word is not itself a
reviewed form. Table 1 shows the rule.

| Sentence | Form | Hits | Why |
| -------- | ---- | ---- | --- |
| 我哋去街。 | 我   | 0    | inside the reviewed form 我哋 |
| 哋去街。  | 我   | 0    | no match                     |
| 我去街。  | 我   | 1    | whole token                  |
| 我食飯。  | 食   | 1    | inside 食飯, which is not a reviewed form |
| 我食飯。  | 食飯 | 1    | whole token                  |

An occurrence inside a false-friend compound never counts. The reviewed list
in `anki_tools/sentence_pack/languages/cantonese.py` holds these compounds:

| Form | Sense    | Blocked compounds                          |
| ---- | -------- | ------------------------------------------ |
| 切   | to cut   | 趕得切 (in time), 一切 (everything)         |
| 聽   | to hear  | 聽日 (tomorrow)                             |
| 數   | to count | 着數 (bargain), 數學 (math), 彩數 (luck), 多數 (mostly) |
| 燒   | to burn  | 發燒 (fever)                                |
| 揸   | to hold  | 揸車 (to drive)                             |

Gate G20 checks the built deck. A card fails when its target sits inside one
of these compounds.

The list does not catch every case. A bare verb can carry a different sense
(揸 alone means to drive). An aspect marker can split a compound (發緊燒 is
fever). A polyphone can change the sense (數 sou3 means number or fate).
Those cases stay for a later review.

The shared English sense test covers all 207 glosses. The table
`vocab/english_senses.json` holds, for each gloss, the English words and phrases
that show its sense. A candidate passes when the paired English holds one of
them. `scripts/build_english_senses.py` writes the table from WordNet plus a
reviewed list of extra words. This test is what catches a bare verb with the
wrong sense, such as 揸 with drive English.

The form-level table adds a check for a compound with the wrong sense. Each
risky form has two English word lists: sense words and false-friend words. A
candidate passes when the paired English holds a sense word, or holds no
false-friend word. The `SENSE_CHECKS` table in `cantonese.py` holds the lists:

| Form | Sense words            | False-friend words                 |
| ---- | ---------------------- | ---------------------------------- |
| 揸    | hold, grip, grasp      | drive, drove, driving, driver      |
| 數    | count, counts, counted | math, mathematics, number, numbers |
| 燒    | burn, burned, fire     | fever, temperature                 |

Gate G21 checks the built deck against the stored English text.

## 7. Filters

A sentence passes when it passes every rule. The rules are the shared F1 to
F12 rules with the Cantonese values below.

| Rule | Value                                                        |
| ---- | ------------------------------------------------------------ |
| F1   | An English link exists                                       |
| F2   | 3 to 12 Han characters                                       |
| F3   | No Latin letter and no digit                                 |
| F4   | Not a repeated pattern                                       |
| F5   | One final mark: `。`, `？`, `！`, `.`, `?`, or `!` at the end  |
| F6   | Exactly one final mark                                       |
| F7   | No quotation mark: `「」『』` and the ASCII quotes            |
| F8   | At most one internal mark: `，：；` or the ASCII marks        |
| F9   | Predicate: off, because no Cantonese tagger exists            |
| F10  | Orphan conjunction: off, for the same reason                 |
| F11  | The English text has 2 to 12 words                           |
| F12  | The sentence text is unique                                  |

The 3 to 12 Han character window is a Cantonese value: Cantonese sentences are
short, so the fallback window of the scaffold tier starts at 3 too.

The known set of the unknown-token budget holds the reviewed forms and the top
1,000 `wordfreq` Chinese words in Traditional characters. A sentence may hold
at most 2 unknown tokens.

Measured effect: F2 removes 2,423 of 7,038 linked sentences. The unknown
budget removes 566 more.

## 8. Selection

The shared selector walks the 207 forms in Swadesh order. For each form it
takes the best unused candidate and rejects a sentence that is too close to a
kept card.

| Setting          | Value                                             |
| ---------------- | ------------------------------------------------- |
| Cards per form   | 2                                                 |
| Hard card cap    | 450                                               |
| Similarity limit | 0.67 character Jaccard, as `anki fix zh`          |
| Order            | Swadesh index, then variant rank, then Tatoeba id |
| Ranking key      | unknown count, audio, Zipf, length, Tatoeba id    |

A form gets no card when the pool holds no candidate that reaches it. The
uncovered list goes to the pool report and the review CSV.

## 9. Audio

| Source  | Rule                                                          |
| ------- | ------------------------------------------------------------- |
| Native  | Use the Tatoeba recording when the row has one and the license allows the pack |
| TTS     | Else synthesize the sentence with gTTS `yue`                  |
| Words   | One TTS file per target word                                  |
| Probe   | `ffprobe` checks the header and the 0.3 s to 60 s duration    |

`gtts` supports Cantonese as language `yue`, so the default engine needs no new
dependency. The optional `edge-tts` package adds the voice
`zh-HK-HiuMaanNeural` with `--tts edge --voice zh-HK-HiuMaanNeural`.

The card CSV keeps `audio_source` and `audio_license` per row. A native
recording keeps its own license; a synthesized file gets the engine and the
voice.

## 10. Interface

```bash
uv run anki swadesh pack cantonese --per-word 2 --max-cards 450 --native-audio auto  # S0 to S5
uv run anki swadesh audit cantonese                          # gates G1 to G21
```

The commands write these files:

| Path                                                    | Content                           |
| ------------------------------------------------------- | --------------------------------- |
| `vocab/english-cantonese.csv`                           | reviewed word table, 207 rows     |
| `vocab/cantonese_word_review.csv`                       | word review, 33 rows              |
| `vocab/cantonese_sentences.csv`                         | one row per card                  |
| `vocab/cantonese_sentences_review.csv`                  | top candidates for the weak words |
| `audio/cantonese_sentences/index.csv`                   | file index of the audio cache     |
| `apkg/swadesh_cantonese_sentences.apkg`                 | the deck                          |
| `apkg/swadesh_cantonese_sentences.manifest.json`        | counts, coverage, and file hashes |
| `ATTRIBUTION.md`                                        | the `## Cantonese pack` section   |

The `sentences` command takes `--min-words` and `--max-words`. Without them the
profile window applies, which is 3 to 12 for Cantonese and 4 to 10 for Greek
and French.

The back template has `class="sentence char-links"` on the `Sentence`
field. The profile field `char_links` adds `templates/shared/char_links.html`, which
links each Han character of the sentence to the CC-Canto dictionary
(`https://cantonese.org/search.php?q=`), the same as the Mandarin sentence
pack. The builder reads the front and back templates from
`templates/sentence_pack/cantonese/front.html` and `back.html`, and the
classes from `templates/sentence_pack/cantonese/card.css`. The pack `card.css`
overrides the Simplified `.char-links` font stack of the shared sheet with a
Traditional stack, so each character uses a Hong Kong face.

## 11. Quality gates

`anki swadesh audit cantonese` runs these gates. A failure stops the command
with a non-zero exit code.

| Gate | Check                                                                          |
| ---- | ------------------------------------------------------------------------------ |
| G1   | The word table passes the row count, index, label, jyutping, and review checks |
| G2   | No two reviewed forms are the same                                             |
| G3   | Every card target is a reviewed form                                           |
| G4   | Every card has a unique Tatoeba id                                             |
| G5   | The target matches exactly one whole token in the sentence                     |
| G6   | No two cards reach the 0.67 character Jaccard limit                            |
| G7   | Every card has a valid sentence MP3                                            |
| G8   | The media count in the manifest matches the APKG                               |
| G9   | No card sentence or target holds a Latin letter or a digit                     |
| G10  | Core coverage is at least 0.70                                                 |
| G11  | The manifest counts and hashes match the files                                 |
| G12  | The `## Cantonese pack` section names every sentence author                    |
| G13  | The APKG opens and its note count matches the CSV                              |
| G14  | Every card passes the proper-sentence rules                                    |
| G15  | The word review file has no `open` row                                         |
| G16  | Every card target has a valid word MP3                                         |
| G17  | Every card has a jyutping reading                                              |
| G18  | Every native card keeps a license                                              |
| G19  | Every card target is a reviewed form with a card                               |
| G20  | No card uses its target inside a false-friend compound                         |
| G21  | No card English holds a different sense than the card word                     |

### 11.1 Tests

`tests/test_cantonese_pack.py` holds 19 offline tests:

| Area              | Test                                                       |
| ----------------- | ---------------------------------------------------------- |
| Word table        | 207 rows, unique forms, non-empty jyutping                 |
| Sew fix           | index 138 is 縫 with `fung4`                               |
| Review file       | an accepted file and an `open` row                         |
| Script            | simplified to Traditional conversion                       |
| Jyutping          | the reading of a sentence                                  |
| Whole-token match | 我 against 我哋 and 食 against 食飯                        |
| False friends     | 聽 against 聽日 and 切 against 一切                        |
| Sense filter      | drive and temperature English reject, hold and burn keep   |
| Unknown budget    | a known and an unknown sentence                            |
| Filters           | F2, F3, F5, F6, F7, F8, and the off rules F9 and F10       |
| Gloss and fields  | the gloss and the `Jyutping` field                         |
| Pool              | a small Tatoeba fixture, the filter counts, and the rank key |

The tests do not download a file and do not need Cifu. A fake tokenizer
replaces the jieba instance.

## 12. Measured result

Measured on 2026-10-22 with the defaults, after the five alternate rows:

| Measure                    | Value                 |
| -------------------------- | --------------------- |
| Cards                      | 286                   |
| Core cards                 | 286                   |
| Scaffold cards             | 0                     |
| Forms with a card          | 161 of 212 (75.9%)    |
| Concepts with a card       | 158 of 207 (76.3%)    |
| Cards with native audio    | 171                   |
| Cards with a TTS file      | 108                   |
| Media files                | 434                   |
| Audio duration             | 0.4 s to 15.0 s       |
| Jyutping on every card     | yes                   |
| Gates G1 to G21            | sentence stage only   |

The 51 forms without a candidate: 闊, 厚, 窄, 薄, 蝨, 棍, 種子, 葉, 根,
樹皮, 皮, 脂肪, 角, 尾, 羽毛, 嘴, 腳, 腿, 膝頭, 翼, 腸, 頸, 胸, 肝, 咬,
吐, 吹, 打獵, 切, 分開, 刺, 搾, 擦, 推, 綁, 數, 浮, 流, 結冰, 脹, 湖, 鹽,
沙, 灰, 燒, 爛, 圓, 利, 鈍, 乾, 裡面. The form 𢯎 has a candidate but no card.

The English sense check removes most of these forms. A candidate must hold the
English gloss sense of the target word, so 揸 with drive English, 數 with math
English, and 燒 with fever English all fail. The cap did not bind: `--per-word
2` stops at 286 cards, before `--max-cards 450`.

Length of the cards: 8 cards have 4 Han characters, 17 have 5, 32 have 6, 37
have 7, 36 have 8, 51 have 9, 43 have 10, 24 have 11, 15 have 12, 11 have 13,
4 have 14, and 1 has 20.

## 13. Risks and open questions

| Risk                                                        | Effect                           | Mitigation                                                              |
| ----------------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| `wordfreq` has no `yue` list                                | A rare Cantonese form looks rare | The value only orders candidates; the Cifu blend gives the spoken rank |
| No Cantonese tagger                                         | F9 and F10 are off               | The length, script, and mark rules still apply                          |
| The reviewed CSV differs from the module terms              | A form may be unusual            | The 32 rows are in the review file with a decision                      |
| A form is a common particle (呢)                             | Wrong cards for `this`           | The review keeps 呢個 for `this`                                          |
| gTTS Cantonese quality is lower than a native voice         | Audio quality                    | The 279-card run had 171 native files; edge-tts adds a HK voice         |
| Cifu is a GPL-3.0 repository with corpora of other licenses | Public build                     | The file is cached, never committed; `data/` is ignored                 |
| 49 concepts have no candidate                               | Incomplete coverage              | Core coverage 0.763 is above the 0.70 gate; the review CSV lists them   |

Open questions:

1. Should each module alternate term get a card? 47 module rows have more than
   one term. A variant row would cover more concepts but raise the card count.
2. Is the whole-token rule too strict for a verb such as 食? Measure the
   candidates that the rule removes on a larger corpus.
3. Should the pack use the Cifu blend rank in the ranking key instead of
   the Chinese Zipf value?

## 14. Milestones

| Milestone | Issue      | Task                                              | Output                                    |
| --------- | ---------- | ------------------------------------------------- | ----------------------------------------- |
| C1        | `anki-55q` | Word table, review file, sew fix                  | `vocab/cantonese_word_review.csv`; G1, G15 |
| C2        | `anki-gkd` | Language profile, tokenizer, jyutping, filters     | profile and hooks                          |
| C3        | `anki-4ng` | Sentence pool and selection                       | `vocab/cantonese_sentences.csv`            |
| C4        | `anki-yzf` | Audio and APKG pack                               | `apkg/swadesh_cantonese_sentences.apkg`    |
| C5        | `anki-y2r` | Gates and tests                                   | `tests/test_cantonese_pack.py`; G1 to G21  |
| C6        | `anki-0wz` | README and `ATTRIBUTION.md`                       | release check                              |

## 15. Appendix: evidence

### 15.1 Candidate counts of the changed forms

The counts are linked sentences that hold the form as a whole token. They
decided the choices of section 5.3.

| Index | Reviewed form | Count | Module term | Count |
| ----- | ------------- | ----- | ----------- | ----- |
| 8     | 嗰個          | 127   | 嗰          | 481   |
| 42    | 媽咪          | 15    | 阿媽        | 74    |
| 43    | 爸爸          | 21    | 阿爸        | 50    |
| 62    | 皮膚          | 7     | 皮          | 41    |
| 66    | 肥            | 37    | 脂肪        | 3     |
| 69    | 尾巴          | 1     | 尾          | 32    |
| 86    | 腸肚          | 0     | 腸          | 25    |
| 96    | 吐痰          | 0     | 吐          | 3     |
| 97    | 嘔吐          | 0     | 嘔          | 1     |
| 101   | 睇見          | 0     | 睇          | 334   |
| 102   | 聽到          | 40    | 聽          | 227   |
| 105   | 聞到          | 2     | 聞          | 17    |
| 106   | 害怕          | 0     | 驚          | 61    |
| 114   | 割            | 6     | 切          | 42    |
| 119   | 遊            | 55    | 游水        | 19    |
| 123   | 瞓低          | 1     | 瞓          | 76    |
| 127   | 跌倒          | 0     | 跌          | 21    |
| 129   | 揸住          | 3     | 揸          | 31    |
| 138   | 聯            | 39    | 縫          | 2     |
| 139   | 數數          | 0     | 數          | 89    |
| 162   | 天空          | 31    | 天          | 203   |
| 169   | 燃燒          | 5     | 燒          | 31    |
| 193   | 平滑          | 0     | 滑          | 17    |
| 196   | 正確          | 10    | 啱          | 97    |

For index 138 the count of the old form 聯 is higher, because 聯 is a common
verb. It is not the verb to sew, so the count did not decide that row.

### 15.2 Filter counts

| Filter | Sentences that pass |
| ------ | ------------------- |
| F1     | 7,038               |
| F2     | 4,615               |
| F3     | 4,329               |
| F4     | 4,329               |
| F5     | 4,273               |
| F6     | 4,234               |
| F7     | 4,228               |
| F8     | 4,219               |
| F9     | 4,219               |
| F10    | 4,219               |
| F11    | 4,184               |
| F12    | 4,184               |
| Unknown | 3,618               |
