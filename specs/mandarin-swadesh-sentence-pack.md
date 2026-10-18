# Spec: Mandarin Swadesh Sentence Pack with Audio

Status: draft for review
Date: 2026-10-18
Revision: 2026-10-18, the word list is Swadesh only, and the correction
sweep of `vocab/english-*.csv` is complete.
Owner: anki-tools

## 1. Purpose

Build a listening-comprehension deck for Mandarin. The deck has two layers:

1. The reviewed Swadesh word list: 207 concepts, 211 forms.
2. Useful sentences that contain those words, with audio on each card.

The word list stays at the Swadesh size, so the deck stays small enough to
review in one pass. The existing `anki swadesh build mandarin` command makes
one isolated-word card per Swadesh entry. This spec adds the sentence layer
on top of it.

## 2. Scope

In scope:

- A reviewed word list with a measured coverage report.
- A sentence corpus with English translations.
- Native or synthesized audio for each sentence.
- A build command that writes one `.apkg` file.
- A manifest with provenance and licenses.

Out of scope:

- Changes to the existing word deck and its templates.
- Grammar explanations or spaced-repetition tuning.
- Cantonese, Japanese, or other languages.
- A web service or a mobile application.

## 3. Research findings

### 3.1 Corpus and word list

All numbers below come from a test run on the Tatoeba export of 2026-10-17
and the local `vocab/english-mandarin.csv` file after the correction sweep.
The test used jieba with a user dictionary that contains the word list.

| Measure                                       | Value            |
| --------------------------------------------- | ---------------- |
| Swadesh concepts                              | 207              |
| Rows in `english-mandarin.csv`                | 207              |
| Swadesh forms after slash expansion           | 211 unique       |
| Tatoeba Mandarin sentences (cmn)              | 89,190           |
| Mandarin to English translation links         | 78,295           |
| Linked sentences with a final sentence mark   | 65,499 of 66,257 |
| Candidate proper sentences, 4 to 10 Han chars | 36,148           |
| Tatoeba Mandarin sentences with native audio  | 5,826            |

### 3.2 Why the word list stays at Swadesh size

An earlier test used a wider word list: Swadesh plus HSK 2.0 levels 1 and
2. These numbers are the evidence for the decision in section 6.

| Measure                                        | Value      |
| ---------------------------------------------- | ---------- |
| HSK 2.0 level 1 words                          | 150        |
| HSK 2.0 level 2 words                          | 151        |
| HSK 2.0 level 3 words                          | 299        |
| Swadesh + HSK 1 + HSK 2, unique forms          | 444        |
| The same set with 2 or more eligible sentences | 397        |
| Swadesh + HSK 1 to 3, unique forms             | 719        |
| Words with 1 or more eligible sentences        | 414 of 444 |
| Words with at least one native audio card      | 274 of 444 |
| Swadesh core words with 1 or more cards        | 187 of 207 |
| Swadesh core words with native audio           | 112 of 207 |

The wider list does not add a Swadesh sentence. 207 concepts give about
414 cards at 2 cards per word, and that count stays small enough to review.
The sentence filter does not need the wider list: the known word set also
holds the top 1000 `wordfreq` words (8.1).

The 187 of 207 figure is an upper bound for the Swadesh-only list, because
the wider list covered more sentences. M1 measures the Swadesh-only value.
Strict mode (`--max-unknown 1`) gives 398 of 444 words and 175 of 207
core words for the wider list.

## 4. Data sources

| Source                       | URL                                                                    | License               | Use                         |
| ---------------------------- | ---------------------------------------------------------------------- | --------------------- | --------------------------- |
| Wiktionary Swadesh Mandarin  | `en.wiktionary.org/wiki/Module:Swadesh/data/cmn`                       | CC BY-SA 4.0          | Base concept list           |
| Local Swadesh CSV            | `vocab/english-mandarin.csv`                                           | derived, CC BY-SA 4.0 | Base word forms             |
| Tatoeba sentences (cmn)      | `downloads.tatoeba.org/exports/per_language/cmn/cmn_sentences.tsv.bz2` | CC BY 2.0 FR          | Sentence text               |
| Tatoeba detailed sentences   | `.../cmn_sentences_detailed.tsv.bz2`                                   | CC BY 2.0 FR          | Author per sentence         |
| Tatoeba translation links    | `.../cmn-eng_links.tsv.bz2`                                            | CC BY 2.0 FR          | English translation pairing |
| Tatoeba sentences with audio | `.../cmn_sentences_with_audio.tsv.bz2`                                 | per recording         | Audio id per sentence       |
| Tatoeba audio download       | `tatoeba.org/audio/download/{audio_id}`                                | per recording         | Audio file                  |
| CC-CEDICT                    | `mdbg.net/chinese/export/cedict/`                                      | CC BY-SA 4.0          | Word validation, pinyin     |
| wordfreq                     | Python package                                                         | Apache 2.0            | Frequency ranks             |

Rules:

- Do not redistribute CC-CEDICT glosses in the deck. Use CC-CEDICT only for
  internal checks, such as homograph lookup and pinyin verification.
- Write the card glosses from the Swadesh English labels. Keep the deck
  license separate.
- Record the Tatoeba sentence author and the audio author in the manifest.
- Check the native audio license before distribution. In the 2026-10-17
  export, 5,742 of 5,826 Mandarin recordings have an empty license column,
  and 84 recordings have the CC BY-NC 4.0 license. The default build uses
  TTS. The native audio tier is opt-in with `--native-audio auto`.
- Leave CC BY-NC recordings out of a public pack. Keep them only in a
  personal pack that sets `--pack-license private`.

Tatoeba audio download method, verified 2026-10-18:

- `cmn_sentences_with_audio.tsv.bz2` columns:
  `sentence_id`, `audio_id`, `author`, `license`, `attribution`.
- `https://tatoeba.org/audio/download/{audio_id}` returns `audio/mpeg`.
- The documented path `https://audio.tatoeba.org/sentences/cmn/{id}.mp3`
  returned HTTP 403 in the test. Do not use it.

## 5. Pipeline

| Stage | Name               | Input                        | Output                                           |
| ----- | ------------------ | ---------------------------- | ------------------------------------------------ |
| S0    | Fetch              | network                      | `data/tatoeba/*.tsv`                             |
| S1    | Word table         | reviewed CSV, wordfreq       | none, checks and an in-memory table              |
| S2    | Candidate pool     | S0, S1                       | in-memory pool, pool report                      |
| S3    | Sentence selection | S2                           | `vocab/mandarin_sentences.csv`, review CSV       |
| S4    | Audio              | S3                           | `audio/mandarin_sentences/*.mp3`, index CSV      |
| S5    | Pack               | S1, S3, S4, license data     | `apkg/swadesh_mandarin_sentences.apkg`, manifest |

Each stage reads its input files and writes deterministic output. Each stage
is safe to re-run. A stage skips work when its cache is complete.

## 6. S1: Swadesh word table

The word list is `vocab/english-mandarin.csv`. The correction sweep fixes
this file, so the build must not regenerate it, and it must not write a
second word file. S1 reads the CSV and makes one row per form in memory.

Do not add HSK words or other words. A wider list adds no Swadesh sentence,
and it makes the deck too large to review (3.2).

Split each slash-separated form into one row. For example, 他/她/它 becomes
three rows. The table has these columns:

| Column            | Type   | Example | Meaning                     |
| ----------------- | ------ | ------- | --------------------------- |
| `swadesh_index`   | int    | 9       | Swadesh 207 index           |
| `english`         | string | here    | Swadesh English label       |
| `mandarin`        | string | 这里    | Simplified form, unique key |
| `pinyin`          | string | zhè lǐ  | Tone-marked pinyin          |
| `traditional`     | string | 這裡    | OpenCC traditional form     |
| `variant`         | string | primary | `primary` or `alternate`    |
| `zipf`            | float  | 5.30    | wordfreq Zipf value         |
| `candidate_count` | int    | 97      | Sentences after S2 filters  |
| `tier`            | string | core    | `core` or `scaffold`        |

`candidate_count` and `tier` come from S2 and S3. They are not input data.
A word with zero candidates keeps its row, and the report lists it as
uncovered. A card that comes from the fallback window (7.1) gets
`tier=scaffold`.

Check on every build:

- 207 input rows, and 205 to 220 forms after slash expansion.
- All indices 1 to 207 exist, with no gap.
- Each row has one simplified form. No duplicate form.
- Every English label matches `vocab/swadesh.txt` at the same index.
- Every `pinyin` value equals the reviewed `transliteration` value, or has
  a manual override row in `vocab/mandarin_pinyin_overrides.csv`.
- `vocab/swadesh_review.csv` has no `open` row.

A failed check stops the build. The audit runs the same checks.

### 6.1 Review against sources

A generated form is not a reviewed form. pypinyin gives the same wrong
reading that a naive generator gives. For example, it returns 长 zhǎng and
切 qiè, which are the wrong senses for long and to cut. Run this review for
every language:

| Check            | Source                                    | Rule                                              |
| ---------------- | ----------------------------------------- | ------------------------------------------------- |
| English label    | `vocab/swadesh.txt`                       | Exact text at the same index                      |
| Canonical form   | `Module:Swadesh/data/{lang}`              | The form appears in the module row set            |
| Mandarin reading | CC-CEDICT gloss                           | The reading whose gloss matches the English label |
| Jyutping         | `tojyutping` via `anki_tools/jyutping.py` | The syllabary matches the written form            |
| Transliteration  | Romanization table for the language       | No pinyin or foreign text in the column           |
| Frequency        | `wordfreq` (`zh`, `de`, `es`, `ar`, `hi`) | Flag below 3.0. Skip Cantonese, see the rule      |

Rules:

- Do not accept a polyphone without a gloss check. Pick the reading that
  matches the English label, not the first reading from pypinyin.
- Keep a module difference when the file form is the more common word.
  Record the reason in the review file.
- Do not use Zipf for Cantonese. `wordfreq` has no `yue` list, and the `zh`
  proxy measures Mandarin only. Use words.hk or CantoDict for the Cantonese
  form, `tojyutping` for the reading, and the module for the canonical form.
- Report the match count per language in the review file.

Output `vocab/swadesh_review.csv` with columns `lang`, `index`,
`english`, `current`, `canonical`, `issue`, `fix`, `status`.
`status` is `fixed`, `kept`, or `open`. The build fails on any `open` row.

Result of the correction sweep, after the fixes (2026-10-18):

| File                  | Canonical match | Errors fixed |
| --------------------- | --------------- | ------------ |
| english-mandarin.csv  | 192 of 207      | 16           |
| english-cantonese.csv | 175 of 207      | 48           |
| english-greek.csv     | 184 of 207      | 42           |
| english-german.csv    | 201 of 207      | 1            |
| english-spanish.csv   | 194 of 207      | 8            |
| english-hindi.csv     | 205 of 207      | 3            |
| english-arabic.csv    | 206 of 207      | 8            |
| all seven files       | 1357 of 1449    | 126          |

The fixes cover forms, Mandarin pinyin readings, and Cantonese jyutping.
The sweep is complete: no row is `open`.
The remaining differences are variants or register choices. Several file
forms are better than the module form, for example 星星 for star and 心 for
heart.

The Cantonese Zipf scan is void. It reports 24 forms below 3.0, and all of
them are Cantonese-specific characters (哋, 嚟, 畀, 脷) that the Mandarin
corpus does not contain. Greek and Hindi scores have low value for a
related reason: the corpora store inflected forms, not citation forms.
For example, δαγκώνω scores 0.00 and δαγκώνει scores 3.56.

## 7. S2: Candidate sentence pool

Input:

- `cmn_sentences.tsv.bz2`: `id`, `lang`, `text`.
- `cmn-eng_links.tsv.bz2`: `cmn_id`, `eng_id`.
- Local `archive/tibetan/eng_sentences.tsv`: `id`, `lang`, `text` for English.
- `cmn_sentences_detailed.tsv.bz2`: `id`, `lang`, `text`, `author`, date.
- `cmn_sentences_with_audio.tsv.bz2`: audio id per sentence.

Converter step:

1. Convert each sentence to Simplified Chinese with OpenCC.
2. Keep the original text in `text_original`.
3. Drop a sentence when the conversion changes a Han character into a
   character that is not in the same word. Report the count.

Filter rules, applied in order:

| Rule | Condition                                                     |
| ---- | ------------------------------------------------------------- |
| F1   | Has at least one English translation link                     |
| F2   | 4 to 10 Han characters                                        |
| F3   | No Latin letters and no digits                                |
| F4   | No repeated-pattern text (`is_repeated_zh_sentence`)          |
| F5   | Last character is `。`, `？`, or `！`                         |
| F6   | Exactly 1 sentence-final mark (no two-sentence string)        |
| F7   | No quotation mark (`"`, `“`, `”`, `「`, `」`, `『`, `』`)     |
| F8   | At most 1 internal punctuation mark (comma, colon, semicolon) |
| F9   | Has a predicate: 1 token tagged `v` or `a` by `jieba.posseg`  |
| F10  | No orphan sentence-initial conjunction (see 7.1)              |
| F11  | English text has 2 to 12 words                                |
| F12  | Not a duplicate after Simplified conversion                   |

### 7.1 Definition of a proper sentence

A card sentence must read as one complete statement. These rules enforce
that requirement:

- Length. 4 to 10 Han characters. Shorter text is a fragment, longer text
  is hard to follow by ear. Prefer 5 to 9.
- Final mark. The text ends with `。`, `？`, or `！`. A trailing quote mark,
  Latin full stop, or a bare noun is a fragment.
- One sentence. Exactly one sentence-final mark. A dialogue pair or a
  two-sentence string makes a bad card.
- No dialogue. Quotation marks signal a nested quote or a dialogue.
- One clause join. At most one internal comma or colon.
- Predicate. jieba tags at least one token as a verb or an adjective.
  A bare noun phrase such as 三只狗。 fails this test. Fallback: pass when
  the sentence contains 是, 有, 在, 不, or a verb from the word list.
- No orphan conjunction. A sentence that starts with 因为, 所以, 但是, 可是,
  而且, 然后, 不过, or 虽然 and has no comma lacks its main clause.

Measured effect of the proper-sentence filters on the linked Mandarin pool:

| Filter set                   | Pool size |
| ---------------------------- | --------- |
| Linked sentences, any length | 66,257    |
| F1 to F8, no predicate test  | 41,967    |
| F1 to F10 (the proper set)   | 36,148    |

The 4 to 10 window removes fragments and still yields 36,148 sentences.
That is enough for 414 of 444 words in the wider list.
Fallback window: when a word gets no candidate in 4 to 10 characters, retry
that word with a 3 to 10 character window and drop F6 and F8. Mark such a
card `tier=scaffold`.

## 8. S3: Sentence selection

### 8.1 Sentence tokenization

- Load the word list into a jieba user dictionary.
- Keep punctuation tokens out of word counts.
- Known word set = word list plus the top 1000 `wordfreq` words for `zh`.
  This set is Mandarin only. Do not use the `zh` list for Cantonese (6.1).
- The known word set does not depend on the word list size. The
  Swadesh-only list does not make this filter stricter.

### 8.2 Hard constraints per card

| Constraint     | Rule                                                           |
| -------------- | -------------------------------------------------------------- |
| Target match   | Target word appears exactly 1 time as a token                  |
| Unknown words  | At most 2 tokens outside the known word set (1 in strict mode) |
| Length         | 4 to 10 Han characters                                         |
| Complete       | Passes all proper-sentence rules F5 to F10                     |
| Script         | Simplified Chinese only                                        |
| Duplicate      | No char-set Jaccard of 0.67 or more with a kept card           |
| Duplicate card | One card per sentence and per target word pair                 |

Reuse `char_similarity_zh` and `is_repeated_zh_sentence` from
`anki_tools/rank.py`.

### 8.3 Ranking key

Sort candidates for each target word by this key, first item first:

1. Unknown token count, low first.
2. Native audio present, yes first.
3. Mean Zipf value of the sentence tokens for `zh`, high first. Omit this
   term for Cantonese (6.1).
4. Han length, 5 to 9 first.
5. Tatoeba sentence id, low first, for determinism.

### 8.4 Selection and coverage

Use greedy set cover:

1. Build a bipartite graph of word-table forms and candidate sentences.
2. Score each candidate with the ranking key.
3. Take the best unused sentence. Remove the words it covers from the
   uncovered set. Increment the count for each covered word.
4. Stop when every word with candidates reaches `--per-word`, or the card
   cap is reached.
5. Process `core` words in Swadesh index order, so the primary form of a
   concept gets a card before its alternate form.

Defaults: `--per-word 2`, `--max-cards 450`, `--max-unknown 2`. The word
list has 211 forms, so 2 cards per word gives about 420 cards. Keep the
deck below about 450 cards for daily review.

Reject a card when the sentence has no English translation after pairing.
Report the words that stay uncovered:

- Words with zero candidates.
- Words with candidates that were cut by the card cap.

### 8.5 Output schema

`vocab/mandarin_sentences.csv`:

| Column           | Type   | Example                |
| ---------------- | ------ | ---------------------- |
| `target`         | string | 水                     |
| `sentence`       | string | 我要喝水。             |
| `text_original`  | string | 我要喝水。             |
| `pinyin`         | string | wǒ yào hē shuǐ         |
| `english`        | string | I want to drink water. |
| `tatoeba_id`     | int    | 1234567                |
| `tatoeba_author` | string | sysko                  |
| `english_id`     | int    | 7654321                |
| `audio_id`       | int    | 1276691                |
| `audio_source`   | string | native or tts          |
| `audio_license`  | string | CC BY 2.0 FR or empty  |
| `unknown_words`  | int    | 0                      |
| `han_len`        | int    | 4                      |
| `zipf_mean`      | float  | 5.12                   |
| `rank_key`       | string | serialized sort key    |

The file has one row per card. One sentence makes one card only. Assign the
sentence to the highest-priority word it covers, so the same sentence does
not appear twice with a different target.

Also write `vocab/mandarin_sentences_review.csv` with the top three candidates
per uncovered or low-confidence word. A human reviews 5 percent of the cards
before release.

### 8.6 Pinyin

- Run pypinyin with jieba segmentation and `Style.TONE`.
- Do not use the pypinyin default reading for a polyphone. Select the
  reading whose CC-CEDICT gloss matches the target gloss.
- Override with CC-CEDICT pinyin when the form has one reading only.
- Apply the 不 and 一 tone sandhi rules.
- Check that the syllable count equals the Han character count.

## 9. S4: Audio

### 9.1 Tiers

| Tier | Source         | Condition                                                                                    |
| ---- | -------------- | -------------------------------------------------------------------------------------------- |
| A    | Tatoeba native | `--native-audio auto` or `only`, an audio id exists, and the license allows the pack license |
| B    | edge-tts       | Default, all other cards                                                                     |
| C    | gTTS           | Fallback when edge-tts fails                                                                 |

Default mode is `--native-audio none`. Ship TTS audio only until a license
check on the two main Mandarin audio contributors completes. Skip a
recording with the CC BY-NC 4.0 license unless `--pack-license private`.

### 9.2 TTS settings

| Setting    | Value                                                  |
| ---------- | ------------------------------------------------------ |
| Engine     | edge-tts, Python package `edge-tts`                    |
| Voice      | `zh-CN-XiaoxiaoNeural` for the default pack            |
| Alt voice  | `zh-CN-YunxiNeural`, `zh-CN-YunyangNeural`             |
| Rate       | `-10%` for sentence audio, `-25%` for the slow variant |
| Format     | MP3, mono, 24 kHz                                      |
| Text input | Simplified sentence, no punctuation change             |
| Retry      | 3 attempts, 2, 4, 8 seconds backoff                    |
| Rate limit | 1 request per second, one worker                       |

Add `--voice-by-card` later for voice variety. One voice per pack keeps the
listening training consistent.

### 9.3 File names and cache

| Type       | Pattern               | Example                     |
| ---------- | --------------------- | --------------------------- |
| Native     | `ta_{audio_id}.mp3`   | `ta_1276691.mp3`            |
| TTS        | `tts_{hash}.mp3`      | `tts_7f3c1a9b2e04.mp3`      |
| TTS slow   | `tts_{hash}_slow.mp3` | `tts_7f3c1a9b2e04_slow.mp3` |
| Word audio | `word_{hash}.mp3`     | `word_2a91bb0c4f11.mp3`     |

The hash input is the sentence text plus the voice name plus the rate.
Keep an index file `audio/mandarin_sentences/index.csv` with columns
`filename`, `text`, `voice`, `source`, `bytes`, `sha1`, `duration_s`.
Use ASCII file names only. Anki handles UTF-8 names, but ASCII names are
shorter and stable on every platform.

### 9.4 Validation

- `ffprobe` reads each file. Duration must be 0.4 to 15 seconds.
- The file must start with `ID3` or `FF FB`.
- A failed file is deleted and the card is marked `audio_missing`.
- A card without audio does not enter the pack.

Measured cost: 450 TTS sentences at 1 request per second take about 8
minutes. 200 native downloads take about 4 minutes.

## 10. S5: Pack build

### 10.1 Note model

Model name: `Mandarin Swadesh Sentence`.

| Order | Field       | Content                       |
| ----- | ----------- | ----------------------------- |
| 0     | `Audio`     | `[sound:tts_...mp3]`          |
| 1     | `Sentence`  | Sentence with `<b>target</b>` |
| 2     | `Pinyin`    | Tone-marked pinyin            |
| 3     | `English`   | English translation           |
| 4     | `Target`    | Target word                   |
| 5     | `Gloss`     | Short gloss plus pinyin       |
| 6     | `WordAudio` | `[sound:word_...mp3]`         |
| 7     | `Source`    | `Tatoeba #1234567 · author`   |

Sort field index: 4 (`Target`). This groups cards by word in the browser.

Implementation note: `AnkiPackage.create` accepts field values only. Add an
optional `tags` key to each card dict and pass it to `genanki.Note`. Keep the
current behavior when the key is absent. Do not change the existing callers.

Front template:

```html
<div class="front-audio">{{Audio}}</div>
<div class="hint">Tap to show the sentence</div>
```

Back template:

```html
{{Audio}}
<hr id=answer>
<div class="sentence">{{Sentence}}</div>
<div class="pinyin">{{Pinyin}}</div>
<div class="english">{{English}}</div>
<div class="target">{{Target}} {{Gloss}} {{WordAudio}}</div>
<div class="source">{{Source}}</div>
```

Use `templates/card.css` and add these classes: `.front-audio`,
`.sentence`, `.pinyin`, `.english`, `.target`, `.source`, `.hint`.
Sentence font 2.5rem, pinyin 1.2rem, English 1.1rem, source 0.8rem gray.

### 10.2 Tags

One tag per dimension:

- `swadesh::water`
- `audio::native` or `audio::tts`
- `tier::core` or `tier::scaffold`

Anki tags cannot contain spaces. Map each gloss to a slug.

### 10.3 Deck

- Deck name: `Mandarin Swadesh Sentences`.
- Add notes in Swadesh index order. Anki shows new cards in note order.
- Do not split the deck. Use tags for study filters.

### 10.4 Manifest

Write `apkg/swadesh_mandarin_sentences.manifest.json`:

```json
{
  "build_date": "2026-10-18",
  "source_dump_date": "2026-10-17",
  "counts": {"words": 211, "cards": 420, "scaffold": 24, "native_audio": 120, "tts_audio": 300},
  "coverage": {"core_with_card": 0.88},
  "voice": "zh-CN-XiaoxiaoNeural",
  "pack_license": "public",
  "audio_license_skipped": {"none": 0, "CC BY-NC 4.0": 3},
  "files": {"source_csv": "sha256:...", "sentences_csv": "sha256:..."},
  "licenses": [
    {"data": "Tatoeba sentences", "license": "CC BY 2.0 FR"},
    {"data": "Tatoeba audio", "license": "per recording"}
  ],
  "excluded_words": ["骨头", "肝"]
}
```

### 10.5 Attribution

Write `ATTRIBUTION.md` at the repo root and add a short line to the deck
description. The file lists:

- Tatoeba sentence ids and authors, grouped by author.
- Tatoeba audio ids and authors, when the native tier is on.
- The Swadesh source page and its license.
- The TTS engine and voice.

Tatoeba text needs this sentence: "Sentences come from Tatoeba
(https://tatoeba.org) and use the CC BY 2.0 FR license."

## 11. CLI

New commands under the existing `anki swadesh` app. Keep `build` unchanged.

```bash
anki swadesh fetch                                   # S0, cache Tatoeba data
anki swadesh sentences mandarin --per-word 2         # S1 to S3
anki swadesh audio mandarin --native none            # S4
anki swadesh pack mandarin                           # S5
anki swadesh audit mandarin                          # gates, no writes
```

Options:

| Option           | Default              | Meaning                      |
| ---------------- | -------------------- | ---------------------------- |
| `--per-word`     | 2                    | Max sentences per word       |
| `--max-cards`    | 450                  | Hard card cap                |
| `--max-unknown`  | 2                    | Unknown tokens per sentence  |
| `--min-han`      | 4                    | Minimum Han characters       |
| `--max-han`      | 10                   | Maximum Han characters       |
| `--native-audio` | none                 | `none`, `auto`, `only`       |
| `--voice`        | zh-CN-XiaoxiaoNeural | edge-tts voice               |
| `--slow-audio`   | off                  | Add a slow variant per card  |
| `--refresh`      | off                  | Ignore the cache and refetch |
| `--pack-license` | public               | `public` or `private`        |

## 12. File layout

```
data/tatoeba/                      cached dumps, gitignored
  cmn_sentences.tsv
  cmn_sentences_detailed.tsv
  cmn-eng_links.tsv
  cmn_sentences_with_audio.tsv
  eng_sentences.tsv                symlink or copy of archive/tibetan/eng_sentences.tsv
vocab/
  english-mandarin.csv            reviewed source list
  mandarin_sentences.csv
  mandarin_sentences_review.csv
  mandarin_pinyin_overrides.csv
  swadesh_review.csv
audio/mandarin_sentences/
  ta_*.mp3  tts_*.mp3  word_*.mp3  index.csv
apkg/
  swadesh_mandarin_sentences.apkg
  swadesh_mandarin_sentences.manifest.json
ATTRIBUTION.md
specs/mandarin-swadesh-sentence-pack.md
```

Add `data/`, `audio/mandarin_sentences/`, and `apkg/*.apkg` to
`.gitignore`. Commit the CSV files and the manifest.

## 13. Quality gates

`anki swadesh audit mandarin` runs these checks and exits with a non-zero
code on a failure.

| Gate | Check                                                             |
| ---- | ----------------------------------------------------------------- |
| G1   | Word table: 207 rows, 205 to 220 forms, no gap                    |
| G2   | No duplicate word form                                            |
| G3   | Every card target exists in the reviewed CSV                      |
| G4   | Every card has a unique Tatoeba id after conversion               |
| G5   | Target appears one time in each sentence                          |
| G6   | No card pair with char-set Jaccard of 0.67 or more                |
| G7   | Every card has a valid audio file with duration 0.4 to 15 seconds |
| G8   | Audio file count equals the media count in the APKG               |
| G9   | Pinyin syllable count equals Han character count                  |
| G10  | Core coverage of 80 percent or more, provisional for M1           |
| G11  | Manifest counts equal the CSV row counts                          |
| G12  | `ATTRIBUTION.md` exists and names every audio author in the pack  |
| G13  | The APKG opens with `AnkiPackage` and lists the expected notes    |
| G14  | Every card passes the proper-sentence rules F2 to F10             |
| G15  | `vocab/swadesh_review.csv` has no open row                        |

G10 is provisional. M1 measures the real coverage. This spec then keeps
the measured value minus 5 points as the gate.

Add unit tests in `tests/test_mandarin_pack.py` for:

- Word-table size, dedupe, and slash expansion.
- Unknown-word filter.
- Set cover on a small fixture.
- Audio filename determinism.
- Proper-sentence filters F2 to F10 on a small fixture.
- Manifest counts.

## 14. Risks

| Risk                                       | Effect                  | Mitigation                                           |
| ------------------------------------------ | ----------------------- | ---------------------------------------------------- |
| Native audio license unclear (84 CC BY-NC) | Cannot distribute       | Default TTS, opt-in native tier, skip NC when public |
| Tatoeba text has traditional forms         | Mixed script deck       | OpenCC conversion, keep original text for checks     |
| jieba segmentation errors                  | Wrong target match      | User dictionary, exact substring check, review CSV   |
| Swadesh tail has no proper sentence        | Missing core words      | Fallback window, scaffold tier, manual frame         |
| Strict length rule drops a natural short   | Fewer cards per word    | 4-char minimum, fallback window, review CSV          |
| edge-tts is an unofficial endpoint         | Voice service can break | Retry, gTTS fallback, cached audio                   |
| Tatoeba rate limits and slow downloads     | Build time              | 1 request per second, cache, resume                  |
| Deck too large for daily review            | Low completion          | `--max-cards 450`, Swadesh-only list, tags           |
| Swadesh-only list gives a small deck       | Few cards to study      | 2 cards per word, scaffold tier, HSK tier later      |
| Sentence pairs from a bad translation      | Wrong English           | English length check, review CSV, 5 percent audit    |
| Wrong Swadesh form or reading              | Bad cards               | Source review in 6.1, gloss check, review CSV        |

## 15. Milestones

| Milestone | Task                                           | Output                   |
| --------- | ---------------------------------------------- | ------------------------ |
| M0        | `swadesh fetch` with cache and checksum report | `data/` populated        |
| M1        | S1 word-table checks and coverage report       | coverage report          |
| M2        | S2 and S3 with review CSV                      | `mandarin_sentences.csv` |
| M3        | S4 TTS only, audio index, validation           | `audio/` populated       |
| M4        | S5 pack, manifest, templates, tags             | first `.apkg`            |
| M5        | `audit`, tests, README and `ATTRIBUTION.md`    | release check            |
| M6        | Native audio tier and slow variant             | optional second pack     |

Acceptance for M4: the audit passes all gates, and 5 percent of the cards
pass a manual review of sentence, pinyin, gloss, and audio.

## 16. Open questions

1. Ship the native audio tier in the public pack, or keep it local only?
2. Use 2 sentences per word, or 1 sentence plus the existing word deck?
3. Add scaffold template sentences for words with no Tatoeba example?
4. Split the pack into two subdecks for review load?
5. Add a slow audio variant by default?
6. Add HSK 1 and 2 words later as an optional extended tier?
