# Spec: Mandarin Swadesh Sentence Pack with Audio

Status: draft for review
Date: 2026-10-18
Owner: anki-tools

## 1. Purpose

Build a listening-comprehension deck for Mandarin. The deck has three layers:

1. A base list of the 207 Swadesh concepts in Mandarin.
2. A master word list of 2x to 3x that size (414 to 621 entries).
3. Useful sentences that contain those words, with audio on each card.

The existing `anki swadesh build mandarin` command makes one isolated-word card
per Swadesh entry. This spec adds the sentence layer on top of it.

## 2. Scope

In scope:

- A master word list with a measured coverage report.
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

All numbers below come from a test run on the Tatoeba export of 2026-10-17
and the local `vocab/english-mandarin.csv` file. The test used jieba with a
user dictionary that contains the master list.

| Measure                                          | Value           |
| ------------------------------------------------ | --------------- |
| Swadesh concepts                                 | 207             |
| Swadesh form entries in `english-mandarin.csv`   | 209, 207 unique |
| Swadesh forms with 1 or more candidates, 3 to 8  | 185 of 207      |
| Swadesh forms with 3 or more candidates, 3 to 8  | 157 of 207      |
| Swadesh forms with 1 or more candidates, 3 to 10 | 194 of 207      |
| Tatoeba Mandarin sentences (cmn)                 | 89,190          |
| Mandarin to English translation links            | 78,295          |
| Candidate sentences, 3 to 8 Han characters       | 30,465          |
| Candidate sentences, 3 to 10 Han characters      | 43,379          |
| Tatoeba Mandarin sentences with native audio     | 5,826           |
| HSK 2.0 level 1 words                            | 150             |
| HSK 2.0 level 2 words                            | 151             |
| HSK 2.0 level 3 words                            | 299             |
| Swadesh + HSK 1 + HSK 2, unique forms            | 444             |
| The same set with 2 or more candidate sentences  | 405             |
| Swadesh + HSK 1 to 3, unique forms               | 719             |

Sentences with at most one word outside the master list and the top 1000
frequency list give coverage as follows:

| Measure                                   | Value      |
| ----------------------------------------- | ---------- |
| Master list, Swadesh + HSK 1 + HSK 2      | 444 words  |
| Words with 1 or more selected sentences   | 408 of 444 |
| Words with at least one native audio card | 248 of 444 |
| Swadesh core words with 1 or more cards   | 183 of 207 |
| Swadesh core words with native audio      | 100 of 207 |

Result: a 500-word master list is feasible. Native audio covers about half of
the words. TTS must cover the rest.

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
| HSK 2.0 word lists           | `raw.githubusercontent.com/glxxyz/hskhsk.com/main/data/lists/`         | MIT (repo)            | Extended vocabulary         |
| CC-CEDICT                    | `mdbg.net/chinese/export/cedict/`                                      | CC BY-SA 4.0          | Word validation, pinyin     |
| wordfreq                     | Python package                                                         | Apache 2.0            | Frequency ranks             |

Rules:

- Do not redistribute CC-CEDICT glosses in the deck. Use CC-CEDICT only for
  internal checks, such as homograph lookup and pinyin verification.
- Write the card glosses from the Swadesh English labels and from the HSK
  official definitions. Keep the deck license separate.
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
| S0    | Fetch              | network                      | `data/tatoeba/*.tsv`, `data/hsk/*.txt`           |
| S1    | Base list          | Swadesh module and local CSV | `vocab/mandarin_swadesh.csv`                     |
| S2    | Master list        | base list, HSK, wordfreq     | `vocab/mandarin_master.csv`, report              |
| S3    | Candidate pool     | S0, S2                       | in-memory pool, pool report                      |
| S4    | Sentence selection | S3                           | `vocab/mandarin_sentences.csv`, review CSV       |
| S5    | Audio              | S4                           | `audio/mandarin_sentences/*.mp3`, index CSV      |
| S6    | Pack               | S2, S4, S5, license data     | `apkg/swadesh_mandarin_sentences.apkg`, manifest |

Each stage reads its input files and writes deterministic output. Each stage
is safe to re-run. A stage skips work when its cache is complete.

## 6. S1: Swadesh base list

Parse `vocab/english-mandarin.csv`, which already has 209 forms for 207
concepts. Add these columns:

| Column        | Type   | Example        | Meaning                  |
| ------------- | ------ | -------------- | ------------------------ |
| `index`       | int    | 9              | Swadesh 207 index        |
| `english`     | string | here           | Swadesh English label    |
| `mandarin`    | string | 这里           | Simplified form          |
| `traditional` | string | 這裡           | Traditional form         |
| `pinyin`      | string | zhèli          | Tone-marked pinyin       |
| `variant`     | string | primary        | `primary` or `alternate` |
| `notes`       | string | Beijing accent | Register or region note  |

Check:

- All indices 1 to 207 exist, with no gap.
- Each row has exactly one simplified form.
- Multi-form entries (for example 他/她/它) become separate rows.

## 7. S2: Master list of 2x to 3x size

### 7.1 Target size

Target size is 500 entries, with a hard limit of 400 to 650. This range is
1.9x to 3.1x of 207. The build fails outside the limit.

### 7.2 Construction rule

Apply these steps in order:

1. Add every Swadesh form. Swadesh rows are always `tier=core`.
2. Add HSK 1 and HSK 2 forms, `tier=extended`.
3. Deduplicate by simplified string. Keep the row with the higher source
   priority: Swadesh before HSK 1, HSK 1 before HSK 2, HSK 2 before HSK 3.
4. Remove an `extended` form when its candidate count is below 2.
5. Repeat these steps until the count is stable:
   add HSK 3 forms in descending Zipf until the count reaches 500, then
   remove `extended` forms with fewer than 2 candidates.
6. Keep a Swadesh form with zero candidates, and set `tier=scaffold`.

The Swadesh part is fixed. The HSK part flexes to reach the target size.

### 7.3 Output schema

`vocab/mandarin_master.csv`:

| Column            | Type   | Example | Meaning                           |
| ----------------- | ------ | ------- | --------------------------------- |
| `rank`            | int    | 1       | Build order, low rank first       |
| `mandarin`        | string | 水      | Simplified form, unique key       |
| `traditional`     | string | 水      | Traditional form                  |
| `pinyin`          | string | shuǐ    | Tone-marked pinyin                |
| `pos`             | string | noun    | Part of speech                    |
| `gloss_en`        | string | water   | Short gloss, 1 to 4 words         |
| `swadesh_index`   | int    | 150     | Swadesh index or empty            |
| `source`          | string | swadesh | `swadesh`, `hsk1`, `hsk2`, `hsk3` |
| `hsk_level`       | int    | 0       | 0 for Swadesh rows                |
| `zipf`            | float  | 5.30    | wordfreq Zipf value               |
| `candidate_count` | int    | 97      | Sentences after S3 filters        |
| `tier`            | string | core    | `core`, `extended`, `scaffold`    |

Sort by `rank`. Assign rank as follows:

1. Swadesh index order, 1 to 207.
2. HSK level, then descending Zipf.

### 7.4 Master list acceptance

- Size in 400 to 650.
- `core` coverage of at least 85 percent with 1 or more sentences. The 3 to 8
  character window gives 185 of 207 Swadesh forms (89 percent).
- No duplicate `mandarin` value.
- Every `pinyin` value matches the pypinyin output for that form, or has a
  manual override row in `vocab/mandarin_pinyin_overrides.csv`.

## 8. S3: Candidate sentence pool

Input:

- `cmn_sentences.tsv.bz2`: `id`, `lang`, `text`.
- `cmn-eng_links.tsv.bz2`: `cmn_id`, `eng_id`.
- Local `eng_sentences.tsv`: `id`, `lang`, `text` for English.
- `cmn_sentences_detailed.tsv.bz2`: `id`, `lang`, `text`, `author`, date.
- `cmn_sentences_with_audio.tsv.bz2`: audio id per sentence.

Converter step:

1. Convert each sentence to Simplified Chinese with OpenCC.
2. Keep the original text in `text_original`.
3. Drop a sentence when the conversion changes a Han character into a
   character that is not in the same word. Report the count.

Filter rules, applied in order:

| Rule | Condition                                            |
| ---- | ---------------------------------------------------- |
| F1   | Has at least one English translation link            |
| F2   | 3 to 8 Han characters                                |
| F3   | No Latin letters and no digits                       |
| F4   | No repeated-pattern text (`is_repeated_zh_sentence`) |
| F5   | No more than one punctuation mark                    |
| F6   | English text has 2 to 12 words                       |
| F7   | Not a duplicate after Simplified conversion          |

Measured pool size after F1 and F2: 30,465 sentences. Record the exact
number in the pool report.

Fallback window: when a word gets no candidate in 3 to 8 characters, retry
that word with a 2 to 10 character window. Mark such a card `tier=scaffold`.

## 9. S4: Sentence selection

### 9.1 Sentence tokenization

- Load the master list into a jieba user dictionary.
- Tokenize each candidate sentence.
- Keep punctuation tokens out of word counts.
- Known word set = master list plus the top 1000 `wordfreq` words for `zh`.

### 9.2 Hard constraints per card

| Constraint     | Rule                                                 |
| -------------- | ---------------------------------------------------- |
| Target match   | Target word appears exactly 1 time as a token        |
| Unknown words  | At most 1 token outside the known word set           |
| Length         | 3 to 8 Han characters                                |
| Script         | Simplified Chinese only                              |
| Duplicate      | No char-set Jaccard of 0.67 or more with a kept card |
| Duplicate card | One card per sentence and per target word pair       |

Reuse `char_similarity_zh` and `is_repeated_zh_sentence` from
`anki_tools/rank.py`.

### 9.3 Ranking key

Sort candidates for each target word by this key, first item first:

1. Unknown token count, low first.
2. Native audio present, yes first.
3. Mean Zipf value of the sentence tokens, high first.
4. Han length, 4 to 6 first.
5. Tatoeba sentence id, low first, for determinism.

### 9.4 Selection and coverage

Use greedy set cover:

1. Build a bipartite graph of master words and candidate sentences.
2. Score each candidate with the ranking key.
3. Take the best unused sentence. Remove the words it covers from the
   uncovered set. Increment the count for each covered word.
4. Stop when every word with candidates reaches `--per-word`, or the card
   cap is reached.
5. Process `core` words before `extended` words, so the Swadesh words get
   their second sentence first.

Defaults: `--per-word 2`, `--max-cards 1200`.

Reject a card when the sentence has no English translation after pairing.
Report the words that stay uncovered:

- `scaffold` words with zero candidates.
- Words with candidates that were cut by the card cap.

### 9.5 Output schema

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
| `han_len`        | int    | 5                      |
| `zipf_mean`      | float  | 5.12                   |
| `rank_key`       | string | serialized sort key    |

The file has one row per card. One sentence makes one card only. Assign the
sentence to the highest-priority word it covers, so the same sentence does
not appear twice with a different target.

Also write `vocab/mandarin_sentences_review.csv` with the top three candidates
per uncovered or low-confidence word. A human reviews 5 percent of the cards
before release.

### 9.6 Pinyin

- Run pypinyin with jieba segmentation and `Style.TONE`.
- Override with CC-CEDICT pinyin when the form has one reading only.
- Apply the 不 and 一 tone sandhi rules.
- Check that the syllable count equals the Han character count.

## 10. S5: Audio

### 10.1 Tiers

| Tier | Source         | Condition                                                                                    |
| ---- | -------------- | -------------------------------------------------------------------------------------------- |
| A    | Tatoeba native | `--native-audio auto` or `only`, an audio id exists, and the license allows the pack license |
| B    | edge-tts       | Default, all other cards                                                                     |
| C    | gTTS           | Fallback when edge-tts fails                                                                 |

Default mode is `--native-audio none`. Ship TTS audio only until a license
check on the two main Mandarin audio contributors completes. Skip a
recording with the CC BY-NC 4.0 license unless `--pack-license private`.

### 10.2 TTS settings

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

### 10.3 File names and cache

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

### 10.4 Validation

- `ffprobe` reads each file. Duration must be 0.4 to 15 seconds.
- The file must start with `ID3` or `FF FB`.
- A failed file is deleted and the card is marked `audio_missing`.
- A card without audio does not enter the pack.

Measured cost: 900 TTS sentences at 1 request per second take about 15
minutes. 400 native downloads take about 7 minutes.

## 11. S6: Pack build

### 11.1 Note model

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

### 11.2 Tags

One tag per dimension:

- `swadesh::water`
- `hsk::1`
- `audio::native` or `audio::tts`
- `tier::core` or `tier::extended`

Anki tags cannot contain spaces. Map each gloss to a slug.

### 11.3 Deck

- Deck name: `Mandarin Swadesh Sentences`.
- Add notes in master `rank` order. Anki shows new cards in note order.
- Do not split the deck. Use tags for study filters.

### 11.4 Manifest

Write `apkg/swadesh_mandarin_sentences.manifest.json`:

```json
{
  "build_date": "2026-10-18",
  "source_dump_date": "2026-10-17",
  "counts": {"master_words": 500, "cards": 900, "native_audio": 210, "tts_audio": 690},
  "coverage": {"core_with_card": 0.93, "all_with_card": 0.88},
  "voice": "zh-CN-XiaoxiaoNeural",
  "pack_license": "public",
  "audio_license_skipped": {"none": 0, "CC BY-NC 4.0": 3},
  "files": {"master_csv": "sha256:...", "sentences_csv": "sha256:..."},
  "licenses": [
    {"data": "Tatoeba sentences", "license": "CC BY 2.0 FR"},
    {"data": "Tatoeba audio", "license": "per recording"}
  ],
  "excluded_words": ["骨头", "肝"]
}
```

### 11.5 Attribution

Write `ATTRIBUTION.md` at the repo root and add a short line to the deck
description. The file lists:

- Tatoeba sentence ids and authors, grouped by author.
- Tatoeba audio ids and authors, when the native tier is on.
- The Swadesh source page and its license.
- The HSK list source.
- The TTS engine and voice.

Tatoeba text needs this sentence: "Sentences come from Tatoeba
(https://tatoeba.org) and use the CC BY 2.0 FR license."

## 12. CLI

New commands under the existing `anki swadesh` app. Keep `build` unchanged.

```bash
anki swadesh fetch                                   # S0, cache Tatoeba and HSK data
anki swadesh master mandarin --target-size 500       # S1 and S2
anki swadesh sentences mandarin --per-word 2         # S3 and S4
anki swadesh audio mandarin --native none            # S5
anki swadesh pack mandarin                           # S6
anki swadesh audit mandarin                          # gates, no writes
```

Options:

| Option           | Default              | Meaning                      |
| ---------------- | -------------------- | ---------------------------- |
| `--target-size`  | 500                  | Master list size, 400 to 650 |
| `--per-word`     | 2                    | Max sentences per word       |
| `--max-cards`    | 1200                 | Hard card cap                |
| `--max-unknown`  | 1                    | Unknown tokens per sentence  |
| `--min-han`      | 3                    | Minimum Han characters       |
| `--max-han`      | 8                    | Maximum Han characters       |
| `--native-audio` | none                 | `none`, `auto`, `only`       |
| `--voice`        | zh-CN-XiaoxiaoNeural | edge-tts voice               |
| `--slow-audio`   | off                  | Add a slow variant per card  |
| `--refresh`      | off                  | Ignore the cache and refetch |
| `--pack-license` | public               | `public` or `private`        |

## 13. File layout

```
data/tatoeba/                      cached dumps, gitignored
  cmn_sentences.tsv
  cmn_sentences_detailed.tsv
  cmn-eng_links.tsv
  cmn_sentences_with_audio.tsv
  eng_sentences.tsv                symlink or copy of repo file
data/hsk/
  hsk1.txt  hsk2.txt  hsk3.txt
vocab/
  mandarin_swadesh.csv
  mandarin_master.csv
  mandarin_sentences.csv
  mandarin_sentences_review.csv
  mandarin_pinyin_overrides.csv
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

## 14. Quality gates

`anki swadesh audit mandarin` runs these checks and exits with a non-zero
code on a failure.

| Gate | Check                                                             |
| ---- | ----------------------------------------------------------------- |
| G1   | Master list size in 400 to 650                                    |
| G2   | No duplicate master form                                          |
| G3   | Every card target exists in the master list                       |
| G4   | Every card has a unique Tatoeba id after conversion               |
| G5   | Target appears one time in each sentence                          |
| G6   | No card pair with char-set Jaccard of 0.67 or more                |
| G7   | Every card has a valid audio file with duration 0.4 to 15 seconds |
| G8   | Audio file count equals the media count in the APKG               |
| G9   | Pinyin syllable count equals Han character count                  |
| G10  | Core coverage of 85 percent or more                               |
| G11  | Manifest counts equal the CSV row counts                          |
| G12  | `ATTRIBUTION.md` exists and names every audio author in the pack  |
| G13  | The APKG opens with `AnkiPackage` and lists the expected notes    |

Add unit tests in `tests/test_mandarin_pack.py` for:

- Master list size and dedupe.
- Unknown-word filter.
- Set cover on a small fixture.
- Audio filename determinism.
- Manifest counts.

## 15. Risks

| Risk                                       | Effect                  | Mitigation                                           |
| ------------------------------------------ | ----------------------- | ---------------------------------------------------- |
| Native audio license unclear (84 CC BY-NC) | Cannot distribute       | Default TTS, opt-in native tier, skip NC when public |
| Tatoeba text has traditional forms         | Mixed script deck       | OpenCC conversion, keep original text for checks     |
| jieba segmentation errors                  | Wrong target match      | User dictionary, exact substring check, review CSV   |
| Swadesh tail has no sentences              | Missing core words      | Scaffold tier, 2-character sentences, manual frame   |
| edge-tts is an unofficial endpoint         | Voice service can break | Retry, gTTS fallback, cached audio                   |
| Tatoeba rate limits and slow downloads     | Build time              | 1 request per second, cache, resume                  |
| Deck too large for daily review            | Low completion          | `--max-cards`, tags, subdeck by rank later           |
| Sentence pairs from a bad translation      | Wrong English           | English length check, review CSV, 5 percent audit    |

## 16. Milestones

| Milestone | Task                                           | Output                   |
| --------- | ---------------------------------------------- | ------------------------ |
| M0        | `swadesh fetch` with cache and checksum report | `data/` populated        |
| M1        | S1 and S2 with size and coverage report        | `mandarin_master.csv`    |
| M2        | S3 and S4 with review CSV                      | `mandarin_sentences.csv` |
| M3        | S5 TTS only, audio index, validation           | `audio/` populated       |
| M4        | S6 pack, manifest, templates, tags             | first `.apkg`            |
| M5        | `audit`, tests, README and `ATTRIBUTION.md`    | release check            |
| M6        | Native audio tier and slow variant             | optional second pack     |

Acceptance for M4: the audit passes all gates, and 5 percent of the cards
pass a manual review of sentence, pinyin, gloss, and audio.

## 17. Open questions

1. Ship the native audio tier in the public pack, or keep it local only?
2. Use 2 sentences per word, or 1 sentence plus the existing word deck?
3. Add scaffold template sentences for words with no Tatoeba example?
4. Split the pack by rank into 250-card subdecks for review load?
5. Add a slow audio variant by default?
