# Spec: French Swadesh Sentence Pack with Audio

Status: proposed
Date: 2026-10-18
Owner: anki-tools
Depends on: `specs/mandarin-swadesh-sentence-pack.md` (pipeline shape)

## 1. Purpose

Build a listening-comprehension deck for French. The deck has two layers:

1. The reviewed Swadesh word list: 207 concepts, 207 forms.
2. Useful sentences that contain those words, with audio on each card.

The word list stays at the Swadesh size, so the deck stays small enough to
review in one pass. The existing `anki swadesh build french` command makes one
isolated-word card per Swadesh entry. This spec adds the sentence layer on top
of it.

The Mandarin pack already implements the same shape. French adds three new
problems: no reviewed word file exists, the language is inflected, and the
Tatoeba sentences use French typography.

## 2. Scope

In scope:

- A reviewed word list, `vocab/english-french.csv`, with a coverage report.
- A sentence corpus from Tatoeba with English translations.
- Audio for each sentence and for each target word.
- A build command that writes one `.apkg` file.
- A manifest with provenance and licenses.
- A shared pipeline layer that the Greek pack also uses.

Out of scope:

- Changes to the existing word deck and its templates.
- Grammar explanations or spaced-repetition tuning.
- New Tatoeba conversion rules for other languages.
- A web service or a mobile application.

## 3. Research findings

### 3.1 Corpus and word list

All numbers below come from a probe run on the Tatoeba export of 2026-10-18,
the `Module:Swadesh/data/fr` wikitext, and the pinned package versions.

| Measure                                        | Value              |
| ---------------------------------------------- | ------------------ |
| Swadesh concepts                               | 207                |
| Terms in `Module:Swadesh/data/fr`              | 217                |
| Unique forms after slash expansion and cleanup | 214                |
| Forms shared by two concepts                   | 3                  |
| Reviewed forms in `english-french.csv`         | 207                |
| Tatoeba French sentences (fra)                 | 728,898            |
| French to English translation links            | 441,888            |
| Source sentences with at least one link        | 378,865            |
| Linked pairs with an English text              | 441,380            |
| Linked sentences with one final mark           | 376,272 of 378,561 |
| Candidate sentences after F1 to F12            | 247,090            |
| Concepts with 1 or more candidates             | 207 of 207         |
| Concepts with 2 or more candidates             | 207 of 207         |
| Tatoeba French sentences with native audio     | 9,833              |
| The same inside the candidate pool             | 4,679              |
| The same with the CC BY-SA 4.0 license         | 644                |
| Concepts with a public native audio card       | 67 of 207          |

The three shared forms are `vous` (indices 2 and 5), `femme` (36 and 40), and
`homme` (37 and 38). The reviewed list keeps one form per concept. Section 6
gives the choices.

### 3.2 Sentence length and the word window

The probe measured the candidate pool for several word windows. The filter set
is F1 to F12 of section 7, with the finite-verb filter F9 applied.

| Window (words) | Candidate sentences |
| -------------- | ------------------- |
| 3 to 9         | 248,617             |
| 3 to 10        | 264,583             |
| 4 to 9         | 231,124             |
| 4 to 10        | 247,090             |
| 5 to 9         | 194,065             |

The window 4 to 10 gives 247,090 sentences, and every concept has at least two
candidates. The deck caps at 450 cards, so the pool is 500 times larger than
the deck. Section 8.3 uses the 5 to 9 range as the preferred band.

### 3.3 Inflection

French verbs inflect, and Tatoeba stores inflected text. Exact form matching
undercounts by a wide margin. For example, `manger` appears as `mange`,
`mangé`, and `mangera`.

The probe tested two matchers. Exact surface matching found `aile` in 187 raw
sentences and `écorce` in 19. Lemma matching alone fails in the other
direction: the spaCy lemmatizer maps the bare noun `pluie` to `plui`, so the
form's own lemma is wrong without context.

Rule: a sentence covers a form when one token surface equals the form, or one
token lemma equals the form surface, or one token lemma equals the head-word
lemma computed inside a carrier phrase (`le ⟨form⟩`, `je ⟨form⟩`). Section 8.2
gives the exact procedure.

### 3.4 Typography

The Tatoeba French dump uses non-breaking spaces. The probe counted 16,606
sentences with U+202F (narrow no-break space) and 26,113 with U+00A0
(no-break space). Both characters appear before `!`, `?`, `:`, and `;`, and
inside the `« »` quote pair.

Rule: normalize U+202F and U+00A0 to a normal space before the word count, the
final-mark test, and the card text. The raw text stays in `text_original`.

## 4. Data sources

| Source                       | URL                                                                    | License               | Use                            |
| ---------------------------- | ---------------------------------------------------------------------- | --------------------- | ------------------------------ |
| Wiktionary Swadesh French    | `en.wiktionary.org/wiki/Module:Swadesh/data/fr`                        | CC BY-SA 4.0          | Base concept and form list     |
| Local Swadesh labels         | `vocab/swadesh.txt`                                                    | derived, CC BY-SA 4.0 | English label per index        |
| Tatoeba sentences (fra)      | `downloads.tatoeba.org/exports/per_language/fra/fra_sentences.tsv.bz2` | CC BY 2.0 FR          | Sentence text                  |
| Tatoeba detailed sentences   | `.../fra/fra_sentences_detailed.tsv.bz2`                               | CC BY 2.0 FR          | Author per sentence            |
| Tatoeba translation links    | `.../fra/fra-eng_links.tsv.bz2`                                        | CC BY 2.0 FR          | English translation pairing    |
| Tatoeba sentences (eng)      | `.../eng/eng_sentences.tsv.bz2`                                        | CC BY 2.0 FR          | English translation text       |
| Tatoeba sentences with audio | `.../fra/fra_sentences_with_audio.tsv.bz2`                             | per recording         | Audio id, author, license      |
| Tatoeba audio download       | `tatoeba.org/audio/download/{audio_id}`                                | per recording         | Audio file                     |
| spaCy French model           | package `fr-core-news-sm` 3.8.0                                        | MIT                   | Tokenization and lemmatization |
| wordfreq                     | Python package                                                         | Apache 2.0            | Frequency ranks                |
| gTTS                         | Python package                                                         | MIT                   | Default audio                  |
| edge-tts (optional)          | Python package                                                         | MIT                   | Alternative audio              |

Rules:

- The module code is `fr`, not `fra`. The `fra` page does not exist. Verified
  2026-10-18.
- Do not redistribute Wiktionary translations as glosses. Write the card gloss
  from the Swadesh English label, and keep the deck license separate.
- Record the Tatoeba sentence author and the audio author in the manifest.
- The French native audio set is mostly non-commercial: 5,754 of 9,833
  recordings carry CC BY-NC 4.0, and 1,953 have an empty license. A public pack
  uses only the 2,126 CC BY-SA 4.0 recordings. Section 9.1 gives the rule.
- The 2026-10-18 export uses U+202F and U+00A0. Section 3.4 gives the rule.

## 5. Pipeline

| Stage | Name               | Input                    | Output                                         |
| ----- | ------------------ | ------------------------ | ---------------------------------------------- |
| S0    | Fetch              | network                  | `data/tatoeba/fra_*.tsv`, `eng_sentences.tsv`  |
| S1    | Word table         | module terms, wordfreq   | `vocab/english-french.csv`, review CSV         |
| S2    | Candidate pool     | S0, S1                   | in-memory pool, pool report                    |
| S3    | Sentence selection | S2                       | `vocab/french_sentences.csv`, review CSV       |
| S4    | Audio              | S3                       | `audio/french_sentences/*.mp3`, index CSV      |
| S5    | Pack               | S1, S3, S4, license data | `apkg/swadesh_french_sentences.apkg`, manifest |

Each stage reads its input files and writes deterministic output. Each stage
is safe to re-run. A stage skips work when its cache is complete.

### 5.1 Shared pipeline layer

The Mandarin modules hold three language-neutral parts: Tatoeba readers, the
sentence pool, and the set cover. The French and Greek packs need the same
parts with different rules.

Add `anki_tools/sentence_pack/` with these modules:

| Module       | Content                                                            |
| ------------ | ------------------------------------------------------------------ |
| `profile.py` | `LanguageProfile` dataclass, profile registry                      |
| `tatoeba.py` | TSV readers, fetch and checksum cache (from `mandarin_fetch.py`)   |
| `pool.py`    | filters F2 to F8, dedupe, pool report (from `mandarin_swadesh.py`) |
| `select.py`  | set cover and ranking (from `mandarin_swadesh.py`)                 |
| `audio.py`   | TTS, native download, probe, index (from `mandarin_audio.py`)      |
| `pack.py`    | genanki model, templates, manifest (from `mandarin_deck.py`)       |
| `audit.py`   | gate runner and gate functions (from `mandarin_audit.py`)          |
| `languages/` | `mandarin.py`, `french.py`, `greek.py`                             |

`LanguageProfile` holds every language-specific value: Tatoeba code, word CSV
path, sentence CSV path, audio directory, deck and model names, word window,
final marks, internal marks, quote marks, `wordfreq` language, tokenizer name,
predicate rule, form-key function, similarity function, dedupe key function,
TTS voice, and the native-audio switch.

Rules:

- The refactor must not change Mandarin output. Rebuild the Mandarin pack and
  compare `vocab/mandarin_sentences.csv`, the note count, and the media count
  with the values in the Mandarin spec, section 17.
- `tests/test_mandarin_*.py` must pass without a change to any assertion.
- A language module must not contain a branch on the language name.

## 6. S1: French word table

`vocab/english-french.csv` does not exist. M1 creates it from the module terms
and reviews it. The build must not write the file.

The module has 207 rows and 217 terms. After cleanup, 214 forms are unique.
Four rows carry two terms: 40 (`femme`, `épouse`), 41 (`mari`, `époux`), 86
(`entrailles`, `intestins`), 179 (`an`, `année`), 183 (`nouveau`, `neuf`). The
review selects one form per concept:

| Index | English     | Module terms              | Reviewed form | Reason                        |
| ----- | ----------- | ------------------------- | ------------- | ----------------------------- |
| 2     | you (sing.) | `tu`, `vous`              | `tu`          | `vous` stays with index 5     |
| 36    | woman       | `femme`                   | `femme`       |                               |
| 37    | man         | `homme`                   | `homme`       |                               |
| 38    | human being | `homme`                   | `être humain` | avoid a duplicate form        |
| 40    | wife        | `femme`, `épouse`         | `épouse`      | `femme` stays with index 36   |
| 41    | husband     | `mari`, `époux`           | `mari`        | more frequent than `époux`    |
| 86    | guts        | `entrailles`, `intestins` | `intestins`   | `entrailles` has 0 candidates |
| 179   | year        | `an`, `année`             | `année`       | general sense of `year`       |
| 183   | new         | `nouveau`, `neuf`         | `nouveau`     | `neuf` means brand new        |

Rules:

- Cleanup. Remove `[[...]]` link markup. Take the text after `|` in a piped
  link. Remove a parenthetical note. Strip the space around the result.
- Multiword forms. Index 16 is `ne pas`, which is discontinuous. Indices 111
  (`se battre`), 125 (`se lever`), and 206 (`parce que`) are contiguous. Store
  the kind in a `form_kind` column with values `single`, `pair`, `discontinuous`.
- Gender. Fill `gender` for a noun and leave it empty for a verb, an adverb,
  or another word class. The module has no gender data, so M1 fills the
  value by hand from a dictionary. Do not infer it from a `le` or `la` probe,
  and do not translate it. The probe expects 120 nouns with a value.
  A missing value is not a build error, but G20 counts the empty noun rows.
- One form per concept. The table has 207 rows and 207 unique forms.
- The `swadesh_index` is the CSV row order, starting at 1. `vocab/swadesh.txt`
  has 207 labels and no trailing newline.
- Do not add HSK-like extra word lists. The deck stays at the Swadesh size.
- Frequency. Flag a form below Zipf 3.0. The probe finds one such form:
  `émoussé` at 2.42. Keep it, and record why in the review file.

Table columns:

| Column            | Type   | Example | Meaning                            |
| ----------------- | ------ | ------- | ---------------------------------- |
| `swadesh_index`   | int    | 36      | Swadesh 207 index                  |
| `english`         | string | woman   | Swadesh English label              |
| `french`          | string | femme   | French form, unique key            |
| `gender`          | string | f       | `m`, `f`, `n`, or empty for a verb |
| `variant`         | string | primary | `primary` or `alternate`           |
| `form_kind`       | string | single  | `single`, `pair`, `discontinuous`  |
| `zipf`            | float  | 4.95    | wordfreq fr Zipf value             |
| `candidate_count` | int    | 1245    | Sentences after S2 filters         |
| `tier`            | string | core    | `core` or `scaffold`               |

`candidate_count` and `tier` come from S2 and S3. They are not input data. A
word with zero candidates keeps its row, and the report lists it as uncovered.
A card that comes from the fallback window (7.1) gets `tier=scaffold`.

Check on every build:

- 207 input rows, and 207 forms after cleanup.
- All indices 1 to 207 exist, with no gap.
- Each row has one form. No duplicate form.
- Every English label matches `vocab/swadesh.txt` at the same index.
- Every form appears in the cleaned module term set at the same index, or has
  a row in `vocab/french_word_review.csv` with a reason.
- `vocab/french_word_review.csv` exists and has no `open` row.
- No form has a character outside the Latin block plus the French marks
  `àâäçéèêëîïôöùûüÿœæ`.

A failed check stops the build. The audit runs the same checks.

### 6.1 Review against sources

A generated form is not a reviewed form. Run this review:

| Check         | Source                   | Rule                                   |
| ------------- | ------------------------ | -------------------------------------- |
| English label | `vocab/swadesh.txt`      | Exact text at the same index           |
| Form          | `Module:Swadesh/data/fr` | The form appears in the module row set |
| Sense         | `wordfreq` fr corpus     | Flag below 3.0 and record the reason   |
| Shared form   | Module row set           | One form must serve one concept only   |
| Spelling      | French orthography       | Accents are present and correct        |

Output `vocab/french_word_review.csv` with columns `index`, `english`,
`module_terms`, `chosen`, `issue`, `fix`, `status`. `status` is `fixed`,
`kept`, or `open`. The build fails on any `open` row.

The probe expects 9 rows: the 9 rows of the table in section 6. The remaining
198 rows are `kept` with no change.

## 7. S2: Candidate sentence pool

Input:

- `fra_sentences.tsv`: `id`, `lang`, `text`.
- `fra-eng_links.tsv`: `fra_id`, `eng_id`.
- `eng_sentences.tsv`: `id`, `lang`, `text` for English.
- `fra_sentences_detailed.tsv`: `id`, `lang`, `text`, `author`,
  `date_added`, `date_modified`.
- `fra_sentences_with_audio.tsv`: audio id per sentence.

Normalization step:

1. Apply Unicode NFC.
2. Replace U+202F and U+00A0 with a normal space.
3. Keep the original text in `text_original`.
4. Do not change the case and do not remove accents.

Filter rules, applied in order:

| Rule | Condition                                                                      |
| ---- | ------------------------------------------------------------------------------ |
| F1   | Has at least one English translation link                                      |
| F2   | 4 to 10 words                                                                  |
| F3   | No digits and no letter outside the Latin block                                |
| F4   | No repeated-pattern text (`is_repeated_latin_sentence`)                        |
| F5   | Last character is `.`, `?`, or `!`                                             |
| F6   | Exactly 1 sentence-final mark (no two-sentence string)                         |
| F7   | No quotation mark and no dialogue dash (`«»"“”`, `—`, `–`)                     |
| F8   | At most 1 internal punctuation mark (comma, colon, semicolon)                  |
| F9   | Has a finite verb: one token with POS `VERB` or `AUX` and morph `VerbForm=Fin` |
| F10  | No orphan sentence-initial conjunction (see 7.1)                               |
| F11  | English text has 2 to 12 words                                                 |
| F12  | Not a duplicate under `latin_text_dedup_key`                                   |

A word is a run of letters, plus `'`, `’`, and `-` inside a word. Count
`aujourd'hui` as one word. Count an elided form as two tokens after the
apostrophe split: `l'eau` gives `l'` and `eau`.

Measured effect of the filters:

| Filter set                   | Pool size |
| ---------------------------- | --------- |
| Linked sentences, any length | 378,865   |
| F1 to F8, no predicate test  | 282,635   |
| F1 to F12 (the proper set)   | 247,090   |

### 7.1 Definition of a proper sentence

A card sentence must read as one complete statement. These rules enforce that
requirement:

- Length. 4 to 10 words. Shorter text is a fragment, and longer text is hard to
  follow by ear. Prefer 5 to 9.
- Final mark. The text ends with `.`, `?`, or `!`. A trailing closing quote, a
  colon, or a bare noun is a fragment.
- One sentence. Exactly one sentence-final mark. A dialogue pair or a
  two-sentence string makes a bad card.
- No dialogue. A quotation mark or a leading dash signals a nested quote or a
  dialogue.
- One clause join. At most one internal comma, colon, or semicolon.
- Predicate. spaCy tags at least one token as `VERB` or `AUX` with morph
  `VerbForm=Fin`. A bare noun phrase such as `Trois chiens.` fails this test.
  Fallback: pass when the sentence contains `est`, `sont`, `était`, `a`, `ont`,
  `y a`, or a verb form of a word-list verb.
- No orphan conjunction. A sentence that starts with `et`, `mais`, `ou`, `car`,
  `donc`, `puis`, `parce que`, `puisque`, or `bien que` and has no comma lacks
  its main clause.

Fallback window: when a word gets no candidate in 4 to 10 words, retry that
word with a 3 to 10 word window and drop F6 and F8. Mark such a card
`tier=scaffold`. A scaffold card must pass F3, F4, F5, F7, F9, and F10. G14
exempts it from F2, F6, and F8.

The probe found no uncovered concept, so the fallback is unused for French. The
code keeps it for the Greek pack.

## 8. S3: Sentence selection

### 8.1 Sentence tokenization

- Run spaCy `fr_core_news_sm` with the tagger, lemmatizer, and parser.
- Keep punctuation tokens out of word counts.
- Known word set = 207 word-list forms plus the top 1000 `wordfreq` words for
  `fr`. Compare a sentence token against the known set by surface and by lemma.
- Add `fr_core_news_sm` to `pyproject.toml` as `fr-core-news-sm>=3.8.0`. A
  missing model stops the build with a clear message.

### 8.2 Hard constraints per card

| Constraint     | Rule                                                              |
| -------------- | ----------------------------------------------------------------- |
| Target match   | Target word appears exactly 1 time as a token or as a lemma       |
| Word boundary  | The match is a whole token, never a substring of a longer word    |
| Unknown words  | At most 2 tokens outside the known set (1 with `--max-unknown 1`) |
| Length         | 4 to 10 words                                                     |
| Complete       | Passes F5 to F10; a scaffold card skips F6 and F8                 |
| Script         | Latin block only, NFC                                             |
| Duplicate      | No word-set Jaccard of 0.67 or more with a kept card              |
| Duplicate card | One card per sentence and per target word pair                    |

The target match uses the form-key rule of section 3.3:

1. Build the form key set: the form surface, plus the head-word lemma from the
   carriers `⟨form⟩`, `le ⟨form⟩`, `je ⟨form⟩`, and `il ⟨form⟩`.
2. A token matches when its surface (casefolded) or its lemma is in the key
   set.
3. For `form_kind=pair`, both parts must appear as adjacent tokens.
4. For `form_kind=discontinuous`, both parts must appear in the sentence, in
   that order. The probe counts 21,163 such sentences for `ne ... pas`.

Reuse `word_similarity_fr` and `latin_text_dedup_key` from
`anki_tools/rank.py`. Do not copy them.

### 8.3 Ranking key

Sort candidates for each target word by this key, first item first:

1. Unknown token count, low first.
2. Native audio present with a public license, yes first.
3. Mean Zipf value of the sentence tokens for `fr`, high first.
4. Word count, 5 to 9 first.
5. Tatoeba sentence id, low first, for determinism.

### 8.4 Selection and coverage

Use greedy set cover:

1. Build a bipartite graph of word-table forms and candidate sentences.
2. Score each candidate with the ranking key.
3. Take the best unused sentence. Remove the words it covers from the
   uncovered set. Increment the count for the assigned target only.
4. Stop when every word with candidates reaches `--per-word`, or the card cap
   is reached.
5. Process `core` words in Swadesh index order, so the primary form of a
   concept gets a card before its alternate form. Assign a sentence to the
   covered word with the lowest Swadesh index.

Defaults: `--per-word 2`, `--max-cards 450`, `--max-unknown 2`. The word list
has 207 forms, so 2 cards per word gives 414 cards. Keep the deck below about
450 cards for daily review.

Reject a card when the sentence has no English translation after pairing.
Report the words that stay uncovered:

- Words with zero candidates.
- Words with candidates that were cut by the card cap.

### 8.5 Output schema

`vocab/french_sentences.csv`:

| Column           | Type   | Example                        |
| ---------------- | ------ | ------------------------------ |
| `target`         | string | eau                            |
| `sentence`       | string | Je voudrais un verre d'eau.    |
| `text_original`  | string | Je voudrais un verre d'eau.    |
| `english`        | string | I would like a glass of water. |
| `tatoeba_id`     | int    | 1234567                        |
| `tatoeba_author` | string | sysko                          |
| `english_id`     | int    | 7654321                        |
| `audio_id`       | int    | 0                              |
| `audio_source`   | string | tts                            |
| `audio_license`  | string | CC BY-SA 4.0 or empty          |
| `unknown_words`  | int    | 0                              |
| `word_len`       | int    | 6                              |
| `zipf_mean`      | float  | 5.12                           |
| `rank_key`       | string | serialized sort key            |
| `tier`           | string | core or scaffold               |
| `form_kind`      | string | single                         |
| `audio_missing`  | bool   | false                          |

The file has one row per card. One sentence makes one card only. The file has
no pinyin column. A native card uses a real audio id. A TTS card sets
`audio_id` to 0.

Also write `vocab/french_sentences_review.csv` with the top three candidates
per uncovered or low-confidence word. A human reviews 5 percent of the cards
before release.

### 8.6 Gloss

- Write the gloss from the Swadesh English label, not from a translation.
- Add the gender in parentheses when the `gender` column is filled: `femme (f)`, `eau (f)`.
- For a verb, add the infinitive: `manger`.
- Store the gloss in the card field `Gloss`. Do not store a second French form.

## 9. S4: Audio

### 9.1 Tiers

| Tier | Source         | Condition                                                                                    |
| ---- | -------------- | -------------------------------------------------------------------------------------------- |
| A    | Tatoeba native | `--native-audio auto` or `only`, an audio id exists, and the license allows the pack license |
| B    | gTTS           | Default, all other cards                                                                     |
| C    | edge-tts       | Optional, when `--tts edge` is set                                                           |

Default mode is `--native-audio none`. The French native set is mostly
non-commercial: 5,754 of 9,833 recordings carry CC BY-NC 4.0, and 1,953 have an
empty license. A public pack uses only the 2,126 CC BY-SA 4.0 recordings. Skip
a recording with the CC BY-NC 4.0 license unless `--pack-license private`.

The public native tier adds audio to 67 of 207 concepts. It is opt-in.

### 9.2 TTS settings

| Setting     | Value                                                           |
| ----------- | --------------------------------------------------------------- |
| Engine      | gTTS, package `gtts`, already in `pyproject.toml`               |
| Voice       | `gtts:fr` for the default pack                                  |
| Alt engine  | edge-tts, package `edge-tts`, optional                          |
| Alt voice   | `fr-FR-DeniseNeural` for edge-tts                               |
| Other voice | `fr-FR-HenriNeural`, `fr-FR-EloiseNeural`                       |
| Rate        | edge-tts `-10%` for sentence audio, `-25%` for the slow variant |
| Format      | MP3, mono, 24 kHz                                               |
| Text input  | NFC sentence, normal spaces, no punctuation change              |
| Retry       | 3 attempts, 2, 4, 8 seconds backoff                             |
| Rate limit  | 1 request per second, one worker                                |

The voice names above come from the edge-tts voice list, read on 2026-10-18.
gTTS has no rate control, so the slow variant needs edge-tts. One voice per
pack keeps the listening training consistent.

### 9.3 File names and cache

| Type       | Pattern               | Example                 |
| ---------- | --------------------- | ----------------------- |
| Native     | `ta_{audio_id}.mp3`   | `ta_1276691.mp3`        |
| TTS        | `tts_{hash}.mp3`      | `tts_7f3c1a9b2e04.mp3`  |
| TTS slow   | `tts_{hash_slow}.mp3` | `tts_4c8d0e2f6a11.mp3`  |
| Word audio | `word_{hash}.mp3`     | `word_2a91bb0c4f11.mp3` |

The hash input is the NFC sentence text, the voice name, and the rate. A slow
file uses the slow rate, so it gets its own hash. The `word_{hash}` input is
the word text, the voice, and the rate.

Keep an index file `audio/french_sentences/index.csv` with columns `filename`,
`text`, `voice`, `source`, `audio_id`, `license`, `bytes`, `sha1`,
`duration_s`. Use ASCII file names only.

### 9.4 Validation

- `ffprobe` reads each file. Duration must be 0.4 to 15 seconds.
- The file must start with `ID3` or an MPEG frame sync (`FF Ex` or `FF Fx`).
- A failed file is deleted and the card is marked `audio_missing`.
- A card without audio does not enter the pack.
- A TTS file for a sentence with an apostrophe must not lose the apostrophe.
  Check that the synthesis input equals the card sentence.

`ffprobe` is an external tool. Document it in the README. A missing `ffprobe`
stops the build.

Measured cost: 450 TTS sentences at 1 request per second take about 8 minutes.

## 10. S5: Pack build

### 10.1 Note model

Model name: `French Swadesh Sentence`.

| Order | Field       | Content                       |
| ----- | ----------- | ----------------------------- |
| 0     | `Audio`     | `[sound:tts_...mp3]`          |
| 1     | `Sentence`  | Sentence with `<b>target</b>` |
| 2     | `English`   | English translation           |
| 3     | `Target`    | Target word                   |
| 4     | `Gloss`     | Short gloss plus gender       |
| 5     | `WordAudio` | `[sound:word_...mp3]`         |
| 6     | `Source`    | `Tatoeba #1234567 · author`   |

Sort field index: 3 (`Target`). This groups cards by word in the browser.

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
<div class="english">{{English}}</div>
<div class="target">{{Target}} {{Gloss}} {{WordAudio}}</div>
<div class="source">{{Source}}</div>
```

Use `templates/card.css` and add these classes: `.front-audio`, `.sentence`,
`.english`, `.target`, `.source`, `.hint`. Sentence font 2.5rem, English
1.1rem, source 0.8rem gray. The French pack has no pinyin line.

The bold target must cover the matched token only. For an elided form, bold the
word after the apostrophe: `un verre d'<b>eau</b>`.

### 10.2 Tags

One tag per dimension:

- `swadesh::water`
- `audio::native` or `audio::tts`
- `tier::core` or `tier::scaffold`
- `lang::fr`

Anki tags cannot contain spaces. Map each gloss to a slug. Replace each space,
`/`, and punctuation mark with `_`.

### 10.3 Deck

- Deck name: `French Swadesh Sentences`.
- Add notes in Swadesh index order. Anki shows new cards in note order.
- Do not split the deck. Use tags for study filters.

### 10.4 Manifest

Write `apkg/swadesh_french_sentences.manifest.json`:

```json
{
  "build_date": "2026-10-19",
  "source_dump_date": "2026-10-18",
  "counts": {
    "words": 207,
    "cards": 414,
    "scaffold": 0,
    "native_audio": 0,
    "tts_audio": 414
  },
  "coverage": {"core_with_card": 1.0},
  "voice": "gtts:fr",
  "pack_license": "public",
  "audio_license_skipped": {"empty": 0, "CC BY-NC 4.0": 0},
  "files": {"source_csv": "sha256:...", "sentences_csv": "sha256:..."},
  "licenses": [
    {"data": "Tatoeba sentences", "license": "CC BY 2.0 FR"},
    {"data": "Tatoeba audio", "license": "per recording"},
    {"data": "spaCy fr_core_news_sm", "license": "MIT"}
  ],
  "excluded_words": []
}
```

`words` is the word-table row count. `cards` is the sentence CSV row count.
`scaffold`, `native_audio`, and `tts_audio` count rows with the matching `tier`
or `audio_source` value. `audio_license_skipped` counts only files that this
build skips. The probe gives 5,754 CC BY-NC 4.0 and 1,953 empty recordings in
the source (section 4).

### 10.5 Attribution

`ATTRIBUTION.md` is one file for all packs. Add a `## French pack` section. The
section lists:

- Tatoeba sentence ids and authors, grouped by author.
- Tatoeba audio ids and authors, when the native tier is on.
- The Swadesh source page and its license.
- The TTS engine and voice.
- The spaCy model and its license.

State the licenses in the deck description and in `ATTRIBUTION.md`: word data
CC BY-SA 4.0, sentence text CC BY 2.0 FR, and audio per recording. A TTS file
names the engine and the voice. A native file keeps its recording license.

Tatoeba text needs this sentence: "Sentences come from Tatoeba
(<https://tatoeba.org>) and use the CC BY 2.0 FR license."

Gate G12 reads only the section for the pack language. Do not change the
Mandarin section.

## 11. CLI

New commands under the existing `anki swadesh` app. Keep `build` unchanged.

```bash
anki swadesh fetch --lang fra --lang eng                 # S0
anki swadesh sentences french --per-word 2               # S1 to S3
anki swadesh audio french --native-audio none            # S4
anki swadesh pack french                                 # S5
anki swadesh audit french                                # gates, no writes
```

Options:

| Command     | Option           | Default      | Meaning                      |
| ----------- | ---------------- | ------------ | ---------------------------- |
| `fetch`     | `--lang`         | `cmn`, `eng` | Tatoeba language codes       |
| `fetch`     | `--refresh`      | off          | Ignore the cache and refetch |
| `sentences` | `--per-word`     | 2            | Max sentences per word       |
| `sentences` | `--max-cards`    | 450          | Hard card cap                |
| `sentences` | `--max-unknown`  | 2            | Unknown tokens per sentence  |
| `sentences` | `--min-words`    | 4            | Minimum word count           |
| `sentences` | `--max-words`    | 10           | Maximum word count           |
| `audio`     | `--native-audio` | none         | `none`, `auto`, `only`       |
| `audio`     | `--tts`          | gtts         | `gtts` or `edge`             |
| `audio`     | `--voice`        | `gtts:fr`    | TTS voice                    |
| `audio`     | `--slow-audio`   | off          | Add a slow variant per card  |
| `pack`      | `--pack-license` | public       | `public` or `private`        |

`anki swadesh fetch` with no `--lang` keeps the current Mandarin behavior. The
command must fetch `eng` once and share it between packs.

## 12. File layout

```text
data/tatoeba/                      cached dumps, gitignored
  fra_sentences.tsv
  fra_sentences_detailed.tsv
  fra-eng_links.tsv
  fra_sentences_with_audio.tsv
  eng_sentences.tsv
vocab/
  english-french.csv               reviewed source list
  french_word_review.csv
  french_sentences.csv
  french_sentences_review.csv
audio/french_sentences/
  ta_*.mp3  tts_*.mp3  word_*.mp3  index.csv
apkg/
  swadesh_french_sentences.apkg
  swadesh_french_sentences.manifest.json
ATTRIBUTION.md
```

Add `data/` and `apkg/*.apkg` to `.gitignore`. `audio/` is already ignored.
Commit the CSV files and the manifest.

## 13. Quality gates

`anki swadesh audit french` runs these checks and exits with a non-zero code on
a failure.

| Gate | Check                                                                                      |
| ---- | ------------------------------------------------------------------------------------------ |
| G1   | Word table: 207 rows, 207 forms, no gap                                                    |
| G2   | No duplicate word form                                                                     |
| G3   | Every card target exists in the reviewed CSV                                               |
| G4   | Every card has a unique Tatoeba id after normalization                                     |
| G5   | Target appears one time in each sentence, as a whole token or lemma                        |
| G6   | No card pair with word-set Jaccard of 0.67 or more                                         |
| G7   | Every card has a valid audio file with duration 0.4 to 15 seconds                          |
| G8   | Audio file count equals the media count in the APKG                                        |
| G9   | No card matches the target inside a longer word                                            |
| G10  | Core coverage of 95 percent or more                                                        |
| G11  | Manifest counts match the word table and the sentence CSV (10.4)                           |
| G12  | `ATTRIBUTION.md` has a French section and names every audio author in the pack             |
| G13  | The APKG opens with `AnkiPackage` and lists the expected notes                             |
| G14  | Core cards pass F2 to F10. Scaffold cards pass F3, F4, F5, F7, F9, F10, and length 3 to 10 |
| G15  | `vocab/french_word_review.csv` exists and has no open row                                  |
| G16  | Every card target has a valid `word_{hash}.mp3` file                                       |
| G17  | No card field contains U+202F or U+00A0, and every field is NFC                            |
| G18  | A public pack has no card with a CC BY-NC 4.0 audio file                                   |
| G19  | No card has `form_kind=discontinuous` without both parts in the sentence (8.2)             |
| G20  | Noun rows with an empty `gender` are reported; the count is in the manifest                |

G10 uses the measured coverage. The probe measures 207 of 207 concepts (1.00).
This spec sets the gate at 0.95, which is the measured value minus 5 points.

Add unit tests in `tests/test_french_pack.py` for:

- Word-table size, dedupe, and the 9 reviewed rows.
- Module cleanup, including `[[ne]] [[pas]]` and `se battre`.
- Whitespace normalization for U+202F and U+00A0.
- The form-key rule with `pluie`, `manger`, and `l'eau`.
- The unknown-word filter with lemma matching.
- Set cover on a small fixture.
- Audio filename determinism.
- Proper-sentence filters F2 to F10 on a small fixture.
- The scaffold exemption in G14.
- Manifest counts.
- Mandarin output is unchanged after the refactor.

## 14. Risks

| Risk                                  | Effect                  | Mitigation                                         |
| ------------------------------------- | ----------------------- | -------------------------------------------------- |
| Native audio is mostly CC BY-NC       | Cannot distribute       | Default TTS, opt-in native tier, skip NC in public |
| No reviewed French word file exists   | Bad forms               | M1 module review, 9 recorded rows, review CSV      |
| Lemma errors on a bare form (`pluie`) | Wrong matcher           | Surface plus carrier lemma key (8.2)               |
| Elision hides the target (`l'eau`)    | Missed match            | Split tokens on the apostrophe                     |
| Non-breaking space before `!` and `?` | Bad final-mark test     | Normalize U+202F and U+00A0 (3.4)                  |
| Inflected verb forms                  | Under-counted coverage  | Lemma matching, `wordfreq` known set               |
| spaCy segmentation errors             | Wrong target match      | Whole-token check, review CSV, 5 percent audit     |
| Two concepts share one form           | Duplicate form gate     | Reviewed alternates (6)                            |
| Sentence pairs from a bad translation | Wrong English           | English length check, review CSV, audit            |
| gTTS is an unofficial endpoint        | Voice service can break | Retry, edge-tts option, cached audio               |
| Deck too large for daily review       | Low completion          | `--max-cards 450`, Swadesh-only list, tags         |

## 15. Milestones

| Milestone | Task                                                 | Output                    |
| --------- | ---------------------------------------------------- | ------------------------- |
| M0        | `swadesh fetch --lang fra` with checksum cache       | `data/` populated         |
| M0b       | Shared `sentence_pack/` layer, Mandarin output equal | refactor done, tests pass |
| M1        | S1 word table, module review, coverage report        | `english-french.csv`      |
| M2        | S2 and S3 with review CSV and `tier`                 | `french_sentences.csv`    |
| M3        | S4 TTS and word audio, index, validation             | `audio/` populated        |
| M4        | S5 pack, manifest, templates, tags                   | first `.apkg`             |
| M5        | `audit` with pool counts, tests, README, attribution | release check             |
| M6        | Native audio tier and slow variant                   | optional second pack      |

Acceptance for M4: the audit passes all gates, and 5 percent of the cards pass
a manual review of sentence, gloss, and audio.

## 16. Open questions

1. Ship the native audio tier in the public pack, or keep it local only?
2. Use 2 sentences per word, or 1 sentence plus the existing word deck?
3. Add scaffold template sentences for words with no Tatoeba example?
4. Split the pack into two subdecks for review load?
5. Add a slow audio variant by default?
6. Add a `lang::fr` tag to the existing word deck?

## 17. Implementation result

Not implemented. This section records the probe baseline from 2026-10-18.

| Measure             | Value                            |
| ------------------- | -------------------------------- |
| Word concepts       | 207                              |
| Reviewed forms      | 207, one per concept             |
| Candidate pool      | 247,090 sentences, 4 to 10 words |
| Concept coverage    | 207 of 207 (1.00)                |
| Uncovered concepts  | none                             |
| Public native audio | 644 recordings, 67 concepts      |
| TTS engine          | gTTS, voice `gtts:fr`            |
| Gate result         | not run                          |
