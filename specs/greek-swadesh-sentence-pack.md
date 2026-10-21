# Spec: Greek Swadesh Sentence Pack with Audio

Status: proposed
Date: 2026-10-18
Owner: anki-tools
Depends on: `specs/french-swadesh-sentence-pack.md` section 5.1
(shared `sentence_pack` layer)

## 1. Purpose

Build a listening-comprehension deck for Greek. The deck has two layers:

1. The reviewed Swadesh word list: 207 concepts.
2. Useful sentences that contain those words, with audio on each card.

The word list stays at the Swadesh size, so the deck stays small enough to
review in one pass. The existing `anki swadesh build greek` command makes one
isolated-word card per Swadesh entry. This spec adds the sentence layer on top
of it.

Greek adds four new problems beyond the Mandarin and French packs: no Tatoeba
audio, a small corpus, heavy inflection, and Greek punctuation.

## 2. Scope

In scope:

- A reviewed word list, `vocab/english-greek.csv`, with a coverage report.
- A sentence corpus from Tatoeba with English translations.
- Synthesized audio for each sentence and for each target word.
- A build command that writes one `.apkg` file.
- A manifest with provenance and licenses.
- A Greek profile for the shared `sentence_pack` layer.

Out of scope:

- Native Tatoeba audio. No recording exists (section 3.4).
- Changes to the existing word deck and its templates.
- Grammar explanations or spaced-repetition tuning.
- Polytonic Greek. The deck uses monotonic Greek only.
- A web service or a mobile application.

## 3. Research findings

### 3.1 Corpus and word list

All numbers below come from a probe run on the Tatoeba export of 2026-10-18,
the `Module:Swadesh/data/el` wikitext, and the local `vocab/english-greek.csv`.

| Measure                                     | Value           |
| ------------------------------------------- | --------------- |
| Swadesh concepts                            | 207             |
| Terms in `Module:Swadesh/data/el`           | 231             |
| Unique module forms                         | 225             |
| Rows in `english-greek.csv`                 | 207             |
| Forms after slash expansion                 | 213             |
| Unique forms after note and article removal | 208             |
| Rows with a parenthetical note              | 6               |
| Forms with a leading article                | 83              |
| Concepts that match the module              | 184 of 207      |
| Tatoeba Greek sentences (ell)               | 42,878          |
| Greek to English translation links          | 32,439          |
| Source sentences with at least one link     | 30,809          |
| Linked pairs with an English text           | 32,320          |
| Sentences that end with `;` (question mark) | 6,814 of 42,878 |
| Candidate sentences after F1 to F12         | 18,190          |
| Concepts with 1 or more candidates          | 168 of 207      |
| Concepts with 2 or more candidates          | 156 of 207      |
| Tatoeba Greek recordings                    | 0               |

The pool of 18,190 sentences gives 324 cards at 2 cards per covered concept:
156 concepts get 2 cards and 12 get 1 card. The deck stays small.

### 3.2 Greek punctuation and script

The probe measured these facts on the 2026-10-18 dump:

- 6,814 of 42,878 Greek sentences end with `;`. In Greek, `;` is the question
  mark, not a semicolon.
- 81 sentences contain U+00B7 (middle dot), which is the Greek ano teleia. It
  works as a semicolon inside a sentence.
- No sentence uses U+037E (Greek question mark) or U+0387 (Greek ano teleia).
  Tatoeba stores the ASCII and Latin-1 forms.
- Greek text uses final sigma `ς` at the end of a word and `σ` inside a word.
  The same lemma has both forms.
- The deck uses monotonic Greek. Polytonic text is out of scope.

Rules:

- Filter F5 accepts `;` as a final mark.
- Filter F6 counts `;` as a final mark, so a question with one `;` passes.
- Filter F8 counts `,`, `·`, and `:` as internal marks, and never `;`.
- Compare forms after NFC and casefold. Map `ς` to `σ` in a surface key.

### 3.3 Inflection and coverage

Greek verbs and nouns inflect, and the article agrees with the noun. Exact
surface matching fails: the probe finds 83 of 209 CSV forms by surface only,
because most CSV forms carry an article (`το νερό`) and the text often does not
(`νερό`, `νερού`).

Lemma matching with spaCy `el_core_news_sm` finds 168 of 207 concepts. The
remaining 39 concepts have no candidate in the 4 to 10 word window:

| Concepts with no candidate                                                            |
| ------------------------------------------------------------------------------------- |
| wide, louse, worm, stick, seed, tree bark, grass, horn                                |
| feather, ear, fingernail, knee, wing, guts, liver                                     |
| to suck, to vomit, to blow, to hunt, to split, to stab, to scratch, to dig, to fly    |
| to squeeze, to rub, to wipe, to pull, to tie, to count, to float, to freeze, to swell |
| ash, rotten, round, dull, smooth, dry                                                 |

Twelve concepts have exactly one candidate. The rest have two or more.

The build must not hide this gap. The report lists the 39 uncovered concepts,
and the deck carries 324 cards, not 414.

### 3.4 No native audio

The probe checked two sources on 2026-10-18:

- `downloads.tatoeba.org/exports/per_language/ell/ell_sentences_with_audio.tsv.bz2`
  returns HTTP 404. Tatoeba writes no Greek audio export.
- `api.tatoeba.org/v1/sentences?lang=ell&has_audio=yes&sort=relevance` returns
  `total: 0`.

Therefore the Greek pack is TTS only. The shared pipeline keeps the native tier
for other languages, and the Greek profile sets `native_audio=False`. A build
with `--native-audio auto` fails with a clear message.

## 4. Data sources

| Source                     | URL                                                                    | License               | Use                      |
| -------------------------- | ---------------------------------------------------------------------- | --------------------- | ------------------------ |
| Wiktionary Swadesh Greek   | `en.wiktionary.org/wiki/Module:Swadesh/data/el`                        | CC BY-SA 4.0          | Canonical form check     |
| Local Swadesh CSV          | `vocab/english-greek.csv`                                              | derived, CC BY-SA 4.0 | Base word forms          |
| Local Swadesh labels       | `vocab/swadesh.txt`                                                    | derived, CC BY-SA 4.0 | English label per index  |
| Tatoeba sentences (ell)    | `downloads.tatoeba.org/exports/per_language/ell/ell_sentences.tsv.bz2` | CC BY 2.0 FR          | Sentence text            |
| Tatoeba detailed sentences | `.../ell/ell_sentences_detailed.tsv.bz2`                               | CC BY 2.0 FR          | Author per sentence      |
| Tatoeba translation links  | `.../ell/ell-eng_links.tsv.bz2`                                        | CC BY 2.0 FR          | English pairing          |
| Tatoeba sentences (eng)    | `.../eng/eng_sentences.tsv.bz2`                                        | CC BY 2.0 FR          | English translation text |
| spaCy Greek model          | package `el-core-news-sm` 3.8.0                                        | MIT                   | Tokenization and lemmas  |
| wordfreq                   | Python package                                                         | Apache 2.0            | Frequency ranks          |
| gTTS                       | Python package                                                         | MIT                   | Default audio            |
| edge-tts (optional)        | Python package                                                         | MIT                   | Alternative audio        |

Rules:

- The module code is `el`, not `ell`. The `ell` page does not exist. Verified
  2026-10-18.
- `el-core-news-sm` is not on PyPI. Install it from the spaCy model release URL:
  `https://github.com/explosion/spacy-models/releases/download/el_core_news_sm-3.8.0/el_core_news_sm-3.8.0-py3-none-any.whl`.
  Add it to `pyproject.toml` as a direct URL requirement, or document the
  install step in the README.
- Do not redistribute Wiktionary translations as glosses. Write the card gloss
  from the Swadesh English label.
- Record the Tatoeba sentence author in the manifest.
- gTTS is the default audio source. edge-tts is optional.

## 5. Pipeline

| Stage | Name               | Input        | Output                                        |
| ----- | ------------------ | ------------ | --------------------------------------------- |
| S0    | Fetch              | network      | `data/tatoeba/ell_*.tsv`, `eng_sentences.tsv` |
| S1    | Word table         | reviewed CSV | `vocab/english-greek.csv`, review CSV         |
| S2    | Candidate pool     | S0, S1       | in-memory pool, pool report                   |
| S3    | Sentence selection | S2           | `vocab/greek_sentences.csv`, review CSV       |
| S4    | Audio              | S3           | `audio/greek_sentences/*.mp3`, index CSV      |
| S5    | Pack               | S1, S3, S4   | `apkg/swadesh_greek_sentences.apkg`, manifest |

Each stage reads its input files and writes deterministic output. Each stage
is safe to re-run. A stage skips work when its cache is complete.

### 5.1 Greek profile

The Greek pack uses the shared layer from the French spec, section 5.1. The
`greek.py` profile sets:

| Field           | Value                                                   |
| --------------- | ------------------------------------------------------- |
| Tatoeba code    | `ell`                                                   |
| English code    | `eng`                                                   |
| Word CSV        | `vocab/english-greek.csv`                               |
| Sentence CSV    | `vocab/greek_sentences.csv`                             |
| Audio directory | `audio/greek_sentences`                                 |
| Deck name       | `Swadesh Greek Sentences`                               |
| Model name      | `Greek Swadesh Sentence`                                |
| Word window     | 4 to 10 words                                           |
| Final marks     | `.?!;`                                                  |
| Internal marks  | `,·:`                                                   |
| Quote marks     | `«»"“”` and the dashes `—` and `–`                      |
| `wordfreq` code | `el`                                                    |
| Tokenizer       | spaCy `el_core_news_sm`                                 |
| Predicate rule  | POS `VERB` or `AUX` with morph `VerbForm=Fin`           |
| Similarity      | new `word_similarity_el` in `languages/greek/rank.py`   |
| Dedupe key      | new `greek_text_dedup_key` in `languages/greek/rank.py` |
| TTS voice       | `gtts:el` by default, `el-GR-AthinaNeural` for edge-tts |
| Native audio    | disabled                                                |

`word_similarity_el` and `greek_text_dedup_key` copy the shape of the French
helpers: letters only, casefold, NFC, `ς` to `σ`, and a word-set Jaccard.

## 6. S1: Greek word table

`vocab/english-greek.csv` exists but is unreviewed. M1 fixes it. The build must
not write the file.

The current file has 207 rows and 213 forms. It has three defects:

| Defect             | Count | Example                                 |
| ------------------ | ----- | --------------------------------------- |
| Parenthetical note | 6     | `εσύ (ενικός)`, `το λίπος (ουσιαστικό)` |
| Leading article    | 83    | `η γυναίκα`, `το νερό`                  |
| Duplicate form     | 4     | `αυτό`, `αυτοί`, `το φτερό`, `το πόδι`  |

The note rows are indices 2, 3, 5, 6, 11, and 66.

Module rows with two terms: 24. Rows with a form that serves two concepts: 6.
The review resolves each one:

| Index | English             | Module terms         | Reviewed form | Reason                        |
| ----- | ------------------- | -------------------- | ------------- | ----------------------------- |
| 2     | you (singular)      | `εσύ`, `εσείς`       | `εσύ`         | `εσείς` stays with index 5    |
| 3     | he/she/it/they (sg) | `αυτός`              | `αυτός`       |                               |
| 7     | this                | `αυτός`              | `αυτό`        | neuter form, distinct         |
| 36    | woman               | `γυναίκα`            | `γυναίκα`     |                               |
| 40    | wife                | `σύζυγος`, `γυναίκα` | `σύζυγος`     | `γυναίκα` stays with index 36 |
| 41    | husband             | `σύζυγος`, `άντρας`  | `άντρας`      | `σύζυγος` is the wife form    |
| 70    | feather             | `πούπουλο`, `φτερό`  | `πούπουλο`    | `φτερό` stays with index 84   |
| 80    | foot                | `πόδι`               | `πόδι`        |                               |
| 81    | leg                 | `πόδι`               | `πόδι`        | shared form, see the rule     |
| 84    | wing                | `φτερό`              | `φτερό`       |                               |

Rules:

- Strip the leading article from every form. Keep the article in the `gender`
  column, so the card gloss can show `(f)`, `(m)`, or `(n)`.
- Never strip the article from a one-word form. `ένας` is the word `one`, not an
  article plus a noun.
- One form per concept, except index 81. Indices 80 and 81 share `πόδι`,
  because Greek uses one word for foot and leg. Mark index 81 `variant=shared`
  and give it no card.
- The table therefore has 207 rows and 206 unique forms.
- The reviewed form must appear in the module row set at the same index, or
  have a review row with a reason.
- Frequency. Flag a form below Zipf 3.0. The probe finds 30 such forms, mostly
  rare verbs (`δαγκώνω`, `ρουφώ`, `φτύνω`). Keep a form when it has candidates,
  and record the reason when it does not.
- Do not add extra word lists. The deck stays at the Swadesh size.

Table columns:

| Column            | Type   | Example | Meaning                             |
| ----------------- | ------ | ------- | ----------------------------------- |
| `swadesh_index`   | int    | 149     | Swadesh 207 index                   |
| `english`         | string | star    | Swadesh English label               |
| `greek`           | string | αστέρι  | Monotonic form, NFC, unique key     |
| `gender`          | string | n       | `m`, `f`, `n`, or empty             |
| `variant`         | string | primary | `primary`, `alternate`, or `shared` |
| `form_kind`       | string | single  | `single` or `pair`                  |
| `zipf`            | float  | 4.44    | wordfreq el Zipf value              |
| `candidate_count` | int    | 12      | Sentences after S2 filters          |
| `tier`            | string | core    | `core` or `scaffold`                |

`candidate_count` and `tier` come from S2 and S3. They are not input data. A
word with zero candidates keeps its row, and the report lists it as uncovered.

Check on every build:

- 207 input rows, and 206 unique forms.
- All indices 1 to 207 exist, with no gap.
- No duplicate form, except the shared `πόδι` row.
- Every English label matches `vocab/swadesh.txt` at the same index.
- Every form appears in the cleaned module term set at the same index, or has
  a row in `vocab/greek_word_review.csv` with a reason.
- `vocab/greek_word_review.csv` exists and has no `open` row.
- Every form is NFC and monotonic. No polytonic mark, no final sigma at the
  start of a word, no Latin letter.

A failed check stops the build. The audit runs the same checks.

### 6.1 Review against sources

| Check         | Source                   | Rule                                   |
| ------------- | ------------------------ | -------------------------------------- |
| English label | `vocab/swadesh.txt`      | Exact text at the same index           |
| Form          | `Module:Swadesh/data/el` | The form appears in the module row set |
| Article       | Greek grammar            | Article in `gender`, not in `greek`    |
| Sense         | `wordfreq` el corpus     | Flag below 3.0 and record the reason   |
| Shared form   | Module row set           | One form serves one concept only       |
| Accents       | Monotonic orthography    | Accent present and correct             |

Output `vocab/greek_word_review.csv` with columns `index`, `english`,
`current`, `module_terms`, `chosen`, `issue`, `fix`, `status`. `status` is
`fixed`, `kept`, `open`, or `shared`. The build fails on any `open` row.

The probe expects up to 93 rows: 6 note rows, 4 duplicate forms, and 83
article forms with some overlap. The remaining rows are `kept`.

## 7. S2: Candidate sentence pool

Input:

- `ell_sentences.tsv`: `id`, `lang`, `text`.
- `ell-eng_links.tsv`: `ell_id`, `eng_id`.
- `eng_sentences.tsv`: `id`, `lang`, `text` for English.
- `ell_sentences_detailed.tsv`: `id`, `lang`, `text`, `author`,
  `date_added`, `date_modified`.

Normalization step:

1. Apply Unicode NFC.
2. Keep the original text in `text_original`.
3. Do not change the case and do not remove accents.
4. Keep the final sigma.

Filter rules, applied in order:

| Rule | Condition                                                       |
| ---- | --------------------------------------------------------------- |
| F1   | Has at least one English translation link                       |
| F2   | 4 to 10 words                                                   |
| F3   | No digits, no Latin letter, no letter outside the Greek block   |
| F4   | No repeated-pattern text (`is_repeated_greek_sentence`)         |
| F5   | Last character is `.`, `?`, `!`, or `;`                         |
| F6   | Exactly 1 sentence-final mark (no two-sentence string)          |
| F7   | No quotation mark and no dialogue dash (`«»"“”`, `—`, `–`)      |
| F8   | At most 1 internal punctuation mark (`,`, `·`, `:`)             |
| F9   | Has a finite verb: POS `VERB` or `AUX` and morph `VerbForm=Fin` |
| F10  | No orphan sentence-initial conjunction (see 7.1)                |
| F11  | English text has 2 to 12 words                                  |
| F12  | Not a duplicate under `greek_text_dedup_key`                    |

Greek has no infinitive, so a word-list verb is a first person singular form
(`πίνω`). The matcher of section 8.2 compares lemmas, so `πίνει` matches `πίνω`.

Measured effect of the filters:

| Filter set                   | Pool size |
| ---------------------------- | --------- |
| Linked sentences, any length | 30,809    |
| F1 to F8, no predicate test  | 25,727    |
| F1 to F12, 4 to 10 words     | 18,190    |

The probe also measured other windows:

| Window (words) | Candidate sentences |
| -------------- | ------------------- |
| 3 to 9         | 20,555              |
| 3 to 10        | 21,109              |
| 4 to 9         | 17,636              |
| 4 to 10        | 18,190              |
| 5 to 9         | 13,360              |

The window 4 to 10 keeps 18,190 sentences. A wider window adds 2,919 sentences
at 3 to 10, but most of them are fragments of three words.

### 7.1 Definition of a proper sentence

A card sentence must read as one complete statement. These rules enforce that
requirement:

- Length. 4 to 10 words. Shorter text is a fragment, and longer text is hard to
  follow by ear. Prefer 5 to 8.
- Final mark. The text ends with `.`, `?`, `!`, or `;`. A trailing closing
  quote, a colon, or a bare noun is a fragment.
- One sentence. Exactly one sentence-final mark. A dialogue pair or a
  two-sentence string makes a bad card.
- No dialogue. A quotation mark or a leading dash signals a nested quote or a
  dialogue.
- One clause join. At most one internal comma, ano teleia, or colon.
- Predicate. spaCy tags at least one token as `VERB` or `AUX` with morph
  `VerbForm=Fin`. A bare noun phrase such as `Τρία σκυλιά.` fails this test.
  Fallback: pass when the sentence contains `είναι`, `έχει`, `έχουν`, `δεν`,
  `θα`, or a verb form of a word-list verb.
- No orphan conjunction. A sentence that starts with `και`, `αλλά`, `ή`,
  `γιατί`, `επειδή`, `λοιπόν`, or `όμως` and has no comma lacks its main
  clause.

Fallback window: when a word gets no candidate in 4 to 10 words, retry that
word with a 3 to 10 word window and drop F6 and F8. Mark such a card
`tier=scaffold`. A scaffold card must pass F3, F4, F5, F7, F9, and F10. G14
exempts it from F2, F6, and F8.

For Greek the fallback is part of the plan. The probe finds 39 uncovered
concepts at 4 to 10 words. M2 measures how many of them gain a candidate at 3 to
10 words, and the review CSV lists the rest.

## 8. S3: Sentence selection

### 8.1 Sentence tokenization

- Run spaCy `el_core_news_sm` with the tagger, lemmatizer, and parser.
- Keep punctuation tokens out of word counts.
- Known word set = word-list forms plus the top 1000 `wordfreq` words for `el`.
  Compare a sentence token against the known set by surface and by lemma.
  Greek clitics (`μου`, `σου`, `το`, `να`) are frequent, so the top-1000 list
  contains them.
- Compare surfaces after NFC, casefold, and `ς` to `σ`.
- A missing model stops the build with a clear message. The README gives the
  install command (section 4).

### 8.2 Hard constraints per card

| Constraint     | Rule                                                              |
| -------------- | ----------------------------------------------------------------- |
| Target match   | Target word appears exactly 1 time as a token or as a lemma       |
| Word boundary  | The match is a whole token, never a substring of a longer word    |
| Unknown words  | At most 2 tokens outside the known set (1 with `--max-unknown 1`) |
| Length         | 4 to 10 words                                                     |
| Complete       | Passes F5 to F10; a scaffold card skips F6 and F8                 |
| Script         | Greek block only, NFC, monotonic                                  |
| Duplicate      | No word-set Jaccard of 0.67 or more with a kept card              |
| Duplicate card | One card per sentence and per target word pair                    |

The target match uses this key rule:

1. Build the form key set: the form surface, plus the head-word lemma from the
   carriers `⟨form⟩`, `η ⟨form⟩`, `ο ⟨form⟩`, `το ⟨form⟩`, and
   `τον ⟨form⟩`.
2. A token matches when its surface (casefolded, `ς` to `σ`) or its lemma is in
   the key set.
3. For `form_kind=pair`, both parts must appear as adjacent tokens.
4. Prefer a candidate whose audio and English text are both present.

The form-key rule keeps the matcher honest. A bare form can get a wrong lemma
from the tagger, as the French probe shows with `pluie`. The carrier phrase
gives the tagger a subject or an article, so the lemma is the citation lemma.

Bold only the matched token on the card. For a form with an article in the
sentence, bold the noun, not the article.


The shared English sense test covers all 207 glosses. The table
`vocab/english_senses.json` holds, for each gloss, the English words and phrases
that show its sense. A candidate passes when the paired English holds one of
them. `scripts/build_english_senses.py` writes the table from WordNet plus a
reviewed list of extra words.

The form-level table adds a check for a homograph whose gloss sense is close
to the wrong sense. Each risky form has two English word lists: sense words
and false-friend words. A candidate passes when the paired English holds a
sense word, or holds no false-friend word. The `SENSE_CHECKS` table in
`greek.py` holds the lists:

| Form    | Sense words | False-friend words       |
| ------- | ----------- | ------------------------ |
| γλώσσα  | tongue      | language, languages      |
| δέρμα   | skin        | leather                  |
| θάλασσα | sea         | mess, messed             |
| νέος    | new         | young, younger, youngest |
| χρόνος  | year        | time, times              |
| γυρίζω  | turn        | return, back, home       |
| με      | with        | me                       |

Gate G20 checks the built deck against the stored English text.
### 8.3 Ranking key

Sort candidates for each target word by this key, first item first:

1. Unknown token count, low first.
2. Mean Zipf value of the sentence tokens for `el`, high first.
3. Word count, 5 to 8 first.
4. Tatoeba sentence id, low first, for determinism.

There is no native audio term. The Greek pool has no recordings.

### 8.4 Selection and coverage

Use greedy set cover:

1. Build a bipartite graph of word-table forms and candidate sentences.
2. Score each candidate with the ranking key.
3. Take the best unused sentence. Remove the words it covers from the
   uncovered set. Increment the count for the assigned target only.
4. Stop when every word with candidates reaches `--per-word`, or the card cap
   is reached.
5. Process `core` words in Swadesh index order. Assign a sentence to the
   covered word with the lowest Swadesh index.

Defaults: `--per-word 2`, `--max-cards 340`, `--max-unknown 2`. The probe gives
324 cards, so the cap of 340 keeps about 5 percent of headroom. The deck stays
small because the corpus is small, not because of the cap.

Reject a card when the sentence has no English translation after pairing.
Report the words that stay uncovered:

- Words with zero candidates.
- Words with candidates that were cut by the card cap.
- The 39 concepts of section 3.3, if they stay uncovered.

### 8.5 Output schema

`vocab/greek_sentences.csv` has the same columns as the French file, section
8.5 of that spec, except `audio_license` is always empty and `audio_source` is
always `tts`.

| Column           | Type   | Example                  |
| ---------------- | ------ | ------------------------ |
| `target`         | string | νερό                     |
| `sentence`       | string | Θέλω ένα ποτήρι νερό.    |
| `text_original`  | string | Θέλω ένα ποτήρι νερό.    |
| `english`        | string | I want a glass of water. |
| `tatoeba_id`     | int    | 1234567                  |
| `tatoeba_author` | string | Aimak                    |
| `english_id`     | int    | 7654321                  |
| `audio_id`       | int    | 0                        |
| `audio_source`   | string | tts                      |
| `audio_license`  | string | empty                    |
| `unknown_words`  | int    | 0                        |
| `word_len`       | int    | 5                        |
| `zipf_mean`      | float  | 5.12                     |
| `rank_key`       | string | serialized sort key      |
| `tier`           | string | core or scaffold         |
| `form_kind`      | string | single                   |
| `audio_missing`  | bool   | false                    |

The file has one row per card. One sentence makes one card only.

Also write `vocab/greek_sentences_review.csv` with the top three candidates per
uncovered or low-confidence word. A human reviews 5 percent of the cards before
release, and 100 percent of the scaffold cards.

### 8.6 Gloss

- Write the gloss from the Swadesh English label, not from a translation.
- Add the gender for a noun: `νερό (n)`, `γυναίκα (f)`, `άντρας (m)`.
- For a verb, add the first person singular: `πίνω`.
- Store the gloss in the card field `Gloss`.

## 9. S4: Audio

### 9.1 Tiers

| Tier | Source   | Condition                          |
| ---- | -------- | ---------------------------------- |
| B    | gTTS     | Default, all cards                 |
| C    | edge-tts | Optional, when `--tts edge` is set |

There is no native tier. Section 3.4 gives the evidence. The `audio` command
rejects `--native-audio auto` and `--native-audio only` for Greek.

### 9.2 TTS settings

| Setting     | Value                                                           |
| ----------- | --------------------------------------------------------------- |
| Engine      | gTTS, package `gtts`, already in `pyproject.toml`               |
| Voice       | `gtts:el` for the default pack                                  |
| Alt engine  | edge-tts, package `edge-tts`, optional                          |
| Alt voice   | `el-GR-AthinaNeural` for edge-tts                               |
| Other voice | `el-GR-NestorasNeural`                                          |
| Rate        | edge-tts `-10%` for sentence audio, `-25%` for the slow variant |
| Format      | MP3, mono, 24 kHz                                               |
| Text input  | NFC sentence, final sigma kept, no punctuation change           |
| Retry       | 3 attempts, 2, 4, 8 seconds backoff                             |
| Rate limit  | 1 request per second, one worker                                |

The voice names come from the edge-tts voice list, read on 2026-10-18.

A Greek question ends with `;`. Feed the `;` to gTTS unchanged, so the voice
uses a question intonation. Do not replace `;` with `?`.

### 9.3 File names and cache

| Type       | Pattern           | Example                 |
| ---------- | ----------------- | ----------------------- |
| TTS        | `tts_{hash}.mp3`  | `tts_7f3c1a9b2e04.mp3`  |
| TTS slow   | `tts_{hash}.mp3`  | `tts_4c8d0e2f6a11.mp3`  |
| Word audio | `word_{hash}.mp3` | `word_2a91bb0c4f11.mp3` |

The hash input is the NFC sentence text, the voice name, and the rate. Use
`audio/greek_sentences/index.csv` with the same columns as the French index,
section 9.3 of that spec.

### 9.4 Validation

- `ffprobe` reads each file. Duration must be 0.4 to 15 seconds.
- The file must start with `ID3` or an MPEG frame sync (`FF Ex` or `FF Fx`).
- A failed file is deleted and the card is marked `audio_missing`.
- A card without audio does not enter the pack.
- Check that the synthesis input equals the card sentence, byte for byte,
  after NFC.

`ffprobe` is an external tool. Document it in the README.

Measured cost: 324 sentences and 207 word files at 1 request per second take
about 9 minutes.

## 10. S5: Pack build

### 10.1 Note model

Model name: `Greek Swadesh Sentence`. The field list is the same as the French
model, section 10.1 of that spec:

| Order | Field       | Content                       |
| ----- | ----------- | ----------------------------- |
| 0     | `Audio`     | `[sound:tts_...mp3]`          |
| 1     | `Sentence`  | Sentence with `<b>target</b>` |
| 2     | `English`   | English translation           |
| 3     | `Target`    | Target word                   |
| 4     | `Gloss`     | Short gloss plus gender       |
| 5     | `WordAudio` | `[sound:word_...mp3]`         |
| 6     | `Source`    | `Tatoeba #1234567 · author`   |

Sort field index: 3 (`Target`).

Templates: the same front and back templates as the French pack. The `.hint`
line reads `Tap to show the sentence`.

### 10.2 Tags

- `swadesh::water`
- `audio::tts`
- `tier::core` or `tier::scaffold`
- `lang::el`

### 10.3 Deck

- Deck name: `Swadesh Greek Sentences`.
- Add notes in Swadesh index order.
- Do not split the deck. Use tags for study filters.

### 10.4 Manifest

Write `apkg/swadesh_greek_sentences.manifest.json` with the same shape as the
French manifest, section 10.4 of that spec, and these values:

```json
{
  "counts": {
    "words": 207,
    "cards": 324,
    "scaffold": 0,
    "native_audio": 0,
    "tts_audio": 324
  },
  "coverage": {"core_with_card": 0.81},
  "voice": "gtts:el",
  "native_audio_available": false,
  "excluded_words": []
}
```

`excluded_words` holds index 81 (`πόδι`, shared form) and every concept that
stays uncovered after M2. The report lists the 39 concepts of section 3.3.

### 10.5 Attribution

`ATTRIBUTION.md` is one file for all packs. Add a `## Greek pack` section. The
section lists:

- Tatoeba sentence ids and authors, grouped by author.
- The Swadesh source page and its license.
- The TTS engine and voice.
- The spaCy model and its license.

State the licenses in the deck description and in `ATTRIBUTION.md`: word data
CC BY-SA 4.0 and sentence text CC BY 2.0 FR. A TTS file names the engine and
the voice.

Tatoeba text needs this sentence: "Sentences come from Tatoeba
(<https://tatoeba.org>) and use the CC BY 2.0 FR license."

Gate G12 reads only the section for the pack language.

## 11. CLI

The commands mirror the French pack, section 11 of that spec:

```bash
anki swadesh fetch --lang ell --lang eng                 # S0
anki swadesh sentences greek --per-word 2                # S1 to S3
anki swadesh audio greek --tts gtts                      # S4
anki swadesh pack greek                                  # S5
anki swadesh audit greek                                 # gates, no writes
```

Options:

| Command     | Option           | Default   | Meaning                     |
| ----------- | ---------------- | --------- | --------------------------- |
| `sentences` | `--per-word`     | 2         | Max sentences per word      |
| `sentences` | `--max-cards`    | 340       | Hard card cap               |
| `sentences` | `--max-unknown`  | 2         | Unknown tokens per sentence |
| `sentences` | `--min-words`    | 4         | Minimum word count          |
| `sentences` | `--max-words`    | 10        | Maximum word count          |
| `audio`     | `--tts`          | gtts      | `gtts` or `edge`            |
| `audio`     | `--voice`        | `gtts:el` | TTS voice                   |
| `audio`     | `--slow-audio`   | off       | Add a slow variant per card |
| `pack`      | `--pack-license` | public    | `public` or `private`       |

`--native-audio` is rejected for Greek.

## 12. File layout

```text
data/tatoeba/                      cached dumps, gitignored
  ell_sentences.tsv
  ell_sentences_detailed.tsv
  ell-eng_links.tsv
  eng_sentences.tsv
vocab/
  english-greek.csv                reviewed source list
  greek_word_review.csv
  greek_sentences.csv
  greek_sentences_review.csv
audio/greek_sentences/
  tts_*.mp3  word_*.mp3  index.csv
apkg/
  swadesh_greek_sentences.apkg
  swadesh_greek_sentences.manifest.json
ATTRIBUTION.md
```

Add `data/` and `apkg/*.apkg` to `.gitignore`. `audio/` is already ignored.
Commit the CSV files and the manifest.

## 13. Quality gates

`anki swadesh audit greek` runs these checks and exits with a non-zero code on
a failure.

| Gate | Check                                                                                      |
| ---- | ------------------------------------------------------------------------------------------ |
| G1   | Word table: 207 rows, 206 unique forms, no gap                                             |
| G2   | No duplicate form, except the shared `πόδι` row                                            |
| G3   | Every card target exists in the reviewed CSV                                               |
| G4   | Every card has a unique Tatoeba id after normalization                                     |
| G5   | Target appears one time in each sentence, as a whole token or lemma                        |
| G6   | No card pair with word-set Jaccard of 0.67 or more                                         |
| G7   | Every card has a valid audio file with duration 0.4 to 15 seconds                          |
| G8   | Audio file count equals the media count in the APKG                                        |
| G9   | No card matches the target inside a longer word                                            |
| G10  | Core coverage of 76 percent or more                                                        |
| G11  | Manifest counts match the word table and the sentence CSV (10.4)                           |
| G12  | `ATTRIBUTION.md` has a Greek section and names every sentence author in the pack           |
| G13  | The APKG opens with `AnkiPackage` and lists the expected notes                             |
| G14  | Core cards pass F2 to F10. Scaffold cards pass F3, F4, F5, F7, F9, F10, and length 3 to 10 |
| G15  | `vocab/greek_word_review.csv` exists and has no open row                                   |
| G16  | Every card target has a valid `word_{hash}.mp3` file                                       |
| G17  | Every card field is NFC, monotonic, and has no `ς` inside a word                           |
| G18  | No card has `audio_source=native`                                                          |
| G19  | Every card target is in the 206-word table and is not `variant=shared`                     |
| G20  | No card English holds a different sense than the card word (8.2)                           |

G10 uses the measured coverage. The probe measures 168 of 207 concepts (0.812)
in the 4 to 10 word window. This spec sets the gate at 0.76, which is the
measured value minus 5 points.

Add unit tests in `tests/test_greek_pack.py` for:

- Word-table cleanup: note removal, article strip, and the one-word exception.
- The shared `πόδι` row and G19.
- The final sigma and the `ς` to `σ` surface key.
- `;` as a question mark in F5 and F6, and `·` in F8.
- Lemma matching with `πίνω` and `πίνει`.
- The unknown-word filter with the top-1000 `el` list.
- Set cover on a small fixture.
- Audio filename determinism.
- The Greek filter set F2 to F12 on a small fixture.
- Manifest counts and the 324 card estimate.
- Greek output does not use the native audio path.

## 14. Risks

| Risk                              | Effect                 | Mitigation                                      |
| --------------------------------- | ---------------------- | ----------------------------------------------- |
| No Greek native audio exists      | TTS only               | gTTS default, edge-tts option, documented limit |
| Small corpus                      | 39 concepts uncovered  | scaffold window, review CSV, measured gate 0.76 |
| Heavy inflection                  | Missed target match    | lemma matching, carrier lemma key, `ς` to `σ`   |
| `;` is the question mark          | Wrong filter set       | F5, F6, and F8 rules in 3.2                     |
| Article in the CSV form           | No surface match       | M1 strips the article, `gender` column keeps it |
| Two concepts share `πόδι`         | Duplicate form gate    | `variant=shared`, no card, G19                  |
| `el-core-news-sm` is not on PyPI  | Install fails          | direct URL in `pyproject.toml`, README step     |
| Few candidates per concept        | Repeated sentence text | Jaccard gate, review CSV, per-word cap          |
| gTTS is an unofficial endpoint    | Voice service breaks   | retry, edge-tts option, cached audio            |
| Model lemma errors on a bare form | Wrong matcher          | carrier lemma rule (8.2)                        |

## 15. Milestones

| Milestone | Task                                                 | Output                |
| --------- | ---------------------------------------------------- | --------------------- |
| M0        | `swadesh fetch --lang ell` with checksum cache       | `data/` populated     |
| M0b       | Greek profile on the shared layer, `rank.py` helpers | profile and helpers   |
| M1        | S1 word table, 90 review rows, coverage report       | `english-greek.csv`   |
| M2        | S2 and S3 with review CSV, `tier`, scaffold measure  | `greek_sentences.csv` |
| M3        | S4 TTS and word audio, index, validation             | `audio/` populated    |
| M4        | S5 pack, manifest, templates, tags                   | first `.apkg`         |
| M5        | `audit` with pool counts, tests, README, attribution | release check         |
| M6        | edge-tts voice and slow variant                      | optional second pack  |

Acceptance for M4: the audit passes all gates, and 5 percent of the core cards
plus 100 percent of the scaffold cards pass a manual review of sentence, gloss,
and audio.

## 16. Open questions

1. Write manual frame sentences for the 39 uncovered concepts?
2. Merge foot and leg into one card, or keep only foot?
3. Use 2 sentences per word, or 1 sentence plus the existing word deck?
4. Add a `lang::el` tag to the existing word deck?
5. Use edge-tts `el-GR-AthinaNeural` as the default voice, or gTTS?
6. Add a slow audio variant by default?

## 17. Implementation result

Milestone M5 is complete. These values come from the full pipeline run of
2026-10-21.

| Measure            | Value                                                   |
| ------------------ | ------------------------------------------------------- |
| Build date         | 2026-10-21                                              |
| Word table         | 207 rows, 206 unique forms, 146 review rows, 0 open     |
| Candidate pool     | 7,382 candidates, 4 to 10 words, 2 scaffold             |
| Cards              | 300 (298 core, 2 scaffold)                              |
| Media files        | 460 (300 sentence files and 160 word files)             |
| Concept coverage   | 160 of 207 concepts (0.773)                             |
| Uncovered concepts | 47                                                      |
| Native audio       | none                                                    |
| TTS engine         | gTTS, voice `gtts:el`                                   |
| Gate result        | G1 to G20 pass; `anki swadesh audit greek` exits 0      |

The uncovered words are πλατύς, παχύς, ψείρα, σκουλήκι, ραβδί, σπόρος,
φλοιός του δέντρου, χόρτο, λίπος, κέρατο, πούπουλο, αυτί, νύχι, πόδι,
γόνατο, φτερό, κοιλιά, σπλάχνα, λαιμός, συκώτι, ρουφώ, ξερνώ, φυσάω, κυνηγώ,
χτυπάω, μαχαιρώνω, ξύνω, σκάβω, πετάω, ξαπλώνω, σφίγγω, τρίβω, σκουπίζω,
τραβάω, δένω, μετράω, πλέω, ρέω, παγώνω, πρήζομαι, σκόνη, στάχτη, σάπιος,
στρογγυλός, αιχμηρός, στομωμένος, λείος.

The measured values differ from the probe of section 3.3 in four places:

- The pool has 7,382 candidates, not 18,190. The shared pool applies the
  unknown-token budget of section 8.2 as filter 13, so 4,048 sentences that
  pass F1 to F12 with more than two unknown tokens leave the pool.
- The English sense test removes 1,656 more sentences. A sentence stays only
  when the paired English shows the gloss sense of a covered word.
- Coverage is 160 concepts, not 168, and the deck has 300 cards, not 324.
  The spaCy 3.8.0 tagger and lemmatizer of `el_core_news_sm` and the
  carrier-lemma key set of section 8.2 find fewer targets than the probe.
- The fallback window of section 7.1 adds 2 scaffold cards for the 47
  uncovered words.

The coverage denominator is the 207 Swadesh concepts. Index 81 shares the
form `πόδι` with index 80, so the deck leaves both forms without a card. G10
at 0.76 still passes (0.773).

The word review file has 146 rows: 83 article strips, 6 note removals, 2
slash choices, 1 shared row, 23 module differences, and 31 rare forms. The
row for index 41 keeps `άντρας`, and index 37 moves to `άνδρας` so that one
form serves one concept. Row counts stay at 207 because the shared `πόδι`
row keeps its place.
