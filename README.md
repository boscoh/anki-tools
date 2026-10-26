# Anki Tools

Read, edit, and build Anki flashcard packages (`.apkg` files) through direct
SQLite access. Build vocabulary and sentence decks for listening
comprehension.

Deck goals:

- **Useful** - common words and phrases that you meet often.
- **Concise** - short sentences that are easy to memorize.
- **Interesting** - natural examples, not textbook filler.
- **Non-redundant** - no duplicate concepts or near-identical cards.

## Features

- **Python API** - read and edit APKG files with direct SQLite access.
- **CLI tools** - one command for each common workflow.
- **Audio** - extract media files, add audio, and delete unused files.
- **Deck management** - create decks, copy cards, and move cards.
- **Sentence ranking** - rank sentences by complexity, frequency, and
  similarity (Chinese, French, Spanish, and Cantonese).
- **Vocabulary decks** - build Swadesh vocabulary decks with gTTS audio.
- **No Anki install** - direct SQLite access with minimal dependencies.

## Installation

```bash
uv sync
```

The `anki` command is then available as `uv run anki`.

Related documents:

- `docs/anki-tools-layout.md` - package layout.
- `docs/swadesh-sentence-flow.md`, `docs/vocab-deck-flow.md`,
  `docs/core-deck-flow.md`, `docs/cull-deck-flow.md` - stage order of each
  deck builder.

## Quick Start

```bash
uv run anki --help                        # list the commands
uv run anki swadesh vocab mandarin        # build a Swadesh vocabulary deck
uv run anki fix zh input.apkg             # rank, reorder, fix pinyin, style
uv run anki fix fr input.apkg             # rank and reorder a French deck
uv run anki fix yue input.apkg            # rank, reorder, fix jyutping
uv run anki fix es input.apkg --select    # cull a Spanish deck
uv run anki inspect deck.apkg             # show the package contents
uv run anki style deck.apkg               # apply the CSS and HTML templates
```

Every `fix` command has a default input path, and every command lists its
options: run `uv run anki fix <lang> --help`.

## The Fix Command

`anki fix <lang>` has up to three modes. The four commands take the same
options, apart from the language additions below.

| Language | Modes                                | Output                                          |
| -------- | ------------------------------------ | ----------------------------------------------- |
| `zh`     | rank (default), `--core`             | `_reordered.apkg`, `_core.apkg`                 |
| `fr`     | rank (default), `--select`, `--core` | `_reordered.apkg`, `_culled.apkg`, `_core.apkg` |
| `es`     | rank (default), `--select`, `--core` | `_reordered.apkg`, `_culled.apkg`, `_core.apkg` |
| `yue`    | rank (default), `--core`             | `_reordered.apkg`, `_core.apkg`                 |

Each command has a default input, so a bare `anki fix <lang>` reads it:

| Language | Default input                                        |
| -------- | ---------------------------------------------------- |
| `zh`     | `apkg/Neri's Chinese Sents Part 2.apkg`              |
| `fr`     | `apkg/6000_French_Sentences_w_Audio.apkg`            |
| `es`     | `apkg/Spanish 9000 sentences with native audio.apkg` |
| `yue`    | `apkg/LTL Cantonese Deck Level 1 - Short.apkg`       |

Run `uv run anki fix <lang> --help` to see the defaults of a command.

The shared options:

| Option            | Applies to           | Meaning                                      |
| ----------------- | -------------------- | -------------------------------------------- |
| `apkg-path`       | every mode           | input deck; each command has a default       |
| `--output`        | every mode           | output deck; default `_reordered` or `_core` |
| `--text-field`    | rank mode            | sentence field; the build infers the default |
| `--select`        | `fr`, `es`           | run the cull instead of the rank and reorder |
| `--core`          | every command        | build the core-word deck                     |
| `--cards`         | every mode           | card cap of the chosen mode                  |
| `--words`         | core mode            | tier F words after the Swadesh set           |
| `--check`         | `--select`, `--core` | run the gates on the written output          |

These options apply to one language:

| Option                   | Language   | Meaning                                     |
| ------------------------ | ---------- | ------------------------------------------- |
| `--targets`              | `fr`, `es` | cull target count, 200 to 500               |
| `--reserve`              | `fr`, `es` | add the Swadesh reserve cards               |

The frequency term and the similarity limit are values of the language module,
so no flag can change them.

Every mode writes `.rank.csv` beside the APKG. The cull and core modes also
write the CSV files and the manifest that the "French 1000-Card Cull" and
"Core-Word Decks" sections list.

### Chinese

```bash
uv run anki fix zh deck.apkg         # rank, reorder, fix pinyin, apply styling
uv run anki fix zh deck.apkg --core  # core mode, see "Core-Word Decks"
```

- Rank mode uses character-based complexity and similarity, with `wordfreq`
  frequency data and an HSK vocabulary bonus.
- Rank mode removes repeated-pattern sentences before it ranks, for example
  `走吧走吧` and `好了，好了`.
- Rank mode fixes the pinyin field and applies `templates/default/`. Both steps
  always run.

### French

```bash
uv run anki fix fr deck.apkg           # rank and reorder
uv run anki fix fr deck.apkg --select  # the 1000-card cull
uv run anki fix fr deck.apkg --core    # core mode
```

- Rank mode uses word-based complexity and similarity, with `wordfreq`
  frequency data.
- Rank mode adds a grammar score. The optional spaCy model
  `fr_core_news_sm` gives a parse-tree score:
  `uv run python -m spacy download fr_core_news_sm`.
- Select mode runs the 1000-card cull. See
  `specs/french-1000-sentence-cull.md`.

### Spanish

```bash
uv run anki fix es deck.apkg           # rank and reorder
uv run anki fix es deck.apkg --select  # the cull
uv run anki fix es deck.apkg --core    # core mode
```

- Rank mode uses word-based complexity and similarity, with `wordfreq`
  frequency data.
- Rank mode adds a grammar score for the subjunctive, the relative clauses, the
  conditionals, and the commas.
- Select mode runs the cull. See `specs/french-1000-sentence-cull.md`.

### Cantonese

```bash
uv run anki fix yue deck.apkg         # rank, reorder, fix jyutping
uv run anki fix yue deck.apkg --core  # core mode
```

- Rank mode uses the character-based complexity, similarity, and
  repeated-pattern filter of Chinese, with a frequency term that blends the Cifu
  spoken count at weight 0.7 with the written count. The written count is the
  mean of the Cifu and Apple Daily counts. See
  `specs/cantonese-frequency-list.md`.
- Rank mode converts Simplified to Traditional characters, then writes the
  jyutping.
- The first run downloads both frequency files to `~/.cache/anki-tools`. Set
  `CIFU_PATH` or `YUE_FREQUENCY_PATH` to use other files.

## Card Templates

A deck builder reads every card template and CSS file from `templates/`. Edit a
file, then run the command again.

| Command                      | Template directory                |
| ---------------------------- | --------------------------------- |
| `anki style`, `anki fix zh`  | `templates/default/`              |
| `anki swadesh pack <lang>`   | `templates/sentence_pack/<lang>/` |
| `anki swadesh vocab <lang>`  | `templates/vocab/`                |
| `anki fix fr --select`       | `templates/french_cull/`          |
| `anki fix fr --core`         | `templates/french_core/`          |
| `anki fix es --select`       | `templates/spanish_cull/`         |
| `anki fix es --core`         | `templates/spanish_core/`         |
| `anki fix yue --core`        | `templates/cantonese_core/`       |
| `anki fix zh --core`         | `templates/mandarin_core/`        |

Rules:

- `templates/shared/card.css` holds the rules for every deck.
  `templates/shared/char_links.html` holds the Han character link script.
- A builder reads the shared sheet first, then the deck `card.css`. A deck rule
  therefore overrides a shared rule. A deck `card.css` holds only its own
  classes.
- The class `char-links` is a script hook. The shared script wraps each Han
  character in a link, and the shared sheet keeps the card color on that link.
- The shared sheet sets a Simplified Chinese font stack. The Cantonese decks
  override it with a Traditional stack, so each character uses a Hong Kong
  face.
- The back template of the vocab deck holds the `{{fields}}` placeholder. The
  command writes the field blocks from the field list of the language.

## Card Previews

`uv run anki preview` writes `preview/cards.html`. Open the file in a browser
to see every card type with sample data. The command reads the real template
and CSS files. Run it again after a template edit.

- `--open-browser` opens the file after the write. `--output <path>` writes
  another file.
- `--apkg <path>` previews the real cards of a package. The command reads the
  note fields, the card templates, and the model CSS of that file. It also
  extracts the media files, so the audio controls play and the images show.
- The package preview shows 50 cards by default. `--limit 0` shows every card.
  `--deck <text>` selects a deck and `--search <text>` selects a field value.
- `--no-media` keeps the audio placeholders. The media files go to a directory
  named after the HTML file, for example `preview/deck_media/`. Use
  `--media-dir <path>` to select another directory.

```bash
uv run anki preview --apkg deck.apkg --output preview/deck.html
```

## Swadesh Vocabulary Decks

```bash
uv run anki swadesh vocab mandarin
uv run anki swadesh vocab all
```

The command builds `.apkg` files from `vocab/english-<language>.csv` with gTTS
audio. See the appendix for the languages. A build does these steps in order:

1. Generates the MP3 file for each form that has no file yet.
2. Deletes each audio file in `audio/audio_<language>/` that the current list
   does not use. `--no-prune` keeps those files.
3. Stops with an error if a form still has no audio. No deck is written.
4. Writes `apkg/swadesh_<language>.apkg` through a temporary file. The old
   deck is replaced only after a complete write.

The audio file name comes from the target-language text. If you correct a form
in the CSV file, the build makes a new MP3 file for it. Step 2 deletes the old
MP3 file.

The deck title is `Swadesh <Language> Vocabulary`. The entry count is in the
command output, not in the title.

## Swadesh Sentence Packs

```bash
uv run anki swadesh pack <language> --per-word 2 --max-cards 450  # S0 to S5
uv run anki swadesh audit <language>                              # the gates
```

`anki swadesh pack` pairs 207 Swadesh concepts with Tatoeba sentences. Each
card holds one sentence, the target word, its gloss, the translation, and two
MP3 files. See `docs/swadesh-sentence-flow.md` for the stage and guard order.

| Language  | Command     | Cards | Media | Coverage         | Native audio |
| --------- | ----------- | ----- | ----- | ---------------- | ------------ |
| Mandarin  | `mandarin`  | 371   | 566   | 191 of 207       | 0            |
| French    | `french`    | 412   | 618   | 206 of 207       | 0            |
| Greek     | `greek`     | 300   | 460   | 160 of 206       | 0            |
| Cantonese | `cantonese` | 279   | 434   | 155 of 207       | 171          |

All four packs need `ffprobe` on the path. The audio step removes a file that
fails the header or duration check. All four use gTTS for the default voice.
The optional `edge-tts` package adds the Microsoft voices.

The commands write these files. A language that has no word review file
(Mandarin) has no `_word_review.csv` row.

| Path                                                   | Content                           |
| ------------------------------------------------------ | --------------------------------- |
| `vocab/english-<language>.csv`                         | reviewed word table, 207 rows     |
| `vocab/<language>_word_review.csv`                     | word review rows                  |
| `vocab/<language>_sentences.csv`                       | one row per card                  |
| `vocab/<language>_sentences_review.csv`                | top candidates for the weak words |
| `audio/<language>_sentences/index.csv`                 | file index of the audio cache     |
| `apkg/swadesh_<language>_sentences.apkg`               | the deck                          |
| `apkg/swadesh_<language>_sentences.manifest.json`      | counts, coverage, and file hashes |
| `ATTRIBUTION.md`                                       | the `## <Language> pack` section  |

Every pack uses word data under CC BY-SA 4.0 and Tatoeba sentence text under
CC BY 2.0 FR. The audio follows its engine and voice.

### Mandarin

- gTTS is the only voice. `edge-tts` is absent.
- Audit gates G1 to G18.
- 2 scaffold cards. Each sentence has 4 to 7 Han characters.
- 16 words have no selected sentence: 虱子, 棍子, 种子, 根, 骨头, 脂肪, 羽毛,
  肠子, 乳房, 肝, 呕吐, 分开, 揉, 绑, 肿, 灰.
- On the back, each Han character is a link to the CC-Canto dictionary at
  `https://cantonese.org`. The script is `templates/shared/char_links.html`.

### Greek

- Requirements:
  1. The spaCy model `el_core_news_sm` 3.8.0. It is not on PyPI.
     `pyproject.toml` pins the release URL, so `uv sync` installs it.
  2. `ffprobe` on the path.
  3. `gtts` for the voice `gtts:el`. `--tts edge --voice
     el-GR-AthinaNeural` selects the Microsoft voice. The voice
     `el-GR-NestorasNeural` is also available.
- Audit gates G1 to G20.
- Greek has no Tatoeba recordings. `ell_sentences_with_audio.tsv.bz2` returns
  HTTP 404, so `--native-audio auto` and `--native-audio only` are rejected.
  The pack is TTS only.
- Greek `;` is the question mark. The filter and the TTS input keep it
  unchanged.
- 2 scaffold cards. 47 words have no selected sentence: πλατύς, παχύς, ψείρα,
  σκουλήκι, ραβδί, σπόρος, φλοιός του δέντρου, χόρτο, λίπος, κέρατο, πούπουλο,
  αυτί, νύχι, πόδι, γόνατο, φτερό, κοιλιά, σπλάχνα, λαιμός, συκώτι, ρουφώ, ξερνώ,
  φυσάω, κυνηγώ, χτυπάω, μαχαιρώνω, ξύνω, σκάβω, πετάω, ξαπλώνω, σφίγγω, τρίβω,
  σκουπίζω, τραβάω, δένω, μετρώ, πλέω, ρέω, παγώνω, πρήζομαι, σκόνη, στάχτη,
  σάπιος, στρογγυλός, αιχμηρός, στομωμένος, λείος.
- Index 81 (`πόδι`, leg) shares its form with index 80 (foot). The shared row
  has `variant=shared`, gets no card. Gate G19 rejects a card on a shared-only
  form.
- The spaCy model is under the MIT license.

### French

- Requirements:
  1. The spaCy model `fr_core_news_sm` 3.8.0, installed by `uv sync` from the
     URL that `pyproject.toml` pins.
  2. `ffprobe` on the path.
  3. `gtts` for the voice `gtts:fr`. `--tts edge --voice fr-FR-DeniseNeural`
     selects the Microsoft voice.
- Audit gates G1 to G21.
- The build reads `vocab/english-french.csv` and `vocab/french_word_review.csv`
  and never writes them. Run `uv run python
  scripts/build_french_word_table.py` to regenerate them from
  `Module:Swadesh/data/fr`. The module code is `fr`; the `fra` page does not
  exist.
- 0 scaffold cards. The word `intestins` (guts, index 86) has no candidate
  sentence.
- The French text uses U+202F and U+00A0 before `!`, `?`, `:`, and `;`. The
  build turns both characters into a normal space before the word count, the
  final-mark test, and the card text, and keeps the raw text in
  `text_original`.
- Tatoeba French has 9,833 recordings, but 5,754 are CC BY-NC 4.0 and 1,953
  have no license. `--native-audio auto` keeps only a recording with a public
  license, so the public pack is TTS only.

### Cantonese

- Requirements:
  1. The Cifu lexicon for the tokenizer. The first `anki swadesh pack
     cantonese` run downloads it to the cache. The build does not commit it.
  2. `ffprobe` on the path.
  3. `gtts` for the voice `gtts:yue`. `--tts edge --voice zh-HK-HiuMaanNeural`
     selects the Microsoft voice.
- Audit gates G1 to G21.
- The build reads `vocab/english-cantonese.csv` and
  `vocab/cantonese_word_review.csv` and never writes them. Run `uv run python
  scripts/build_cantonese_word_table.py` to rebuild them from
  `Module:Swadesh/data/yue`.
- The card text uses Traditional characters. The build converts a Simplified
  Tatoeba row with OpenCC and keeps the raw row in `text_original`.
- A form matches only on a token boundary. The card for 我 does not take a
  sentence that holds 我哋.
- 108 synthesized files. 51 forms have no selected sentence: 闊, 厚, 窄, 薄,
  蝨, 棍, 種子, 葉, 根, 樹皮, 皮, 脂肪, 角, 尾, 羽毛, 嘴, 腳, 腿, 膝頭, 翼, 腸,
  頸, 胸, 肝, 咬, 吐, 吹, 打獵, 切, 分開, 刺, 搾, 擦, 推, 綁, 數, 浮, 流, 結冰,
  脹, 湖, 鹽, 沙, 灰, 燒, 爛, 圓, 利, 鈍, 乾, 裡面. The form 𢯎 has a candidate
  but no card.
- The Cifu data is cached only. It is under the GPL-3.0 license.

## Core-Word Decks

The core deck of Mandarin, French, Spanish, and Cantonese comes from `anki
fix <lang> --core`. Each core word gets 2 examples. A sentence that brings many
words outside the core costs more, so the deck adds few other words.

```bash
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --core
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --core \
    --output apkg/french_core_300.apkg --check
```

The flags:

- `--words` adds content lemmas in blended frequency order.
- `--cards` sets the card cap.
- `--output` sets the path. The default is `<input>_core.apkg`.
- `--check` runs the gates K1 to K20 on the built deck.

The core list starts with the reviewed Swadesh forms of the language. The
selection is a greedy cover. Each card has a value from the target examples
that it adds, minus a cost for the words outside the core. A new rare word
costs 3 times a frequent one. A target that more than 0.25 of the candidates
hold gets one example, because a second example adds no skill.

The command writes the deck and four side files beside it:

| Suffix           | Content                                                                           |
| ---------------- | --------------------------------------------------------------------------------- |
| `.apkg`          | the kept cards in core order, with the input model and audio                      |
| `.core.csv`      | one row per input card: `kept`, `pass`, `reason`, `targets`, `assigned`           |
| `.coverage.csv`  | one row per target: `key`, `word`, `tier`, `position`, `quota`, `count`, `status` |
| `.manifest.json` | counts, options, frequency source, and hashes                                     |
| `.rank.csv`      | the ranking CSV of `reorder_deck`                                                 |

A target `status` is `covered` for two or more cards, `partial` for one card,
`absent` for no candidate, `absent_sense` when every candidate fails the sense
test, and `unchecked` when the target has no sense row.

Each note has one card. The front shows only the play button. The back shows
the play button, the sentence, the translation, and the core word. The model
gains a `Core` field for the core word. The assigned target is bold in the
sentence.

### French Core

- Defaults: 50 tier F words, 300 cards, 2 examples per word, and a tail of
  120 cards.
- A card must pass the cull filter F1 to F9 and the guards C1 to C5: a
  predicate, a main clause, no name, no loan word, and an audio file.
- The sentence window is 5 to 9 words, set by the `min_units` and `max_units`
  fields of `CoreRequest`.
- See `specs/core-word-small-deck.md` sections 4 and 5.

Measured result on 2026-10-21: 300 cards for 209 targets (159 Swadesh forms
and 50 frequency lemmas). 183 targets have 2 cards, 21 have one, and 5 are
`absent_sense`. The core cards hold 247 support words, of which 18 are rare,
and the tail adds 346 new words. Gates K1 to K19 pass.

### Cantonese Core

- Input: the LTL Cantonese deck. The deck is small, so two examples per target
  are a best effort: 70 of the 212 word-table rows have two or more usable
  cards. The selection takes the second card where the pool holds one, and
  keeps one card where it does not.
- Gate K3 reports the two-card share and does not gate, because the share
  measures the input deck.
- The 50 tier F targets need the sense rows of
  `vocab/cantonese_core_senses.tsv`, so `--words` defaults to 0.

Measured result on 2026-10-24: 126 cards for 130 targets. 60 targets have two
cards, 43 have one, and 27 are `absent_sense`. All 20 gates pass.

### Mandarin Core

- Input: the Neri Part 2 deck. The tier S form is the reviewed word-table
  form. A form hits a card when it occurs as a character span and the paired
  English holds the Swadesh sense. Thus 水 hits inside 喝水 and 来 does not hit
  inside 出来.
- The first run builds `data/mandarin_frequency.tsv` and runs gates Z1 to Z3.
- Defaults: 50 tier F words, 300 cards, and 2 examples per word.
- The sense rows are `vocab/mandarin_core_senses.tsv`. Rule C4 reads
  `vocab/mandarin_loanwords.txt`.
- The deck keeps the input model, adds the `Core` and `CoreGloss` fields, and
  bolds the assigned target.

Measured result on 2026-10-24: 147 cards for 140 targets (90 Swadesh forms and
50 tier F words). 96 of 117 coverable targets hold two cards, 106 cards hold
the bold target, and all 20 gates pass. The input deck is small, so 23 targets
stay `absent_sense`.

### Spanish Core

- Input: the Spanish 9000 deck, with the `SpanishSentence` and
  `EnglishSentence` fields.
- Defaults: 50 tier F words, 300 cards, 2 examples per word, and a tail of
  0 cards.
- The sentence window is 4 to 10 words.
- The sense rows are `vocab/spanish_core_senses.tsv`. Rule C4 reads
  `vocab/spanish_loanwords.txt`.

Measured result on 2026-10-24: 265 cards for 226 targets. 198 targets have two
cards, 220 have one, and 6 are `absent_sense`. The cards hold 177 support
words, of which 11 are rare. 201 cards hold the bold target.

The four core decks keep the input audio, under the license of the source
deck. Mandarin and Spanish also use the `wordfreq` list under Apache License
2.0.

## French 1000-Card Cull

`anki fix fr --select` culls the 5,928-card audio deck to 1,000 cards with a
wide and diverse vocabulary. The selection is a greedy cover of the 300 most
frequent French content lemmas, weighted to the spoken frequency list. No
non-exempt word appears in more than 4 cards.

```bash
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --select \
    --output apkg/french_1000_sentences.apkg

# Run gates G1 to G15 on the written output
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --select \
    --output apkg/french_1000_sentences.apkg --check
```

The defaults build the measured deck. The tuning values, such as the spoken
weight and the similarity limit, are fields of `CullRequest`.
`specs/french-1000-sentence-cull.md` records them.

| Path                                       | Content                                                     |
| ------------------------------------------ | ----------------------------------------------------------- |
| `apkg/french_1000_sentences.apkg`          | the culled deck with the input model, fields, and audio     |
| `apkg/french_1000_sentences.cull.csv`      | one row per input card: `kept`, `reason`, `targets`, `gain` |
| `apkg/french_1000_sentences.coverage.csv`  | one row per target: `word`, `rank`, `count`, `status`       |
| `apkg/french_1000_sentences.manifest.json` | counts, parameters, frequency source, and hashes            |

A target `status` is `covered` for two or more cards, `partial` for one card,
and `absent` for no card.

The cull adds a Swadesh reserve from `apkg/swadesh_french_sentences.apkg`: one
card per Swadesh concept, with its audio. The frequency pass fills the rest of
the 1,000 cards. `--no-reserve` runs the frequency cull alone. The reserve
path is a `CullRequest` field.

The cull keeps the audio files and the note model of the input deck. It does
not make new TTS files. The output deck holds one card for each note: the
comprehension card with the audio on the front. The input deck has a second
card direction, so the command removes that template and its cards. The deck
has 1,000 cards for 1,000 notes.

Measured result on 2026-10-21: 1,000 cards, 194 reserve cards, 806 frequency
cards, 2,002 distinct content lemmas, 299 of 300 frequency targets covered,
287 targets with 2 cards, and 206 of 207 Swadesh concepts covered. `intestins`
is the only Swadesh gap. The most-used non-exempt frequency lemma is in 4
cards, the cap. The reserve has a separate maximum of 34. The `spoken_mass` is
0.540, and the output APKG is about 29 MB.

Licenses: Tatoeba sentence text CC BY 2.0 FR, the `wordfreq` list under Apache
License 2.0, and the input audio under the license of the source deck.

## Python API

```python
from anki_tools import AnkiPackage

# Read and extract audio
with AnkiPackage('deck.apkg') as reader:
    cards = reader.get_cards()
    stats = reader.get_audio_statistics()
    reader.extract_audio_files('./audio_output')

    # Add an audio file to the package
    filename = reader.add_media_file('/path/to/audio.mp3')
    reader.add_audio_to_card(cards[0]['nid'], 0, filename)

    # Edit cards
    reader.update_note_field(cards[0]['nid'], 0, "Updated")
    reader.move_card(cards[0]['id'], new_deck_id)

    # Delete cards and clean up unused audio
    result = reader.delete_card(cards[0]['id'], cleanup_audio=True)
    print(f"Deleted {result['cards_deleted']} cards, "
          f"{len(result['audio_files_deleted'])} audio files")

    reader.save('updated.apkg')
```

### `AnkiPackage` Methods

Reading: `get_decks()`, `get_models()`, `get_notes()`, `get_cards()`,
`parse_card()`.

Audio: `get_media_mapping()`, `extract_audio_files()`,
`get_audio_statistics()`, `add_media_file()`.

Editing: `update_note_field()`, `move_card()`, `delete_card()`,
`delete_cards()`, `save()`.

### Utility Functions

```python
from anki_tools import (
    fix_pinyin,           # Fix pinyin formatting
    extract_sentences,    # Extract sentences from a deck
    rank_sentences_zh,    # Rank Chinese sentences
    write_ranking_csv,    # Write a ranking to a CSV file
    complexity_score_zh,  # Calculate a complexity score
    frequency_score_zh,   # Calculate a frequency score
    char_similarity_zh,   # Calculate a character similarity
    reorder_deck,         # Reorder a deck by its ranking
    load_ranking,         # Load a ranking from a CSV file
)
```

## Testing

```bash
uv run pytest tests/ -v                                    # all tests
uv run pytest tests/test_apkg_package.py -v                # one file
uv run pytest tests/ --cov=anki_tools --cov-report=term-missing
```

The suite holds 632 tests. They cover the package API, the deck gates, the
language pack rules, and the ranking. A test that needs a large input file or
a cache skips when the file is absent. The tests build their own small APKG
files, so the repository holds no deck binary.

## Package Format

An APKG file is a ZIP archive with three parts:

- `collection.anki2` - the SQLite database with notes, cards, decks, and
  models.
- `media` - a JSON map from numeric IDs to filenames:
  `{"0": "我.mp3", "1": "你.mp3"}`.
- Numbered media files `0`, `1`, `2`, and so on.

The library extracts the ZIP, queries SQLite, parses the JSON metadata, and
repackages the result.

Key tables:

- `notes` - `id`, `mid`, `flds`, `tags`. The fields in `flds` are separated by
  `\x1f`.
- `cards` - `id`, `nid`, `did`, `ord`.
- `col` - collection metadata, with the decks and models as JSON.

A note field refers to audio with `[sound:filename.mp3]`.

### Text Import Format (`.anki.txt`)

```text
#separator:Pipe
#html:true
#deck:Swadesh Mandarin Vocabulary
#notetype:Basic
I | <span style='font-size: 3rem'>我</span> <br> wǒ [sound:我.mp3]
```

This format holds one card per line, with the fields separated by `|` or tab.

## Alternative Methods

| Method            | Time   | Dependencies | Read | Write | Best for                           |
| ----------------- | ------ | ------------ | ---- | ----- | ---------------------------------- |
| **Direct SQLite** | 0.073s | None         | Yes  | Yes   | Reading, editing, audio extraction |
| genanki           | N/A    | 1            | No   | Yes   | Creating new decks                 |

This library uses direct SQLite access. It is fast and needs no dependency.
`genanki` creates decks from scratch. It is already installed: `uv add
genanki`.

## References

- [Anki Package Format](https://docs.ankiweb.net/exporting.html)
- [AnkiDroid Database Structure](https://github.com/ankidroid/Anki-Android/wiki/Database-Structure)
- [genanki Documentation](https://github.com/kerrickstaley/genanki)
- [Neri Frequency List](https://frequencylists.blogspot.com/2018/02/welcome.html)
- [Apple Daily Cantonese Frequency](https://github.com/chaaklau/appledaily-frequency)
- [Tatoeba Project](https://tatoeba.org/en/)
- [Xefjord's Complete Languages](https://xefjord.wixsite.com/xefscompletelangs)

## Appendix: gTTS Languages

These languages are available for the Swadesh vocabulary audio. The `uv run
anki swadesh vocab` error message lists the languages with a deck.

| Code  | Language              | Code   | Language              |
| ----- | --------------------- | ------ | --------------------- |
| af    | Afrikaans             | mr     | Marathi               |
| am    | Amharic               | ms     | Malay                 |
| ar    | Arabic                | my     | Myanmar (Burmese)     |
| bg    | Bulgarian             | ne     | Nepali                |
| bn    | Bengali               | nl     | Dutch                 |
| bs    | Bosnian               | no     | Norwegian             |
| ca    | Catalan               | pa     | Punjabi (Gurmukhi)    |
| cs    | Czech                 | pl     | Polish                |
| cy    | Welsh                 | pt     | Portuguese (Brazil)   |
| da    | Danish                | pt-PT  | Portuguese (Portugal) |
| de    | German                | ro     | Romanian              |
| el    | Greek                 | ru     | Russian               |
| en    | English               | si     | Sinhala               |
| es    | Spanish               | sk     | Slovak                |
| et    | Estonian              | sq     | Albanian              |
| eu    | Basque                | sr     | Serbian               |
| fi    | Finnish               | su     | Sundanese             |
| fr    | French                | sv     | Swedish               |
| fr-CA | French (Canada)       | sw     | Swahili               |
| gl    | Galician              | ta     | Tamil                 |
| gu    | Gujarati              | te     | Telugu                |
| ha    | Hausa                 | th     | Thai                  |
| hi    | Hindi                 | tl     | Filipino              |
| hr    | Croatian              | tr     | Turkish               |
| hu    | Hungarian             | uk     | Ukrainian             |
| id    | Indonesian            | ur     | Urdu                  |
| is    | Icelandic             | vi     | Vietnamese            |
| it    | Italian               | yue    | Cantonese             |
| iw    | Hebrew                | zh-CN  | Chinese (Simplified)  |
| ja    | Japanese              | zh-TW  | Chinese (Taiwan)      |
| jw    | Javanese              | zh     | Chinese (Mandarin)    |
| km    | Khmer                 |        |                       |
| kn    | Kannada               |        |                       |
| ko    | Korean                |        |                       |
| la    | Latin                 |        |                       |
| lt    | Lithuanian            |        |                       |
| lv    | Latvian               |        |                       |
| ml    | Malayalam             |        |                       |
