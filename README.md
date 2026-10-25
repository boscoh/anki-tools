# Anki Tools

Read, edit, and manipulate Anki flashcard packages (.apkg files) using direct SQLite access. Build high-quality vocabulary and sentence decks optimized for **listening comprehension**.

**Goals:**
- **Useful** - common words and phrases you'll actually encounter
- **Concise** - short sentences that are easy to memorize
- **Interesting** - natural examples, not textbook filler
- **Non-redundant** - no duplicate concepts or near-identical cards

## Features

- **Python API**: Read and edit APKG files with direct SQLite access
- **CLI Tools**: Command-line interface for common workflows
- **Audio Extraction**: Extract audio files with proper filenames
- **Deck Management**: Create decks, copy cards, move cards between decks
- **Sentence Ranking**: Rank and filter sentences by complexity, frequency, and similarity (Chinese, French, and Cantonese)
- **Cantonese Sentence Selection**: Filter fragments and keep the fewest sentences that show the most frequent spoken words
- **Vocabulary Generation**: Build Swadesh vocabulary decks with audio (multiple languages)
- **Card Deletion**: Delete cards with automatic cleanup of unused audio files
- **Minimal Dependencies**: Direct SQLite access, no Anki installation required

## Installation

```bash
uv sync
```

After installation, the `anki` command is available via `uv run anki`.

See `docs/anki-tools-layout.md` for the package layout.

See `docs/anki-input-sources.md` for the local input file and the upstream
source of each `anki fix` command.

## Quick Start

### CLI Commands

```bash
# List available commands
uv run anki --help

# Build Swadesh vocabulary decks
uv run anki swadesh list-languages
uv run anki swadesh build mandarin

# Fix Chinese deck (rank, reorder, fix pinyin, apply styling)
uv run anki fix zh input.apkg

# Fix French deck (rank and reorder, or select with --select)
uv run anki fix fr input.apkg

# Fix Cantonese deck (convert to traditional, fix jyutping, select, rank, reorder)
uv run anki fix yue input.apkg

# Fix Spanish deck (rank and reorder, optionally reduce to 3000 sentences)
uv run anki fix es input.apkg --max-sentences 3000

# Inspect APKG files
uv run anki inspect deck.apkg

# Apply CSS/HTML templates
uv run anki style deck.apkg
```

### Card Templates

The deck builders read every card template and CSS file from `templates/`.
Edit a file, then run the command again to change the deck.

- Shared layer: `templates/shared/card.css` holds the rules that every deck
  gets, and `templates/shared/char_links.html` holds the Han character link
  script.
- Default deck style: `templates/default/` with `card.css`, `front.html`, and
  `back.html`. The `anki style` command and `anki fix zh` read it.
- Swadesh sentence packs: `templates/sentence_pack/<language>/` with
  `front.html`, `back.html`, and `card.css`.
- `anki swadesh build <language>`: `templates/vocab/front.html`,
  `templates/vocab/back.html`, `templates/vocab/card.css`, and an optional
  `templates/vocab/<language>/card.css`.
- `anki fix fr --select`: `templates/french_cull/front.html` and `back.html`.
- `anki fix fr --core`: `templates/french_core/front.html` and `back.html`.
- `anki fix yue --core`: `templates/cantonese_core/front.html` and `back.html`.
- `anki fix zh --core`: `templates/mandarin_core/front.html` and `back.html`.

A builder reads the shared sheet first, then the deck `card.css`, so a deck
rule overrides a shared rule. A deck `card.css` holds only its own classes.

The class `char-links` is a script hook: the shared script wraps each Han
character in a link, and the shared sheet keeps the card color on that link.
The shared sheet sets a Simplified Chinese font stack. The Cantonese deck
overrides it with a Traditional stack, so each character uses a Hong Kong
face.

The back template of the vocab deck holds the `{{fields}}` placeholder for
the field blocks. The command writes these blocks from the field list of the
language.

### Card Previews

`uv run anki preview` writes `preview/cards.html`. Open the file in a browser
to see every card type with sample data. The command reads the real template
and CSS files. Run it again after a template edit.

`uv run anki preview --open-browser` writes the file and opens it. Use
`--output <path>` to write another file.

Use `--apkg <path>` to preview the real cards of a package. The command
reads the note fields, the card templates, and the model CSS of that file.
It also extracts the media files that the cards use, so the audio controls
play and the images show.

```bash
uv run anki preview --apkg deck.apkg --output preview/deck.html
```

The package preview shows 50 cards by default. Use `--limit 0` to show every
card, `--deck <text>` to select a deck, and `--search <text>` to select a
field value. Use `--no-media` to keep the audio placeholders. The media
files go to a directory named after the HTML file, for example
`preview/deck_media/`. Use `--media-dir <path>` to select another directory.

### Python API

```python
from anki_tools import AnkiPackage

# Read and extract audio
with AnkiPackage('deck.apkg') as reader:
    cards = reader.get_cards()
    stats = reader.get_audio_statistics()
    reader.extract_audio_files('./audio_output')
    
    # Add audio file to APKG
    filename = reader.add_media_file('/path/to/audio.mp3')
    reader.add_audio_to_card(cards[0]['nid'], 0, filename)
    
    # Edit cards
    reader.update_note_field(cards[0]['nid'], 0, "Updated")
    reader.move_card(cards[0]['id'], new_deck_id)
    
    # Delete cards and cleanup unused audio
    result = reader.delete_card(cards[0]['id'], cleanup_audio=True)
    print(f"Deleted {result['cards_deleted']} cards, {len(result['audio_files_deleted'])} audio files")
    
    reader.save('updated.apkg')
```

### Fix Command

The `fix` command optimizes decks by analyzing sentences by complexity, frequency, and similarity, then reorders optimally:

**Chinese (zh):**
```bash
uv run anki fix zh deck.apkg        # Rank, reorder, fix pinyin, apply styling
uv run anki fix zh deck.apkg --core # Core mode: a small core-word deck
```
- Character-based complexity and similarity
- Uses wordfreq for frequency data with HSK vocabulary bonus
- Strips repeated-pattern sentences before ranking (e.g. '走吧走吧', '好了，好了')
- Fixes pinyin formatting and applies card styling
- Outputs `_reordered.apkg` and `.rank.csv`
- Core mode builds a small deck from the Swadesh word table. It writes
  `_core.apkg` with its CSV files and manifest. See "Mandarin Core-Word
  Deck" below.

**French (fr):**
```bash
uv run anki fix fr deck.apkg           # Rank mode: rank and reorder
uv run anki fix fr deck.apkg --select  # Select mode: the 1000-card cull
uv run anki fix fr deck.apkg --core    # Core mode: a small core-word deck
```
- Rank mode ranks by word-based complexity and similarity and reorders the
  deck. It writes `_reordered.apkg` and `.rank.csv`.
- Select mode runs the 1000-card cull. It writes `_culled.apkg` with its CSV
  files and manifest. See "French 1000-Card Cull" below.
- Core mode builds a small deck from the core word list. It writes
  `_core.apkg` with its CSV files and manifest. See "French Core-Word Deck"
  below.
- Uses wordfreq for frequency data
- Optional spaCy grammar scoring: `uv run python -m spacy download fr_core_news_sm`

**Cantonese (yue):**
```bash
uv run anki fix yue deck.apkg        # Convert, fix jyutping, select, rank, reorder
uv run anki fix yue deck.apkg --core # Core mode: a small core-word deck
```
- Converts simplified to traditional Chinese
- Generates accurate jyutping romanization
- The frequency term blends spoken and written counts. The spoken term is the
  Cifu `SpokenAdult` count; the written term is the mean of the Cifu and Apple
  Daily counts. The default spoken weight is 0.7.
  - `--freq written` uses the Apple Daily list alone. This is the old ranking.
  - The list is built on demand at the first `anki fix yue` or
    `anki swadesh core cantonese`. The first run downloads both files to
    `~/.cache/anki-tools`. Set `CIFU_PATH` or `YUE_FREQUENCY_PATH` to use
    other files.
  - A variant map in `vocab/cantonese_variants.tsv` and the overrides in
    `vocab/cantonese_frequency_overrides.tsv` fix common spoken forms and
    greetings.
  - `--freq-report` writes `deck.freq_change.csv` for the sentences that the
    blend moves by more than 100 places.
- The filter removes fragments and hard sentences. A sentence passes when it
  has 3 to 12 Han characters, no Latin letter or digit, no repeated pattern,
  one final mark at the end, no quotation mark, at most one internal mark, at
  most 2 words outside the top 2000, and no foreign name from
  `vocab/cantonese_names.tsv` (add a row for a new name). Latin names and
  words fail the Latin-letter rule.
- The selection keeps the fewest sentences that cover the 1500 most frequent
  words of the deck. Each target word gets 2 cards. One word outside the top
  100 goes in at most 8 kept sentences. The cards go from easy to hard.
  - `--cards` caps the card count of the chosen mode. `--max-uses` sets the
    most kept sentences for one word outside the top 100. `--freq` and
    `--freq-report` set the frequency term and write its report. `--output`
    names the output file, `--model-id` filters by model, and
    `--keep-filtered` keeps the similar cards of rank mode.
  - `--no-select` gives the old output: rank and reorder only.
  - The other selection values are defaults in `SelectRequest`, in
    `anki_tools/languages/cantonese/select_pipeline.py`. `--order` is no
    longer a flag: the selection always uses the coverage order.
  - `--core` builds the core deck instead. `--words`, `--examples`, and
    `--tail` set its values, and `--check` runs its gates on the output.
- Outputs `_reordered.apkg`, `.rank.csv` (adds `hardest_rank` and `kept`),
  `.select.csv` (one row per input sentence with the reason), and
  `.coverage.csv` (one row per target word)
- Core mode builds a small deck from the Swadesh word table. It writes
  `_core.apkg` with its CSV files and manifest. See "Cantonese Core-Word
  Deck" below.

**Spanish (es):**
```bash
uv run anki fix es deck.apkg  # Rank and reorder
uv run anki fix es deck.apkg --max-sentences 3000  # Reduce to 3000 with stratified sampling
```
- Word-based complexity and similarity (like French)
- Uses wordfreq for frequency data
- Word-based deduplication (Jaccard similarity threshold 0.5)
- Optional stratified reduction by data-driven complexity percentile bins:
  - Bin 1 (P0–P40):  35% simple sentences
  - Bin 2 (P40–P70): 35% elementary sentences
  - Bin 3 (P70–P90): 20% intermediate sentences
  - Bin 4 (P90–P100): 10% advanced sentences
- Outputs `_reordered.apkg` and `.rank.csv`

### Swadesh Vocabulary Decks

Build vocabulary flashcards with audio:

```bash
uv run anki swadesh list-languages
uv run anki swadesh build mandarin
uv run anki swadesh build all
```

Builds `.apkg` files from `vocab/english-<language>.csv` with gTTS audio.
See gTTS language table below for supported languages.

A build does these steps in order:

1. Generates the MP3 file for each form that has no file yet.
2. Deletes each audio file in `audio/audio_<language>/` that the current list
   does not use. Use `--no-prune` to keep those files.
3. Stops with an error if a form still has no audio. No deck is written.
4. Writes `apkg/swadesh_<language>.apkg` through a temporary file. The old
   deck is replaced only after a complete write.

The audio file name comes from the target-language text. If you correct a
form in the CSV file, the build makes a new MP3 file for it. The old MP3 file
has no use, and step 2 deletes it.

The deck title is `Swadesh <Language> Vocabulary`; the entry count is in the
command output, not in the title.

### Swadesh Mandarin Sentences

Build a listening-comprehension deck for Mandarin. The deck pairs 207 Swadesh
concepts with 371 Tatoeba sentences. Each card holds one sentence, the
tone-marked pinyin, the English translation, the target word, and two MP3
files.

```bash
uv run anki swadesh fetch mandarin                          # cache the Tatoeba data
uv run anki swadesh sentences mandarin --per-word 2 --max-cards 450
uv run anki swadesh audio mandarin --native-audio none      # make the audio
uv run anki swadesh pack mandarin                           # write the APKG
uv run anki swadesh audit mandarin                          # gates G1 to G18
```

The core deck of mandarin, french, spanish, and cantonese also has a Swadesh
entry point: `uv run anki swadesh core <language> <input.apkg>`. It reads the
same input as `anki fix <lang> --core` and takes the same `--output`,
`--cards`, `--words`, `--examples`, `--tail`, and `--check` flags.

See `docs/swadesh-sentence-flow.md` for the stage order and the guard order.

The commands write these files:

| Path                                            | Content                            |
| ----------------------------------------------- | ---------------------------------- |
| `vocab/mandarin_sentences.csv`                  | one row per card                   |
| `vocab/mandarin_sentences_review.csv`           | top candidates for the weak words  |
| `audio/mandarin_sentences/index.csv`            | file index of the audio cache      |
| `apkg/swadesh_mandarin_sentences.apkg`          | the deck                           |
| `apkg/swadesh_mandarin_sentences.manifest.json` | counts, coverage, and file hashes  |
| `ATTRIBUTION.md`                                | sentence authors and licenses      |

The audio step uses gTTS because `edge-tts` is absent. `ffprobe` must be on
the path. The audio step removes a file that fails the header or duration
check.

Measured result: 371 cards, 566 media files, and core coverage of 191 of 207
concepts (0.923). The pack has 2 scaffold cards and 0 native recordings. Each
sentence has 4 to 7 Han characters. These 16 words have no selected sentence:
虱子, 棍子, 种子, 根, 骨头, 脂肪, 羽毛, 肠子, 乳房, 肝, 呕吐, 分开, 揉, 绑,
肿, and 灰.

On the back of a card, each Han character of the sentence is a link to the
CC-Canto dictionary at `https://cantonese.org`. The script is
`templates/shared/char_links.html`.

Licenses: word data CC BY-SA 4.0, Tatoeba sentence text CC BY 2.0 FR, and
audio per recording. This pack uses synthesized audio only.



### Swadesh Greek Sentences

Build a listening-comprehension deck for Greek. The deck pairs 207 Swadesh
concepts with 300 Tatoeba sentences. Each card holds one sentence, the English
translation, the target word with a gender gloss, and two MP3 files.

```bash
uv run anki swadesh fetch greek                             # cache the Tatoeba data
uv run anki swadesh sentences greek --per-word 2 --max-cards 340
uv run anki swadesh audio greek --tts gtts                   # make the audio
uv run anki swadesh pack greek                               # write the APKG
uv run anki swadesh audit greek                              # gates G1 to G20
```

The commands write these files:

| Path                                          | Content                            |
| --------------------------------------------- | ---------------------------------- |
| `vocab/english-greek.csv`                     | reviewed word table, 207 rows      |
| `vocab/greek_word_review.csv`                 | word review, 146 rows              |
| `vocab/greek_sentences.csv`                   | one row per card                   |
| `vocab/greek_sentences_review.csv`            | top candidates for the weak words  |
| `audio/greek_sentences/index.csv`             | file index of the audio cache      |
| `apkg/swadesh_greek_sentences.apkg`           | the deck                           |
| `apkg/swadesh_greek_sentences.manifest.json`  | counts, coverage, and file hashes  |
| `ATTRIBUTION.md`                              | the `## Greek pack` section        |

Greek needs three external things:

1. The spaCy model `el_core_news_sm` 3.8.0. It is not on PyPI. `pyproject.toml`
   pins the release URL, so `uv sync` installs it.
2. `ffprobe` on the path. The audio step removes a file that fails the header or
   duration check.
3. `gtts` for the default voice `gtts:el`. The optional `edge-tts` package adds
   the voice `el-GR-AthinaNeural` (and `el-GR-NestorasNeural`) with
   `--tts edge --voice el-GR-AthinaNeural`.

Greek has no Tatoeba recordings. `ell_sentences_with_audio.tsv.bz2` returns HTTP
404, so `--native-audio auto` and `--native-audio only` are rejected for Greek.
The pack is TTS only. Greek `;` is the question mark, so the filter and the TTS
input keep it unchanged.

Measured result: 300 cards, 460 media files, and core coverage of 160 of 206
card-capable concepts (0.777). The pack has 2 scaffold cards and 0 native
recordings. These 47 words have no selected sentence: πλατύς, παχύς, ψείρα,
σκουλήκι, ραβδί, σπόρος, φλοιός του δέντρου, χόρτο, λίπος, κέρατο, πούπουλο,
αυτί, νύχι, πόδι, γόνατο, φτερό, κοιλιά, σπλάχνα, λαιμός, συκώτι, ρουφώ,
ξερνώ, φυσάω, κυνηγώ, χτυπάω, μαχαιρώνω, ξύνω, σκάβω, πετάω, ξαπλώνω, σφίγγω,
τρίβω, σκουπίζω, τραβάω, δένω, μετρώ, πλέω, ρέω, παγώνω, πρήζομαι, σκόνη,
στάχτη, σάπιος, στρογγυλός, αιχμηρός, στομωμένος, λείος.

Index 81 (`πόδι`, leg) shares its form with index 80 (foot). The shared row has
`variant=shared`, gets no card, and gate G19 rejects a card on a shared-only
form.

Licenses: word data CC BY-SA 4.0, Tatoeba sentence text CC BY 2.0 FR, spaCy
model MIT, and synthesized audio named by its engine and voice.

### Swadesh French Sentences

Build a listening-comprehension deck for French. The deck pairs 207 Swadesh
concepts with 412 Tatoeba sentences. Each card holds one sentence with the
matched token in bold, the English translation, the target word with a gender
gloss, and two MP3 files.

```bash
uv run anki swadesh fetch french                            # cache the Tatoeba data
uv run anki swadesh sentences french --per-word 2 --max-cards 450
uv run anki swadesh audio french --native-audio none         # make the audio
uv run anki swadesh pack french                              # write the APKG
uv run anki swadesh audit french                             # gates G1 to G21
```

The commands write these files:

| Path                                            | Content                            |
| ----------------------------------------------- | ---------------------------------- |
| `vocab/english-french.csv`                      | reviewed word table, 207 rows      |
| `vocab/french_word_review.csv`                  | word review, 94 rows               |
| `vocab/french_sentences.csv`                    | one row per card                   |
| `vocab/french_sentences_review.csv`             | top candidates for the weak words  |
| `audio/french_sentences/index.csv`              | file index of the audio cache      |
| `apkg/swadesh_french_sentences.apkg`            | the deck                           |
| `apkg/swadesh_french_sentences.manifest.json`   | counts, coverage, and file hashes  |
| `ATTRIBUTION.md`                                | the `## French pack` section       |

French needs three external things:

1. The spaCy model `fr_core_news_sm` 3.8.0. It is not on PyPI. `pyproject.toml`
   pins the release URL, so `uv sync` installs it.
2. `ffprobe` on the path. The audio step removes a file that fails the header or
   duration check.
3. `gtts` for the voice `gtts:fr`. The optional `edge-tts` package adds the
   voice `fr-FR-DeniseNeural` with `--tts edge --voice fr-FR-DeniseNeural`.

The build reads `vocab/english-french.csv` and `vocab/french_word_review.csv`
and never writes them. Run `uv run python scripts/build_french_word_table.py`
to regenerate them from `Module:Swadesh/data/fr`. The module code is `fr`; the
`fra` page does not exist.

Measured result on 2026-10-21: 412 cards, 618 media files, and core coverage
of 206 of 207 concepts (0.995). The pack has 0 scaffold cards and 0 native
recordings. The word `intestins` (guts, index 86) has no candidate sentence.

The French text uses U+202F and U+00A0 before `!`, `?`, `:`, and `;`. The build
turns both characters into a normal space before the word count, the final-mark
test, and the card text, and keeps the raw text in `text_original`.

Tatoeba French has 9,833 recordings, but 5,754 are CC BY-NC 4.0 and 1,953 have
no license. `--native-audio auto` keeps only a recording with a public license,
so the public pack is TTS only.

Licenses: word data CC BY-SA 4.0, Tatoeba sentence text CC BY 2.0 FR, spaCy
model MIT, and synthesized audio named by its engine and voice.

### French 1000-Card Cull

Cull the 5,928-card audio deck to 1,000 cards with a wide and diverse
vocabulary. The selection is a greedy cover of the 300 most frequent French
content lemmas, weighted to the spoken frequency list. No non-exempt word
appears in more than 4 cards.

`anki fix fr` has two modes. Rank mode is the default: it ranks the sentences
and writes `<input>_reordered.apkg`. Select mode adds `--select` and runs the
cull: it writes `<input>_culled.apkg`.

```bash
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --select \
    --output apkg/french_1000_sentences.apkg

# Run gates G1 to G15 on the written output
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --select \
    --output apkg/french_1000_sentences.apkg --check
```

The defaults build the measured 1000-card deck. The tuning values, such as
the spoken weight and the similarity limit, are fields of `CullRequest`, and
`specs/french-1000-sentence-cull.md` records them.

The command writes these files:

| Path                                       | Content                                                     |
| ------------------------------------------ | ----------------------------------------------------------- |
| `apkg/french_1000_sentences.apkg`          | the culled deck with the input model, fields, and audio     |
| `apkg/french_1000_sentences.cull.csv`      | one row per input card: `kept`, `reason`, `targets`, `gain` |
| `apkg/french_1000_sentences.coverage.csv`  | one row per target: `word`, `rank`, `count`, `status`       |
| `apkg/french_1000_sentences.manifest.json` | counts, parameters, frequency source, and hashes            |

The `status` of a target is `covered` for two or more cards, `partial` for
one card, and `absent` for no card.

The cull adds a Swadesh reserve from
`apkg/swadesh_french_sentences.apkg`: one card per Swadesh concept, with its
audio. The frequency pass fills the rest of the 1,000 cards. Use
`--no-reserve` for the frequency cull alone. The reserve pack path is a
`CullRequest` field.

Measured result on 2026-10-21: 1,000 cards, 194 reserve cards, 806 frequency
cards, 2,002 distinct content lemmas, 299 of 300 frequency targets covered,
287 targets with 2 cards, and 206 of 207 Swadesh concepts covered. `intestins`
is the only Swadesh gap. The most-used non-exempt frequency lemma is in 4
cards, the cap; the reserve has a separate maximum of 34. The `spoken_mass` is
0.540, and the output APKG is about 29 MB.

The cull keeps the audio files and the note model of the input deck. It does
not make new TTS files. The input file is
`apkg/6000_French_Sentences_w_Audio.apkg`.

The output deck holds one card for each note: the comprehension card with the
audio on the front. The input deck has a second card direction, so the
command removes that template and its cards. The deck has 1,000 cards for
1,000 notes.

Licenses: Tatoeba sentence text CC BY 2.0 FR, `wordfreq` list Apache License
2.0, and input audio under the license of the source deck.

### French Core-Word Deck

Build a small deck from a fixed core word list. Each core word gets 2
examples. A sentence that brings many words outside the core costs more, so
the deck adds few other words.

```bash
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --core
# Run gates K1 to K19 on the built deck
uv run anki fix fr "apkg/6000_French_Sentences_w_Audio.apkg" --core \
    --output apkg/french_core_300.apkg --check
```

The defaults are the measured preset: 50 tier F words, 300 cards, 2 examples
per word, and a tail of 120 cards.

The core list starts with the reviewed Swadesh forms of the language. It then
adds `--words` content lemmas in blended frequency order. A target that
more than 0.25 of the candidates hold gets one example, because a second
example adds no skill.

A card must pass the cull filter F1 to F9 and the guards C1 to C5: a
predicate, a main clause, no name, no loan word, and an audio file. The
window is 5 to 9 words, set by the `min_units` and `max_units` fields of
`CoreRequest`.

The selection is a greedy cover. Each card has a value from the target
examples that it adds, minus a cost for the words outside the core. A new
rare word costs 3 times a frequent one. See
`specs/core-word-small-deck.md` sections 4 and 5.

The default output is `<input>_core.apkg`. `--output` sets another path. The
command writes the deck and four side files beside it:

| Suffix           | Content                                                                           |
| ---------------- | --------------------------------------------------------------------------------- |
| `.apkg`          | the kept cards in core order, with the input model and audio                      |
| `.core.csv`      | one row per input card: `kept`, `pass`, `reason`, `targets`, `assigned`           |
| `.coverage.csv`  | one row per target: `key`, `word`, `tier`, `position`, `quota`, `count`, `status` |
| `.manifest.json` | counts, options, frequency source, and hashes                                     |
| `.rank.csv`      | the ranking CSV of `reorder_deck`                                                 |

A target `status` is `covered` for two or more cards, `partial` for one
card, `absent` for no candidate, `absent_sense` when every candidate fails
the sense test, and `unchecked` when the target has no sense row.

Each note has one card. The front shows only the play button. The back shows
the play button, the French sentence, the English translation, and the core
word. The model gains a `Core` field for the core word. The assigned target
is bold in the sentence.

Measured result on 2026-10-21 with the command above: 300 cards for 209
targets (159 Swadesh forms and 50 frequency lemmas). 183 targets have 2
cards, 21 have one, and 5 are `absent_sense`. The core cards hold 247
support words, of which 18 are rare, and the tail adds 346 new words. Gates
K1 to K19 pass.

Licenses: Tatoeba sentence text CC BY 2.0 FR, `wordfreq` list Apache License
2.0, and input audio under the license of the source deck.

### Cantonese Core-Word Deck

Build a core deck from the LTL Cantonese deck. The LTL deck is small, so two
examples per target are a best effort: 70 of the 212 word-table rows have two
or more usable cards. The selection takes the second card where the pool holds
one, and keeps one card where it does not.

```bash
uv run anki fix yue "apkg/LTL Cantonese Deck Level 1 - Short.apkg" --core
# Run gates K1 to K20 on the built deck
uv run anki fix yue "apkg/LTL Cantonese Deck Level 1 - Short.apkg" --core \
    --output "apkg/LTL Cantonese Deck Level 1 - Short_core.apkg" --check
```

Gate K3 reports the two-card share and does not gate, because the share
measures the input deck. The 50 tier F targets need the sense rows of
`vocab/cantonese_core_senses.tsv`, so `--words` defaults to 0.

Measured result on 2026-10-24: 126 cards for 130 targets. 60 targets have two
cards, 43 have one, and 27 are `absent_sense`. All 20 gates pass.

Licenses: Cifu frequency data (GPL-3.0) cached only, and input audio under the
license of the source deck.

### Mandarin Core-Word Deck

Build a core deck from the Neri Part 2 deck. The tier S form is the reviewed
word-table form. A form hits a card when it occurs as a character span and the
paired English holds the Swadesh sense, so 水 hits inside 喝水 and 来 does not
hit inside 出来.

```bash
uv run anki fix zh "apkg/Neri's Chinese Sents Part 2.apkg" --core
# Run gates K1 to K20 on the built deck
uv run anki fix zh "apkg/Neri's Chinese Sents Part 2.apkg" --core --check
```

The first run builds `data/mandarin_frequency.tsv` and runs gates Z1 to Z3.

The defaults are 50 tier F words, 300 cards, and 2 examples per word. The
sense rows are `vocab/mandarin_core_senses.tsv`, and rule C4 reads
`vocab/mandarin_loanwords.txt`. The deck keeps the input model, adds the `Core`
and `CoreGloss` fields, and bolds the assigned target.

Measured result on 2026-10-24: 147 cards for 140 targets (90 Swadesh forms and
50 tier F words). 96 of 117 coverable targets hold two cards, 106 cards hold
the bold target, and all 20 gates pass. The input deck is small, so 23 targets
stay `absent_sense`.

Licenses: wordfreq list Apache License 2.0, and input audio under the license
of the source deck.

### Swadesh Cantonese Sentences

Build a listening-comprehension deck for Cantonese. The deck pairs 207 Swadesh
concepts with 279 Tatoeba sentences. Each card holds one sentence with the
matched token in bold, the jyutping reading, the English translation, the
target word with its reading, and two MP3 files.

```bash
uv run anki swadesh fetch cantonese                         # cache the Tatoeba data
uv run anki swadesh sentences cantonese --per-word 2 --max-cards 450
uv run anki swadesh audio cantonese --native-audio auto      # make the audio
uv run anki swadesh pack cantonese                           # write the APKG
uv run anki swadesh audit cantonese                          # gates G1 to G21
```

The commands write these files:

| Path                                              | Content                            |
| ------------------------------------------------- | ---------------------------------- |
| `vocab/english-cantonese.csv`                     | reviewed word table, 207 rows      |
| `vocab/cantonese_word_review.csv`                 | word review, 33 rows               |
| `vocab/cantonese_sentences.csv`                   | one row per card                   |
| `vocab/cantonese_sentences_review.csv`            | top candidates for the weak words  |
| `audio/cantonese_sentences/index.csv`             | file index of the audio cache      |
| `apkg/swadesh_cantonese_sentences.apkg`           | the deck                           |
| `apkg/swadesh_cantonese_sentences.manifest.json`  | counts, coverage, and file hashes  |
| `ATTRIBUTION.md`                                  | the `## Cantonese pack` section    |

Cantonese needs three external things:

1. The Cifu lexicon for the tokenizer. The first `anki swadesh sentences
   cantonese` downloads it to the cache. The build does not commit it.
2. `ffprobe` on the path. The audio step removes a file that fails the header or
   the duration check.
3. `gtts` for the voice `gtts:yue`. The optional `edge-tts` package adds the
   voice `zh-HK-HiuMaanNeural` with `--tts edge --voice zh-HK-HiuMaanNeural`.

The build reads `vocab/english-cantonese.csv` and
`vocab/cantonese_word_review.csv` and never writes them. Run
`uv run python scripts/build_cantonese_word_table.py` to rebuild them from
`Module:Swadesh/data/yue`.

The card text uses Traditional characters. The build converts a simplified
Tatoeba row with OpenCC and keeps the raw row in `text_original`. A form
matches only on a token boundary, so the card for 我 does not take a sentence
that holds 我哋.

Measured result on 2026-10-21: 279 cards, 434 media files, 155 of 207 forms
(0.749), 171 native recordings, and 108 synthesized files. These 51 forms have
no selected sentence: 闊, 厚, 窄, 薄, 蝨, 棍, 種子, 葉, 根, 樹皮, 皮, 脂肪, 角,
尾, 羽毛, 嘴, 腳, 腿, 膝頭, 翼, 腸, 頸, 胸, 肝, 咬, 吐, 吹, 打獵, 切, 分開,
刺, 搾, 擦, 推, 綁, 數, 浮, 流, 結冰, 脹, 湖, 鹽, 沙, 灰, 燒, 爛, 圓, 利,
鈍, 乾, 裡面. The form 𢯎 has a candidate but no card.

Licenses: word data CC BY-SA 4.0, Tatoeba sentence text CC BY 2.0 FR, Cifu
data cached only, and audio named by its source, engine, and voice.

## API Reference

### AnkiPackage Class

Import: `from anki_tools import AnkiPackage`

**Reading Methods:**
- `get_decks()` - Get all decks
- `get_models()` - Get all note types (card templates)
- `get_notes()` - Get all notes
- `get_cards()` - Get all cards
- `parse_card()` - Parse card HTML fields

**Audio Methods:**
- `get_media_mapping()` - Get media ID to filename mapping
- `extract_audio_files()` - Extract audio files to directory
- `get_audio_for_card()` - Get audio files for specific card
- `get_audio_statistics()` - Get audio usage statistics
- `add_media_file()` - Add audio file to package

**Editing Methods:**
- `update_note_field()` - Update note field content
- `move_card()` - Move card to different deck
- `add_audio_to_card()` - Add audio reference to card
- `create_deck()` - Create new deck
- `copy_cards_to_deck()` - Copy cards to deck
- `create_deck_from_cards()` - Create deck and copy cards
- `delete_card()` - Delete single card with optional audio cleanup
- `delete_cards()` - Delete multiple cards with optional audio cleanup
- `delete_note()` - Delete note and all its cards
- `save()` - Save changes to new APKG file

### Utility Functions

```python
from anki_tools import (
    fix_pinyin,              # Fix pinyin formatting
    extract_sentences,       # Extract sentences from deck
    rank_sentences_zh,       # Rank Chinese sentences
    write_ranking_csv,       # Write ranking to CSV
    complexity_score_zh,     # Calculate complexity score
    frequency_score_zh,      # Calculate frequency score
    char_similarity_zh,      # Calculate character similarity
    reorder_deck,            # Reorder deck by ranking
    load_ranking,            # Load ranking from CSV
)
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_apkg_package.py -v

# Run with coverage
uv run pytest tests/ --cov=anki_tools --cov-report=term-missing
```

**Test coverage:** 450 tests. They cover the package API, the deck gates, the
language pack rules, and the ranking. A test that needs a large input file or
a cache skips when the file is absent. The tests build their own small APKG
files, so the repository holds no deck binary.

## Alternative Methods

| Method | Time | Dependencies | Read | Write | Best For |
|--------|------|--------------|------|-------|----------|
| **Direct SQLite** | 0.073s | None | ✅ | ✅ | Reading, editing, audio extraction |
| genanki | N/A | 1 | ❌ | ✅ | Creating new decks |

### Quick Comparison

**Direct SQLite** (current): Fast, no dependencies, full control. Best for most use cases.

**genanki**: Create decks from scratch. `uv add genanki` (already included)

## How It Works

APKG files are ZIP archives containing:
- `collection.anki2` - SQLite database (cards, notes, decks, models)
- `media` - JSON mapping file IDs to filenames
- Numbered media files (`0`, `1`, `2`, etc.)

The library extracts the ZIP, queries SQLite, parses JSON metadata, and optionally repackages.

## Anki Format Details

### Package Format (.apkg)

APKG files are ZIP archives containing:
- `collection.anki2` - SQLite database (notes, cards, decks, models)
- `media` - JSON mapping numeric IDs to filenames: `{"0": "我.mp3", "1": "你.mp3"}`
- Media files stored with numeric filenames (0, 1, 2, ...)

**Key Tables:**
- `notes` - Note data with fields separated by `\x1f` (id, mid, flds, tags)
- `cards` - Card review data (id, nid, did, ord)
- `col` - Collection metadata (decks, models as JSON)

**Audio:** Referenced as `[sound:filename.mp3]` in note fields.

### Text Import Format (.anki.txt)

```text
#separator:Pipe
#html:true
#deck:Swadesh Mandarin Vocabulary
#notetype:Basic
I | <span style='font-size: 3rem'>我</span> <br> wǒ [sound:我.mp3]
```

One card per line, fields separated by `|` or tab. Audio embedded with `[sound:file.mp3]` tags.

## References

- [Anki Package Format](https://docs.ankiweb.net/exporting.html)
- [AnkiDroid Database Structure](https://github.com/ankidroid/Anki-Android/wiki/Database-Structure)
- [genanki Documentation](https://github.com/kerrickstaley/genanki)
- [Neri Frequency List](https://frequencylists.blogspot.com/2018/02/welcome.html)
- [Apple Daily Cantonese Frequency](https://github.com/chaaklau/appledaily-frequency)
- [Tatoeba Project](https://tatoeba.org/en/)
- [Xefjord's Complete Languages](https://xefjord.wixsite.com/xefscompletelangs)

## Appendix: Supported Languages

Languages supported by gTTS for Swadesh vocabulary audio generation. Use `uv run anki swadesh list-languages` for the current list.

| Code  | Language                 | Code   | Language                 |
|-------|--------------------------|--------|--------------------------|
| af    | Afrikaans                | mr     | Marathi                  |
| am    | Amharic                  | ms     | Malay                    |
| ar    | Arabic                   | my     | Myanmar (Burmese)        |
| bg    | Bulgarian                | ne     | Nepali                   |
| bn    | Bengali                  | nl     | Dutch                    |
| bs    | Bosnian                  | no     | Norwegian                |
| ca    | Catalan                  | pa     | Punjabi (Gurmukhi)       |
| cs    | Czech                    | pl     | Polish                   |
| cy    | Welsh                    | pt     | Portuguese (Brazil)      |
| da    | Danish                   | pt-PT  | Portuguese (Portugal)    |
| de    | German                   | ro     | Romanian                 |
| el    | Greek                    | ru     | Russian                  |
| en    | English                  | si     | Sinhala                  |
| es    | Spanish                  | sk     | Slovak                   |
| et    | Estonian                 | sq     | Albanian                 |
| eu    | Basque                   | sr     | Serbian                  |
| fi    | Finnish                  | su     | Sundanese                |
| fr    | French                   | sv     | Swedish                  |
| fr-CA | French (Canada)          | sw     | Swahili                  |
| gl    | Galician                 | ta     | Tamil                    |
| gu    | Gujarati                 | te     | Telugu                   |
| ha    | Hausa                    | th     | Thai                     |
| hi    | Hindi                    | tl     | Filipino                 |
| hr    | Croatian                 | tr     | Turkish                  |
| hu    | Hungarian                | uk     | Ukrainian                |
| id    | Indonesian               | ur     | Urdu                     |
| is    | Icelandic                | vi     | Vietnamese               |
| it    | Italian                  | yue    | Cantonese                |
| iw    | Hebrew                   | zh-CN  | Chinese (Simplified)     |
| ja    | Japanese                 | zh-TW  | Chinese (Taiwan)         |
| jw    | Javanese                 | zh     | Chinese (Mandarin)       |
| km    | Khmer                    |        |                          |
| kn    | Kannada                  |        |                          |
| ko    | Korean                   |        |                          |
| la    | Latin                    |        |                          |
| lt    | Lithuanian               |        |                          |
| lv    | Latvian                  |        |                          |
| ml    | Malayalam                |        |                          |
