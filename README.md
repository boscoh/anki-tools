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

# Fix French deck (rank and reorder)
uv run anki fix fr input.apkg

# Build the Cantonese frequency list (blended spoken and written counts)
uv run anki freq yue

# Fix Cantonese deck (convert to traditional, fix jyutping, select, rank, reorder)
uv run anki fix yue input.apkg

# Fix Spanish deck (rank and reorder, optionally reduce to 3000 sentences)
uv run anki fix es input.apkg --max-sentences 3000

# Inspect APKG files
uv run anki inspect deck.apkg

# Apply CSS/HTML templates
uv run anki style deck.apkg
```

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
uv run anki fix zh deck.apkg                        # Rank, reorder, fix pinyin, apply styling
uv run anki fix zh deck.apkg --max-sentences 2600   # Also deduplicate and reduce
```
- Character-based complexity and similarity
- Uses wordfreq for frequency data with HSK vocabulary bonus
- Strips repeated-pattern sentences before ranking (e.g. '走吧走吧', '好了，好了')
- Fixes pinyin formatting and applies card styling
- Outputs `_reordered.apkg` and `.rank.csv`
- With `--max-sentences`: stratified reduction across 4 complexity bins (35/35/20/10%)
  with same-unique-char-length deduplication (threshold 0.67).
  - Threshold 0.67 is intentional: for 3-unique-char sentences Jaccard jumps from 0.5
    directly to 1.0 with nothing in between, so 0.67 catches only true anagrams
    ('你要我'/'我要你') and avoids false positives like '我在这'/'你在这'.
  - Dedup runs within each complexity bin; cross-bin near-duplicates are not removed.

**French (fr):**
```bash
uv run anki fix fr deck.apkg  # Rank and reorder
```
- Word-based complexity and similarity
- Uses wordfreq for frequency data
- Optional spaCy grammar scoring: `uv run python -m spacy download fr_core_news_sm`
- Outputs `_reordered.apkg` and `.rank.csv`

**Cantonese (yue):**
```bash
uv run anki freq yue          # Write data/yue_frequency.tsv
uv run anki freq yue --check  # Run gates G1, G3, and G4
uv run anki fix yue deck.apkg # Convert, fix jyutping, select, rank, reorder
```
- Converts simplified to traditional Chinese
- Generates accurate jyutping romanization
- The frequency term blends spoken and written counts. The spoken term is the
  Cifu `SpokenAdult` count; the written term is the mean of the Cifu and Apple
  Daily counts. The default spoken weight is 0.7.
  - `--freq written` uses the Apple Daily list alone. This is the old ranking.
  - The first run downloads both files to `~/.cache/anki-tools`. Set
    `CIFU_PATH` or `YUE_FREQUENCY_PATH` to use other files.
  - A variant map in `vocab/yue_variants.tsv` and the overrides in
    `vocab/yue_frequency_overrides.tsv` fix common spoken forms and greetings.
  - `--freq-report` writes `deck.freq_change.csv` for the sentences that the
    blend moves by more than 100 places.
- The filter removes fragments and hard sentences. A sentence passes when it
  has 3 to 12 Han characters, no Latin letter or digit, no repeated pattern,
  one final mark at the end, no quotation mark, at most one internal mark, at
  most 2 words outside the top 2000, and no foreign name from
  `vocab/yue_names.tsv` (add a row for a new name). Latin names and English
  words fail the Latin-letter rule.
- The selection keeps the fewest sentences that cover the 1500 most frequent
  words of the deck. Each target word gets 2 cards. One word outside the top
  100 goes in at most 8 kept sentences. The cards go from easy to hard.
  - `--max-sentences` caps the card count. `--per-word`, `--max-uses`,
    `--top-words`, and `--order score` control the selection.
  - `--no-select` gives the old output: rank and reorder only.
- Outputs `_reordered.apkg`, `.rank.csv` (adds `hardest_rank` and `kept`),
  `.select.csv` (one row per input sentence with the reason), and
  `.coverage.csv` (one row per target word)

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

The deck name has the entry count (for example, `Swadesh Mandarin 207`).

### Mandarin Swadesh Sentences

Build a listening-comprehension deck for Mandarin. The deck pairs 207 Swadesh
concepts with 387 Tatoeba sentences. Each card holds one sentence, the
tone-marked pinyin, the English translation, the target word, and two MP3
files.

```bash
uv run anki swadesh fetch                                   # cache the Tatoeba data
uv run anki swadesh sentences mandarin --per-word 2 --max-cards 450
uv run anki swadesh audio mandarin --native-audio none      # make the audio
uv run anki swadesh pack mandarin                           # write the APKG
uv run anki swadesh audit mandarin                          # gates G1 to G16
```

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

Measured result: 387 cards, 586 media files, and core coverage of 195 of 207
concepts (0.942). The pack has 0 scaffold cards and 0 native recordings. Each
sentence has 4 to 7 Han characters. These 12 words have no selected sentence:
虱子, 棍子, 种子, 骨头, 脂肪, 羽毛, 舌头, 肠子, 乳房, 呕吐, 揉, and 绑.

On the back of a card, each Han character of the sentence is a link to the
CC-Canto dictionary at `https://cantonese.org`. The script is
`templates/char_links.html`.

Licenses: word data CC BY-SA 4.0, Tatoeba sentence text CC BY 2.0 FR, and
audio per recording. This pack uses synthesized audio only.



### Greek Swadesh Sentences

Build a listening-comprehension deck for Greek. The deck pairs 207 Swadesh
concepts with 306 Tatoeba sentences. Each card holds one sentence, the English
translation, the target word with a gender gloss, and two MP3 files.

```bash
uv run anki swadesh fetch --lang ell --lang eng              # cache the Tatoeba data
uv run anki swadesh sentences greek --per-word 2 --max-cards 340
uv run anki swadesh audio greek --tts gtts                   # make the audio
uv run anki swadesh pack greek                               # write the APKG
uv run anki swadesh audit greek                              # gates G1 to G19
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

Measured result: 306 cards, 468 media files, and core coverage of 162 of 206
card-capable concepts (0.786). The pack has 0 scaffold cards and 0 native
recordings. These 44 words have no selected sentence: πλατύς, παχύς, ψείρα,
σκουλήκι, ραβδί, σπόρος, φλοιός του δέντρου, χόρτο, λίπος, κέρατο, πούπουλο,
αυτί, νύχι, γόνατο, φτερό, κοιλιά, σπλάχνα, συκώτι, ρουφώ, ξερνώ, φυσάω,
κυνηγώ, χτυπάω, μαχαιρώνω, ξύνω, σκάβω, πετάω, σφίγγω, τρίβω, σκουπίζω,
τραβάω, δένω, μετρώ, πλέω, ρέω, παγώνω, πρήζομαι, σκόνη, στάχτη, σάπιος,
στρογγυλός, αιχμηρός, στομωμένος, λείος.

Index 81 (`πόδι`, leg) shares its form with index 80 (foot). The shared row has
`variant=shared`, gets no card, and gate G19 rejects a card on a shared-only
form.

Licenses: word data CC BY-SA 4.0, Tatoeba sentence text CC BY 2.0 FR, spaCy
model MIT, and synthesized audio named by its engine and voice.

### French Swadesh Sentences

Build a listening-comprehension deck for French. The deck pairs 207 Swadesh
concepts with 412 Tatoeba sentences. Each card holds one sentence with the
matched token in bold, the English translation, the target word with a gender
gloss, and two MP3 files.

```bash
uv run anki swadesh fetch --lang fra --lang eng              # cache the Tatoeba data
uv run anki swadesh sentences french --per-word 2 --max-cards 450
uv run anki swadesh audio french --native-audio none         # make the audio
uv run anki swadesh pack french                              # write the APKG
uv run anki swadesh audit french                             # gates G1 to G20
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

Measured result on 2026-10-19: 412 cards, 618 media files, and core coverage
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

### Cantonese Swadesh Sentences

Build a listening-comprehension deck for Cantonese. The deck pairs 207 Swadesh
concepts with 333 Tatoeba sentences. Each card holds one sentence with the
matched token in bold, the jyutping reading, the English translation, the
target word with its reading, and two MP3 files.

```bash
uv run anki swadesh fetch --lang yue --lang eng              # cache the Tatoeba data
uv run anki swadesh sentences cantonese --per-word 2 --max-cards 450
uv run anki swadesh audio cantonese --native-audio auto      # make the audio
uv run anki swadesh pack cantonese                           # write the APKG
uv run anki swadesh audit cantonese                          # gates G1 to G19
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

1. The Cifu lexicon for the tokenizer. `uv run anki freq yue` downloads it to
   the cache. The build does not commit it.
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

Measured result on 2026-10-20: 333 cards, 178 of 207 forms (0.860), 208
native recordings, and 125 synthesized files. These 29 forms have no selected
sentence: 闊, 蝨, 棍, 種子, 葉, 樹皮, 脂肪, 羽毛, 腿, 膝頭, 翼, 腸, 胸, 肝,
咬, 吐, 打獵, 分開, 𢯎, 搾, 掟, 綁, 浮, 結冰, 脹, 湖, 鹽, 鈍, 裡面.

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
uv run pytest tests/test_card_deletion.py -v

# Run with coverage
uv run pytest tests/ --cov=anki_tools --cov-report=term-missing
```

**Test coverage:** 54 tests covering media mapping, audio extraction, card linking, deletion, and sentence ranking.

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

```
#separator:Pipe
#html:true
#deck:Mandarin Swadesh 207
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
