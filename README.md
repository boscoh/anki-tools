# Anki Tools

Tools for building high-quality Anki vocabulary and sentence decks, with a focus on **listening comprehension** - audio-first flashcards where you hear the sentence and try to understand it.

The goal is to create flashcards that are:

- **Useful** - common words and phrases you'll actually encounter
- **Concise** - short sentences that are easy to memorize
- **Interesting** - natural examples, not textbook filler
- **Non-redundant** - no duplicate concepts or near-identical cards

Read, edit, and manipulate Anki flashcard packages (.apkg files) using direct SQLite access, plus command-line tools for ranking, reordering, and building vocabulary decks.

## Features

- **Python API**: Read and edit APKG files with direct SQLite access
- **CLI Tools**: Command-line interface for common workflows
- **Audio Extraction**: Extract audio files with proper filenames
- **Deck Management**: Create decks, copy cards, move cards between decks
- **Sentence Ranking**: Rank and filter sentences by complexity, frequency, and similarity (Chinese, French, and Cantonese)
- **Vocabulary Generation**: Build Swadesh vocabulary decks with audio (multiple languages)
- **Card Deletion**: Delete cards with automatic cleanup of unused audio files
- **Minimal Dependencies**: Direct SQLite access, no Anki installation required

## Installation

```bash
uv sync
```

After installation, the `anki` command is available via `uv run anki`.

## CLI Commands

The `anki` command provides several subcommands:

### swadesh - Build Vocabulary Decks

```bash
uv run anki swadesh list-languages    # List supported languages
uv run anki swadesh build LANGUAGE    # Build deck for language
```

### zh - Chinese Deck Processing

```bash
uv run anki zh rank INPUT.apkg        # Rank by frequency/complexity
uv run anki zh reorder INPUT.apkg     # Reorder by ranking
uv run anki zh fix INPUT.apkg         # Fix pinyin formatting
uv run anki zh all INPUT.apkg         # Run complete pipeline
```

### fr - French Deck Processing

```bash
uv run anki fr rank INPUT.apkg        # Rank by difficulty
uv run anki fr similar INPUT.apkg     # Find similar sentences
uv run anki fr reorder INPUT.apkg     # Reorder by ranking
```

### yue - Cantonese Deck Processing

```bash
uv run anki yue rank INPUT.apkg       # Rank by difficulty
uv run anki yue similar INPUT.apkg    # Find similar sentences
uv run anki yue reorder INPUT.apkg    # Reorder by ranking
```

### inspect - Inspect APKG Files

```bash
uv run anki inspect INPUT.apkg        # Show deck contents
```

### style - Apply Templates

```bash
uv run anki style INPUT.apkg          # Apply CSS and card templates
```

## Quick Start

### Command Line Interface

After installation, use the `anki` command:

```bash
# List available commands
uv run anki --help

# Build Swadesh vocabulary decks
uv run anki swadesh build mandarin
uv run anki swadesh list-languages

# Rank Chinese sentences by difficulty
uv run anki zh rank input.apkg
uv run anki zh reorder input.apkg

# Rank French sentences
uv run anki fr rank input.apkg
uv run anki fr similar input.apkg

# Rank Cantonese sentences
uv run anki yue rank input.apkg
uv run anki yue similar input.apkg

# Inspect APKG files
uv run anki inspect deck.apkg
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

### Swadesh Vocabulary Generation

Generate Swadesh vocabulary flashcards with audio:

```bash
# List available languages
uv run anki swadesh list-languages

# Build vocabulary deck for a specific language
uv run anki swadesh build mandarin
uv run anki swadesh build spanish
uv run anki swadesh build greek
uv run anki swadesh build cantonese
uv run anki swadesh build arabic
uv run anki swadesh build hindi
uv run anki swadesh build german
```

The `swadesh` command:
- Builds `.apkg` files from CSV vocabulary lists in `vocab/` directory
- Generates audio using gTTS (Google Text-to-Speech)
- Creates flashcards with clickable characters (for character-based languages)
- Supports multiple languages with proper formatting and styling

### French Sentence Ranking

Rank French decks by difficulty and similarity:

```bash
# Rank sentences by complexity, frequency, and uniqueness
uv run anki fr rank deck.apkg

# Show high-similarity pairs (candidates for deletion)
uv run anki fr similar deck.apkg

# Reorder deck based on ranking
uv run anki fr reorder deck.apkg
```

**Frequency:** Uses wordfreq (no download; ships with the package).

**Grammar:** No download needed for basic scoring (heuristics). For dependency-based grammar complexity, download the spaCy model once:

```bash
uv run python -m spacy download fr_core_news_sm
```

### Cantonese Sentence Ranking

Rank Cantonese decks by difficulty and similarity:

```bash
# Rank sentences by complexity, frequency, and uniqueness
uv run anki yue rank deck.apkg

# Show high-similarity pairs (candidates for deletion)
uv run anki yue similar deck.apkg

# Reorder deck based on ranking
uv run anki yue reorder deck.apkg
```

**Method:** Uses character-based complexity and similarity scoring (like Chinese) with Cantonese-specific weighting.

**Note:** Frequency data uses Chinese (Mandarin) from wordfreq as an approximation, since Cantonese-specific frequency data is not available in wordfreq. This is reasonable because both languages use the same Chinese characters with similar frequency distributions.

### Chinese Sentence Processing

Process Chinese decks with ranking, reordering, and pinyin fixes:

```bash
# Rank sentences by frequency and complexity
uv run anki zh rank deck.apkg

# Reorder deck based on ranking
uv run anki zh reorder deck.apkg

# Fix pinyin formatting
uv run anki zh fix deck.apkg

# Run complete pipeline: rank + reorder + fix pinyin + style
uv run anki zh all deck.apkg
```

### gTTS Available Languages

The following languages are supported by gTTS for audio generation:

| Code    | Language                   | Code    | Language                   |
|---------|----------------------------|---------|----------------------------|
| af      | Afrikaans                  | mr      | Marathi                    |
| am      | Amharic                    | ms      | Malay                      |
| ar      | Arabic                     | my      | Myanmar (Burmese)          |
| bg      | Bulgarian                  | ne      | Nepali                     |
| bn      | Bengali                    | nl      | Dutch                      |
| bs      | Bosnian                    | no      | Norwegian                  |
| ca      | Catalan                    | pa      | Punjabi (Gurmukhi)         |
| cs      | Czech                      | pl      | Polish                     |
| cy      | Welsh                      | pt      | Portuguese (Brazil)        |
| da      | Danish                     | pt-PT   | Portuguese (Portugal)      |
| de      | German                     | ro      | Romanian                   |
| el      | Greek                      | ru      | Russian                    |
| en      | English                    | si      | Sinhala                    |
| es      | Spanish                    | sk      | Slovak                     |
| et      | Estonian                   | sq      | Albanian                   |
| eu      | Basque                     | sr      | Serbian                    |
| fi      | Finnish                    | su      | Sundanese                  |
| fr      | French                     | sv      | Swedish                    |
| fr-CA   | French (Canada)            | sw      | Swahili                    |
| gl      | Galician                   | ta      | Tamil                      |
| gu      | Gujarati                   | te      | Telugu                     |
| ha      | Hausa                      | th      | Thai                       |
| hi      | Hindi                      | tl      | Filipino                   |
| hr      | Croatian                   | tr      | Turkish                    |
| hu      | Hungarian                  | uk      | Ukrainian                  |
| id      | Indonesian                 | ur      | Urdu                       |
| is      | Icelandic                  | vi      | Vietnamese                 |
| it      | Italian                    | yue     | Cantonese                  |
| iw      | Hebrew                     | zh-CN   | Chinese (Simplified)       |
| ja      | Japanese                   | zh-TW   | Chinese (Mandarin/Taiwan)  |
| jw      | Javanese                   | zh      | Chinese (Mandarin)         |
| km      | Khmer                      |         |                            |
| kn      | Kannada                    |         |                            |
| ko      | Korean                     |         |                            |
| la      | Latin                      |         |                            |
| lt      | Lithuanian                 |         |                            |
| lv      | Latvian                    |         |                            |
| ml      | Malayalam                  |         |                            |

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

### Quick Start

```bash
# Run all tests
uv run pytest tests/ -v

# Run all tests (from anki-package directory)
cd anki-package
uv run pytest tests/ -v
```

### Test Structure

Tests are organized in the `tests/` directory:

- `tests/conftest.py` - Shared fixtures (apkg_path, pkg, temp_output_dir)
- `tests/test_media_mapping.py` - Media mapping tests (4 tests)
- `tests/test_audio_statistics.py` - Audio statistics tests (5 tests)
- `tests/test_audio_extraction.py` - Audio extraction tests (9 tests)
- `tests/test_card_audio_linking.py` - Card-to-audio linking tests (6 tests)
- `tests/test_integration.py` - Integration workflow tests (3 tests)
- `tests/test_card_deletion.py` - Card deletion tests (10 tests)
- `tests/test_rank_sentences.py` - Sentence ranking tests (7 tests)

**Total:** 54 tests covering media mapping, audio extraction, card linking, deletion, sentence ranking, and Chinese filenames

### Running Specific Tests

```bash
# Run specific test file
uv run pytest tests/test_card_deletion.py -v

# Run specific test class
uv run pytest tests/test_card_deletion.py::TestCardDeletion -v

# Run specific test method
uv run pytest tests/test_card_deletion.py::TestCardDeletion::test_delete_single_card -v

# Run tests matching a pattern
uv run pytest tests/ -k "audio" -v
uv run pytest tests/ -k "deletion" -v
```

### Test Coverage

```bash
# Generate coverage report
uv run pytest tests/ --cov=anki_tools --cov-report=html

# View coverage in terminal
uv run pytest tests/ --cov=anki_tools --cov-report=term-missing
```

### Prerequisites

Tests require:
- An `.apkg` file in the `tests/` directory
- pytest installed: `uv add pytest` or `uv add --dev pytest`

### Test Output

```bash
# Verbose output (shows each test)
uv run pytest tests/ -v

# Very verbose (shows test docstrings)
uv run pytest tests/ -vv

# Stop on first failure
uv run pytest tests/ -x

# Show print statements
uv run pytest tests/ -s
```

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

### Text Import Format (.anki.txt)

Anki text import files use a simple format with header directives followed by card data:

**Header Directives:**
- `#separator:Pipe` or `#separator:tab` - Field separator (Pipe `|` or Tab `\t`)
- `#html:true` - Enable HTML rendering in fields
- `#deck:Deck Name` - Target deck name (created if doesn't exist)
- `#notetype:Basic` or `#notetype:Basic (and reversed card)` - Note type template

**Card Format:**
- Each line represents one card
- Fields separated by the specified separator (Pipe `|` or Tab)
- HTML tags supported when `#html:true` is set
- Audio references: `[sound:filename.mp3]` embedded in field text

**Example:**
```
#separator:Pipe
#html:true
#deck:Mandarin Swadesh 207
#notetype:Basic
I | <span style='font-size: 3rem'>我</span> <br> wǒ [sound:我.mp3]
```

### Anki Package Format (.apkg)

Anki package files are ZIP archives containing:

**Structure:**
- `collection.anki2` - SQLite database with all notes, cards, decks, and models
- `media` - JSON file mapping numeric IDs to media filenames (e.g., `{"0": "我.mp3", "1": "你.mp3"}`)
- `media/` directory - Media files stored with numeric filenames (0, 1, 2, ...)

**Database Schema:**
- **notes table**: Stores note data with fields separated by `\x1f` (unit separator character)
  - `id` - Note ID
  - `mid` - Model ID (references note type)
  - `flds` - Field values separated by `\x1f`
  - `tags` - Space-separated tags
- **cards table**: Links notes to cards for review
  - `id` - Card ID
  - `nid` - Note ID
  - `did` - Deck ID
  - `ord` - Card ordinal (for multi-card note types)
- **col table**: Collection metadata
  - `decks` - JSON object mapping deck IDs to deck configs
  - `models` - JSON object mapping model IDs to note type templates

**Audio References:**
- Audio files referenced in note fields as `[sound:filename.mp3]`
- When importing, Anki matches the filename to the media mapping
- Audio files must be included in the package's media directory
- Media files are stored with numeric IDs, not original filenames

**Field Separators:**
- In database: `\x1f` (unit separator, ASCII 31)
- In text import: Configurable (Pipe `|` or Tab `\t`)

**HTML Support:**
- HTML tags are preserved and rendered when `#html:true` is set
- Common tags: `<br>`, `<span>`, `<div>`, inline styles supported
- Useful for formatting, font sizes, colors

**Note Types:**
- `Basic` - Single card (Front → Back)
- `Basic (and reversed card)` - Two cards (Front → Back, Back → Front)
- Custom note types can have multiple fields and card templates

## File Format

**Database tables:** `notes` (id, mid, flds, tags), `cards` (id, nid, did), `col` (models, decks as JSON)

**Media:** Files stored as numbers, mapped via `media` JSON: `{"0": "哪.mp3"}`

**Audio references:** `[sound:filename.mp3]` in card fields

## References

- [Anki Package Format](https://docs.ankiweb.net/exporting.html)
- [AnkiDroid Database Structure](https://github.com/ankidroid/Anki-Android/wiki/Database-Structure)
- [genanki Documentation](https://github.com/kerrickstaley/genanki)
- [Neri Frequency List](https://frequencylists.blogspot.com/2018/02/welcome.html)
- [Tatoeba Project](https://tatoeba.org/en/)
- [Xefjord's Complete Languages](https://xefjord.wixsite.com/xefscompletelangs)