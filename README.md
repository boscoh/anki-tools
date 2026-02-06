# Anki Package

Tools for building high-quality Anki vocabulary and sentence decks, with a focus on **listening comprehension** - audio-first flashcards where you hear the sentence and try to understand it.

The goal is to create flashcards that are:

- **Useful** - common words and phrases you'll actually encounter
- **Concise** - short sentences that are easy to memorize
- **Interesting** - natural examples, not textbook filler
- **Non-redundant** - no duplicate concepts or near-identical cards

Read, edit, and manipulate Anki flashcard packages (.apkg files) using direct SQLite access.

## Features

- Read and edit APKG files
- Extract audio files with proper filenames
- Create decks and copy cards
- Rank and filter sentences by complexity, frequency, and similarity
- Generate vocabulary flashcards from CSV files
- Direct SQLite access with minimal dependencies

## Installation

```bash
uv sync
```

## Quick Start

### Basic Usage

```python
from anki_package import AnkiPackage

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

### Vocabulary Generation

Generate vocabulary flashcards from CSV files:

```bash
# Generate .anki.txt import files
uv run python vocab.py txt mandarin
uv run python vocab.py txt spanish
uv run python vocab.py txt greek
uv run python vocab.py txt cantonese

# Build .apkg files with audio
uv run python vocab.py apkg mandarin
uv run python vocab.py apkg spanish
uv run python vocab.py apkg greek
uv run python vocab.py apkg cantonese
```

The `vocab.py` script supports:
- **txt subcommand**: Generate `.anki.txt` files from CSV (for Anki import)
- **apkg subcommand**: Build `.apkg` files with audio generation using gTTS

Supported languages: Mandarin, Spanish, Greek, Cantonese

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

### AnkiPackage

**Reading:** `get_decks()`, `get_models()`, `get_notes()`, `get_cards()`, `parse_card()`

**Audio:** `get_media_mapping()`, `extract_audio_files()`, `get_audio_for_card()`, `get_audio_statistics()`, `add_media_file()`

**Editing:** `update_note_field()`, `move_card()`, `add_audio_to_card()`, `create_deck()`, `copy_cards_to_deck()`, `create_deck_from_cards()`, `delete_card()`, `delete_cards()`, `delete_note()`, `save()`

## Examples

Run examples with:
```bash
python main.py read          # Read cards
python main.py audio         # Extract audio
python main.py edit          # Edit cards
python main.py deck          # Create decks
python main.py add-audio     # Add audio files
python main.py delete        # Delete cards and cleanup audio
```

Or run `python main.py` for default card reading example.

- `examples.py` - All examples in one file
- `compare_methods.py` - Performance comparison

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

**Total:** 37 tests covering media mapping, audio extraction, card linking, deletion, and Chinese filenames

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
uv run pytest tests/ --cov=anki --cov-report=html

# View coverage in terminal
uv run pytest tests/ --cov=anki --cov-report=term-missing
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
