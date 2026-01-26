# Anki Reader

A Python project to read and parse Anki flashcard packages (.apkg files).

## Features

- Extracts and reads Anki package files
- Displays deck information
- Lists all note types/templates
- Parses and displays flashcards with their fields
- Handles HTML content in cards
- **NEW: Audio extraction and processing**
  - Extract all audio files from .apkg packages
  - Get audio statistics (counts, formats)
  - Link cards to their audio files
  - Preserve Chinese character filenames

## Installation

This project uses [UV](https://docs.astral.sh/uv/) for Python package management.

```bash
# The dependencies are already installed
uv sync
```

## Usage

Place your .apkg file in the parent directory, then run:

### Basic Card Reading

```bash
uv run read_anki.py
```

The script will:
1. Find the .apkg file in the parent directory
2. Extract and parse the database
3. Display deck information, note types, and cards

### Audio Extraction

```bash
uv run demo_audio_extraction.py
```

This demonstrates:
1. Getting audio statistics from the package
2. Extracting all audio files to `./extracted_audio`
3. Linking cards to their audio files
4. Displaying cards with audio references

## Output

The script displays:
- **Decks**: All decks in the package
- **Note Types**: Card templates/models used
- **Cards**: First 10 cards with all fields (shows total count)

## How It Works

Anki packages (.apkg) are ZIP files containing:
- `collection.anki2`: SQLite database with cards, decks, and metadata
- Media files (images, audio, etc.)

The script:
1. Extracts the ZIP to a temporary directory
2. Opens the SQLite database
3. Queries the cards and notes tables
4. Parses JSON metadata for decks and models
5. Combines the data into readable output

## Audio Extraction API

The `AnkiReader` class now includes methods for audio extraction:

```python
from read_anki import AnkiReader

with AnkiReader('deck.apkg') as reader:
    # Get audio statistics
    stats = reader.get_audio_statistics()
    print(f"Audio files: {stats['audio_files']}")
    print(f"Formats: {stats['audio_formats']}")

    # Extract all audio files
    extracted = reader.extract_audio_files('./audio_output')
    print(f"Extracted {len(extracted)} files")

    # Get audio for specific cards
    cards = reader.get_cards()
    models = reader.get_models()

    for card in cards:
        audio = reader.get_audio_for_card(card, models)
        if audio:
            print(f"Card has audio: {audio}")
```

### Available Methods

- `get_media_mapping()` - Returns dict mapping file IDs to filenames
- `extract_audio_files(output_dir, audio_only=True)` - Extracts audio to directory
- `get_audio_for_card(card, models)` - Returns list of audio files for a card
- `get_audio_statistics()` - Returns stats about audio files in package

## Alternative Methods

This project uses **Direct SQLite** access for reading .apkg files. See `RESEARCH.md` for a comprehensive comparison of different approaches:

- **Direct SQLite** (current) - Fast, no dependencies, full control
- **anki-export** - Simple API, easy data export
- **ankipandas** - Pandas integration for data analysis
- **ankisync2** - ORM approach for editing decks
- **genanki** - For creating new decks

Run `compare_methods.py` to see a side-by-side comparison with performance metrics.

## Example Scripts

- `read_anki.py` - Main script using direct SQLite (card reading)
- `demo_audio_extraction.py` - **NEW: Audio extraction demonstration**
- `examples_anki_export.py` - Using anki-export library
- `examples_ankisync2.py` - Using ankisync2 ORM
- `compare_methods.py` - Performance comparison

## Current Package

Reading: `A_Course_in_Contemporary_Chinese_1.apkg`
- 576 total cards
- 538 audio files (MP3, 32 kbps, 16 kHz, Mono)
- Chinese language learning flashcards
- Fields include: Hanzi, Pinyin, Meaning, Examples, Audio
- All cards include audio pronunciation
