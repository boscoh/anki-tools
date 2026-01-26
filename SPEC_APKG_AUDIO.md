# Specification: Reading Anki .apkg Files and Processing Audio

**Version:** 1.1
**Date:** 2026-01-26
**Status:** Phase 1 Complete

## Overview

This specification defines approaches for reading Anki package (.apkg) files and extracting/processing embedded audio files for Chinese language learning flashcards.

## Background

### What is an .apkg file?

An Anki package (`.apkg`) is a **ZIP archive** containing:
- `collection.anki2` or `collection.anki21` - SQLite database with cards, notes, models, and decks
- `media` - JSON file mapping numeric IDs to actual media filenames
- Numbered media files (`0`, `1`, `2`, etc.) - actual audio/image files without extensions
- Database schema versions: v11 (older) or v18 (newer with zstd compression)

### Current State

**Existing Implementation:**
- Direct SQLite reading via `anki-reader/read_anki.py`
- Extracts and parses card data (Hanzi, Pinyin, Meaning, etc.)
- Does NOT currently handle audio files
- Works with `A_Course_in_Contemporary_Chinese_1.apkg` (576 cards, 540 media files)

**Media File Structure:**
```
A_Course_in_Contemporary_Chinese_1/
├── collection.anki2         # SQLite database
├── media                    # JSON: {"344": "下課.mp3", "0": "哪.mp3", ...}
├── 0                        # MPEG audio (32 kbps, 16 kHz, Mono)
├── 1                        # MPEG audio
├── 2                        # MPEG audio
└── ...                      # 540 total media files
```

## Requirements

### Functional Requirements

#### FR1: Read .apkg Files
- **FR1.1**: Extract ZIP archive to temporary directory
- **FR1.2**: Parse SQLite database (collection.anki2)
- **FR1.3**: Query cards, notes, decks, and models
- **FR1.4**: Map fields to card data (Hanzi, Pinyin, Meaning, Examples, Audio)
- **FR1.5**: Handle HTML content in fields

#### FR2: Extract Audio Files
- **FR2.1**: Parse `media` JSON file mapping
- **FR2.2**: Map numeric file IDs to actual filenames
- **FR2.3**: Identify audio references in card fields
- **FR2.4**: Copy/rename audio files with proper extensions
- **FR2.5**: Validate audio file integrity

#### FR3: Process Audio Files
- **FR3.1**: Detect audio format (MP3, WAV, OGG)
- **FR3.2**: Read audio metadata (bitrate, sample rate, channels)
- **FR3.3**: Extract audio from HTML `[sound:filename.mp3]` tags
- **FR3.4**: Optional: Convert between audio formats
- **FR3.5**: Optional: Normalize audio levels
- **FR3.6**: Optional: Transcribe audio using speech-to-text

### Non-Functional Requirements

#### NFR1: Performance
- Extract and parse 1000-card deck in < 5 seconds
- Process 500 audio files in < 10 seconds

#### NFR2: Dependencies
- Minimize external dependencies
- Use `uv` for package management
- Prefer Python standard library where possible

#### NFR3: Compatibility
- Support Anki format versions v11 and v18
- Handle both `.anki2` and `.anki21` databases
- Work with Python 3.10+

## Design Approaches

### Approach 1: Direct SQLite + Manual Audio Handling (Recommended)

**Pros:**
- Minimal dependencies (zipfile, sqlite3, json - all stdlib)
- Full control over extraction and processing
- Fast and lightweight
- Already implemented for card reading

**Cons:**
- Manual parsing of media mappings
- Need to handle audio format detection manually
- More boilerplate code

**Implementation:**
```python
class AnkiAudioReader(AnkiReader):
    def get_media_mapping(self):
        """Parse media JSON file."""
        media_path = os.path.join(self.temp_dir, 'media')
        with open(media_path, 'r') as f:
            return json.load(f)

    def extract_audio_files(self, output_dir):
        """Extract and rename audio files."""
        mapping = self.get_media_mapping()
        for file_id, filename in mapping.items():
            if filename.endswith(('.mp3', '.wav', '.ogg')):
                src = os.path.join(self.temp_dir, file_id)
                dst = os.path.join(output_dir, filename)
                shutil.copy(src, dst)

    def get_card_audio(self, card_fields):
        """Extract audio references from card HTML."""
        import re
        audio_pattern = r'\[sound:(.*?)\]'
        audio_files = []
        for field_value in card_fields.values():
            matches = re.findall(audio_pattern, field_value)
            audio_files.extend(matches)
        return audio_files
```

**Dependencies:**
```toml
# pyproject.toml - No additional dependencies needed!
[project]
dependencies = []  # Uses only Python stdlib
```

### Approach 2: ankisync2 with Audio Support

**Pros:**
- ORM-based database access (Peewee)
- Built-in media file handling
- Can create/modify decks with audio

**Cons:**
- Additional dependencies (peewee, ankisync2)
- More complex for simple reading tasks
- Learning curve for ORM

**Implementation:**
```python
from ankisync2 import Apkg

with Apkg("deck.apkg") as apkg:
    for card in apkg:
        # Access media files
        media_files = apkg.media_files
        # Process audio...
```

**Dependencies:**
```toml
[project]
dependencies = [
    "ankisync2>=0.3.4",
    "peewee>=3.14.0",
]
```

### Approach 3: anki-export for Audio Extraction

**Pros:**
- Simple API for reading
- Easy export to formats with media references

**Cons:**
- Limited audio processing capabilities
- Less control over extraction process

**Implementation:**
```python
from anki_export import ApkgReader

with ApkgReader('deck.apkg') as apkg:
    data = apkg.export()
    # Extract media references from data
```

**Dependencies:**
```toml
[project]
dependencies = [
    "anki-export>=0.6.0",
]
```

### Approach 4: Audio Processing Libraries

For advanced audio processing (optional):

**pydub** - Audio manipulation
```python
from pydub import AudioSegment

audio = AudioSegment.from_mp3("file.mp3")
normalized = audio.normalize()
converted = audio.export("output.wav", format="wav")
```

**Dependencies:**
```toml
[project.optional-dependencies]
audio = [
    "pydub>=0.25.1",
]
```

**whisper** - Speech-to-text transcription
```python
import whisper

model = whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])
```

**Dependencies:**
```toml
[project.optional-dependencies]
transcription = [
    "openai-whisper>=20231117",
]
```

## Recommended Architecture

### Phase 1: Basic Audio Extraction (Immediate)

**Goal:** Extract audio files from .apkg with correct filenames

**Components:**
1. Extend existing `AnkiReader` class
2. Add `get_media_mapping()` method
3. Add `extract_audio_files(output_dir)` method
4. Add `get_card_audio(card)` to link cards to audio

**Files:**
- `anki-reader/read_anki.py` (extend existing)
- `anki-reader/audio_extractor.py` (new)

**No additional dependencies needed**

### Phase 2: Audio Analysis (Future)

**Goal:** Analyze audio properties and validate files

**Components:**
1. Audio format detection
2. Metadata extraction (duration, bitrate, sample rate)
3. File integrity validation
4. Basic statistics (total duration, file sizes)

**Dependencies:**
```bash
uv add "mutagen>=1.47.0"  # For audio metadata
```

### Phase 3: Audio Processing (Optional)

**Goal:** Advanced audio manipulation

**Components:**
1. Format conversion (MP3 ↔ WAV ↔ OGG)
2. Audio normalization
3. Speech-to-text transcription
4. Silence trimming

**Dependencies:**
```bash
uv add --optional audio "pydub>=0.25.1"
uv add --optional transcription "openai-whisper>=20231117"
```

## Data Flow

```
.apkg file (ZIP)
    ↓
Extract to temp directory
    ↓
┌─────────────────┬──────────────────┐
│                 │                  │
collection.anki2  media (JSON)      0, 1, 2, ... (audio files)
│                 │                  │
SQLite query      Parse mapping     Identify audio
│                 │                  │
Cards + Notes     {0: "哪.mp3"}     MPEG files
│                 │                  │
└─────────────────┴──────────────────┘
                  ↓
      Link card fields to audio files
                  ↓
      Extract/rename to output directory
                  ↓
      Optional: Process/analyze audio
```

## Implementation Example

### Basic Audio Extraction

```python
class AnkiAudioReader(AnkiReader):
    """Extends AnkiReader with audio extraction capabilities."""

    def get_media_mapping(self):
        """Get mapping of file IDs to filenames."""
        media_path = os.path.join(self.temp_dir, 'media')
        if not os.path.exists(media_path):
            return {}
        with open(media_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_audio_files(self, output_dir, audio_only=True):
        """
        Extract media files from .apkg to output directory.

        Args:
            output_dir: Destination directory for audio files
            audio_only: If True, only extract audio files (.mp3, .wav, .ogg)

        Returns:
            Dict mapping filenames to extracted file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        mapping = self.get_media_mapping()
        extracted = {}

        for file_id, filename in mapping.items():
            # Filter for audio files if requested
            if audio_only and not filename.endswith(('.mp3', '.wav', '.ogg')):
                continue

            src = os.path.join(self.temp_dir, file_id)
            if not os.path.exists(src):
                continue

            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
            extracted[filename] = dst

        return extracted

    def get_audio_for_card(self, card, models):
        """
        Extract audio filenames referenced in a card's fields.

        Args:
            card: Card row from database
            models: Models dict from get_models()

        Returns:
            List of audio filenames referenced in card
        """
        import re
        audio_pattern = r'\[sound:(.*?)\]'

        # Get all field values
        field_values = card['flds'].split('\x1f')

        audio_files = []
        for field_value in field_values:
            matches = re.findall(audio_pattern, field_value)
            audio_files.extend(matches)

        return audio_files

    def get_audio_statistics(self):
        """Get statistics about audio files in the package."""
        mapping = self.get_media_mapping()
        audio_files = {k: v for k, v in mapping.items()
                      if v.endswith(('.mp3', '.wav', '.ogg'))}

        stats = {
            'total_media_files': len(mapping),
            'audio_files': len(audio_files),
            'image_files': len(mapping) - len(audio_files),
            'audio_formats': {},
        }

        # Count audio formats
        for filename in audio_files.values():
            ext = filename.split('.')[-1].lower()
            stats['audio_formats'][ext] = stats['audio_formats'].get(ext, 0) + 1

        return stats

# Usage example
def main():
    with AnkiAudioReader('deck.apkg') as reader:
        # Get statistics
        stats = reader.get_audio_statistics()
        print(f"Audio files: {stats['audio_files']}")
        print(f"Formats: {stats['audio_formats']}")

        # Extract all audio
        extracted = reader.extract_audio_files('./audio_output')
        print(f"Extracted {len(extracted)} audio files")

        # Get audio for specific cards
        cards = reader.get_cards()
        models = reader.get_models()

        for card in cards[:5]:
            audio = reader.get_audio_for_card(card, models)
            if audio:
                print(f"Card {card['id']} has audio: {audio}")
```

## Testing Strategy

### Unit Tests

```python
def test_media_mapping():
    """Test parsing of media JSON file."""
    assert mapping['0'] == '哪.mp3'
    assert len(mapping) == 540

def test_audio_extraction():
    """Test audio file extraction."""
    extracted = reader.extract_audio_files('./test_output')
    assert os.path.exists('./test_output/哪.mp3')
    assert len(extracted) > 0

def test_audio_for_card():
    """Test extracting audio references from cards."""
    audio = reader.get_audio_for_card(card, models)
    assert '[sound:' in card['flds']
    assert len(audio) > 0
```

### Integration Tests

- Extract full `A_Course_in_Contemporary_Chinese_1.apkg`
- Verify 540 media files extracted
- Validate MP3 format (32 kbps, 16 kHz, Mono)
- Check filename encoding (Chinese characters)

## Migration Path

### Current State → Phase 1

1. Extend `AnkiReader` class with audio methods
2. Add audio extraction script
3. Test with existing .apkg file
4. Document usage in README.md

### Phase 1 → Phase 2

1. Add `mutagen` dependency for metadata
2. Implement audio analysis methods
3. Add validation checks

### Phase 2 → Phase 3

1. Add optional dependencies group
2. Implement format conversion
3. Add transcription support
4. Create CLI tool for batch processing

## Open Questions

1. **Audio format support**: Should we support all formats or just MP3?
   - **Answer**: Support MP3, WAV, OGG (most common in Anki)

2. **Output structure**: Flat directory or organize by deck/card?
   - **Options**:
     - Flat: `./audio/哪.mp3`
     - By deck: `./audio/DeckName/哪.mp3`
     - By card: `./audio/card_123/哪.mp3`
   - **Recommendation**: Start with flat, add organization later

3. **Duplicate filenames**: How to handle same audio used by multiple cards?
   - **Answer**: Keep one copy (audio files are referenced, not duplicated)

4. **Audio processing**: Which features are high priority?
   - **Priority 1**: Extraction and mapping
   - **Priority 2**: Metadata and validation
   - **Priority 3**: Transcription and conversion

5. **Performance**: Should we process audio in parallel?
   - **Answer**: Not needed for hundreds of files; consider for 1000+ files

## Success Criteria

### Phase 1 (Basic Extraction) - ✅ COMPLETED 2026-01-26
- ✅ Extract all audio files from .apkg
- ✅ Preserve original filenames (Chinese characters)
- ✅ Link cards to their audio files
- ✅ No additional dependencies beyond stdlib

**Implementation:**
- Added `get_media_mapping()` - Parses media JSON file
- Added `extract_audio_files(output_dir, audio_only=True)` - Extracts audio with proper filenames
- Added `get_audio_for_card(card, models)` - Links cards to audio via [sound:...] tags
- Added `get_audio_statistics()` - Reports counts and formats
- Created `demo_audio_extraction.py` - Full demonstration script
- Tested with 538 MP3 files, all Chinese character filenames preserved
- Location: `anki-reader/read_anki.py`

### Phase 2 (Analysis)
- ✅ Report audio statistics (count, formats, durations)
- ✅ Validate audio file integrity
- ✅ Extract metadata (bitrate, sample rate)

### Phase 3 (Processing)
- ✅ Convert between audio formats
- ✅ Normalize audio levels
- ✅ Transcribe Chinese audio to text

## References

- [Anki Manual - Package Format](https://docs.ankiweb.net/exporting.html)
- [AnkiDroid Database Structure](https://github.com/ankidroid/Anki-Android/wiki/Database-Structure)
- Existing research: `anki-reader/RESEARCH.md`
- Current implementation: `anki-reader/read_anki.py`
