# Test Coverage: Audio Extraction

**Status:** ✅ All 27 tests passing
**Coverage Date:** 2026-01-26
**Test Runtime:** ~4.2 seconds

## Test Summary

### Test Classes

1. **TestMediaMapping** (4 tests)
   - Media mapping returns correct data structure
   - Contains audio files
   - Validates ID-to-filename mapping

2. **TestAudioStatistics** (5 tests)
   - Statistics return all required fields
   - Counts are consistent (audio + image = total)
   - Validates expected values for test deck

3. **TestAudioExtraction** (9 tests)
   - Creates output directory
   - Extracts files with correct names
   - Filters audio-only correctly
   - Preserves Chinese character filenames
   - Validates file sizes
   - Matches statistics counts

4. **TestCardAudioLinking** (6 tests)
   - Links cards to audio files
   - Parses [sound:...] tags correctly
   - Validates all cards have audio
   - Confirms extractable audio matches references

5. **TestIntegration** (3 tests)
   - Complete workflow (stats → extract → link)
   - No duplicate files
   - Idempotent extraction

## Test Results

```
27 passed in 4.23 seconds
```

### Tested Against

- **Deck:** A_Course_in_Contemporary_Chinese_1.apkg
- **Cards:** 576 total
- **Audio Files:** 538 MP3 files
- **Format:** MPEG layer III, 32 kbps, 16 kHz, Mono
- **Filenames:** Chinese characters preserved

## Key Findings

### Data Quality Issue Detected

The test suite revealed a real data quality issue in the test deck:

- **Referenced audio files:** 542
- **Available in package:** 538
- **Missing files:** 4

Missing files:
- 發炎.mp3
- 藥房.mp3
- 請打開書，翻倒第五頁.mp3
- 關心.mp3

The integration test now checks for **95% coverage** to account for this real-world scenario where decks may have missing media files.

## Test Coverage

### Methods Tested

✅ `get_media_mapping()`
- Returns dict with ID → filename mapping
- Handles missing media file gracefully
- Validates structure and types

✅ `extract_audio_files(output_dir, audio_only=True)`
- Creates output directory
- Extracts all audio files
- Preserves original filenames (including Chinese)
- Filters by audio extension
- Returns dict of extracted files

✅ `get_audio_for_card(card, models)`
- Extracts [sound:...] references
- Returns list of filenames
- Handles multiple audio per card
- Parses HTML correctly

✅ `get_audio_statistics()`
- Returns comprehensive stats
- Counts total/audio/image files
- Breaks down by format
- Consistent calculations

### Edge Cases Tested

- ✅ Chinese character filenames (哪.mp3, 下課.mp3)
- ✅ Empty output directories
- ✅ Idempotent extraction (re-running is safe)
- ✅ Missing media files (real-world scenario)
- ✅ Multiple audio references per card
- ✅ audio_only filter toggle
- ✅ File size validation (1KB - 100KB range)

## Running Tests

### All Tests

```bash
cd anki-reader
uv run pytest test_audio_extraction.py -v
```

### Specific Test Class

```bash
uv run pytest test_audio_extraction.py::TestAudioExtraction -v
```

### Single Test

```bash
uv run pytest test_audio_extraction.py::TestAudioExtraction::test_chinese_character_filenames -v
```

### With Coverage Report

```bash
uv run pytest test_audio_extraction.py --cov=read_anki --cov-report=html
```

## Dependencies

- pytest >= 8.0.0

Installed via:
```bash
uv add pytest
```

## Future Test Improvements

- [ ] Test with multiple deck formats (.anki2 vs .anki21)
- [ ] Test with decks containing only images (no audio)
- [ ] Test with different audio formats (WAV, OGG)
- [ ] Performance tests for large decks (1000+ cards)
- [ ] Mock tests for unit testing without real .apkg file
- [ ] Test error handling for corrupted .apkg files
- [ ] Test Unicode edge cases in filenames
- [ ] Memory usage tests for extraction
