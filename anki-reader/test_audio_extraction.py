#!/usr/bin/env python3
"""
Unit and integration tests for audio extraction from .apkg files.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
from read_anki import AnkiReader


@pytest.fixture
def apkg_path():
    """Find the .apkg file in the parent directory."""
    apkg_files = list(Path('..').glob('*.apkg'))
    if not apkg_files:
        pytest.skip("No .apkg file found in parent directory")
    return str(apkg_files[0])


@pytest.fixture
def reader(apkg_path):
    """Create an AnkiReader instance."""
    with AnkiReader(apkg_path) as r:
        yield r


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for audio extraction."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


class TestMediaMapping:
    """Tests for get_media_mapping() method."""

    def test_get_media_mapping_returns_dict(self, reader):
        """Test that get_media_mapping returns a dictionary."""
        mapping = reader.get_media_mapping()
        assert isinstance(mapping, dict)

    def test_media_mapping_not_empty(self, reader):
        """Test that media mapping contains entries."""
        mapping = reader.get_media_mapping()
        assert len(mapping) > 0

    def test_media_mapping_structure(self, reader):
        """Test that media mapping has correct structure (ID -> filename)."""
        mapping = reader.get_media_mapping()
        # Keys should be numeric strings
        for file_id, filename in mapping.items():
            assert isinstance(file_id, str)
            assert isinstance(filename, str)
            # Filename should have an extension
            assert '.' in filename

    def test_media_mapping_contains_audio(self, reader):
        """Test that media mapping contains audio files."""
        mapping = reader.get_media_mapping()
        audio_files = [f for f in mapping.values()
                      if f.endswith(('.mp3', '.wav', '.ogg'))]
        assert len(audio_files) > 0


class TestAudioStatistics:
    """Tests for get_audio_statistics() method."""

    def test_get_audio_statistics_returns_dict(self, reader):
        """Test that get_audio_statistics returns a dictionary."""
        stats = reader.get_audio_statistics()
        assert isinstance(stats, dict)

    def test_statistics_has_required_fields(self, reader):
        """Test that statistics contains required fields."""
        stats = reader.get_audio_statistics()
        assert 'total_media_files' in stats
        assert 'audio_files' in stats
        assert 'image_files' in stats
        assert 'audio_formats' in stats

    def test_statistics_counts_are_consistent(self, reader):
        """Test that audio + image counts match total."""
        stats = reader.get_audio_statistics()
        assert stats['audio_files'] + stats['image_files'] == stats['total_media_files']

    def test_audio_formats_is_dict(self, reader):
        """Test that audio_formats is a dictionary."""
        stats = reader.get_audio_statistics()
        assert isinstance(stats['audio_formats'], dict)

    def test_statistics_expected_values(self, reader):
        """Test expected statistics for A_Course_in_Contemporary_Chinese_1.apkg."""
        stats = reader.get_audio_statistics()
        # This deck has 538 audio files, all MP3
        assert stats['audio_files'] == 538
        assert stats['total_media_files'] == 538
        assert 'mp3' in stats['audio_formats']
        assert stats['audio_formats']['mp3'] == 538


class TestAudioExtraction:
    """Tests for extract_audio_files() method."""

    def test_extract_audio_files_creates_directory(self, reader, temp_output_dir):
        """Test that extract_audio_files creates the output directory."""
        output_dir = os.path.join(temp_output_dir, 'audio')
        reader.extract_audio_files(output_dir)
        assert os.path.exists(output_dir)
        assert os.path.isdir(output_dir)

    def test_extract_audio_files_returns_dict(self, reader, temp_output_dir):
        """Test that extract_audio_files returns a dictionary."""
        extracted = reader.extract_audio_files(temp_output_dir)
        assert isinstance(extracted, dict)

    def test_extracted_files_exist(self, reader, temp_output_dir):
        """Test that extracted files actually exist on disk."""
        extracted = reader.extract_audio_files(temp_output_dir)
        for filename, path in extracted.items():
            assert os.path.exists(path)
            assert os.path.isfile(path)

    def test_extracted_filenames_match(self, reader, temp_output_dir):
        """Test that extracted filenames match the mapping."""
        extracted = reader.extract_audio_files(temp_output_dir)
        for filename, path in extracted.items():
            assert os.path.basename(path) == filename

    def test_audio_only_filters_correctly(self, reader, temp_output_dir):
        """Test that audio_only=True filters non-audio files."""
        extracted = reader.extract_audio_files(temp_output_dir, audio_only=True)
        for filename in extracted.keys():
            assert filename.endswith(('.mp3', '.wav', '.ogg'))

    def test_extract_all_media(self, reader, temp_output_dir):
        """Test that audio_only=False extracts all media."""
        extracted = reader.extract_audio_files(temp_output_dir, audio_only=False)
        stats = reader.get_audio_statistics()
        assert len(extracted) == stats['total_media_files']

    def test_chinese_character_filenames(self, reader, temp_output_dir):
        """Test that Chinese character filenames are preserved."""
        extracted = reader.extract_audio_files(temp_output_dir)
        # Check for some known Chinese filenames
        chinese_files = [f for f in extracted.keys() if any('\u4e00' <= c <= '\u9fff' for c in f)]
        assert len(chinese_files) > 0
        # Verify specific files exist
        assert any('哪' in f for f in extracted.keys())

    def test_extracted_file_size(self, reader, temp_output_dir):
        """Test that extracted files have reasonable sizes."""
        extracted = reader.extract_audio_files(temp_output_dir)
        for path in extracted.values():
            size = os.path.getsize(path)
            # Audio files should be between 1KB and 100KB
            assert 1000 < size < 100000

    def test_extract_count_matches_statistics(self, reader, temp_output_dir):
        """Test that extracted count matches statistics."""
        stats = reader.get_audio_statistics()
        extracted = reader.extract_audio_files(temp_output_dir, audio_only=True)
        assert len(extracted) == stats['audio_files']


class TestCardAudioLinking:
    """Tests for get_audio_for_card() method."""

    def test_get_audio_for_card_returns_list(self, reader):
        """Test that get_audio_for_card returns a list."""
        cards = reader.get_cards()
        models = reader.get_models()
        if cards:
            audio = reader.get_audio_for_card(cards[0], models)
            assert isinstance(audio, list)

    def test_cards_have_audio(self, reader):
        """Test that cards contain audio references."""
        cards = reader.get_cards()
        models = reader.get_models()
        cards_with_audio = 0
        for card in cards[:10]:
            audio = reader.get_audio_for_card(card, models)
            if audio:
                cards_with_audio += 1
        assert cards_with_audio > 0

    def test_audio_filenames_are_strings(self, reader):
        """Test that audio filenames are strings."""
        cards = reader.get_cards()
        models = reader.get_models()
        for card in cards[:10]:
            audio = reader.get_audio_for_card(card, models)
            for filename in audio:
                assert isinstance(filename, str)
                assert filename.endswith(('.mp3', '.wav', '.ogg'))

    def test_audio_references_extractable(self, reader):
        """Test that audio references match extractable files."""
        cards = reader.get_cards()
        models = reader.get_models()
        mapping = reader.get_media_mapping()

        for card in cards[:10]:
            audio = reader.get_audio_for_card(card, models)
            for filename in audio:
                # Audio filename should be in the media mapping
                assert filename in mapping.values()

    def test_all_cards_have_audio(self, reader):
        """Test that all cards in this deck have audio (specific to test deck)."""
        cards = reader.get_cards()
        models = reader.get_models()

        cards_with_audio = 0
        for card in cards:
            audio = reader.get_audio_for_card(card, models)
            if audio:
                cards_with_audio += 1

        # A_Course_in_Contemporary_Chinese_1.apkg has 576 cards all with audio
        assert cards_with_audio == 576

    def test_sound_tag_parsing(self, reader):
        """Test that [sound:...] tags are correctly parsed."""
        cards = reader.get_cards()
        models = reader.get_models()

        # Find a card with audio
        for card in cards:
            if '[sound:' in card['flds']:
                audio = reader.get_audio_for_card(card, models)
                assert len(audio) > 0
                # The filename should not include the [sound:] wrapper
                for filename in audio:
                    assert not filename.startswith('[sound:')
                    assert not filename.endswith(']')
                break


class TestIntegration:
    """Integration tests for complete audio extraction workflow."""

    def test_complete_workflow(self, reader, temp_output_dir):
        """Test the complete audio extraction workflow."""
        # Step 1: Get statistics
        stats = reader.get_audio_statistics()
        assert stats['audio_files'] > 0

        # Step 2: Extract audio
        extracted = reader.extract_audio_files(temp_output_dir)
        assert len(extracted) == stats['audio_files']

        # Step 3: Link cards to audio
        cards = reader.get_cards()
        models = reader.get_models()

        # Collect all audio files referenced by cards
        referenced_audio = set()
        for card in cards:
            audio = reader.get_audio_for_card(card, models)
            referenced_audio.update(audio)

        # Get media mapping to check what's available
        mapping = reader.get_media_mapping()
        available_audio = set(mapping.values())

        # Most referenced audio should be in extracted files
        # (Some decks may have missing media files)
        found_count = sum(1 for audio_file in referenced_audio if audio_file in extracted)
        coverage = found_count / len(referenced_audio) if referenced_audio else 0
        assert coverage > 0.95  # At least 95% coverage

        # All extracted files should be accessible
        for audio_file in extracted.keys():
            assert os.path.exists(extracted[audio_file])

    def test_no_duplicate_extraction(self, reader, temp_output_dir):
        """Test that same audio file is not duplicated."""
        extracted = reader.extract_audio_files(temp_output_dir)

        # Check that all filenames are unique
        filenames = list(extracted.keys())
        assert len(filenames) == len(set(filenames))

    def test_idempotent_extraction(self, reader, temp_output_dir):
        """Test that extracting twice produces same results."""
        # First extraction
        extracted1 = reader.extract_audio_files(temp_output_dir)

        # Second extraction (should overwrite)
        extracted2 = reader.extract_audio_files(temp_output_dir)

        # Results should be identical
        assert extracted1.keys() == extracted2.keys()
        for filename in extracted1.keys():
            assert os.path.getsize(extracted1[filename]) == os.path.getsize(extracted2[filename])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
