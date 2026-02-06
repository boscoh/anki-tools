"""
Tests for extract_audio_files() method.
"""

import os


class TestAudioExtraction:
    """Tests for extract_audio_files() method."""

    def test_extract_audio_files_creates_directory(self, pkg, temp_output_dir):
        """Test that extract_audio_files creates the output directory."""
        output_dir = os.path.join(temp_output_dir, 'audio')
        pkg.extract_audio_files(output_dir)
        assert os.path.exists(output_dir)
        assert os.path.isdir(output_dir)

    def test_extract_audio_files_returns_dict(self, pkg, temp_output_dir):
        """Test that extract_audio_files returns a dictionary."""
        extracted = pkg.extract_audio_files(temp_output_dir)
        assert isinstance(extracted, dict)

    def test_extracted_files_exist(self, pkg, temp_output_dir):
        """Test that extracted files actually exist on disk."""
        extracted = pkg.extract_audio_files(temp_output_dir)
        for filename, path in extracted.items():
            assert os.path.exists(path)
            assert os.path.isfile(path)

    def test_extracted_filenames_match(self, pkg, temp_output_dir):
        """Test that extracted filenames match the mapping."""
        extracted = pkg.extract_audio_files(temp_output_dir)
        for filename, path in extracted.items():
            assert os.path.basename(path) == filename

    def test_audio_only_filters_correctly(self, pkg, temp_output_dir):
        """Test that audio_only=True filters non-audio files."""
        extracted = pkg.extract_audio_files(temp_output_dir, audio_only=True)
        for filename in extracted.keys():
            assert filename.endswith(('.mp3', '.wav', '.ogg'))

    def test_extract_all_media(self, pkg, temp_output_dir):
        """Test that audio_only=False extracts all media."""
        extracted = pkg.extract_audio_files(temp_output_dir, audio_only=False)
        stats = pkg.get_audio_statistics()
        assert len(extracted) == stats['total_media_files']

    def test_chinese_character_filenames(self, pkg, temp_output_dir):
        """Test that Chinese character filenames are preserved."""
        extracted = pkg.extract_audio_files(temp_output_dir)
        # Check for some known Chinese filenames
        chinese_files = [f for f in extracted.keys() if any('\u4e00' <= c <= '\u9fff' for c in f)]
        assert len(chinese_files) > 0
        # Verify specific files exist
        assert any('哪' in f for f in extracted.keys())

    def test_extracted_file_size(self, pkg, temp_output_dir):
        """Test that extracted files have reasonable sizes."""
        extracted = pkg.extract_audio_files(temp_output_dir)
        for path in extracted.values():
            size = os.path.getsize(path)
            # Audio files should be between 1KB and 100KB
            assert 1000 < size < 100000

    def test_extract_count_matches_statistics(self, pkg, temp_output_dir):
        """Test that extracted count matches statistics."""
        stats = pkg.get_audio_statistics()
        extracted = pkg.extract_audio_files(temp_output_dir, audio_only=True)
        assert len(extracted) == stats['audio_files']
