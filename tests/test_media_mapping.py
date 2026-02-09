"""
Tests for get_media_mapping() method.
"""


class TestMediaMapping:
    """Tests for get_media_mapping() method."""

    def test_get_media_mapping_returns_dict(self, pkg):
        """Test that get_media_mapping returns a dictionary."""
        mapping = pkg.get_media_mapping()
        assert isinstance(mapping, dict)

    def test_media_mapping_not_empty(self, pkg):
        """Test that media mapping contains entries."""
        mapping = pkg.get_media_mapping()
        assert len(mapping) > 0

    def test_media_mapping_structure(self, pkg):
        """Test that media mapping has correct structure (ID -> filename)."""
        mapping = pkg.get_media_mapping()
        # Keys should be numeric strings
        for file_id, filename in mapping.items():
            assert isinstance(file_id, str)
            assert isinstance(filename, str)
            # Filename should have an extension
            assert "." in filename

    def test_media_mapping_contains_audio(self, pkg):
        """Test that media mapping contains audio files."""
        mapping = pkg.get_media_mapping()
        audio_files = [
            f for f in mapping.values() if f.endswith((".mp3", ".wav", ".ogg"))
        ]
        assert len(audio_files) > 0
