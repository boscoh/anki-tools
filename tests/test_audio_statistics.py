"""
Tests for get_audio_statistics() method.
"""



class TestAudioStatistics:
    """Tests for get_audio_statistics() method."""

    def test_get_audio_statistics_returns_dict(self, pkg):
        """Test that get_audio_statistics returns a dictionary."""
        stats = pkg.get_audio_statistics()
        assert isinstance(stats, dict)

    def test_statistics_has_required_fields(self, pkg):
        """Test that statistics contains required fields."""
        stats = pkg.get_audio_statistics()
        assert 'total_media_files' in stats
        assert 'audio_files' in stats
        assert 'image_files' in stats
        assert 'audio_formats' in stats

    def test_statistics_counts_are_consistent(self, pkg):
        """Test that audio + image counts match total."""
        stats = pkg.get_audio_statistics()
        assert stats['audio_files'] + stats['image_files'] == stats['total_media_files']

    def test_audio_formats_is_dict(self, pkg):
        """Test that audio_formats is a dictionary."""
        stats = pkg.get_audio_statistics()
        assert isinstance(stats['audio_formats'], dict)

    def test_statistics_expected_values(self, pkg):
        """Test expected statistics for A_Course_in_Contemporary_Chinese_1.apkg."""
        stats = pkg.get_audio_statistics()
        # This deck has 538 audio files, all MP3
        assert stats['audio_files'] == 538
        assert stats['total_media_files'] == 538
        assert 'mp3' in stats['audio_formats']
        assert stats['audio_formats']['mp3'] == 538
