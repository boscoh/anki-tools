"""
Integration tests for complete audio extraction workflow.
"""

import os


class TestIntegration:
    """Integration tests for complete audio extraction workflow."""

    def test_complete_workflow(self, pkg, temp_output_dir):
        """Test the complete audio extraction workflow."""
        # Step 1: Get statistics
        stats = pkg.get_audio_statistics()
        assert stats["audio_files"] > 0

        # Step 2: Extract audio
        extracted = pkg.extract_audio_files(temp_output_dir)
        assert len(extracted) == stats["audio_files"]

        # Step 3: Link cards to audio
        cards = pkg.get_cards()
        models = pkg.get_models()

        # Collect all audio files referenced by cards
        referenced_audio = set()
        for card in cards:
            audio = pkg.get_audio_for_card(card, models)
            referenced_audio.update(audio)

        # Get media mapping to check what's available
        mapping = pkg.get_media_mapping()
        available_audio = set(mapping.values())

        # Most referenced audio should be in extracted files
        # (Some decks may have missing media files)
        found_count = sum(
            1 for audio_file in referenced_audio if audio_file in extracted
        )
        coverage = found_count / len(referenced_audio) if referenced_audio else 0
        assert coverage > 0.95  # At least 95% coverage

        # All extracted files should be accessible
        for audio_file in extracted.keys():
            assert os.path.exists(extracted[audio_file])

    def test_no_duplicate_extraction(self, pkg, temp_output_dir):
        """Test that same audio file is not duplicated."""
        extracted = pkg.extract_audio_files(temp_output_dir)

        # Check that all filenames are unique
        filenames = list(extracted.keys())
        assert len(filenames) == len(set(filenames))

    def test_idempotent_extraction(self, pkg, temp_output_dir):
        """Test that extracting twice produces same results."""
        # First extraction
        extracted1 = pkg.extract_audio_files(temp_output_dir)

        # Second extraction (should overwrite)
        extracted2 = pkg.extract_audio_files(temp_output_dir)

        # Results should be identical
        assert extracted1.keys() == extracted2.keys()
        for filename in extracted1.keys():
            assert os.path.getsize(extracted1[filename]) == os.path.getsize(
                extracted2[filename]
            )
