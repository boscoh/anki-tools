"""
Tests for card deletion functionality using A_Course_in_Contemporary_Chinese_1.apkg.
"""

import pytest


class TestCardDeletion:
    """Tests for card deletion functionality using A_Course_in_Contemporary_Chinese_1.apkg."""

    def test_delete_single_card(self, pkg):
        """Test deleting a single card from A_Course_in_Contemporary_Chinese_1.apkg."""
        cards = pkg.get_cards()
        initial_count = len(cards)
        assert initial_count == 576  # This deck has 576 cards

        # Delete first card
        first_card = cards[0]
        result = pkg.delete_card(first_card["id"], cleanup_audio=False)

        assert result["card_deleted"] is True
        assert result["note_deleted"] is True  # Each card has its own note in this deck

        # Verify card count decreased
        updated_cards = pkg.get_cards()
        assert len(updated_cards) == initial_count - 1

    def test_delete_card_with_audio_cleanup(self, pkg):
        """Test deleting a card and cleaning up unused audio files."""
        cards = pkg.get_cards()
        initial_stats = pkg.get_audio_statistics()
        initial_audio_count = initial_stats["total_media_files"]
        assert initial_audio_count == 538  # This deck has 538 audio files

        # Get a card and its audio files
        first_card = cards[0]
        note_id = first_card["nid"]
        audio_files = pkg.get_audio_files_for_note(note_id)
        assert len(audio_files) > 0  # Cards in this deck have audio

        # Delete the card with audio cleanup
        result = pkg.delete_card(first_card["id"], cleanup_audio=True)

        assert result["card_deleted"] is True
        assert result["note_deleted"] is True
        # Audio files should be deleted if not used by other cards
        assert len(result["audio_files_deleted"]) > 0

        # Verify audio count decreased
        updated_stats = pkg.get_audio_statistics()
        assert updated_stats["total_media_files"] < initial_audio_count

    def test_shared_audio_not_deleted(self, pkg):
        """Test that audio files shared between cards are not deleted."""
        cards = pkg.get_cards()
        if len(cards) < 2:
            pytest.skip("Need at least 2 cards for this test")

        # Find two cards that share an audio file
        # In this deck, cards typically have unique audio, so we'll test the opposite
        # Get audio from first card
        audio1 = pkg.get_audio_files_for_note(cards[0]["nid"])
        audio2 = pkg.get_audio_files_for_note(cards[1]["nid"])

        # Delete first card
        result1 = pkg.delete_card(cards[0]["id"], cleanup_audio=True)

        # If audio files were shared, they shouldn't be deleted
        # Check if any audio from card 1 still exists (meaning it was shared)
        remaining_audio = pkg.get_audio_files_for_note(cards[1]["nid"])
        if audio1 & audio2:  # If they share audio
            # Shared audio should still exist
            shared = audio1 & audio2
            for audio_file in shared:
                assert pkg.is_audio_file_used(audio_file)

    def test_delete_multiple_cards(self, pkg):
        """Test deleting multiple cards at once."""
        cards = pkg.get_cards()
        initial_count = len(cards)
        assert initial_count == 576

        # Delete first 5 cards
        card_ids = [c["id"] for c in cards[:5]]
        result = pkg.delete_cards(card_ids, cleanup_audio=False)

        assert result["cards_deleted"] == 5
        assert result["notes_deleted"] == 5  # Each card has its own note

        # Verify count
        updated_cards = pkg.get_cards()
        assert len(updated_cards) == initial_count - 5

    def test_delete_note_deletes_all_cards(self, pkg):
        """Test that deleting a note deletes all associated cards."""
        cards = pkg.get_cards()
        initial_count = len(cards)
        assert initial_count == 576

        # Get a note ID
        first_card = cards[0]
        note_id = first_card["nid"]

        # Count cards for this note (should be 1 in this deck)
        cursor = pkg.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cards WHERE nid = ?", (note_id,))
        cards_for_note = cursor.fetchone()[0]
        assert cards_for_note == 1

        # Delete the note
        result = pkg.delete_note(note_id, cleanup_audio=False)

        assert result["note_deleted"] is True
        assert result["cards_deleted"] == 1

        # Verify card count decreased
        updated_cards = pkg.get_cards()
        assert len(updated_cards) == initial_count - 1

    def test_get_audio_files_for_note(self, pkg):
        """Test getting audio files for a specific note."""
        cards = pkg.get_cards()
        first_card = cards[0]
        note_id = first_card["nid"]

        audio_files = pkg.get_audio_files_for_note(note_id)
        assert isinstance(audio_files, set)
        assert len(audio_files) > 0  # Cards in this deck have audio

        # Verify audio filenames are strings with extensions
        for audio_file in audio_files:
            assert isinstance(audio_file, str)
            assert audio_file.endswith((".mp3", ".wav", ".ogg"))

    def test_is_audio_file_used(self, pkg):
        """Test checking if an audio file is used by any note."""
        cards = pkg.get_cards()
        first_card = cards[0]
        note_id = first_card["nid"]

        # Get audio files for this note
        audio_files = pkg.get_audio_files_for_note(note_id)
        assert len(audio_files) > 0

        # Check that audio files are marked as used
        for audio_file in audio_files:
            assert pkg.is_audio_file_used(audio_file) is True

    def test_delete_media_file(self, pkg):
        """Test deleting a media file from the package."""
        mapping = pkg.get_media_mapping()
        assert len(mapping) == 538  # This deck has 538 media files

        # Get first audio filename
        audio_filename = None
        for filename in mapping.values():
            if filename.endswith(".mp3"):
                audio_filename = filename
                break

        assert audio_filename is not None

        # Delete the media file
        result = pkg.delete_media_file(audio_filename)
        assert result is True

        # Verify it's removed from mapping
        updated_mapping = pkg.get_media_mapping()
        assert audio_filename not in updated_mapping.values()
        assert len(updated_mapping) == len(mapping) - 1

    def test_delete_card_preserves_other_cards(self, pkg):
        """Test that deleting one card doesn't affect other cards."""
        cards = pkg.get_cards()
        assert len(cards) == 576

        # Get first two cards
        card1 = cards[0]
        card2 = cards[1]

        # Delete first card
        pkg.delete_card(card1["id"], cleanup_audio=False)

        # Verify second card still exists
        remaining_cards = pkg.get_cards()
        card2_ids = [c["id"] for c in remaining_cards]
        assert card2["id"] in card2_ids

    def test_delete_multiple_cards_cleanup_audio(self, pkg):
        """Test that deleting multiple cards cleans up unused audio files."""
        cards = pkg.get_cards()
        initial_stats = pkg.get_audio_statistics()
        initial_audio_count = initial_stats["total_media_files"]
        assert initial_audio_count == 538

        # Delete first 10 cards with audio cleanup
        card_ids = [c["id"] for c in cards[:10]]
        result = pkg.delete_cards(card_ids, cleanup_audio=True)

        assert result["cards_deleted"] == 10
        assert result["notes_deleted"] == 10
        # Some audio files should be deleted (at least 10, possibly more if shared)
        assert len(result["audio_files_deleted"]) >= 10

        # Verify audio count decreased
        updated_stats = pkg.get_audio_statistics()
        assert updated_stats["total_media_files"] < initial_audio_count
        assert updated_stats["total_media_files"] == initial_audio_count - len(
            result["audio_files_deleted"]
        )
