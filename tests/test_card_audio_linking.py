"""
Tests for get_audio_for_card() method.
"""



class TestCardAudioLinking:
    """Tests for get_audio_for_card() method."""

    def test_get_audio_for_card_returns_list(self, pkg):
        """Test that get_audio_for_card returns a list."""
        cards = pkg.get_cards()
        models = pkg.get_models()
        if cards:
            audio = pkg.get_audio_for_card(cards[0], models)
            assert isinstance(audio, list)

    def test_cards_have_audio(self, pkg):
        """Test that cards contain audio references."""
        cards = pkg.get_cards()
        models = pkg.get_models()
        cards_with_audio = 0
        for card in cards[:10]:
            audio = pkg.get_audio_for_card(card, models)
            if audio:
                cards_with_audio += 1
        assert cards_with_audio > 0

    def test_audio_filenames_are_strings(self, pkg):
        """Test that audio filenames are strings."""
        cards = pkg.get_cards()
        models = pkg.get_models()
        for card in cards[:10]:
            audio = pkg.get_audio_for_card(card, models)
            for filename in audio:
                assert isinstance(filename, str)
                assert filename.endswith(('.mp3', '.wav', '.ogg'))

    def test_audio_references_extractable(self, pkg):
        """Test that audio references match extractable files."""
        cards = pkg.get_cards()
        models = pkg.get_models()
        mapping = pkg.get_media_mapping()

        for card in cards[:10]:
            audio = pkg.get_audio_for_card(card, models)
            for filename in audio:
                # Audio filename should be in the media mapping
                assert filename in mapping.values()

    def test_all_cards_have_audio(self, pkg):
        """Test that all cards in this deck have audio (specific to A_Course_in_Contemporary_Chinese_1.apkg)."""
        cards = pkg.get_cards()
        models = pkg.get_models()

        cards_with_audio = 0
        for card in cards:
            audio = pkg.get_audio_for_card(card, models)
            if audio:
                cards_with_audio += 1

        # A_Course_in_Contemporary_Chinese_1.apkg has 576 cards all with audio
        assert cards_with_audio == 576

    def test_sound_tag_parsing(self, pkg):
        """Test that [sound:...] tags are correctly parsed."""
        cards = pkg.get_cards()
        models = pkg.get_models()

        # Find a card with audio
        for card in cards:
            if '[sound:' in card['flds']:
                audio = pkg.get_audio_for_card(card, models)
                assert len(audio) > 0
                # The filename should not include the [sound:] wrapper
                for filename in audio:
                    assert not filename.startswith('[sound:')
                    assert not filename.endswith(']')
                break
