#!/usr/bin/env python3
"""
Demonstration of audio extraction from Anki .apkg files.

This script shows how to:
1. Get audio statistics from a package
2. Extract all audio files
3. Link cards to their audio files
"""

from pathlib import Path
from read_anki import AnkiReader


def main():
    # Find the .apkg file in the parent directory
    apkg_files = list(Path('..').glob('*.apkg'))

    if not apkg_files:
        print("No .apkg files found in parent directory")
        return

    apkg_path = apkg_files[0]
    print(f"Reading: {apkg_path.name}\n")

    with AnkiReader(str(apkg_path)) as reader:
        # Step 1: Get audio statistics
        print("=" * 60)
        print("AUDIO STATISTICS")
        print("=" * 60)
        stats = reader.get_audio_statistics()
        print(f"Total media files: {stats['total_media_files']}")
        print(f"Audio files: {stats['audio_files']}")
        print(f"Image files: {stats['image_files']}")
        print(f"Audio formats: {stats['audio_formats']}")

        # Step 2: Extract all audio files
        print(f"\n" + "=" * 60)
        print("EXTRACTING AUDIO FILES")
        print("=" * 60)
        output_dir = './extracted_audio'
        extracted = reader.extract_audio_files(output_dir)
        print(f"Extracted {len(extracted)} audio files to: {output_dir}")

        # Show first 10 extracted files
        print("\nFirst 10 audio files:")
        for i, (filename, path) in enumerate(list(extracted.items())[:10], 1):
            print(f"  {i}. {filename}")

        # Step 3: Link cards to their audio
        print(f"\n" + "=" * 60)
        print("CARDS WITH AUDIO")
        print("=" * 60)
        cards = reader.get_cards()
        models = reader.get_models()
        decks = reader.get_decks()

        cards_with_audio = []
        for card in cards:
            audio = reader.get_audio_for_card(card, models)
            if audio:
                cards_with_audio.append((card, audio))

        print(f"Found {len(cards_with_audio)} cards with audio\n")

        # Show first 10 cards with audio
        print("First 10 cards with audio:")
        for i, (card, audio) in enumerate(cards_with_audio[:10], 1):
            parsed = reader.parse_card(card, models, decks)
            print(f"\nCard {i}:")

            # Get the Hanzi field (first field typically)
            fields = parsed['fields']
            if fields:
                first_field = list(fields.values())[0]
                # Clean HTML
                import re
                clean_value = re.sub('<[^<]+?>', '', first_field)
                print(f"  Content: {clean_value[:50]}")

            print(f"  Audio: {', '.join(audio)}")

        if len(cards_with_audio) > 10:
            print(f"\n... and {len(cards_with_audio) - 10} more cards with audio")

        print(f"\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total cards: {len(cards)}")
        print(f"Cards with audio: {len(cards_with_audio)}")
        print(f"Audio extraction: {output_dir}")
        print(f"Extracted files: {len(extracted)}")


if __name__ == '__main__':
    main()
