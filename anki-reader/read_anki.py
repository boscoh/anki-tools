#!/usr/bin/env python3
"""
Read and display cards from an Anki package (.apkg) file.
"""

import sqlite3
import zipfile
import tempfile
import os
import json
import shutil
import re
from pathlib import Path


class AnkiReader:
    def __init__(self, apkg_path):
        self.apkg_path = apkg_path
        self.temp_dir = None
        self.conn = None

    def __enter__(self):
        self.temp_dir = tempfile.mkdtemp()

        # Extract the .apkg file (it's a ZIP file)
        with zipfile.ZipFile(self.apkg_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)

        # Connect to the SQLite database
        db_path = os.path.join(self.temp_dir, 'collection.anki2')
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()

        # Clean up temp directory
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)

    def get_decks(self):
        """Get all decks in the collection."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT decks FROM col")
        decks_json = cursor.fetchone()[0]
        decks = json.loads(decks_json)
        return decks

    def get_models(self):
        """Get all note models/templates."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT models FROM col")
        models_json = cursor.fetchone()[0]
        models = json.loads(models_json)
        return models

    def get_notes(self):
        """Get all notes from the collection."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, mid, flds, tags FROM notes")
        return cursor.fetchall()

    def get_cards(self):
        """Get all cards with their note information."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT cards.id, cards.nid, cards.did, cards.ord,
                   notes.flds, notes.tags, notes.mid
            FROM cards
            JOIN notes ON cards.nid = notes.id
        """)
        return cursor.fetchall()

    def parse_card(self, card, models, decks):
        """Parse a card into a readable format."""
        model_id = str(card['mid'])
        model = models.get(model_id, {})
        model_name = model.get('name', 'Unknown')

        # Get field names from the model
        fields = model.get('flds', [])
        field_names = [f['name'] for f in fields]

        # Parse the field values (separated by \x1f)
        field_values = card['flds'].split('\x1f')

        # Create a dictionary of field name -> value
        card_data = dict(zip(field_names, field_values))

        # Get deck name
        deck_id = str(card['did'])
        deck = decks.get(deck_id, {})
        deck_name = deck.get('name', 'Unknown')

        return {
            'card_id': card['id'],
            'note_id': card['nid'],
            'deck': deck_name,
            'model': model_name,
            'fields': card_data,
            'tags': card['tags']
        }

    def get_media_mapping(self):
        """
        Get mapping of file IDs to filenames from the media JSON file.

        Returns:
            Dict mapping numeric file IDs (as strings) to actual filenames.
            Returns empty dict if media file doesn't exist.
        """
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
            card: Card row from database (with 'flds' and 'mid' fields)
            models: Models dict from get_models()

        Returns:
            List of audio filenames referenced in card (e.g., ['哪.mp3'])
        """
        audio_pattern = r'\[sound:(.*?)\]'

        # Get all field values
        field_values = card['flds'].split('\x1f')

        audio_files = []
        for field_value in field_values:
            matches = re.findall(audio_pattern, field_value)
            audio_files.extend(matches)

        return audio_files

    def get_audio_statistics(self):
        """
        Get statistics about audio files in the package.

        Returns:
            Dict with counts of total media, audio files, formats, etc.
        """
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


def main():
    # Find the .apkg file in the parent directory
    apkg_files = list(Path('..').glob('*.apkg'))

    if not apkg_files:
        print("No .apkg files found in parent directory")
        return

    apkg_path = apkg_files[0]
    print(f"Reading: {apkg_path.name}\n")

    with AnkiReader(str(apkg_path)) as reader:
        # Get metadata
        decks = reader.get_decks()
        models = reader.get_models()

        print("=" * 60)
        print("DECKS:")
        print("=" * 60)
        for deck_id, deck_info in decks.items():
            print(f"- {deck_info.get('name', 'Unknown')}")

        print("\n" + "=" * 60)
        print("NOTE TYPES:")
        print("=" * 60)
        for model_id, model_info in models.items():
            print(f"- {model_info.get('name', 'Unknown')}")

        # Get all cards
        cards = reader.get_cards()
        print(f"\n" + "=" * 60)
        print(f"CARDS (Total: {len(cards)})")
        print("=" * 60)

        # Display first 10 cards as examples
        for i, card in enumerate(cards[:10], 1):
            parsed = reader.parse_card(card, models, decks)
            print(f"\nCard {i}:")
            print(f"  Deck: {parsed['deck']}")
            print(f"  Type: {parsed['model']}")
            print(f"  Fields:")
            for field_name, field_value in parsed['fields'].items():
                # Clean HTML tags for display
                import re
                clean_value = re.sub('<[^<]+?>', '', field_value)
                print(f"    {field_name}: {clean_value[:100]}")
            if parsed['tags']:
                print(f"  Tags: {parsed['tags']}")

        if len(cards) > 10:
            print(f"\n... and {len(cards) - 10} more cards")


if __name__ == '__main__':
    main()
