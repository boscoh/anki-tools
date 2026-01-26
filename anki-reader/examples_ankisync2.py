#!/usr/bin/env python3
"""
Example: Reading Anki cards using ankisync2 library
Install: uv add ankisync2
"""

from pathlib import Path


def example_ankisync2():
    """
    ankisync2 uses Peewee ORM to safely work with .apkg files.
    """
    try:
        from ankisync2 import Apkg

        # Find .apkg file
        apkg_files = list(Path('..').glob('*.apkg'))
        if not apkg_files:
            print("No .apkg files found in parent directory")
            return

        apkg_path = apkg_files[0]
        print("=" * 60)
        print("ANKISYNC2 EXAMPLE")
        print("=" * 60)
        print(f"\nReading: {apkg_path.name}\n")

        with Apkg(str(apkg_path)) as apkg:
            # Access database tables via ORM
            print("Database structure:")
            print(f"  apkg.db.Notes: {apkg.db.Notes}")
            print(f"  apkg.db.Cards: {apkg.db.Cards}")
            print(f"  apkg.db.Decks: {apkg.db.Decks}")
            print(f"  apkg.db.Models: {apkg.db.Models}")

            # Count cards
            card_count = apkg.db.Cards.select().count()
            print(f"\nTotal cards: {card_count}")

            # Count notes
            note_count = apkg.db.Notes.select().count()
            print(f"Total notes: {note_count}")

            # List decks
            print("\nDecks:")
            for deck in apkg.db.Decks.select():
                print(f"  - {deck.name}")

            # List models
            print("\nModels:")
            for model in apkg.db.Models.select():
                print(f"  - {model.name}")
                # Fields are stored as JSON array
                if hasattr(model, 'flds'):
                    print(f"    Fields: {model.flds}")

            # Iterate through first 5 notes
            print("\nFirst 5 notes:")
            for i, note in enumerate(apkg.db.Notes.select().limit(5), 1):
                print(f"\nNote {i}:")
                print(f"  ID: {note.id}")
                print(f"  Model ID: {note.mid}")
                # Fields are separated by \x1f
                fields = note.flds.split('\x1f') if hasattr(note, 'flds') else []
                for j, field in enumerate(fields[:3], 1):  # Show first 3 fields
                    # Clean HTML
                    import re
                    clean = re.sub('<[^<]+?>', '', field)
                    print(f"  Field {j}: {clean[:80]}")
                if hasattr(note, 'tags'):
                    print(f"  Tags: {note.tags}")

            # Iterate through cards (alternative syntax)
            print("\n" + "=" * 60)
            print("Iterating through cards (first 3):")
            for i, card in enumerate(apkg):
                if i >= 3:
                    break
                print(f"\nCard {i + 1}:")
                print(f"  Card ID: {card.id}")
                print(f"  Note ID: {card.nid}")
                print(f"  Deck ID: {card.did}")
                print(f"  Order: {card.ord}")

    except ImportError as e:
        print(f"ankisync2 not installed. Install with: uv add ankisync2")
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    example_ankisync2()
