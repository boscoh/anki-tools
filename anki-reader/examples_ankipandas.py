#!/usr/bin/env python3
"""
Example: Reading Anki cards using ankipandas library
Install: uv add ankipandas
"""

def example_ankipandas():
    """
    ankipandas provides pandas DataFrames for easy analysis.
    Note: This reads from your Anki user directory, not from .apkg files directly.
    """
    try:
        from ankipandas import Collection

        # Load collection (finds Anki database automatically)
        col = Collection()

        print("=" * 60)
        print("ANKIPANDAS EXAMPLE")
        print("=" * 60)

        # Access notes as DataFrame
        notes = col.notes
        print(f"\nTotal notes: {len(notes)}")
        print(f"\nNotes columns: {list(notes.columns)}")

        # Access cards as DataFrame
        cards = col.cards
        print(f"\nTotal cards: {len(cards)}")
        print(f"\nCards columns: {list(cards.columns)}")

        # Example: Merge cards with notes to get full information
        merged = cards.merge_notes()
        print(f"\nMerged data shape: {merged.shape}")

        # Example: Filter by deck
        if 'cdeck' in cards.columns:
            deck_counts = cards['cdeck'].value_counts()
            print("\nCards per deck:")
            print(deck_counts)

        # Example: Get notes with specific fields
        notes_with_fields = notes.fields_as_columns()
        print(f"\nNotes with expanded fields: {list(notes_with_fields.columns)[:10]}")

        # Example: Statistics
        if 'creps' in cards.columns:
            print("\nReview statistics:")
            print(cards['creps'].describe())

        # Example: Find cards with tags
        if 'ntags' in merged.columns:
            tagged = merged[merged['ntags'].str.len() > 0]
            print(f"\nCards with tags: {len(tagged)}")

    except ImportError:
        print("ankipandas not installed. Install with: uv add ankipandas")
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: ankipandas reads from Anki's user directory,")
        print("not from standalone .apkg files. You need to import")
        print("the .apkg into Anki first, or use a different library.")


if __name__ == '__main__':
    example_ankipandas()
