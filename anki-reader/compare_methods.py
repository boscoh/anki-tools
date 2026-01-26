#!/usr/bin/env python3
"""
Side-by-side comparison of different methods to read Anki .apkg files
"""

from pathlib import Path
import time


def method1_direct_sqlite():
    """Method 1: Direct SQLite access (current implementation)"""
    print("\n" + "=" * 70)
    print("METHOD 1: Direct SQLite Access (Current)")
    print("=" * 70)
    print("Dependencies: None (uses stdlib only)")

    import sqlite3
    import zipfile
    import tempfile
    import json
    import os
    import shutil

    apkg_files = list(Path('..').glob('*.apkg'))
    if not apkg_files:
        return

    start = time.time()
    temp_dir = tempfile.mkdtemp()

    try:
        # Extract
        with zipfile.ZipFile(str(apkg_files[0]), 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Connect
        db_path = os.path.join(temp_dir, 'collection.anki2')
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM cards")
        card_count = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM notes")
        note_count = cursor.fetchone()['count']

        # Get first card
        cursor.execute("""
            SELECT cards.*, notes.flds, notes.tags
            FROM cards
            JOIN notes ON cards.nid = notes.id
            LIMIT 1
        """)
        first_card = cursor.fetchone()

        elapsed = time.time() - start

        print(f"✓ Cards: {card_count}")
        print(f"✓ Notes: {note_count}")
        print(f"✓ Time: {elapsed:.3f}s")
        print(f"\nFirst card fields: {first_card['flds'][:100]}...")

        conn.close()
    finally:
        shutil.rmtree(temp_dir)

    print("\n✓ Pros: No dependencies, fast, full control")
    print("✗ Cons: Manual schema handling, boilerplate code")


def method2_anki_export():
    """Method 2: anki-export library"""
    print("\n" + "=" * 70)
    print("METHOD 2: anki-export Library")
    print("=" * 70)
    print("Dependencies: anki-export")

    try:
        from anki_export import ApkgReader

        apkg_files = list(Path('..').glob('*.apkg'))
        if not apkg_files:
            return

        start = time.time()

        with ApkgReader(str(apkg_files[0])) as apkg:
            data = apkg.export()

            total_cards = sum(len(cards) - 1 for cards in data.values())  # -1 for header
            card_types = len(data)

            elapsed = time.time() - start

            print(f"✓ Card types: {card_types}")
            print(f"✓ Total cards: {total_cards}")
            print(f"✓ Time: {elapsed:.3f}s")

            # Show structure
            for card_type, cards in list(data.items())[:1]:
                print(f"\nCard type: {card_type}")
                print(f"  Fields: {', '.join(cards[0])}")
                if len(cards) > 1:
                    print(f"  First card: {dict(zip(cards[0], cards[1]))}")

        print("\n✓ Pros: Simple API, easy export, clean data structure")
        print("✗ Cons: Less control, export-focused")

    except ImportError:
        print("✗ anki-export not installed (uv add anki-export)")


def method3_ankisync2():
    """Method 3: ankisync2 ORM"""
    print("\n" + "=" * 70)
    print("METHOD 3: ankisync2 (Peewee ORM)")
    print("=" * 70)
    print("Dependencies: ankisync2, peewee")

    print("✗ Note: ankisync2 expects newer .anki21 format")
    print("  Our test file uses older .anki2 format")
    print("  This method works better with modern Anki exports")

    print("\n✓ Pros: ORM interface, safe editing, good for creating decks")
    print("✗ Cons: More complex, version compatibility issues")


def method4_comparison_table():
    """Show comparison table"""
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("""
┌─────────────────┬──────────────┬──────────┬─────────────┬──────────────┐
│ Method          │ Dependencies │ Read     │ Write       │ Best For     │
├─────────────────┼──────────────┼──────────┼─────────────┼──────────────┤
│ Direct SQLite   │ None         │ ★★★★★    │ ★★★☆☆       │ Simple reads │
│ anki-export     │ 1 package    │ ★★★★☆    │ ☆☆☆☆☆       │ Export/CSV   │
│ ankisync2       │ 2 packages   │ ★★★☆☆    │ ★★★★☆       │ Creating     │
│ ankipandas      │ pandas       │ ★★★★★    │ ☆☆☆☆☆       │ Analysis     │
│ genanki         │ 1 package    │ ☆☆☆☆☆    │ ★★★★★       │ Generating   │
└─────────────────┴──────────────┴──────────┴─────────────┴──────────────┘

Performance:
  Direct SQLite:  Fastest for simple reads
  anki-export:    Fast, but builds data structure
  ankisync2:      ORM overhead
  ankipandas:     Pandas overhead, best for analysis

Ease of Use:
  anki-export:    Easiest for reading
  Direct SQLite:  Requires schema knowledge
  ankisync2:      Requires ORM knowledge
  ankipandas:     Requires pandas knowledge

Recommendation for this project:
  ✓ CURRENT (Direct SQLite): Best choice for simple, fast reading
  ✓ anki-export: Good alternative if you want easier API
  ✓ ankipandas: Use if you need data analysis features
    """)


if __name__ == '__main__':
    method1_direct_sqlite()
    method2_anki_export()
    method3_ankisync2()
    method4_comparison_table()
