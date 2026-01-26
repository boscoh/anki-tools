#!/usr/bin/env python3
"""
Test anki-export to see actual card data
"""

from pathlib import Path
from anki_export import ApkgReader

apkg_files = list(Path('..').glob('*.apkg'))
apkg_path = apkg_files[0]

with ApkgReader(str(apkg_path)) as apkg:
    data = apkg.export()

    print("Card types found:", list(data.keys()))

    for card_type, cards in data.items():
        print(f"\n{'='*60}")
        print(f"Card Type: {card_type}")
        print(f"Number of cards: {len(cards)}")
        print(f"\nField names (first row):")
        print(cards[0])

        print(f"\nFirst 3 cards:")
        for i, card in enumerate(cards[1:4], 1):  # Skip header row
            print(f"\nCard {i}:")
            for field_name, value in zip(cards[0], card):
                print(f"  {field_name}: {value[:100] if isinstance(value, str) else value}")
