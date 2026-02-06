#!/usr/bin/env python3
"""Backwards compatibility - imports from anki_tools package."""

from anki_tools.reorder import reorder_deck, load_ranking

__all__ = ["reorder_deck", "load_ranking"]


def main():
    """Run reordering from command line."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(
        description='Reorder Anki deck based on sentence ranking'
    )
    parser.add_argument('input_apkg', help='Input .apkg file')
    parser.add_argument('ranking_csv', help='Ranking CSV from rank_sentences.py')
    parser.add_argument('-o', '--output', help='Output .apkg file')
    parser.add_argument('--keep-filtered', action='store_true',
                       help='Keep filtered cards instead of removing')
    
    args = parser.parse_args()
    
    if args.output:
        output = args.output
    else:
        input_path = Path(args.input_apkg)
        output = str(input_path.parent / f"{input_path.stem}_ranked{input_path.suffix}")
    
    stats = reorder_deck(
        args.input_apkg,
        output,
        args.ranking_csv,
        remove_filtered=not args.keep_filtered
    )
    
    print(f"\nDone!")
    print(f"  Cards reordered: {stats['cards_reordered']}")
    print(f"  Cards removed: {stats['cards_removed']}")
    print(f"  Output: {output}")


if __name__ == '__main__':
    main()
