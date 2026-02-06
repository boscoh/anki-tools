#!/usr/bin/env python3
"""
Process a Chinese Anki deck: rank sentences, reorder cards, fix pinyin.

Combines rank_sentences.py, reorder_deck.py, and fix_pinyin.py into one pipeline.
"""

import argparse
import tempfile
from pathlib import Path

from rank_sentences import (
    extract_sentences,
    load_frequency_data,
    rank_sentences,
    export_csv,
)
from reorder_deck import reorder_deck
from fix_pinyin import fix_pinyin


def process_deck(
    input_apkg: str,
    output_apkg: str | None = None,
    model_id: int | None = None,
    remove_filtered: bool = True,
    fix_pinyin_enabled: bool = True,
    ranking_csv: str | None = None,
    vocab_dir: str = "vocab",
    verbose: bool = False,
) -> dict:
    """
    Process a Chinese Anki deck with ranking, reordering, and pinyin fixes.
    
    Args:
        input_apkg: Path to input .apkg file
        output_apkg: Path to output .apkg file (default: input_processed.apkg)
        model_id: Filter by specific model ID (None = all models)
        remove_filtered: Remove foreign names and invalid characters
        fix_pinyin_enabled: Apply pinyin corrections
        ranking_csv: Path to save ranking CSV (None = temp file)
        vocab_dir: Directory containing frequency data
        verbose: Print detailed progress
    
    Returns:
        Dict with processing statistics
    """
    input_path = Path(input_apkg)
    
    if output_apkg is None:
        output_apkg = str(input_path.parent / f"{input_path.stem}_processed.apkg")
    
    stats = {
        'input': str(input_apkg),
        'output': output_apkg,
        'sentences_extracted': 0,
        'cards_reordered': 0,
        'cards_removed': 0,
        'pinyin_corrected': 0,
    }
    
    # Step 1: Extract and rank sentences
    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(input_apkg, model_id)
    stats['sentences_extracted'] = len(sentences)
    
    if not sentences:
        print("No sentences found!")
        return stats
    
    freq_data = load_frequency_data(vocab_dir)
    ranked = rank_sentences(sentences, freq_data)
    
    # Save ranking CSV
    if ranking_csv is None:
        ranking_csv = tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False
        ).name
    export_csv(ranked, ranking_csv)
    
    # Step 2: Reorder deck
    print("\nStep 2: Reordering deck...")
    reorder_stats = reorder_deck(
        input_apkg,
        output_apkg,
        ranking_csv,
        remove_filtered=remove_filtered
    )
    stats['cards_reordered'] = reorder_stats['cards_reordered']
    stats['cards_removed'] = reorder_stats['cards_removed']
    
    # Step 3: Fix pinyin
    if fix_pinyin_enabled:
        print("\nStep 3: Fixing pinyin...")
        # Fix pinyin in place (overwrite the reordered deck)
        # Use /dev/null for csv_output to suppress file creation
        csv_out = Path("/dev/null") if not verbose else None
        fix_pinyin(
            Path(output_apkg),
            output=Path(output_apkg),
            csv_output=csv_out,
            verbose=verbose
        )
    
    print(f"\n{'='*50}")
    print("DONE!")
    print(f"  Input:  {input_apkg}")
    print(f"  Output: {output_apkg}")
    print(f"  Sentences ranked:  {stats['sentences_extracted']}")
    print(f"  Cards reordered:   {stats['cards_reordered']}")
    print(f"  Cards removed:     {stats['cards_removed']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Process Chinese Anki deck: rank, reorder, fix pinyin'
    )
    parser.add_argument('input_apkg', help='Input .apkg file')
    parser.add_argument('-o', '--output', help='Output .apkg file')
    parser.add_argument('--model-id', type=int, help='Filter by model ID')
    parser.add_argument('--keep-filtered', action='store_true',
                       help='Keep filtered cards (names, invalid)')
    parser.add_argument('--no-pinyin-fix', action='store_true',
                       help='Skip pinyin correction')
    parser.add_argument('--ranking-csv', help='Save ranking CSV to this path')
    parser.add_argument('--vocab-dir', default='vocab',
                       help='Directory with frequency data')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    process_deck(
        args.input_apkg,
        output_apkg=args.output,
        model_id=args.model_id,
        remove_filtered=not args.keep_filtered,
        fix_pinyin_enabled=not args.no_pinyin_fix,
        ranking_csv=args.ranking_csv,
        vocab_dir=args.vocab_dir,
        verbose=args.verbose,
    )


if __name__ == '__main__':
    main()
