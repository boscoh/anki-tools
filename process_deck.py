#!/usr/bin/env python3
"""
Process Chinese Anki decks: rank sentences, reorder cards, fix pinyin.

Commands:
    rank    - Rank sentences by frequency and complexity
    reorder - Reorder deck based on ranking CSV
    fix     - Fix pinyin formatting
    all     - Run complete pipeline (rank + reorder + fix)
"""

import tempfile
from pathlib import Path

import cyclopts

from rank_sentences import (
    extract_sentences,
    load_frequency_data,
    rank_sentences,
    export_csv,
)
from reorder_deck import reorder_deck
from fix_pinyin import fix_pinyin

app = cyclopts.App(help="Process Chinese Anki decks")


# =============================================================================
# Rank command - analyze and rank sentences
# =============================================================================


@app.command
def rank(
    apkg_path: Path,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    vocab_dir: str = "vocab",
):
    """
    Rank sentences by frequency and complexity.

    Outputs a CSV with sentences ranked for optimal learning order.

    Args:
        apkg_path: Input .apkg file
        output: Output CSV file (default: ranked_sentences.csv)
        model_id: Filter by specific model ID
        vocab_dir: Directory containing frequency data
    """
    if output is None:
        output = Path("ranked_sentences.csv")

    print(f"Ranking sentences from: {apkg_path}")

    sentences = extract_sentences(str(apkg_path), model_id)
    if not sentences:
        print("No sentences found!")
        return

    freq_data = load_frequency_data(vocab_dir)
    ranked = rank_sentences(sentences, freq_data)
    export_csv(ranked, str(output))

    print(f"\nRanked {len(ranked)} sentences -> {output}")


# =============================================================================
# Reorder command - reorder deck based on ranking
# =============================================================================


@app.command
def reorder(
    apkg_path: Path,
    ranking_csv: Path,
    *,
    output: Path | None = None,
    keep_filtered: bool = False,
):
    """
    Reorder deck based on ranking CSV.

    Updates card order and optionally removes filtered cards (names, invalid).

    Args:
        apkg_path: Input .apkg file
        ranking_csv: Ranking CSV from 'rank' command
        output: Output .apkg file (default: input_reordered.apkg)
        keep_filtered: Keep filtered cards instead of removing
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    print(f"Reordering deck: {apkg_path}")
    print(f"Using ranking: {ranking_csv}")

    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
    )

    print(f"\nReordered {stats['cards_reordered']} cards")
    print(f"Removed {stats['cards_removed']} filtered cards")
    print(f"Output: {output}")


# =============================================================================
# Fix command - fix pinyin formatting
# =============================================================================


@app.command
def fix(
    apkg_path: Path,
    *,
    output: Path | None = None,
    verbose: bool = False,
):
    """
    Fix pinyin formatting in deck.

    Corrects: lowercase, syllable separation, tone marks, common errors.

    Args:
        apkg_path: Input .apkg file
        output: Output .apkg file (default: input_fixed.apkg)
        verbose: Show all corrections
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_fixed.apkg"

    print(f"Fixing pinyin in: {apkg_path}")

    fix_pinyin(
        apkg_path,
        output=output,
        csv_output=None,
        verbose=verbose,
    )

    print(f"\nOutput: {output}")


# =============================================================================
# All command - complete pipeline
# =============================================================================


@app.command(name="all")
def process_all(
    apkg_path: Path,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    no_pinyin_fix: bool = False,
    ranking_csv: Path | None = None,
    vocab_dir: str = "vocab",
    verbose: bool = False,
):
    """
    Run complete pipeline: rank + reorder + fix pinyin.

    This is the main command for processing a Chinese Anki deck.

    Args:
        apkg_path: Input .apkg file
        output: Output .apkg file (default: input_processed.apkg)
        model_id: Filter by specific model ID
        keep_filtered: Keep filtered cards (names, invalid)
        no_pinyin_fix: Skip pinyin correction
        ranking_csv: Save ranking CSV to this path
        vocab_dir: Directory with frequency data
        verbose: Show detailed output
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_processed.apkg"

    print("=" * 50)
    print("Processing Chinese Anki Deck")
    print("=" * 50)
    print(f"Input:  {apkg_path}")
    print(f"Output: {output}")
    print()

    # Step 1: Rank sentences
    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id)

    if not sentences:
        print("No sentences found!")
        return

    freq_data = load_frequency_data(vocab_dir)
    ranked = rank_sentences(sentences, freq_data)

    # Save ranking CSV (temp file if not specified)
    if ranking_csv is None:
        csv_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ).name
    else:
        csv_file = str(ranking_csv)
    export_csv(ranked, csv_file)

    # Step 2: Reorder deck
    print("\nStep 2: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        csv_file,
        remove_filtered=not keep_filtered,
    )

    # Step 3: Fix pinyin
    if not no_pinyin_fix:
        print("\nStep 3: Fixing pinyin...")
        fix_pinyin(
            output,
            output=output,
            csv_output=Path("/dev/null"),
            verbose=verbose,
        )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:  {len(ranked)}")
    print(f"  Cards reordered:   {stats['cards_reordered']}")
    print(f"  Cards removed:     {stats['cards_removed']}")
    print(f"  Output: {output}")


if __name__ == "__main__":
    app()
