#!/usr/bin/env python3
"""
Process Chinese Anki decks: rank sentences, reorder cards, fix pinyin.

Commands:
    rank    - Rank sentences by frequency and complexity
    reorder - Reorder deck based on ranking CSV
    fix     - Fix pinyin formatting
    all     - Run complete pipeline (rank + reorder + fix)
"""

import csv
import tempfile
from pathlib import Path

import cyclopts

from rank_sentences import (
    Sentence,
    extract_sentences,
    load_frequency_data,
    rank_sentences,
    export_csv,
)
from reorder_deck import reorder_deck, load_ranking
from fix_pinyin import fix_pinyin, check_pinyin

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


def export_summary_csv(
    ranked: list[Sentence],
    remove_ranks: set[int],
    output_path: str,
    fix_pinyin_enabled: bool = True,
) -> int:
    """
    Export summary CSV with all changes.
    
    Returns number of pinyin corrections.
    """
    pinyin_corrections = 0
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank",
            "sentence",
            "pinyin",
            "pinyin_corrected",
            "pinyin_reason",
            "english",
            "original_order",
            "complexity",
            "frequency",
            "similarity_penalty",
            "similar_to",
            "final_score",
            "removed",
        ])
        
        for i, sent in enumerate(ranked, 1):
            removed = "yes" if i in remove_ranks else ""
            
            pinyin_corrected = ""
            pinyin_reason = ""
            if fix_pinyin_enabled and not removed:
                correction, reason = check_pinyin(sent.chinese, sent.pinyin)
                if correction:
                    pinyin_corrected = correction
                    pinyin_reason = reason
                    pinyin_corrections += 1
            
            writer.writerow([
                i,
                sent.chinese,
                sent.pinyin,
                pinyin_corrected,
                pinyin_reason,
                sent.english,
                sent.original_order,
                f"{sent.complexity_score:.1f}",
                f"{sent.frequency_score:.1f}",
                f"{sent.similarity_penalty:.1f}",
                sent.similar_to if sent.similar_to else "",
                f"{sent.final_score:.2f}",
                removed,
            ])
    
    return pinyin_corrections


@app.command(name="all")
def process_all(
    apkg_path: Path,
    *,
    output: Path | None = None,
    summary_csv: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    no_pinyin_fix: bool = False,
    vocab_dir: str = "vocab",
    verbose: bool = False,
):
    """
    Run complete pipeline: rank + reorder + fix pinyin.

    This is the main command for processing a Chinese Anki deck.

    Args:
        apkg_path: Input .apkg file
        output: Output .apkg file (default: input_processed.apkg)
        summary_csv: Output summary CSV (default: input_summary.csv)
        model_id: Filter by specific model ID
        keep_filtered: Keep filtered cards (names, invalid)
        no_pinyin_fix: Skip pinyin correction
        vocab_dir: Directory with frequency data
        verbose: Show detailed output
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_processed.apkg"
    if summary_csv is None:
        summary_csv = apkg_path.parent / f"{apkg_path.stem}_summary.csv"

    print("=" * 50)
    print("Processing Chinese Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Summary: {summary_csv}")
    print()

    # Step 1: Rank sentences
    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id)

    if not sentences:
        print("No sentences found!")
        return

    freq_data = load_frequency_data(vocab_dir)
    ranked = rank_sentences(sentences, freq_data)

    # Save ranking to temp file for reorder_deck
    ranking_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ).name
    export_csv(ranked, ranking_file)

    # Load remove_ranks for summary
    _, remove_ranks = load_ranking(ranking_file)

    # Step 2: Reorder deck
    print("\nStep 2: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        ranking_file,
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

    # Step 4: Export summary CSV
    print("\nStep 4: Exporting summary...")
    pinyin_corrections = export_summary_csv(
        ranked,
        remove_ranks if not keep_filtered else set(),
        str(summary_csv),
        fix_pinyin_enabled=not no_pinyin_fix,
    )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Pinyin corrections:  {pinyin_corrections}")
    print(f"  Output:  {output}")
    print(f"  Summary: {summary_csv}")


if __name__ == "__main__":
    app()
