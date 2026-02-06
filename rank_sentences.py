#!/usr/bin/env python3
"""Backwards compatibility - imports from anki_tools package."""

from anki_tools.rank import (
    Sentence,
    FrequencyData,
    extract_sentences,
    load_frequency_data,
    rank_sentences,
    export_csv,
    complexity_score,
    frequency_score,
    char_similarity,
    get_chinese_chars,
    is_chinese_char,
    get_word_rank,
    compute_similarity_penalties,
)

__all__ = [
    "Sentence",
    "FrequencyData",
    "extract_sentences",
    "load_frequency_data",
    "rank_sentences",
    "export_csv",
    "complexity_score",
    "frequency_score",
    "char_similarity",
    "get_chinese_chars",
    "is_chinese_char",
    "get_word_rank",
    "compute_similarity_penalties",
]


def main():
    """Run ranking from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rank Chinese sentences by complexity and frequency"
    )
    parser.add_argument("apkg", help="Path to .apkg file")
    parser.add_argument(
        "-o", "--output", default="ranked_sentences.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--model-id", type=int, help="Filter by specific model ID"
    )
    parser.add_argument(
        "--vocab-dir", default="vocab", help="Directory containing frequency data"
    )
    
    args = parser.parse_args()
    
    sentences = extract_sentences(args.apkg, args.model_id)
    if not sentences:
        print("No sentences found!")
        return
    
    freq_data = load_frequency_data(args.vocab_dir)
    ranked = rank_sentences(sentences, freq_data)
    export_csv(ranked, args.output)


if __name__ == "__main__":
    main()
