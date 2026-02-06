"""
Rank Chinese sentences by complexity, word frequency, and uniqueness.

See SPEC_RANK_SENTENCES.md for full specification.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import jieba

from anki_tools.package import AnkiPackage


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Sentence:
    """A Chinese sentence extracted from Anki deck."""

    note_id: int
    chinese: str
    pinyin: str
    english: str
    original_order: int
    complexity_score: float = 0.0
    frequency_score: float = 0.0
    similarity_penalty: float = 0.0
    similar_to: int | None = None  # Rank of most similar higher-ranked sentence
    final_score: float = 0.0


@dataclass
class FrequencyData:
    """Word and character frequency lookup tables."""

    word_freq: dict[str, int] = field(default_factory=dict)
    char_freq: dict[str, int] = field(default_factory=dict)
    hsk_vocab: dict[str, int] = field(default_factory=dict)  # word -> HSK level


# =============================================================================
# Phase 1: Data Extraction
# =============================================================================


def extract_sentences(apkg_path: str, model_id: int | None = None) -> list[Sentence]:
    """
    Extract sentences from an Anki package.

    Args:
        apkg_path: Path to the .apkg file
        model_id: Optional model ID to filter by (extracts all if None)

    Returns:
        List of Sentence objects
    """
    sentences = []

    with AnkiPackage(apkg_path) as pkg:
        models = pkg.get_models()
        decks = pkg.get_decks()
        cards = pkg.get_cards()

        print(f"Available models in {apkg_path}:")
        for mid, model in models.items():
            print(f"  {mid}: {model['name']} ({len(model['flds'])} fields)")
            for i, fld in enumerate(model["flds"]):
                print(f"    [{i}] {fld['name']}")

        for card in cards:
            card_model_id = str(card["mid"])

            if model_id and card_model_id != str(model_id):
                continue

            model = models.get(card_model_id, {})
            field_names = [f["name"] for f in model.get("flds", [])]
            field_values = card["flds"].split("\x1f")
            fields = dict(zip(field_names, field_values))

            chinese = fields.get("Sentence", fields.get("Front", ""))
            pinyin = fields.get("Sentence (Latin)", fields.get("Pinyin", ""))
            english = fields.get(
                "Sentence (Translation)", fields.get("Translation", "")
            )

            counter_str = fields.get("[counter 2]", "0")
            try:
                original_order = int(counter_str) if counter_str else 0
            except ValueError:
                original_order = 0

            if chinese:
                sentences.append(
                    Sentence(
                        note_id=card["nid"],
                        chinese=chinese,
                        pinyin=pinyin,
                        english=english,
                        original_order=original_order,
                    )
                )

    seen = set()
    unique = []
    for s in sentences:
        if s.note_id not in seen:
            seen.add(s.note_id)
            unique.append(s)

    print(f"Extracted {len(unique)} unique sentences")
    return unique


# =============================================================================
# Phase 2: Frequency Data Loading
# =============================================================================


def load_frequency_data(vocab_dir: str = "vocab") -> FrequencyData:
    """
    Load word frequency and HSK vocabulary data.

    Expected files in vocab_dir:
        - subtlex_ch.txt: Tab-separated word frequency list
        - hsk_vocab.json: JSON mapping words to HSK levels

    Returns:
        FrequencyData with loaded lookup tables
    """
    vocab_path = Path(vocab_dir)
    data = FrequencyData()

    subtlex_path = vocab_path / "subtlex_ch.txt"
    if subtlex_path.exists():
        with open(subtlex_path, encoding="utf-8") as f:
            for rank, line in enumerate(f, 1):
                parts = line.strip().split("\t")
                if parts:
                    word = parts[0]
                    data.word_freq[word] = rank
        print(f"Loaded {len(data.word_freq)} words from SUBTLEX-CH")
    else:
        print(f"Warning: {subtlex_path} not found, frequency scoring disabled")

    hsk_path = vocab_path / "hsk_vocab.json"
    if hsk_path.exists():
        with open(hsk_path, encoding="utf-8") as f:
            data.hsk_vocab = json.load(f)
        print(f"Loaded {len(data.hsk_vocab)} HSK words")
    else:
        print(f"Warning: {hsk_path} not found, HSK scoring disabled")

    for word, rank in data.word_freq.items():
        for char in word:
            if is_chinese_char(char) and char not in data.char_freq:
                data.char_freq[char] = rank

    return data


def is_chinese_char(char: str) -> bool:
    """Check if a character is a Chinese character."""
    return "\u4e00" <= char <= "\u9fff"


def get_chinese_chars(text: str) -> list[str]:
    """Extract Chinese characters from text."""
    return [c for c in text if is_chinese_char(c)]


# =============================================================================
# Phase 3: Scoring Functions
# =============================================================================


def complexity_score(sentence: str) -> float:
    """
    Calculate complexity score (0-100) based on structural features.

    Metrics (see SPEC_RANK_SENTENCES.md):
        - Character count (15%)
        - Unique character count (15%)
        - Word count (20%)
        - Average word length (10%)
        - Character stroke count (15%) - TODO
        - HSK level of characters (25%) - TODO

    For now, simplified version using available metrics.
    """
    chars = get_chinese_chars(sentence)
    if not chars:
        return 0.0

    words = list(jieba.cut(sentence))
    chinese_words = [w for w in words if any(is_chinese_char(c) for c in w)]

    char_count = len(chars)
    unique_chars = len(set(chars))
    word_count = len(chinese_words)
    avg_word_len = sum(len(w) for w in chinese_words) / max(len(chinese_words), 1)

    char_norm = min(char_count / 30, 1.0)
    unique_norm = min(unique_chars / 20, 1.0)
    word_norm = min(word_count / 15, 1.0)
    avg_len_norm = min(avg_word_len / 4, 1.0)

    score = (
        char_norm * 0.25 + unique_norm * 0.25 + word_norm * 0.30 + avg_len_norm * 0.20
    )

    return score * 100


def get_word_rank(word: str, freq_data: FrequencyData, max_rank: int = 50000) -> int:
    """
    Get frequency rank for a word, falling back to character average if not found.

    If the word isn't in the frequency list (e.g., jieba combined common words),
    calculate average rank of its characters.
    """
    if word in freq_data.word_freq:
        return freq_data.word_freq[word]

    chars = get_chinese_chars(word)
    if not chars:
        return max_rank

    char_ranks = [freq_data.char_freq.get(c, max_rank) for c in chars]
    return int(sum(char_ranks) / len(char_ranks))


def frequency_score(sentence: str, freq_data: FrequencyData) -> float:
    """
    Calculate frequency score (0-100) based on word commonality.

    Higher score = more common vocabulary = easier to learn first.

    Metrics:
        - Average word frequency rank (40%)
        - % words in top 1000 (25%)
        - % words in top 5000 (15%)
        - HSK coverage (20%)
    """
    words = list(jieba.cut(sentence))
    chinese_words = [w for w in words if any(is_chinese_char(c) for c in w)]

    if not chinese_words:
        return 0.0

    if not freq_data.word_freq:
        return 50.0

    max_rank = 50000
    ranks = [get_word_rank(w, freq_data, max_rank) for w in chinese_words]

    avg_rank = sum(ranks) / len(ranks)
    avg_score = 1.0 - min(avg_rank / max_rank, 1.0)

    top_1000 = sum(1 for r in ranks if r <= 1000) / len(ranks)
    top_5000 = sum(1 for r in ranks if r <= 5000) / len(ranks)

    if freq_data.hsk_vocab:
        hsk_count = 0
        for w in chinese_words:
            if freq_data.hsk_vocab.get(w, 7) <= 4:
                hsk_count += 1
            else:
                chars = get_chinese_chars(w)
                if chars and all(freq_data.hsk_vocab.get(c, 7) <= 4 for c in chars):
                    hsk_count += 1
        hsk_coverage = hsk_count / len(chinese_words)
    else:
        hsk_coverage = 0.5

    score = avg_score * 0.40 + top_1000 * 0.25 + top_5000 * 0.15 + hsk_coverage * 0.20

    return score * 100


# =============================================================================
# Phase 4: Similarity Analysis
# =============================================================================


def char_similarity(sent_a: str, sent_b: str) -> float:
    """
    Calculate Jaccard similarity between character sets.

    Returns value between 0 (no overlap) and 1 (identical).
    """
    chars_a = set(get_chinese_chars(sent_a))
    chars_b = set(get_chinese_chars(sent_b))

    if not chars_a or not chars_b:
        return 0.0

    intersection = len(chars_a & chars_b)
    union = len(chars_a | chars_b)

    return intersection / union


def compute_similarity_penalties(
    sentences: list[Sentence], top_n: int = 100
) -> tuple[list[float], list[int | None]]:
    """
    Compute similarity penalties for each sentence.

    For each sentence, find maximum similarity to higher-ranked sentences
    (those with better preliminary scores) and apply penalty.

    Args:
        sentences: List of sentences sorted by preliminary score (best first)
        top_n: Number of higher-ranked sentences to check (for efficiency)

    Returns:
        Tuple of (penalties, similar_to_indices):
        - penalties: List of penalties (0-50 scale) in same order as input
        - similar_to_indices: List of indices (1-based rank) of most similar sentence
    """
    penalties = []
    similar_to_indices = []

    for i, sent in enumerate(sentences):
        if i == 0:
            penalties.append(0.0)
            similar_to_indices.append(None)
            continue

        higher_ranked_indices = list(range(min(i, top_n)))
        similarities = [
            char_similarity(sent.chinese, sentences[j].chinese)
            for j in higher_ranked_indices
        ]

        max_sim = max(similarities)
        max_idx = higher_ranked_indices[similarities.index(max_sim)]

        penalty = max_sim * 50
        penalties.append(penalty)
        similar_to_indices.append(max_idx + 1)

    return penalties, similar_to_indices


# =============================================================================
# Phase 5: Ranking and Output
# =============================================================================


def rank_sentences(
    sentences: list[Sentence],
    freq_data: FrequencyData,
    weights: dict[str, float] | None = None,
) -> list[Sentence]:
    """
    Rank sentences by combined score.

    Default weights:
        - complexity: 0.3 (simpler first)
        - frequency: 0.5 (common vocab first)
        - similarity_penalty: 0.2 (variety)

    Note: Complexity is inverted (lower complexity = higher score for learning)
    """
    if weights is None:
        weights = {
            "complexity": 0.3,
            "frequency": 0.5,
            "similarity_penalty": 0.2,
        }

    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity_score(sent.chinese)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency_score(sent.chinese, freq_data)

    for sent in sentences:
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - sent.complexity_score) * weights["complexity"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    print("Calculating similarity penalties...")
    penalties, similar_to_indices = compute_similarity_penalties(sentences)
    for sent, penalty, similar_to in zip(sentences, penalties, similar_to_indices):
        sent.similarity_penalty = penalty
        sent.similar_to = similar_to

    for sent in sentences:
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - sent.complexity_score) * weights["complexity"]
            - sent.similarity_penalty * weights["similarity_penalty"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    return sentences


def export_csv(sentences: list[Sentence], output_path: str) -> None:
    """Export ranked sentences to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "sentence",
                "pinyin",
                "english",
                "original_order",
                "complexity",
                "frequency",
                "similarity_penalty",
                "similar_to",
                "final_score",
            ]
        )

        for rank, sent in enumerate(sentences, 1):
            writer.writerow(
                [
                    rank,
                    sent.chinese,
                    sent.pinyin,
                    sent.english,
                    sent.original_order,
                    f"{sent.complexity_score:.1f}",
                    f"{sent.frequency_score:.1f}",
                    f"{sent.similarity_penalty:.1f}",
                    sent.similar_to if sent.similar_to else "",
                    f"{sent.final_score:.2f}",
                ]
            )

    print(f"Exported {len(sentences)} sentences to {output_path}")
