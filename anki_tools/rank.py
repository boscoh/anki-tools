"""
Rank sentences by complexity, frequency, and similarity (ZH and FR pipelines).

See SPEC_RANK_SENTENCES.md for full specification.
"""

import csv
import re
from dataclasses import dataclass, field
from typing import Callable

import jieba
import opencc
from wordfreq import iter_wordlist

from anki_tools.jyutping import get_jyutping, simplified_to_traditional
from anki_tools.package import AnkiPackage

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Sentence:
    """A sentence extracted from an Anki deck (ZH or FR pipeline).

    :param note_id: Note ID in the deck.
    :param text: Main sentence text (Chinese or French).
    :param original_order: Original card order index.
    :param complexity_score: Structural complexity (0-100).
    :param frequency_score: Word frequency score (0-100).
    :param similarity_penalty: Penalty for similarity to higher-ranked sentences.
    :param similar_to_rank: Rank (1-based) of most similar higher-ranked sentence.
    :param final_score: Combined ranking score.
    """

    note_id: int
    text: str
    original_order: int
    complexity_score: float = 0.0
    frequency_score: float = 0.0
    similarity_penalty: float = 0.0
    similar_to_rank: int | None = None  # Rank of most similar higher-ranked sentence
    similar_to_text: str | None = None  # Closest sentence text (French pipeline)
    grammar_score: float = 0.0  # Grammar complexity 0-100 (French pipeline)
    romanization: str = ""  # Romanization (jyutping for Cantonese)
    final_score: float = 0.0


@dataclass
class FrequencyData:
    """Word and character frequency lookup tables.

    :param word_freq: Word to frequency rank.
    :param char_freq: Character to frequency rank.
    :param hsk_vocab: Word or character to HSK level.
    """

    word_freq: dict[str, int] = field(default_factory=dict)
    char_freq: dict[str, int] = field(default_factory=dict)
    hsk_vocab: dict[str, int] = field(default_factory=dict)  # word -> HSK level


# =============================================================================
# Phase 1: Data Extraction
# =============================================================================


def _resolve_extract_fields(
    field_names: list[str],
    text_prefer: str | None,
) -> tuple[str, str | None, str | None]:
    deck = set(field_names)
    text = None
    if text_prefer and text_prefer in deck:
        text = text_prefer
    if not text:
        for c in ("Sentence", "French", "sentence", "Front"):
            if c in deck:
                text = c
                break
    if not text:
        text = field_names[0] if field_names else "Front"
    order = "[counter 2]" if "[counter 2]" in deck else None
    romanization = None
    for c in ("Jyutping", "Pinyin", "Romanization", "jyutping", "pinyin"):
        if c in deck:
            romanization = c
            break
    return (text, order, romanization)


def extract_sentences(
    apkg_path: str,
    model_id: int | None = None,
    *,
    text_field: str | None = None,
) -> list[Sentence]:
    """Extract sentences from an Anki package.

    Main text and order fields are inferred from each model's field names.
    Pass text_field to prefer a specific main field (e.g. "French").

    :param apkg_path: Path to the .apkg file.
    :param model_id: Optional model ID to filter by; extracts all if None.
    :param text_field: Preferred main sentence field (inferred if None).
    :returns: List of :class:`Sentence` objects.
    """
    sentences = []

    with AnkiPackage(apkg_path) as pkg:
        models = pkg.get_models()
        cards = pkg.get_cards()

        print(f"Available models in {apkg_path}:")
        for mid, model in models.items():
            print(f"  {mid}: {model['name']} ({len(model['flds'])} fields)")
            for i, fld in enumerate(model["flds"]):
                print(f"    [{i}] {fld['name']}")

        resolved_by_model: dict[str, tuple[str, str | None, str | None]] = {}
        card_index = 0
        for card in cards:
            card_model_id = str(card["mid"])

            if model_id and card_model_id != str(model_id):
                continue

            model = models.get(card_model_id, {})
            field_names = [f["name"] for f in model.get("flds", [])]
            if card_model_id not in resolved_by_model:
                resolved_by_model[card_model_id] = _resolve_extract_fields(
                    field_names, text_field
                )
            text_f, order_f, romanization_f = resolved_by_model[card_model_id]

            field_values = card["flds"].split("\x1f")
            fields = dict(zip(field_names, field_values))

            text = fields.get(text_f, fields.get("Front", ""))
            romanization = fields.get(romanization_f, "") if romanization_f else ""
            romanization = re.sub(r"<[^>]+>", "", romanization).strip()

            original_order = card_index
            if order_f and order_f in fields and fields.get(order_f):
                try:
                    original_order = int(fields[order_f])
                except ValueError:
                    pass

            if text:
                sentences.append(
                    Sentence(
                        note_id=card["nid"],
                        text=text,
                        original_order=original_order,
                        romanization=romanization,
                    )
                )
                card_index += 1

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


def _load_frequency_data_wordfreq(
    lang: str,
    *,
    lower_keys: bool = False,
    build_char_freq: bool = False,
) -> FrequencyData:
    data = FrequencyData()
    for rank, word in enumerate(iter_wordlist(lang, wordlist="best"), 1):
        key = word.lower() if lower_keys else word
        data.word_freq[key] = rank
    if build_char_freq:
        for word, rank in data.word_freq.items():
            for char in word:
                if is_chinese_char(char) and char not in data.char_freq:
                    data.char_freq[char] = rank
    print(f"Loaded {len(data.word_freq)} words from wordfreq")
    return data


def load_frequency_data_zh() -> FrequencyData:
    """Load Chinese word frequency data from wordfreq."""
    data = _load_frequency_data_wordfreq("zh", build_char_freq=True)
    return data


def load_frequency_data_fr() -> FrequencyData:
    """Load French word frequency data from wordfreq."""
    return _load_frequency_data_wordfreq("fr", lower_keys=True)


def load_frequency_data_es() -> FrequencyData:
    """Load Spanish word frequency data from wordfreq."""
    return _load_frequency_data_wordfreq("es", lower_keys=True)


def is_chinese_char(char: str) -> bool:
    """Check if a character is a Chinese character.

    :param char: Single character.
    :returns: True if in CJK Unified Ideographs range.
    """
    return "\u4e00" <= char <= "\u9fff"


def get_chinese_chars(text: str) -> list[str]:
    """Extract Chinese characters from text.

    :param text: Arbitrary text.
    :returns: List of Chinese characters in order.
    """
    return [c for c in text if is_chinese_char(c)]


# =============================================================================
# Phase 3: Scoring Functions
# =============================================================================


def complexity_score_zh(sentence: str) -> float:
    """Calculate complexity score (0-100) based on structural features.

    Metrics (see SPEC_RANK_SENTENCES.md):
        - Character count, unique characters, word count, average word length.

    :param sentence: Chinese sentence.
    :returns: Score from 0 to 100 (higher = more complex).
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


def get_word_rank_zh(word: str, freq_data: FrequencyData, max_rank: int = 50000) -> int:
    """Get frequency rank for a word, falling back to character average if not found.

    :param word: Word to look up.
    :param freq_data: Loaded frequency data.
    :param max_rank: Default rank when not found.
    :returns: Frequency rank (lower = more common).
    """
    if word in freq_data.word_freq:
        return freq_data.word_freq[word]

    chars = get_chinese_chars(word)
    if not chars:
        return max_rank

    char_ranks = [freq_data.char_freq.get(c, max_rank) for c in chars]
    return int(sum(char_ranks) / len(char_ranks))


def frequency_score_zh(sentence: str, freq_data: FrequencyData) -> float:
    """Calculate frequency score (0-100) based on word commonality.

    Higher score = more common vocabulary = easier to learn first.

    :param sentence: Chinese sentence.
    :param freq_data: Loaded frequency data.
    :returns: Score from 0 to 100.
    """
    words = list(jieba.cut(sentence))
    chinese_words = [w for w in words if any(is_chinese_char(c) for c in w)]

    if not chinese_words:
        return 0.0

    if not freq_data.word_freq:
        return 50.0

    max_rank = 50000
    ranks = [get_word_rank_zh(w, freq_data, max_rank) for w in chinese_words]

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


def _jaccard_similarity(items_a: set, items_b: set) -> float:
    """Calculate Jaccard similarity between two sets.

    :param items_a: First set of items.
    :param items_b: Second set of items.
    :returns: Similarity between 0 (no overlap) and 1 (identical).
    """
    if not items_a or not items_b:
        return 0.0
    return len(items_a & items_b) / len(items_a | items_b)


def char_similarity_zh(sent_a: str, sent_b: str) -> float:
    """Calculate Jaccard similarity between character sets.

    :param sent_a: First Chinese sentence.
    :param sent_b: Second Chinese sentence.
    :returns: Similarity between 0 (no overlap) and 1 (identical).
    """
    chars_a = set(get_chinese_chars(sent_a))
    chars_b = set(get_chinese_chars(sent_b))
    return _jaccard_similarity(chars_a, chars_b)


def _compute_similarity_penalties(
    sentences: list[Sentence],
    similarity_fn: Callable[[str, str], float],
    top_n: int = 100,
) -> tuple[list[float], list[int | None], list[str | None]]:
    penalties: list[float] = []
    similar_to_rank_indices: list[int | None] = []
    similar_to_rank_texts: list[str | None] = []

    for i, sent in enumerate(sentences):
        if i == 0:
            penalties.append(0.0)
            similar_to_rank_indices.append(None)
            similar_to_rank_texts.append(None)
            continue

        higher_ranked_indices = list(range(min(i, top_n)))
        similarities = [
            similarity_fn(sent.text, sentences[j].text)
            for j in higher_ranked_indices
        ]
        max_sim = max(similarities)
        max_idx = higher_ranked_indices[similarities.index(max_sim)]
        penalty = max_sim * SIMILARITY_PENALTY_SCALE
        penalties.append(penalty)
        similar_to_rank_indices.append(max_idx + 1)
        similar_to_rank_texts.append(sentences[max_idx].text)

    return penalties, similar_to_rank_indices, similar_to_rank_texts


def compute_similarity_penalties_zh(
    sentences: list[Sentence], top_n: int = 100
) -> tuple[list[float], list[int | None], list[str | None]]:
    """Compute similarity penalties (character-based, ZH)."""
    return _compute_similarity_penalties(sentences, char_similarity_zh, top_n)


SIMILARITY_PENALTY_SCALE = 20

SIMILARITY_DELETE_THRESHOLD_RATIO = 0.7
SIMILARITY_CONSIDER_DELETE_PENALTY = SIMILARITY_PENALTY_SCALE * SIMILARITY_DELETE_THRESHOLD_RATIO  # flag for deletion when Jaccard >= ratio

STRUCTURAL_COMPLEXITY_BLEND = 0.6


def _structural_score(
    complexity: float, grammar: float, blend: float = STRUCTURAL_COMPLEXITY_BLEND
) -> float:
    return blend * complexity + (1.0 - blend) * grammar


# =============================================================================
# Phase 5: Ranking and Output
# =============================================================================


def _rank_character_based(
    sentences: list[Sentence],
    weights: dict[str, float],
    preprocess_fn: Callable[[list[Sentence]], None] | None = None,
) -> list[Sentence]:
    """Shared character-based ranking logic for Chinese and Cantonese.

    :param sentences: Sentences to rank.
    :param weights: Dict with complexity, frequency, similarity_penalty weights.
    :param preprocess_fn: Optional preprocessing function (e.g., for traditional conversion).
    :returns: Same list sorted by final_score (best first).
    """
    if preprocess_fn:
        preprocess_fn(sentences)

    freq_data = load_frequency_data_zh()

    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity_score_zh(sent.text)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency_score_zh(sent.text, freq_data)

    for sent in sentences:
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - sent.complexity_score) * weights["complexity"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    print("Calculating similarity penalties...")
    penalties, similar_to_rank_indices, similar_to_rank_texts = compute_similarity_penalties_zh(sentences)
    for sent, penalty, similar_to_rank, sim_text in zip(
        sentences, penalties, similar_to_rank_indices, similar_to_rank_texts
    ):
        sent.similarity_penalty = penalty
        sent.similar_to_rank = similar_to_rank
        sent.similar_to_text = sim_text

    for sent in sentences:
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - sent.complexity_score) * weights["complexity"]
            - sent.similarity_penalty * weights["similarity_penalty"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    return sentences


def rank_sentences_zh(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
) -> list[Sentence]:
    """Rank sentences by combined score (Chinese pipeline).

    Default weights: complexity 0.3, frequency 0.5, similarity_penalty 0.2.
    Complexity is inverted (lower complexity = higher score for learning).

    :param sentences: Sentences to rank.
    :param weights: Optional dict with complexity, frequency, similarity_penalty.
    :returns: Same list sorted by final_score (best first).
    """
    if weights is None:
        weights = {
            "complexity": 0.3,
            "frequency": 0.5,
            "similarity_penalty": 0.2,
        }

    return _rank_character_based(sentences, weights)


def write_ranking_csv(
    sentences: list[Sentence],
    output_path: str,
    *,
    structural_blend: float = STRUCTURAL_COMPLEXITY_BLEND,
    similar_to_threshold: float = SIMILARITY_CONSIDER_DELETE_PENALTY,
) -> None:
    """Write ranking CSV for reorder_deck and inspection (ZH and FR).

    Columns: rank, note_id, sentence, romanization (if present), original_order, frequency,
    complexity, grammar, structural, similarity, similar_to, final_score.
    """
    has_romanization = any(sent.romanization for sent in sentences)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "rank",
            "note_id",
            "sentence",
        ]
        if has_romanization:
            header.append("romanization")
        header.extend([
            "original_order",
            "frequency",
            "complexity",
            "grammar",
            "structural",
            "similarity",
            "similar_to",
            "final_score",
        ])
        writer.writerow(header)

        for rank, sent in enumerate(sentences, 1):
            structural = _structural_score(
                sent.complexity_score, sent.grammar_score, structural_blend
            )
            similar_to_text = (
                (sent.similar_to_text or "")
                if sent.similarity_penalty > similar_to_threshold
                else ""
            )
            row = [
                rank,
                sent.note_id,
                sent.text,
            ]
            if has_romanization:
                row.append(sent.romanization)
            row.extend([
                sent.original_order,
                f"{sent.frequency_score:.1f}",
                f"{sent.complexity_score:.1f}",
                f"{sent.grammar_score:.1f}",
                f"{structural:.1f}",
                f"{sent.similarity_penalty:.1f}",
                similar_to_text,
                f"{sent.final_score:.2f}",
            ])
            writer.writerow(row)
    print(f"Wrote ranking for {len(sentences)} sentences to {output_path}")


# =============================================================================
# Cantonese ranking (character-based, similar to ZH but with YUE weighting)
# =============================================================================


def rank_sentences_yue(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
) -> list[Sentence]:
    """Rank sentences for Cantonese by combined score.

    Converts simplified to traditional Chinese and generates jyutping romanization.
    Uses Chinese character-based complexity and similarity (like ZH)
    with Cantonese-specific weighting. Frequency data uses Chinese (Mandarin)
    as an approximation since Cantonese is not available in wordfreq.

    :param sentences: Sentences to rank.
    :param weights: Optional dict with complexity, frequency, similarity_penalty.
    :returns: Same list sorted by final_score (best first).
    """
    if weights is None:
        weights = {
            "complexity": 0.3,
            "frequency": 0.5,
            "similarity_penalty": 0.2,
        }

    def preprocess_cantonese(sentences: list[Sentence]) -> None:
        print("Converting to traditional Chinese and generating jyutping...")
        for sent in sentences:
            sent.text = simplified_to_traditional(sent.text)
            sent.romanization = get_jyutping(sent.text)

    return _rank_character_based(sentences, weights, preprocess_cantonese)
# =============================================================================
# French ranking (word-based, reusable structure)
# =============================================================================


def _tokenize_words_fr(text: str) -> list[str]:
    """Tokenize text into words (letters only, lowercase for comparison).

    :param text: Sentence or phrase.
    :returns: List of non-empty word tokens.
    """
    cleaned = re.sub(r"<[^>]+>", "", text)
    tokens = re.findall(r"[a-zA-Z\u00c0-\u024f]+", cleaned)
    return [t.lower() for t in tokens if t]


def complexity_score_fr(sentence: str) -> float:
    """Calculate complexity score (0-100) for French/latin text from structural features.

    :param sentence: French sentence.
    :returns: Score from 0 to 100 (higher = more complex).
    """
    words = _tokenize_words_fr(sentence)
    if not words:
        return 0.0

    word_count = len(words)
    unique_words = len(set(words))
    avg_word_len = sum(len(w) for w in words) / len(words)
    char_count = sum(len(w) for w in words)

    word_norm = min(word_count / 25, 1.0)
    unique_norm = min(unique_words / 20, 1.0)
    avg_len_norm = min(avg_word_len / 8, 1.0)
    char_norm = min(char_count / 150, 1.0)

    score = (
        word_norm * 0.25
        + unique_norm * 0.25
        + avg_len_norm * 0.25
        + char_norm * 0.25
    )
    return score * 100


def get_word_rank_fr(word: str, freq_data: FrequencyData, max_rank: int = 100000) -> int:
    """Get frequency rank for a word (French).

    :param word: Word to look up (lowercase).
    :param freq_data: Loaded frequency data.
    :param max_rank: Default rank when not found.
    :returns: Frequency rank (lower = more common).
    """
    return freq_data.word_freq.get(word.lower(), max_rank)


def frequency_score_fr(sentence: str, freq_data: FrequencyData) -> float:
    """Calculate frequency score (0-100) for French based on word commonality.

    :param sentence: French sentence.
    :param freq_data: Loaded frequency data (from load_frequency_data_fr).
    :returns: Score from 0 to 100 (higher = more common = easier first).
    """
    words = _tokenize_words_fr(sentence)
    if not words:
        return 0.0
    if not freq_data.word_freq:
        return 50.0

    max_rank = 100000
    ranks = [get_word_rank_fr(w, freq_data, max_rank) for w in words]
    avg_rank = sum(ranks) / len(ranks)
    avg_score = 1.0 - min(avg_rank / max_rank, 1.0)
    top_5k = sum(1 for r in ranks if r <= 5000) / len(ranks)
    return (avg_score * 0.6 + top_5k * 0.4) * 100


def _grammar_score_fr_spacy(sentence: str) -> float | None:
    """Use spaCy fr_core_news_sm for grammar complexity (dependency depth, clause count). Returns None if unavailable."""
    if not HAS_SPACY:
        return None
    try:
        nlp = spacy.load("fr_core_news_sm")
    except OSError:
        return None
    cleaned = re.sub(r"<[^>]+>", "", sentence)
    if not cleaned.strip():
        return 0.0
    doc = nlp(cleaned)
    if not doc:
        return 0.0
    max_depth = 0
    for token in doc:
        depth = 0
        t = token
        while t.head != t:
            depth += 1
            t = t.head
        max_depth = max(max_depth, depth)
    verb_count = sum(1 for t in doc if t.pos_ == "VERB" or t.pos_ == "AUX")
    subj_count = sum(1 for t in doc if t.dep_ == "nsubj" or t.dep_ == "csubj")
    score = min(100.0, max_depth * 8 + verb_count * 5 + subj_count * 3)
    return score


def grammar_score_fr(sentence: str) -> float:
    """Grammar complexity score (0-100) for French.

    Uses spaCy fr_core_news_sm if available (dependency depth, verbs, subjects);
    otherwise heuristics (subjunctive, relatives, conditionals, commas).

    :param sentence: French sentence.
    :returns: Score from 0 to 100.
    """
    spacy_score = _grammar_score_fr_spacy(sentence)
    if spacy_score is not None:
        return min(100.0, spacy_score)

    cleaned = re.sub(r"<[^>]+>", "", sentence)
    lower = cleaned.lower()
    score = 0.0

    subjunctive_triggers = [
        "il faut que", "il faut qu'", "pour que", "bien que", "avant que",
        "quoique", "à moins que", "jusqu'à ce que", "pourvu que", "afin que",
        "bien qu'", "quoiqu'", "pour qu'", "avant qu'", "à moins qu'",
    ]
    for trigger in subjunctive_triggers:
        if trigger in lower:
            score += 12
            break

    relative_markers = [
        "lequel", "laquelle", "lesquels", "lesquelles", "dont",
        " auquel", " à laquelle", " auxquels", " auxquelles",
    ]
    for m in relative_markers:
        if m in lower:
            score += 10
            break

    if re.search(r"\bqui\b", lower):
        score += 3
    que_count = lower.count(" que ") + lower.count(" qu'")
    if que_count >= 2:
        score += 6
    elif que_count >= 1:
        score += 2
    if re.search(r"\bsi\b.*\b(était|avait|pouvait|faisait|allait|venait|voulait)", lower):
        score += 8
    comma_count = lower.count(",") + lower.count(";")
    if comma_count >= 2:
        score += min(comma_count * 3, 15)
    elif comma_count == 1:
        score += 3

    return min(score, 100.0)


def word_similarity_fr(sent_a: str, sent_b: str) -> float:
    """Jaccard similarity between word sets (for French/latin text).

    :param sent_a: First sentence.
    :param sent_b: Second sentence.
    :returns: Similarity between 0 and 1.
    """
    words_a = set(_tokenize_words_fr(sent_a))
    words_b = set(_tokenize_words_fr(sent_b))
    return _jaccard_similarity(words_a, words_b)


def compute_similarity_penalties_fr(
    sentences: list[Sentence], top_n: int = 100
) -> tuple[list[float], list[int | None], list[str | None]]:
    """Compute similarity penalties (word-based, FR)."""
    return _compute_similarity_penalties(sentences, word_similarity_fr, top_n)


def rank_sentences_fr(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
    structural_blend: float = STRUCTURAL_COMPLEXITY_BLEND,
) -> list[Sentence]:
    """Rank sentences for French (or other word-based languages) by combined score.

    Combines complexity (length, lexical diversity) and grammar (clauses, mood) into
    a single structural score to avoid double-counting correlated difficulty.

    :param sentences: Sentences to rank (text = main sentence).
    :param weights: Optional dict with structural, frequency, similarity_penalty.
    :param structural_blend: Fraction of structural from complexity (rest from grammar); 0.6 = 60% complexity, 40% grammar.
    :returns: Same list sorted by final_score (best first).
    """
    if weights is None:
        weights = {
            "structural": 0.35,
            "frequency": 0.45,
            "similarity_penalty": 0.2,
        }

    freq_data = load_frequency_data_fr()

    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity_score_fr(sent.text)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency_score_fr(sent.text, freq_data)

    print("Calculating grammar scores...")
    for sent in sentences:
        sent.grammar_score = grammar_score_fr(sent.text)

    for sent in sentences:
        structural = _structural_score(sent.complexity_score, sent.grammar_score, structural_blend)
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    print("Calculating similarity penalties...")
    penalties, similar_to_rank_indices, similar_to_texts = compute_similarity_penalties_fr(
        sentences
    )
    for sent, penalty, similar_to_rank, sim_text in zip(
        sentences, penalties, similar_to_rank_indices, similar_to_texts
    ):
        sent.similarity_penalty = penalty
        sent.similar_to_rank = similar_to_rank
        sent.similar_to_text = sim_text

    for sent in sentences:
        structural = _structural_score(sent.complexity_score, sent.grammar_score, structural_blend)
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
            - sent.similarity_penalty * weights["similarity_penalty"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)
    return sentences


# =============================================================================
# Sentence reduction with stratified complexity sampling
# =============================================================================


def reduce_sentences_stratified(
    sentences: list[Sentence],
    target_count: int,
    similarity_threshold: float = 0.5,
    similarity_fn: Callable[[str, str], float] | None = None,
) -> list[Sentence]:
    """Reduce sentences using stratified complexity sampling with aggressive deduplication.

    Strategy:
    1. Compute complexity bins from actual data percentiles
    2. Within each bin, remove similar sentences (Jaccard > threshold)
    3. Sample proportionally from each bin to reach target_count

    Bins are computed dynamically based on actual complexity distribution:
    - Bin 1 (P0-P40): 35% of target (beginner focus)
    - Bin 2 (P40-P70): 35% of target (elementary)
    - Bin 3 (P70-P90): 20% of target (intermediate)
    - Bin 4 (P90-P100): 10% of target (advanced)

    :param sentences: Already ranked sentences (sorted by final_score).
    :param target_count: Target number of sentences (e.g., 3000).
    :param similarity_threshold: Jaccard threshold for deduplication (default 0.5).
    :param similarity_fn: Optional similarity function (defaults to word_similarity).
    :returns: Reduced list of sentences maintaining complexity distribution.
    """
    if not sentences:
        return []

    if len(sentences) <= target_count:
        return sentences

    if similarity_fn is None:
        similarity_fn = word_similarity_fr

    # Compute complexity percentiles from actual data
    complexities = sorted([s.complexity_score for s in sentences])
    n = len(complexities)

    p40 = complexities[int(n * 0.40)] if n > 0 else 0
    p70 = complexities[int(n * 0.70)] if n > 0 else 0
    p90 = complexities[int(n * 0.90)] if n > 0 else 0
    p100 = complexities[-1] if n > 0 else 100

    # Define complexity bins based on percentiles and target proportions
    bins = [
        (0.0, p40, 0.35),      # Simple (P0-P40): 35%
        (p40, p70, 0.35),      # Elementary (P40-P70): 35%
        (p70, p90, 0.20),      # Intermediate (P70-P90): 20%
        (p90, p100 + 0.1, 0.10),  # Advanced (P90-P100): 10%
    ]

    print(f"\nReducing from {len(sentences)} to {target_count} sentences...")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"\nComputed complexity bins from actual data:")
    print(f"  P40  = {p40:.1f}")
    print(f"  P70  = {p70:.1f}")
    print(f"  P90  = {p90:.1f}")
    print(f"  P100 = {p100:.1f}")

    # Assign sentences to bins
    binned = [[] for _ in bins]
    for sent in sentences:
        for i, (low, high, _) in enumerate(bins):
            if i == len(bins) - 1:  # Last bin includes upper bound
                if sent.complexity_score >= low:
                    binned[i].append(sent)
                    break
            elif low <= sent.complexity_score < high:
                binned[i].append(sent)
                break

    # Print bin statistics
    bin_labels = ["Simple (P0-P40)", "Elementary (P40-P70)", "Intermediate (P70-P90)", "Advanced (P90-P100)"]
    print("\nComplexity distribution before reduction:")
    for i, ((low, high, target_prop), label) in enumerate(zip(bins, bin_labels)):
        pct = len(binned[i]) / len(sentences) * 100 if sentences else 0
        print(f"  Bin {i+1} {label:24s} [{low:5.1f}-{high:5.1f}]: {len(binned[i]):>5} sentences ({pct:4.1f}%)")

    # Deduplicate within each bin
    deduplicated_bins = []
    total_removed = 0

    for i, bin_sentences in enumerate(binned):
        if not bin_sentences:
            deduplicated_bins.append([])
            continue

        # Sort by final_score within bin (best first)
        bin_sentences.sort(key=lambda s: s.final_score, reverse=True)

        kept = []
        for sent in bin_sentences:
            # Check similarity to all kept sentences in this bin
            is_duplicate = False
            for kept_sent in kept:
                sim = similarity_fn(sent.text, kept_sent.text)
                if sim >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept.append(sent)
            else:
                total_removed += 1

        deduplicated_bins.append(kept)
        low, high, _ = bins[i]
        label = bin_labels[i]
        print(f"  Bin {i+1} {label:24s} [{low:5.1f}-{high:5.1f}]: removed {len(bin_sentences) - len(kept)}, kept {len(kept)}")

    print(f"\nTotal removed by similarity: {total_removed}")

    # Calculate target count per bin
    bin_targets = []
    for i, (low, high, target_prop) in enumerate(bins):
        target = int(target_count * target_prop)
        available = len(deduplicated_bins[i])
        final_target = min(target, available)
        bin_targets.append(final_target)

    # Adjust if we're short due to rounding or insufficient sentences
    total_allocated = sum(bin_targets)
    if total_allocated < target_count:
        # Distribute remaining quota to bins with surplus
        remaining = target_count - total_allocated
        for i in range(len(bins)):
            if remaining <= 0:
                break
            available = len(deduplicated_bins[i])
            surplus = available - bin_targets[i]
            if surplus > 0:
                add = min(surplus, remaining)
                bin_targets[i] += add
                remaining -= add

    print("\nComplexity distribution after reduction:")
    selected = []
    for i, (low, high, _) in enumerate(bins):
        target = bin_targets[i]
        bin_selected = deduplicated_bins[i][:target]
        selected.extend(bin_selected)
        pct = len(bin_selected) / target_count * 100 if target_count else 0
        label = bin_labels[i]
        print(f"  Bin {i+1} {label:24s} [{low:5.1f}-{high:5.1f}]: {len(bin_selected):>5} sentences ({pct:4.1f}%)")

    # Sort selected sentences by original rank (final_score)
    selected.sort(key=lambda s: s.final_score, reverse=True)

    print(f"\nFinal count: {len(selected)} sentences")

    return selected


# =============================================================================
# Spanish ranking (word-based, similar to French)
# =============================================================================


def complexity_score_es(sentence: str) -> float:
    """Calculate complexity score (0-100) for Spanish from structural features.

    Uses same word-based metrics as French.

    :param sentence: Spanish sentence.
    :returns: Score from 0 to 100 (higher = more complex).
    """
    return complexity_score_fr(sentence)


def frequency_score_es(sentence: str, freq_data: FrequencyData) -> float:
    """Calculate frequency score (0-100) for Spanish based on word commonality.

    :param sentence: Spanish sentence.
    :param freq_data: Loaded frequency data (from load_frequency_data_es).
    :returns: Score from 0 to 100 (higher = more common = easier first).
    """
    words = _tokenize_words_fr(sentence)
    if not words:
        return 0.0
    if not freq_data.word_freq:
        return 50.0

    max_rank = 100000
    ranks = [get_word_rank_fr(w, freq_data, max_rank) for w in words]
    avg_rank = sum(ranks) / len(ranks)
    avg_score = 1.0 - min(avg_rank / max_rank, 1.0)
    top_5k = sum(1 for r in ranks if r <= 5000) / len(ranks)
    return (avg_score * 0.6 + top_5k * 0.4) * 100


def grammar_score_es(sentence: str) -> float:
    """Grammar complexity score (0-100) for Spanish.

    Uses heuristics for subjunctive, relative clauses, conditionals.

    :param sentence: Spanish sentence.
    :returns: Score from 0 to 100.
    """
    cleaned = re.sub(r"<[^>]+>", "", sentence)
    lower = cleaned.lower()
    score = 0.0

    # Subjunctive triggers
    subjunctive_triggers = [
        "es necesario que", "es importante que", "es posible que",
        "para que", "aunque", "antes de que", "sin que", "con tal de que",
        "ojalá", "tal vez", "quizás", "puede que",
    ]
    for trigger in subjunctive_triggers:
        if trigger in lower:
            score += 12
            break

    # Relative clauses
    relative_markers = ["el cual", "la cual", "los cuales", "las cuales", "cuyo", "cuya"]
    for m in relative_markers:
        if m in lower:
            score += 10
            break

    if re.search(r"\bque\b", lower):
        score += 3

    # Conditionals
    if re.search(r"\bsi\b.*\b(fuera|hubiera|sería|haría|tendría)", lower):
        score += 8

    # Clause complexity
    comma_count = lower.count(",") + lower.count(";")
    if comma_count >= 2:
        score += min(comma_count * 3, 15)
    elif comma_count == 1:
        score += 3

    return min(score, 100.0)


def word_similarity_es(sent_a: str, sent_b: str) -> float:
    """Jaccard similarity between word sets (for Spanish).

    Reuses word tokenization from French (both are Romance languages).

    :param sent_a: First sentence.
    :param sent_b: Second sentence.
    :returns: Similarity between 0 and 1.
    """
    return word_similarity_fr(sent_a, sent_b)


def compute_similarity_penalties_es(
    sentences: list[Sentence], top_n: int = 100
) -> tuple[list[float], list[int | None], list[str | None]]:
    """Compute similarity penalties (word-based, ES)."""
    return _compute_similarity_penalties(sentences, word_similarity_es, top_n)


def rank_sentences_es(
    sentences: list[Sentence],
    weights: dict[str, float] | None = None,
    structural_blend: float = STRUCTURAL_COMPLEXITY_BLEND,
) -> list[Sentence]:
    """Rank sentences for Spanish by combined score.

    Uses word-based complexity, frequency, and grammar scoring.
    Combines complexity and grammar into structural score.

    :param sentences: Sentences to rank (text = main sentence).
    :param weights: Optional dict with structural, frequency, similarity_penalty.
    :param structural_blend: Fraction of structural from complexity (rest from grammar).
    :returns: Same list sorted by final_score (best first).
    """
    if weights is None:
        weights = {
            "structural": 0.35,
            "frequency": 0.45,
            "similarity_penalty": 0.2,
        }

    freq_data = load_frequency_data_es()

    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity_score_es(sent.text)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency_score_es(sent.text, freq_data)

    print("Calculating grammar scores...")
    for sent in sentences:
        sent.grammar_score = grammar_score_es(sent.text)

    for sent in sentences:
        structural = _structural_score(sent.complexity_score, sent.grammar_score, structural_blend)
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    print("Calculating similarity penalties...")
    penalties, similar_to_rank_indices, similar_to_texts = compute_similarity_penalties_es(
        sentences
    )
    for sent, penalty, similar_to_rank, sim_text in zip(
        sentences, penalties, similar_to_rank_indices, similar_to_texts
    ):
        sent.similarity_penalty = penalty
        sent.similar_to_rank = similar_to_rank
        sent.similar_to_text = sim_text

    for sent in sentences:
        structural = _structural_score(sent.complexity_score, sent.grammar_score, structural_blend)
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
            - sent.similarity_penalty * weights["similarity_penalty"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)
    return sentences


