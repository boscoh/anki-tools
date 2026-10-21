"""Shared sentence ranking helpers.

The module holds the sentence data class, the frequency tables, the download
helpers, the Han character helpers, the similarity engine, the ranking CSV
writer, and the stratified reduction. It also holds
:func:`rank_character_based`, the driver that the character-based languages
share.

The module holds no language data and imports no language package. The word
metrics of a language live in ``anki_tools/languages/<lang>/rank.py``.

See SPEC_RANK_SENTENCES.md for the specification.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from wordfreq import iter_wordlist

from anki_tools.apkg.package import AnkiPackage

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
    :param hardest_rank: Cantonese selection: rank of the least frequent known token.
    :param reject_reason: Cantonese selection: rule or step that removed the sentence.
    :param kept: Cantonese selection: True when kept, None when no selection ran.
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
    complexity_bin: int = 0  # Complexity bin (0-3) for stratified reduction
    hardest_rank: int = 0  # Rank of the least frequent known token (Cantonese selection)
    reject_reason: str = ""  # Rule or selection step that removed the sentence
    kept: bool | None = None  # None when no selection ran


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
    """Load the ``wordfreq`` list of one language.

    :param lang: ``wordfreq`` language code, for example ``fr``.
    :param lower_keys: Store the words with lowercase keys.
    :param build_char_freq: Fill the character ranks from the word ranks.
    :returns: Filled :class:`FrequencyData`.
    """
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


YUE_FREQUENCY_URL = (
    "https://raw.githubusercontent.com/chaaklau/appledaily-frequency"
    "/master/release/word-freq.tsv"
)
YUE_FREQUENCY_ATTRIBUTION = (
    "Cantonese word frequency: chaaklau/appledaily-frequency (CC BY 4.0)"
)
YUE_FREQUENCY_MAX_WORDS = 50000


def yue_frequency_path() -> Path:
    """Return the cache path of the Cantonese frequency list.

    Set the environment variable ``YUE_FREQUENCY_PATH`` to use another file.

    :returns: Path to the cached TSV file.
    """
    override = os.environ.get("YUE_FREQUENCY_PATH")
    if override:
        return Path(override)
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "anki-tools" / "appledaily-word-freq.tsv"


def download_cached_file(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download a file into the cache.

    The download writes a temporary file first, then renames it.

    :param url: Source URL.
    :param dest: Cache destination.
    :param force: Download again when the file already exists.
    :returns: Path to the cached file.
    """
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    print(f"Downloading {url}")
    try:
        with urlopen(url, timeout=120) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Cannot download {url}.") from exc
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")
    return dest


def download_yue_frequency(path: Path | None = None, *, force: bool = False) -> Path:
    """Download the Apple Daily Cantonese frequency list into the cache.

    The file has two tab-separated columns: word and occurrence count.

    :param path: Cache destination. Defaults to :func:`yue_frequency_path`.
    :param force: Download again when the cached file already exists.
    :returns: Path to the cached TSV file.
    """
    dest = Path(path) if path else yue_frequency_path()
    try:
        return download_cached_file(YUE_FREQUENCY_URL, dest, force=force)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Cannot download {YUE_FREQUENCY_URL}. "
            "Set YUE_FREQUENCY_PATH to a local copy of the file."
        ) from exc


def _read_word_count_tsv(path: Path) -> dict[str, int]:
    """Read a two-column word and count TSV file.

    Lines with no integer count are skipped, so a header line is ignored.

    :param path: Path to the TSV file.
    :returns: Word to occurrence count.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read the Cantonese frequency file {path}. "
            "Set YUE_FREQUENCY_PATH to a valid file."
        ) from exc
    counts: dict[str, int] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        word, count = parts
        counts[word] = counts.get(word, 0) + int(count)
    return counts


def frequency_data_from_ranks(ranks: dict[str, int]) -> FrequencyData:
    """Build word and character rank tables from word ranks.

    Characters take the rank of the first word that contains them, as in
    :func:`_load_frequency_data_wordfreq`.

    :param ranks: Word to rank, 1 is most frequent.
    :returns: Filled :class:`FrequencyData`.
    """
    data = FrequencyData()
    data.word_freq.update(ranks)
    for word, rank in sorted(ranks.items(), key=lambda item: item[1]):
        for char in get_chinese_chars(word):
            data.char_freq.setdefault(char, rank)
    return data


def _build_frequency_data(counts: dict[str, int], max_words: int) -> FrequencyData:
    """Build rank tables from word counts.

    :param counts: Word to occurrence count.
    :param max_words: Number of most common words to keep.
    :returns: Filled :class:`FrequencyData`.
    """
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ranks = {word: rank for rank, (word, _) in enumerate(ordered[:max_words], 1)}
    return frequency_data_from_ranks(ranks)


# =============================================================================
# Han character helpers
# =============================================================================


def is_repeated_zh_sentence(text: str) -> bool:
    """Check if Chinese chars in text form a pure repetition pattern.

    Catches '是 是', '走吧走吧', '好了，好了', '谢谢你，谢谢你', etc.

    :param text: Sentence text.
    :returns: True if the Chinese character sequence is a repeated sub-sequence.
    """
    chars = get_chinese_chars(text)
    n = len(chars)
    if n < 2:
        return False
    for period in range(1, n // 2 + 1):
        if n % period == 0 and chars == chars[:period] * (n // period):
            return True
    return False


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
# Phase 4: Similarity Analysis
# =============================================================================


def jaccard_similarity(items_a: set, items_b: set) -> float:
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
    return jaccard_similarity(chars_a, chars_b)


def char_similarity_zh_same_len(sent_a: str, sent_b: str) -> float:
    """Jaccard similarity of character sets, only for equal unique-character counts.

    The unique-character count avoids stutter anchors: '是 是' does not match
    '是的' only because both have 2 Chinese characters.

    :param sent_a: First Chinese sentence.
    :param sent_b: Second Chinese sentence.
    :returns: Similarity between 0 (no overlap or different sizes) and 1.
    """
    chars_a = set(get_chinese_chars(sent_a))
    chars_b = set(get_chinese_chars(sent_b))
    if len(chars_a) != len(chars_b) or not chars_a:
        return 0.0
    return jaccard_similarity(chars_a, chars_b)


def compute_similarity_penalties(
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
    """Compute similarity penalties (character-based, ZH and YUE).

    :param sentences: Sentences in rank order.
    :param top_n: Number of higher-ranked sentences to compare.
    :returns: Penalties, similar ranks, and similar texts.
    """
    return compute_similarity_penalties(sentences, char_similarity_zh, top_n)


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


def rank_character_based(
    sentences: list[Sentence],
    weights: dict[str, float],
    *,
    complexity: Callable[[str], float],
    frequency: Callable[[str, FrequencyData], float],
    penalties: Callable[
        [list[Sentence]], tuple[list[float], list[int | None], list[str | None]]
    ],
    preprocess: Callable[[list[Sentence]], None] | None = None,
    freq_loader: Callable[[], FrequencyData],
) -> list[Sentence]:
    """Rank sentences with a character-based score.

    The caller gives the word metrics of its language, so the driver holds no
    language data. Mandarin and Cantonese use it.

    :param sentences: Sentences to rank.
    :param weights: Weights of complexity, frequency, and similarity_penalty.
    :param complexity: Complexity score of one sentence, 0 to 100.
    :param frequency: Frequency score of one sentence, 0 to 100.
    :param penalties: Similarity penalties of the ranked sentences.
    :param preprocess: Optional preprocessing, for example the traditional
        conversion and the romanization of the Cantonese pipeline.
    :param freq_loader: Frequency data loader of the language.
    :returns: The same list, sorted by final score, best first.
    """
    if preprocess:
        preprocess(sentences)

    before = len(sentences)
    sentences = [s for s in sentences if not is_repeated_zh_sentence(s.text)]
    removed = before - len(sentences)
    if removed:
        print(f"Filtered {removed} repeated-character sentences")

    freq_data = freq_loader()

    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity(sent.text)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency(sent.text, freq_data)

    for sent in sentences:
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - sent.complexity_score) * weights["complexity"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    print("Calculating similarity penalties...")
    (
        penalty_values,
        similar_to_rank_indices,
        similar_to_rank_texts,
    ) = penalties(sentences)
    for sent, penalty, similar_to_rank, sim_text in zip(
        sentences, penalty_values, similar_to_rank_indices, similar_to_rank_texts
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


def _open_for_write(output_path: str):
    try:
        return open(output_path, "w", newline="", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"cannot write {output_path}: {error}") from error


def write_ranking_csv(
    sentences: list[Sentence],
    output_path: str,
    *,
    structural_blend: float = STRUCTURAL_COMPLEXITY_BLEND,
    similar_to_threshold: float = SIMILARITY_CONSIDER_DELETE_PENALTY,
) -> None:
    """Write ranking CSV for reorder_deck and inspection (ZH and FR).

    Columns: rank, note_id, sentence, romanization (if present), original_order, frequency,
    complexity, grammar, structural, similarity, similar_to, final_score, and
    hardest_rank and kept when a selection ran.
    """
    has_romanization = any(sent.romanization for sent in sentences)
    has_selection = any(sent.kept is not None for sent in sentences)

    with _open_for_write(output_path) as f:
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
        if has_selection:
            header.extend(["hardest_rank", "kept"])
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
            if has_selection:
                row.extend([sent.hardest_rank, int(bool(sent.kept))])
            writer.writerow(row)
    print(f"Wrote ranking for {len(sentences)} sentences to {output_path}")


# =============================================================================
# Sentence reduction with stratified complexity sampling
# =============================================================================


def reduce_sentences_stratified(
    sentences: list[Sentence],
    target_count: int,
    similarity_threshold: float = 0.5,
    similarity_fn: Callable[[str, str], float] | None = None,
    min_chars: int = 0,
) -> list[Sentence]:
    """Reduce sentences using stratified complexity sampling with aggressive deduplication.

    Strategy:
    1. Compute complexity bins from actual data percentiles
    2. Within each bin, rank by frequency (highest first)
    3. Remove similar sentences (Jaccard > threshold) — within each bin only
    4. Sample proportionally from each bin to reach target_count
    5. Order final deck: Bin 1 (by freq) → Bin 2 (by freq) → Bin 3 (by freq) → Bin 4 (by freq)

    Note: deduplication runs within each bin independently. Cross-bin near-duplicates
    (e.g. a simple sentence in Bin 1 and its repetition-pattern variant in Bin 2) are
    not caught here. For Chinese, is_repeated_zh_sentence() removes the worst offenders
    before this step is reached.

    Bins are computed dynamically based on actual complexity distribution:
    - Bin 1 (P0-P40): 35% of target (beginner focus)
    - Bin 2 (P40-P70): 35% of target (elementary)
    - Bin 3 (P70-P90): 20% of target (intermediate)
    - Bin 4 (P90-P100): 10% of target (advanced)

    :param sentences: Sentences with complexity/frequency scores computed.
    :param target_count: Target number of sentences (e.g., 3000).
    :param similarity_threshold: Jaccard threshold for deduplication (default 0.5).
    :param similarity_fn: Optional similarity function (defaults to the Latin
        word similarity, the historical default of the French pipeline).
    :param min_chars: Skip deduplication for sentences with fewer characters than this (default 0 = disabled).
    :returns: Reduced list of sentences ordered by complexity bins, frequency within bins.
    """
    if not sentences:
        return []

    if len(sentences) <= target_count:
        return sentences

    similarity: Callable[[str, str], float]
    if similarity_fn is None:
        from anki_tools.ranking.latin import latin_word_similarity

        similarity = latin_word_similarity
    else:
        similarity = similarity_fn

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
    print("\nComputed complexity bins from actual data:")
    print(f"  P40  = {p40:.1f}")
    print(f"  P70  = {p70:.1f}")
    print(f"  P90  = {p90:.1f}")
    print(f"  P100 = {p100:.1f}")

    # Assign sentences to bins and sort by frequency within each bin
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

    # Sort each bin by frequency (highest first)
    for bin_sentences in binned:
        bin_sentences.sort(key=lambda s: s.frequency_score, reverse=True)

    # Print bin statistics
    bin_labels = ["Simple (P0-P40)", "Elementary (P40-P70)", "Intermediate (P70-P90)", "Advanced (P90-P100)"]
    print("\nComplexity distribution before reduction:")
    for i, ((low, high, target_prop), label) in enumerate(zip(bins, bin_labels)):
        pct = len(binned[i]) / len(sentences) * 100 if sentences else 0
        print(f"  Bin {i+1} {label:24s} [{low:5.1f}-{high:5.1f}]: {len(binned[i]):>5} sentences ({pct:4.1f}%)")

    # Deduplicate within each bin (already sorted by frequency)
    deduplicated_bins = []
    total_removed = 0

    for i, bin_sentences in enumerate(binned):
        if not bin_sentences:
            deduplicated_bins.append([])
            continue

        # Sentences already sorted by frequency (highest first)
        kept = []
        for sent in bin_sentences:
            # Short sentences have unreliable Jaccard similarity — skip dedup
            if min_chars and len(sent.text.replace(" ", "")) < min_chars:
                kept.append(sent)
                continue

            # Check similarity to all kept sentences in this bin
            is_duplicate = False
            for kept_sent in kept:
                # Also skip if the reference is short — it shouldn't anchor removals
                if min_chars and len(kept_sent.text.replace(" ", "")) < min_chars:
                    continue
                sim = similarity(sent.text, kept_sent.text)
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

        # Assign bin number for final ordering
        for sent in bin_selected:
            sent.complexity_bin = i

        selected.extend(bin_selected)
        pct = len(bin_selected) / target_count * 100 if target_count else 0
        label = bin_labels[i]
        print(f"  Bin {i+1} {label:24s} [{low:5.1f}-{high:5.1f}]: {len(bin_selected):>5} sentences ({pct:4.1f}%)")

    # Sort by bin, then by final_score within bin (includes similarity penalties for variety)
    # Bin 1 (best scores) → Bin 2 (best scores) → Bin 3 (best scores) → Bin 4 (best scores)
    selected.sort(key=lambda s: (s.complexity_bin, -s.final_score))

    print(f"\nFinal count: {len(selected)} sentences")
    print("Final ordering: By complexity bin (ascending), then by final score (descending) within bin")

    return selected
