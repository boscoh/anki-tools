"""Shared sentence ranking helpers.

The module holds the sentence data class, the frequency tables, the download
helpers, the similarity engine, the ranking CSV writer, and the stratified
reduction. It also holds :func:`rank_character_based`, the score driver that
a character-based language calls with its own metrics.

The module holds no language data and imports no language package. The word
metrics of a language live in ``anki_tools/languages/<lang>/rank.py``, and the
Han character helpers live in ``anki_tools/languages/mandarin/characters.py``.

See SPEC_RANK_SENTENCES.md for the specification.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from wordfreq import iter_wordlist

from anki_tools.apkg.package import AnkiPackage
from anki_tools.ranking.latin import (
    complexity_score_latin,
    frequency_score_latin,
    latin_word_similarity,
)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Sentence:
    """A sentence extracted from an Anki deck.

    :param note_id: Note ID in the deck.
    :param text: Main sentence text.
    :param original_order: Original card order index.
    :param complexity_score: Structural complexity (0-100).
    :param frequency_score: Word frequency score (0-100).
    :param similarity_penalty: Penalty for similarity to higher-ranked sentences.
    :param similar_to_rank: Rank (1-based) of most similar higher-ranked sentence.
    :param final_score: Combined ranking score.
    :param hardest_rank: Rank of the least frequent known token.
    :param reject_reason: Rule or step that removed the sentence.
    :param kept: True when kept, None when no selection ran.
    """

    note_id: int
    text: str
    original_order: int
    complexity_score: float = 0.0
    frequency_score: float = 0.0
    similarity_penalty: float = 0.0
    similar_to_rank: int | None = None  # Rank of most similar higher-ranked sentence
    similar_to_text: str | None = None  # Closest sentence text
    grammar_score: float = 0.0  # Grammar complexity 0-100
    romanization: str = ""  # Romanization of the sentence text
    final_score: float = 0.0
    complexity_bin: int = 0  # Complexity bin (0-3) for stratified reduction
    hardest_rank: int = 0  # Rank of the least frequent known token
    reject_reason: str = ""  # Rule or selection step that removed the sentence
    kept: bool | None = None  # None when no selection ran


@dataclass
class FrequencyData:
    """Word and character frequency lookup tables.

    :param word_freq: Word to frequency rank.
    :param char_freq: Character to frequency rank.
    :param vocab_levels: Word or character to vocabulary level.
    """

    word_freq: dict[str, int] = field(default_factory=dict)
    char_freq: dict[str, int] = field(default_factory=dict)
    vocab_levels: dict[str, int] = field(default_factory=dict)  # word -> level


# =============================================================================
# Phase 1: Data Extraction
# =============================================================================


#: Main sentence field names of an Anki deck that use no language name.
TEXT_FIELD_FALLBACKS = ("Sentence", "sentence", "Front")

#: Romanization field names of an Anki deck that use no language name.
ROMANIZATION_FIELD_FALLBACKS = ("Romanization", "romanization")


def _resolve_extract_fields(
    field_names: list[str],
    text_prefer: str | None,
    text_candidates: Sequence[str],
    romanization_candidates: Sequence[str],
) -> tuple[str, str | None, str | None]:
    deck = set(field_names)
    text = None
    if text_prefer and text_prefer in deck:
        text = text_prefer
    if not text:
        for c in (*text_candidates, *TEXT_FIELD_FALLBACKS):
            if c in deck:
                text = c
                break
    if not text:
        text = field_names[0] if field_names else "Front"
    order = "[counter 2]" if "[counter 2]" in deck else None
    romanization = None
    for c in (*romanization_candidates, *ROMANIZATION_FIELD_FALLBACKS):
        if c in deck:
            romanization = c
            break
    return (text, order, romanization)


def extract_sentences(
    apkg_path: str,
    model_id: int | None = None,
    *,
    text_field: str | None = None,
    text_candidates: Sequence[str] = (),
    romanization_candidates: Sequence[str] = (),
) -> list[Sentence]:
    """Extract sentences from an Anki package.

    The main text and order fields are inferred from each model's field names.
    Pass text_field to name the main field, or pass the field names of the
    language as a candidate list.

    :param apkg_path: Path to the .apkg file.
    :param model_id: Optional model ID to filter by; extracts all if None.
    :param text_field: Preferred main sentence field (inferred if None).
    :param text_candidates: Main field names of the language, tried in order.
    :param romanization_candidates: Romanization field names of the language.
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
                    field_names, text_field, text_candidates, romanization_candidates
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


def load_wordfreq_data(
    lang: str,
    *,
    lower_keys: bool = False,
    char_filter: Callable[[str], bool] | None = None,
) -> FrequencyData:
    """Load the ``wordfreq`` list of one language.

    :param lang: ``wordfreq`` language code, for example ``fr``.
    :param lower_keys: Store the words with lowercase keys.
    :param char_filter: Fill the character ranks from the letters that pass
        this test. No character ranks are built when it is None.
    :returns: Filled :class:`FrequencyData`.
    """
    data = FrequencyData()
    for rank, word in enumerate(iter_wordlist(lang, wordlist="best"), 1):
        key = word.lower() if lower_keys else word
        data.word_freq[key] = rank
    if char_filter is not None:
        for word, rank in data.word_freq.items():
            for char in word:
                if char_filter(char) and char not in data.char_freq:
                    data.char_freq[char] = rank
    print(f"Loaded {len(data.word_freq)} words from wordfreq")
    return data







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




SIMILARITY_PENALTY_SCALE = 20

SIMILARITY_DELETE_THRESHOLD_RATIO = 0.7
SIMILARITY_CONSIDER_DELETE_PENALTY = SIMILARITY_PENALTY_SCALE * SIMILARITY_DELETE_THRESHOLD_RATIO  # flag for deletion when Jaccard >= ratio

STRUCTURAL_COMPLEXITY_BLEND = 0.6


def structural_score(
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
    repeated_filter: Callable[[str], bool] | None = None,
) -> list[Sentence]:
    """Rank sentences with the scores of one language.

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
    :param repeated_filter: Test that marks a sentence with a repeated pattern.
        Repeating sentences are dropped before the scores are computed.
    :returns: The same list, sorted by final score, best first.
    """
    if preprocess:
        preprocess(sentences)

    if repeated_filter is not None:
        before = len(sentences)
        sentences = [s for s in sentences if not repeated_filter(s.text)]
        removed = before - len(sentences)
        if removed:
            print(f"Filtered {removed} repeated-pattern sentences")

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


#: Default weights of a word-based language: structural, frequency, similarity.
WORD_WEIGHTS = {"structural": 0.35, "frequency": 0.45, "similarity_penalty": 0.2}


def rank_word_based(
    sentences: list[Sentence],
    weights: dict[str, float],
    *,
    grammar: Callable[[str], float],
    freq_loader: Callable[[], FrequencyData],
    structural_blend: float = STRUCTURAL_COMPLEXITY_BLEND,
) -> list[Sentence]:
    """Rank the sentences of a word-based Latin-script language.

    The structural term blends the complexity score and the grammar score, so
    the two correlated measures count once. The caller gives the grammar score
    and the frequency loader of its language. The Latin-script word metrics and
    the similarity engine are shared.

    :param sentences: Sentences to rank.
    :param weights: Weights of structural, frequency, and similarity_penalty.
    :param grammar: Grammar score of one sentence, 0 to 100.
    :param freq_loader: Frequency data loader of the language.
    :param structural_blend: Share of the complexity term in the structural
        score.
    :returns: The same list, sorted by final score, best first.
    """
    freq_data = freq_loader()

    print("Calculating complexity scores...")
    for sent in sentences:
        sent.complexity_score = complexity_score_latin(sent.text)

    print("Calculating frequency scores...")
    for sent in sentences:
        sent.frequency_score = frequency_score_latin(sent.text, freq_data)

    print("Calculating grammar scores...")
    for sent in sentences:
        sent.grammar_score = grammar(sent.text)

    for sent in sentences:
        structural = structural_score(
            sent.complexity_score, sent.grammar_score, structural_blend
        )
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
        )

    sentences.sort(key=lambda s: s.final_score, reverse=True)

    print("Calculating similarity penalties...")
    penalties, similar_to_rank_indices, similar_to_texts = (
        compute_similarity_penalties(sentences, latin_word_similarity)
    )
    for sent, penalty, similar_to_rank, sim_text in zip(
        sentences, penalties, similar_to_rank_indices, similar_to_texts
    ):
        sent.similarity_penalty = penalty
        sent.similar_to_rank = similar_to_rank
        sent.similar_to_text = sim_text

    for sent in sentences:
        structural = structural_score(
            sent.complexity_score, sent.grammar_score, structural_blend
        )
        sent.final_score = (
            sent.frequency_score * weights["frequency"]
            + (100 - structural) * weights["structural"]
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
    """Write the ranking CSV for reorder_deck and for inspection.

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
            structural = structural_score(
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
    not caught here. A language with a repeated-pattern filter runs that filter
    before this step, so the worst offenders are already gone.

    Bins are computed dynamically based on actual complexity distribution:
    - Bin 1 (P0-P40): 35% of target (beginner focus)
    - Bin 2 (P40-P70): 35% of target (elementary)
    - Bin 3 (P70-P90): 20% of target (intermediate)
    - Bin 4 (P90-P100): 10% of target (advanced)

    :param sentences: Sentences with complexity/frequency scores computed.
    :param target_count: Target number of sentences (e.g., 3000).
    :param similarity_threshold: Jaccard threshold for deduplication (default 0.5).
    :param similarity_fn: Optional similarity function (defaults to the Latin
        word similarity).
    :param min_chars: Skip deduplication for sentences with fewer characters than this (default 0 = disabled).
    :returns: Reduced list of sentences ordered by complexity bins, frequency within bins.
    """
    if not sentences:
        return []

    if len(sentences) <= target_count:
        return sentences

    similarity: Callable[[str, str], float]
    if similarity_fn is None:
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
