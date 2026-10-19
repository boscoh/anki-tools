"""Greedy set cover and ranking of the candidate sentences (stage S3).

The selector walks the word table in Swadesh index order. For each word it takes
the best unused sentence that still needs no full quota, assigns that sentence to
the covered word with the lowest index, and rejects a sentence that is too close
to a kept card.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Callable, Iterable, Sequence

from anki_tools.sentence_pack.profile import (
    LanguageProfile,
    WordRow,
    duplicate_limit,
)

RankKey = tuple[int, int, float, int, int]


def rank_key_parts(candidate: Any, profile: LanguageProfile) -> RankKey:
    """Build the sort key of a candidate, first item first.

    :param candidate: Candidate sentence.
    :param profile: Language profile.
    :returns: Unknown count, audio penalty, negated Zipf, length penalty, id.
    """
    audio_penalty = (
        0 if candidate.audio_id or not profile.prefer_native_audio else 1
    )
    length = profile.target_length(candidate)
    length_penalty = (
        0
        if profile.preferred_min_units <= length <= profile.preferred_max_units
        else 1
    )
    return (
        candidate.unknown_words,
        audio_penalty,
        -candidate.zipf_mean,
        length_penalty,
        candidate.tatoeba_id,
    )


def serialize_rank_key(parts: RankKey) -> str:
    """Serialize a ranking key tuple into the CSV string.

    :param parts: Tuple from :func:`rank_key_parts`.
    :returns: Pipe-separated key, for example ``0|1|5.120|0|123``.
    """
    unknown_words, audio_penalty, negated_zipf, length_penalty, tatoeba_id = parts
    return (
        f"{unknown_words}|{audio_penalty}|{negated_zipf:.3f}|"
        f"{length_penalty}|{tatoeba_id}"
    )


def rank_order(candidate: Any, profile: LanguageProfile) -> RankKey:
    """Sort candidates with the ranking key.

    :param candidate: Candidate sentence.
    :param profile: Language profile.
    :returns: Sort key, lowest value first.
    """
    return rank_key_parts(candidate, profile)


class SimilarityIndex:
    """Reject a sentence that is too close to a kept card.

    Only kept cards that share one bucket key with the sentence are compared, so
    the guard stays linear for a large deck.
    """

    def __init__(
        self,
        similarity: Callable[[str, str], float],
        keys_of: Callable[[str], set[str]],
        limit: float = 0.67,
    ) -> None:
        """Start an empty index.

        :param similarity: Jaccard function of two sentences.
        :param keys_of: Function that gives the bucket keys of a sentence.
        :param limit: Jaccard value at which a sentence counts as a duplicate.
        """
        self.similarity = similarity
        self.keys_of = keys_of
        self.limit = limit
        self._texts: list[str] = []
        self._by_key: dict[str, list[int]] = defaultdict(list)

    def too_similar(self, sentence: str) -> bool:
        """Check a sentence against the kept cards.

        :param sentence: Candidate sentence.
        :returns: True when a kept card reaches the Jaccard limit.
        """
        shared: set[int] = set()
        for key in self.keys_of(sentence):
            shared.update(self._by_key.get(key, ()))
        return any(
            self.similarity(sentence, self._texts[index]) >= self.limit
            for index in shared
        )

    def add(self, sentence: str) -> None:
        """Add a kept card sentence to the index.

        :param sentence: Kept card sentence.
        """
        position = len(self._texts)
        self._texts.append(sentence)
        for key in self.keys_of(sentence):
            self._by_key[key].append(position)


def assign_target(
    covered: frozenset[str],
    positions: dict[str, tuple[int, int]],
    counts: dict[str, int],
    per_word: int,
) -> str | None:
    """Pick the lowest covered form that still needs a card.

    :param covered: Forms that the sentence contains one time.
    :param positions: Mapping from form to ``(swadesh_index, variant_rank)``.
    :param counts: Current card count per form.
    :param per_word: Card limit per form.
    :returns: The target form, or None when every covered form is full.
    """
    for form in sorted(covered, key=lambda name: positions[name]):
        if counts.get(form, per_word) < per_word:
            return form
    return None


def select_sentences(
    pool: Sequence[Any],
    word_table: Sequence[WordRow],
    profile: LanguageProfile,
    *,
    per_word: int = 2,
    max_cards: int = 450,
    finalize: Callable[[Any, str], Any] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Select cards with greedy set cover.

    Forms are processed in Swadesh index order, primary form first. A sentence
    goes to the lowest covered form that still needs a card, so a sentence never
    becomes two cards. Only the assigned form gains a card.

    :param pool: Candidate list from the pool stage.
    :param word_table: Reviewed word table of the language.
    :param profile: Language profile.
    :param per_word: Card limit per form.
    :param max_cards: Hard card cap.
    :param finalize: Hook that fills the language-specific card fields.
    :returns: The cards in Swadesh order and the selection report.
    """
    positions = profile.form_positions(word_table)
    ordered_forms = sorted(positions, key=lambda form: positions[form])
    counts: dict[str, int] = dict.fromkeys(ordered_forms, 0)
    by_form: dict[str, list[Any]] = defaultdict(list)
    for candidate in pool:
        for form in candidate.covered:
            if form in counts:
                by_form[form].append(candidate)
    for candidates in by_form.values():
        candidates.sort(key=lambda item: rank_order(item, profile))

    used: set[int] = set()
    kept: list[Any] = []
    index = SimilarityIndex(
        profile.similarity, profile.similarity_keys, duplicate_limit(profile)
    )
    for form in ordered_forms:
        for candidate in by_form.get(form, ()):
            if len(kept) >= max_cards or counts[form] >= per_word:
                break
            if candidate.tatoeba_id in used:
                continue
            target = assign_target(candidate.covered, positions, counts, per_word)
            if target is None or index.too_similar(candidate.sentence):
                continue
            counts[target] += 1
            used.add(candidate.tatoeba_id)
            index.add(candidate.sentence)
            card = replace(candidate, target=target)
            if finalize is not None:
                card = finalize(card, target)
            kept.append(card)

    kept.sort(key=lambda card: (positions[card.target], rank_order(card, profile)))
    concepts = {form.swadesh_index: form.english for form in word_table}
    covered_concepts = {
        positions[form][0] for form in ordered_forms if counts[form] > 0
    }
    uncovered = [
        {
            "target": form,
            "swadesh_index": positions[form][0],
            "english": concepts.get(positions[form][0], ""),
            "pool_candidates": len(by_form.get(form, ())),
        }
        for form in ordered_forms
        if counts[form] == 0
    ]
    forms_with_card = sum(1 for form in ordered_forms if counts[form] > 0)
    report: dict[str, Any] = {
        "per_word": per_word,
        "max_cards": max_cards,
        "cards": len(kept),
        "tier_counts": dict(Counter(card.tier for card in kept)),
        "forms": {
            "total": len(ordered_forms),
            "with_card": forms_with_card,
            "ratio": round(forms_with_card / len(ordered_forms), 4)
            if ordered_forms
            else 0.0,
        },
        "core_coverage": {
            "concepts": len(concepts),
            "with_card": len(covered_concepts),
            "ratio": round(len(covered_concepts) / len(concepts), 4)
            if concepts
            else 0.0,
        },
        "uncovered": uncovered,
        "zero_candidates": [
            item["target"] for item in uncovered if not item["pool_candidates"]
        ],
        "cut_by_cap": [item["target"] for item in uncovered if item["pool_candidates"]],
    }
    return kept, report


def review_status(
    form: WordRow,
    pool_count: int,
    core_count: int,
    per_word: int,
    zipf_min: float,
) -> str | None:
    """Give the review status of one word.

    :param form: Reviewed word form.
    :param pool_count: Candidate count of the form.
    :param core_count: Core candidate count of the form.
    :param per_word: Card limit per form.
    :param zipf_min: Zipf value below which a form is rare.
    :returns: The status text, or None when the word needs no review.
    """
    if not pool_count:
        return "uncovered"
    if not core_count:
        return "scaffold_only"
    if pool_count < per_word:
        return "thin"
    if form.zipf < zipf_min:
        return "low_zipf"
    return None


def review_candidates(
    pool: Iterable[Any], profile: LanguageProfile
) -> tuple[Counter[str], Counter[str], dict[str, list[Any]]]:
    """Group the pool by form for the review CSV.

    :param pool: Candidate list.
    :param profile: Language profile.
    :returns: Pool counts, core counts, and the ranked candidates per form.
    """
    pool_counts: Counter[str] = Counter()
    core_counts: Counter[str] = Counter()
    by_form: dict[str, list[Any]] = defaultdict(list)
    for candidate in pool:
        for form in candidate.covered:
            pool_counts[form] += 1
            by_form[form].append(candidate)
            if candidate.tier == "core":
                core_counts[form] += 1
    for candidates in by_form.values():
        candidates.sort(key=lambda item: rank_order(item, profile))
    return pool_counts, core_counts, by_form
