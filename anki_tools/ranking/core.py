"""Core-word selection: a greedy cover with an example quota and a word cost.

The cover keeps the card with the best score, where the score is the value of
the target examples that the card adds, minus the cost of the words outside the
core. The passes of the spec fill the quota in stages, then the tail adds the
wide vocabulary.

The module holds no language data. The language module gives the targets of a
card, its outside words, and its unit count. See
``specs/core-word-small-deck.md`` sections 4 and 6.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

#: The value of example 1, 2, and 3 of one target.
DEMAND = (1.0, 0.35, 0.12)

#: The passes in order, with the reason label of each.
PASSES = ("seed", "depth", "completion", "tail", "topup")

#: A candidate and a kept card share a bucket when one lemma Jaccard value
#: reaches this limit.
SIMILARITY_LIMIT = 0.67


@dataclass(frozen=True)
class CoreOptions:
    """Settings of the selection.

    :param max_cards: Card cap of the whole run.
    :param per_word: Examples wanted for a content target, 1 to 3.
    :param third_examples: Number of top targets that get a third example.
    :param ubiquitous_share: Share of the candidates above which a target gets
        quota 1, because a second example adds no skill.
    :param completion_cards: Card cap of the completion pass.
    :param tail_cards: Card cap of the tail pass, 0 for no tail.
    :param outside_budget: Cap on the distinct outside lemmas, 0 for no cap.
    :param support_top: Outside rank that costs 1.0.
    :param hard_top: Outside rank that costs 1.8; above it the cost is 3.0.
    :param lambda_new: Cost of one new outside lemma.
    :param lambda_repeat: Cost of one repeated outside lemma.
    :param lambda_len: Cost per unit above ``simple_units``.
    :param lambda_seed: Cost factor of the seed pass.
    :param simple_units: Free unit count of the length cost.
    :param similarity: Lemma Jaccard limit of a duplicate pair.
    """

    max_cards: int = 300
    per_word: int = 2
    third_examples: int = 0
    ubiquitous_share: float = 0.25
    completion_cards: int = 120
    tail_cards: int = 0
    outside_budget: int = 0
    support_top: int = 1000
    hard_top: int = 3000
    lambda_new: float = 0.1
    lambda_repeat: float = 0.03
    lambda_len: float = 0.02
    lambda_seed: float = 0.01
    simple_units: int = 7
    similarity: float = SIMILARITY_LIMIT

    def __post_init__(self) -> None:
        """Check the values that the algorithm depends on."""
        if self.per_word < 1:
            raise ValueError("per_word must be at least 1")
        if self.per_word > 3:
            raise ValueError("per_word must be at most 3")
        if self.max_cards < 1:
            raise ValueError("max_cards must be at least 1")
        if self.tail_cards < 0:
            raise ValueError("tail_cards must be 0 or more")


@dataclass(frozen=True)
class CoreCandidate:
    """One card that passed the filter.

    :param index: Input index of the card.
    :param targets: Target keys of the card, after the sense guard.
    :param outside: Content lemmas of the card that are not targets.
    :param units: Word or Han character count of the card.
    """

    index: int
    targets: tuple[str, ...]
    outside: tuple[str, ...]
    units: int

    @property
    def lemmas(self) -> frozenset[str]:
        """Give the content lemmas of the card, targets and outside words."""
        return frozenset(self.targets) | frozenset(self.outside)


@dataclass
class CoreResult:
    """Result of the core selection.

    :param kept: Input indices of the kept cards, in pick order.
    :param uses: Kept examples per target key.
    :param quotas: Quota per target key.
    :param assigned: Assigned target key per kept card.
    :param credited: Target keys that the card credited toward a quota.
    :param score: Score of each kept card at the moment of the pick.
    :param cost: Outside cost of each kept card at the moment of the pick.
    :param pick_pass: Pass that kept each card.
    :param reasons: Reason per unkept card that the selection skipped.
    :param order: Kept cards in the final card order.
    """

    kept: list[int]
    uses: Counter[str]
    quotas: dict[str, int]
    assigned: dict[int, str] = field(default_factory=dict)
    credited: dict[int, tuple[str, ...]] = field(default_factory=dict)
    score: dict[int, float] = field(default_factory=dict)
    cost: dict[int, float] = field(default_factory=dict)
    pick_pass: dict[int, str] = field(default_factory=dict)
    reasons: dict[int, str] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)


def target_weights(positions: dict[str, int]) -> dict[str, float]:
    """Give the value weight of each target.

    :param positions: 1-based core position per target key.
    :returns: ``1 / log2(position + 1)`` per target key.
    """
    return {
        key: 1 / math.log2(position + 1) for key, position in positions.items()
    }


def target_quotas(
    positions: dict[str, int],
    shares: dict[str, float],
    options: CoreOptions,
) -> dict[str, int]:
    """Give the example quota of each target.

    A target in more than ``--ubiquitous-share`` of the candidates gets quota
    1. The first ``--third-examples`` targets of the core list get quota 3 when
    they are not ubiquitous.

    :param positions: 1-based core position per target key.
    :param shares: Share of the candidates that hold the target.
    :param options: Selection settings.
    :returns: Quota per target key, at least 1.
    """
    quotas: dict[str, int] = {}
    for key, position in positions.items():
        if shares.get(key, 0.0) > options.ubiquitous_share:
            quotas[key] = 1
        else:
            quotas[key] = options.per_word
    if options.third_examples:
        top = sorted(positions, key=lambda key: (positions[key], key))
        for key in top[: options.third_examples]:
            if quotas[key] > 1:
                quotas[key] = 3
    return quotas


def _rank_multiplier(rank: int, options: CoreOptions) -> float:
    """Give the cost multiplier of one outside lemma."""
    if rank <= options.support_top:
        return 1.0
    if rank <= options.hard_top:
        return 1.8
    return 3.0


def card_cost(
    candidate: CoreCandidate,
    introduced: set[str],
    ranks: dict[str, int],
    options: CoreOptions,
    lambda_new: float | None = None,
) -> float:
    """Give the cost of one card at the current state.

    :param candidate: Card to cost.
    :param introduced: Outside lemmas that a kept card holds.
    :param ranks: Lemma to blended frequency rank.
    :param options: Selection settings.
    :param lambda_new: Cost of a new outside word, or None for the option.
    :returns: The sum of the new-word, repeat-word, and length terms.
    """
    new = [word for word in candidate.outside if word not in introduced]
    repeats = len(candidate.outside) - len(new)
    new_cost = sum(
        _rank_multiplier(ranks.get(word, 10**6), options) for word in new
    )
    cost_new = options.lambda_new if lambda_new is None else lambda_new
    return (
        cost_new * new_cost
        + options.lambda_repeat * repeats
        + options.lambda_len * max(0, candidate.units - options.simple_units)
    )


def _value(
    candidate: CoreCandidate,
    uses: Counter[str],
    stop: dict[str, int],
    weights: dict[str, float],
    demand: Sequence[float],
) -> float:
    """Give the value of the examples that a card adds."""
    total = 0.0
    for key in candidate.targets:
        count = uses[key]
        if count < stop.get(key, 0):
            total += weights[key] * demand[min(count, len(demand) - 1)]
    return total


def _tail_gain(
    candidate: CoreCandidate,
    introduced: set[str],
    ranks: dict[str, int],
) -> float:
    """Give the sum of the frequency weights of the new outside lemmas."""
    return sum(
        1 / math.log2(ranks.get(word, 10**6) + 1)
        for word in candidate.outside
        if word not in introduced
    )


def _is_duplicate(
    candidate: CoreCandidate,
    kept_sets: dict[int, frozenset[str]],
    shared: dict[str, list[int]],
    limit: float,
) -> bool:
    """Tell whether the card is too close to a kept card that shares a lemma."""
    buckets = {index for word in candidate.lemmas for index in shared.get(word, ())}
    lemmas = candidate.lemmas
    for index in buckets:
        other = kept_sets[index]
        union = len(lemmas | other)
        if union and len(lemmas & other) / union >= limit:
            return True
    return False


def _pick(
    candidates: dict[int, CoreCandidate],
    order: Sequence[int],
    kept: set[int],
    score_of: Callable[[int], float | None],
    accept: float,
    reject: Callable[[int], str | None],
) -> int | None:
    """Return the best card of one scan, or None.

    A candidate that ``reject`` names is skipped. A card that scores at or
    below ``accept`` is skipped.
    """
    best_index = None
    best_key = None
    for index in order:
        if index in kept:
            continue
        score = score_of(index)
        if score is None:
            continue
        if score <= accept:
            continue
        reason = reject(index)
        if reason is not None:
            continue
        key = (round(score, 9), -len(candidates[index].outside), -candidates[index].units, -index)
        if best_key is None or key > best_key:
            best_index, best_key = index, key
    return best_index


def select(
    candidates: Sequence[CoreCandidate],
    *,
    positions: dict[str, int],
    quotas: dict[str, int],
    ranks: dict[str, int],
    options: CoreOptions,
) -> CoreResult:
    """Run the seed, depth, completion, tail, and top-up passes.

    :param candidates: Cards that passed the filter, in input order.
    :param positions: 1-based core position per target key.
    :param quotas: Quota per target key.
    :param ranks: Lemma to blended frequency rank.
    :param options: Selection settings.
    :raises ValueError: A kept-card count that cannot hold the quota.
    :returns: The kept cards, the uses, and the order.
    """
    weights = target_weights(positions)
    uses: Counter[str] = Counter()
    introduced: set[str] = set()
    kept: list[int] = []
    kept_set: set[int] = set()
    score: dict[int, float] = {}
    cost: dict[int, float] = {}
    pick_pass: dict[int, str] = {}
    reasons: dict[int, str] = {}
    credited: dict[int, tuple[str, ...]] = {}
    kept_sets: dict[int, frozenset[str]] = {}
    shared: dict[str, list[int]] = {}
    order = [candidate.index for candidate in candidates]
    by_index = {candidate.index: candidate for candidate in candidates}

    def reject(index: int) -> str | None:
        if _is_duplicate(
            by_index[index], kept_sets, shared, options.similarity
        ):
            reasons[index] = "similar"
            return "similar"
        if (
            options.outside_budget
            and len(introduced | set(by_index[index].outside))
            > options.outside_budget
        ):
            reasons[index] = "outside_budget"
            return "outside_budget"
        return None

    def on_pick(index: int, pass_name: str, card_score: float) -> None:
        kept.append(index)
        kept_set.add(index)
        score[index] = card_score
        cost[index] = card_cost(by_index[index], introduced, ranks, options)
        pick_pass[index] = pass_name
        candidate = by_index[index]
        credited[index] = tuple(
            key for key in candidate.targets if uses[key] < quotas.get(key, 0)
        )
        for key in credited[index]:
            uses[key] += 1
        kept_sets[index] = candidate.lemmas
        for word in candidate.lemmas:
            shared.setdefault(word, []).append(index)
        introduced.update(candidate.outside)

    def run_pass(
        name: str,
        stop: dict[str, int],
        demand: Sequence[float],
        lambda_new: float,
        accept: float,
        limit: int,
        tail: bool = False,
    ) -> None:
        picked = 0
        while len(kept) < options.max_cards and picked < limit:
            def score_of(index: int) -> float | None:
                candidate = by_index[index]
                if tail:
                    gain = _tail_gain(candidate, introduced, ranks)
                    return gain if gain > 0 else None
                value = _value(candidate, uses, stop, weights, demand)
                if value <= 0:
                    return None
                return value - card_cost(
                    candidate, introduced, ranks, options, lambda_new
                )

            index = _pick(
                by_index, order, kept_set, score_of, accept, reject
            )
            if index is None:
                return
            on_pick(index, name, score_of(index) or 0.0)
            picked += 1

    # Seed: one example of every target, with a small cost as a tie-break.
    seed_stop = {key: 1 for key in positions}
    run_pass(
        "seed", seed_stop, (1.0,), options.lambda_seed, 0.0, len(positions)
    )
    # Depth: the full quota, with the outside cost.
    run_pass(
        "depth", quotas, DEMAND, options.lambda_new, 0.0, len(positions) * 3
    )
    # Completion: fill the quota, even at a negative score.
    if options.completion_cards:
        run_pass(
            "completion",
            quotas,
            DEMAND,
            options.lambda_new,
            -math.inf,
            options.completion_cards,
        )
    # Top-up: a third example for the top targets.
    if options.third_examples:
        run_pass(
            "topup", quotas, DEMAND, options.lambda_new, 0.0, options.max_cards
        )
    # Tail: the widest vocabulary in the rest of the budget.
    if options.tail_cards:
        tail_stop = {key: 0 for key in positions}
        run_pass(
            "tail", tail_stop, DEMAND, 0.0, 0.0, options.tail_cards, tail=True
        )

    for index in order:
        if index not in kept_set:
            reasons.setdefault(index, "max_cards")

    result = CoreResult(
        kept=kept,
        uses=uses,
        quotas=quotas,
        score=score,
        cost=cost,
        pick_pass=pick_pass,
        reasons=reasons,
        credited=credited,
    )
    result.assigned = assign_targets(kept, candidates, positions)
    result.order = order_cards(result, candidates, positions, ranks)
    return result


def assign_targets(
    kept: Iterable[int],
    candidates: Sequence[CoreCandidate],
    positions: dict[str, int],
) -> dict[int, str]:
    """Give each kept card the lowest-position target that is still free.

    :param kept: Kept cards in pick order.
    :param candidates: Cards that passed the filter.
    :param positions: 1-based core position per target key.
    :returns: Assigned target key per kept card. A card with no free target is
        not in the mapping.
    """
    by_index = {candidate.index: candidate for candidate in candidates}
    taken: set[str] = set()
    assigned: dict[int, str] = {}
    for index in kept:
        options = [key for key in by_index[index].targets if key not in taken]
        if not options:
            continue
        key = min(options, key=lambda item: (positions[item], item))
        assigned[index] = key
        taken.add(key)
    return assigned


def order_cards(
    result: CoreResult,
    candidates: Sequence[CoreCandidate],
    positions: dict[str, int],
    ranks: dict[str, int],
) -> list[int]:
    """Order the kept cards as a curriculum.

    The keys are the position of the assigned target, the example number of
    that target, the new outside words, the unit count, and the input index.

    :param result: Selection result with the kept cards and the assignment.
    :param candidates: Cards that passed the filter.
    :param positions: 1-based core position per target key.
    :param ranks: Lemma to blended frequency rank.
    :returns: Kept cards in the final order.
    """
    by_index = {candidate.index: candidate for candidate in candidates}
    examples: dict[tuple[int, str], int] = {}
    seen: Counter[str] = Counter()
    for index in result.kept:
        for key in by_index[index].targets:
            seen[key] += 1
            examples[(index, key)] = seen[key]

    introduced: set[str] = set()
    keys: list[tuple[tuple, int]] = []
    for index in result.kept:
        candidate = by_index[index]
        target = result.assigned.get(index)
        new_words = [
            word for word in candidate.outside if word not in introduced
        ]
        introduced.update(candidate.outside)
        new_count = len(new_words)
        if target is None:
            key = (len(positions) + 1, 0, new_count, candidate.units, index)
        else:
            key = (
                positions[target],
                examples.get((index, target), 1),
                new_count,
                candidate.units,
                index,
            )
        keys.append((key, index))
    keys.sort()
    return [index for _key, index in keys]
