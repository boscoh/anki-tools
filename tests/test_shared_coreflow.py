"""Tests for the shared core flow, the shared card filter, and the Spanish adapter.

Every test is offline. The token lists are built by hand, so no test loads the
frequency list or ffprobe.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import replace
from typing import Collection

import pytest

from anki_tools.languages.spanish import core as spanish_core
from anki_tools.ranking import coreflow
from anki_tools.ranking.core import CoreCandidate, CoreOptions, CoreResult
from anki_tools.ranking.corelist import CoreTarget
from anki_tools.ranking.scan import CoreProfile
from anki_tools.sentence_pack.english_sense import SenseRow
from tests.shape_fixture import PROFILES, shape_rules

WORD_PATTERN = re.compile(r"[^\W\d_]+")
TWO = ("El perro come pan.", "El gato come pan.")


def _card(
    sentence: str = "El perro come pan.",
    index: int = 0,
    english: str = "The dog eats bread.",
    audio: str = "a.mp3",
) -> coreflow.InputCard:
    return coreflow.InputCard(
        note_id=1000 + index,
        sentence=sentence,
        english=english,
        audio=audio,
        index=index,
    )


def _target(
    key: str,
    tier: str = "S",
    position: int = 1,
    form: str = "",
    gloss: str = "",
    senses: frozenset[str] = frozenset(),
    false_friends: frozenset[str] = frozenset(),
) -> CoreTarget:
    return CoreTarget(
        key=key,
        tier=tier,
        position=position,
        form=form or key.split(":")[-1],
        gloss=gloss,
        sense=SenseRow(senses=senses, false_friends=false_friends),
    )


def _hits(sentence: str, forms: Collection[str]) -> dict[str, int]:
    """Count the whole words of each form in the sentence."""
    words = WORD_PATTERN.findall(sentence.casefold())
    return {form: words.count(form) for form in forms if form in words}


def _profile(**overrides: object) -> CoreProfile:
    """Give the Spanish profile with the given hooks replaced."""
    values: dict[str, object] = {"hits": _hits}
    values.update(overrides)
    return replace(spanish_core.SPANISH_CORE_PROFILE, **values)


def _hit_fixture() -> tuple[
    list[coreflow.InputCard], list[list[tuple[str, str]]], list[CoreTarget]
]:
    cards = [_card("Yo bebo agua.", english="I drink water.")]
    tokens = [[("beber", "VERB")]]
    targets = [
        _target("S1:agua", position=1, form="agua", gloss="water"),
        _target("F:beber", tier="F", position=2, senses=frozenset({"drink"})),
    ]
    return cards, tokens, targets


class TestCoreHits:
    def test_a_hit_marks_the_card_and_its_targets(self) -> None:
        cards, tokens, targets = _hit_fixture()

        hits, failures, present = coreflow.core_hits(
            cards, tokens, [0], targets, _profile()
        )

        assert hits == {0: ("F:beber", "S1:agua")}
        assert failures == {}
        assert present == {"S1:agua": True, "F:beber": True}

    def test_a_tier_s_hit_without_the_gloss_is_a_sense_failure(self) -> None:
        _fixture_cards, tokens, targets = _hit_fixture()
        cards = [_card("Yo bebo agua.", english="I drink wine.")]

        hits, failures, _present = coreflow.core_hits(
            cards, tokens, [0], targets, _profile()
        )

        assert hits == {0: ("F:beber",)}
        assert failures == {"S1:agua": 1}

    def test_a_tier_f_hit_needs_its_sense(self) -> None:
        cards, tokens, _targets = _hit_fixture()
        targets = [
            _target("F:beber", tier="F", position=1, senses=frozenset({"eat"}))
        ]

        hits, failures, present = coreflow.core_hits(
            cards, tokens, [0], targets, _profile()
        )

        assert hits == {0: ()}
        assert failures == {"F:beber": 1}
        assert present == {"F:beber": True}

    def test_a_tier_s_false_friend_blocks_the_form_hit(self) -> None:
        cards, tokens, _targets = _hit_fixture()
        targets = [
            _target(
                "S1:agua",
                position=1,
                form="agua",
                gloss="water",
                false_friends=frozenset({"drink"}),
            )
        ]

        hits, failures, present = coreflow.core_hits(
            cards, tokens, [0], targets, _profile()
        )

        assert "S1:agua" not in hits[0]
        assert failures["S1:agua"] == 1
        assert present["S1:agua"] is True

    def test_a_token_outside_the_core_list_is_ignored(self) -> None:
        _cards, _tokens, targets = _hit_fixture()
        cards = [_card("Yo bebo vino.", english="I drink wine.")]
        tokens = [[("perro", "NOUN")]]

        hits, failures, present = coreflow.core_hits(
            cards, tokens, [0], targets, _profile()
        )

        assert hits == {0: ()}
        assert failures == {}
        assert present == {}

    def test_a_function_word_token_is_ignored(self) -> None:
        _cards, _tokens, targets = _hit_fixture()
        cards = [_card("Yo bebo vino.", english="I drink wine.")]
        tokens = [[("beber", "DET")]]

        hits, _failures, present = coreflow.core_hits(
            cards, tokens, [0], targets, _profile()
        )

        assert hits == {0: ()}
        assert present == {}


def _manifest_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    CoreResult,
    list[CoreCandidate],
    dict[str, int],
    CoreOptions,
]:
    rows: list[dict[str, object]] = [
        {"index": 0, "kept": 1, "pass": "seed", "new_lemmas": 1},
        {"index": 1, "kept": 1, "pass": "tail", "new_lemmas": 2},
        {"index": 2, "kept": 0, "pass": "", "new_lemmas": 0},
    ]
    coverage: list[dict[str, object]] = [
        {
            "key": "S1:a",
            "word": "a",
            "tier": "S",
            "count": 1,
            "quota": 2,
            "status": "partial",
        },
        {
            "key": "F:b",
            "word": "b",
            "tier": "F",
            "count": 0,
            "quota": 2,
            "status": "absent",
        },
    ]
    result = CoreResult(
        kept=[0, 1], uses=Counter({"S1:a": 1}), quotas={"S1:a": 2, "F:b": 2}
    )
    candidates = [
        CoreCandidate(0, ("S1:a",), ("x",), 5),
        CoreCandidate(1, ("F:b",), ("y",), 5),
        CoreCandidate(2, ("F:b",), ("z",), 5),
    ]
    ranks = {"x": 10}
    return rows, coverage, result, candidates, ranks, CoreOptions(hard_top=100)


class TestManifestCounts:
    def test_the_counts_follow_the_rows_and_the_coverage(self) -> None:
        rows, coverage, result, candidates, ranks, options = _manifest_fixture()

        counts = coreflow.manifest_counts(
            rows,
            coverage,
            result,
            candidates,
            [0, 1],
            ranks,
            options,
            bolded=1,
            unbolded=1,
        )

        assert counts["cards"] == 2
        assert counts["targets"] == 1
        assert counts["covered1"] == 1
        assert counts["covered2"] == 0
        assert counts["full"] == 0
        assert counts["absent"] == 1
        assert counts["outside"] == 1
        assert counts["outside_rare"] == 0
        assert counts["tail_words"] == 2
        assert counts["unranked"] == 1
        assert counts["bolded"] == 1
        assert counts["unbolded"] == 1
        assert counts["sense_rows"] == 0
        assert counts["rows"] == 3

    def test_a_rare_outside_word_is_counted(self) -> None:
        rows, coverage, result, candidates, _ranks, options = _manifest_fixture()

        counts = coreflow.manifest_counts(
            rows,
            coverage,
            result,
            candidates,
            [0, 1],
            {"x": 500},
            options,
        )

        assert counts["outside"] == 1
        assert counts["outside_rare"] == 1


class TestCardFilter:
    @pytest.mark.parametrize("language, sentence, expected", shape_rules("spanish"))
    def test_a_card_gets_the_first_failed_shape_rule(
        self, language: str, sentence: str, expected: str
    ) -> None:
        profile = PROFILES[language]

        passed, reasons = coreflow.filter_core_cards([_card(sentence)], profile)

        assert reasons == [expected]
        assert passed == ([] if expected else [0])

    @pytest.mark.parametrize(
        ("sentences", "english", "expected"),
        [
            pytest.param(TWO, ("The dog eats.", " the  DOG eats."), [0], id="same english"),
            pytest.param(TWO, ("The dog eats.", "The cat eats."), [0, 1], id="different english"),
            pytest.param(TWO, ("", ""), [0, 1], id="empty english"),
            pytest.param(TWO[:1] * 2, ("A.", "B."), [0], id="same sentence"),
        ],
    )
    def test_a_card_is_unique_by_its_sentence_and_its_english(
        self, sentences: tuple[str, str], english: tuple[str, str], expected: list[int]
    ) -> None:
        cards = [_card(sentences[i], i, english[i]) for i in (0, 1)]

        passed, reasons = coreflow.filter_core_cards(cards, PROFILES["spanish"])

        assert passed == expected
        assert (reasons[1] == "unique") == (expected == [0])
