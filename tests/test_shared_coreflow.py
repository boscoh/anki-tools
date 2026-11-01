"""Tests for the shared core flow, the shared card filter, and the Spanish adapter.

Every test is offline. The token lists are built by hand, so no test loads the
frequency list or ffprobe. The one test that needs spaCy skips when the Spanish
model is absent.
"""

from __future__ import annotations

import importlib.util
import re
from collections import Counter
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Collection

from anki_tools.languages.spanish import core as spanish_core
from anki_tools.ranking import cardfilter, coreflow
from anki_tools.ranking.core import CoreCandidate, CoreOptions, CoreResult
from anki_tools.ranking.corelist import CoreTarget
from anki_tools.ranking.scan import CoreProfile
from anki_tools.sentence_pack.english_sense import SenseRow, read_senses

SPANISH_TOKENIZER = "es_core_news_sm"
WORD_PATTERN = re.compile(r"[^\W\d_]+")
HAS_SPANISH_MODEL = importlib.util.find_spec(SPANISH_TOKENIZER) is not None


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
        cards, tokens, targets = _hit_fixture()
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


class TestFilterCoreCards:
    def test_a_clean_card_passes(self) -> None:
        cards = [_card()]

        passed, reasons = coreflow.filter_core_cards(
            cards, spanish_core.SPANISH_CORE_PROFILE
        )

        assert passed == [0]
        assert reasons == [""]



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


class TestFiles:
    def test_read_senses_gives_the_first_sense_word(self, tmp_path: Path) -> None:
        path = tmp_path / "senses.tsv"
        path.write_text(
            "word\tsenses\tfalse_friends\nBeber\tDrink,Eat\tHave\n", encoding="utf-8"
        )

        assert read_senses(path) == {
            ("F", "beber"): SenseRow(
                frozenset({"drink", "eat"}), frozenset({"have"}), (), "drink"
            )
        }


def _filter_cards(
    cards: list[coreflow.InputCard],
) -> tuple[list[int], list[str]]:
    return cardfilter.filter_cards(
        cards,
        card_rule=partial(
            cardfilter.filter_reason, profile=spanish_core.SPANISH_CORE_PROFILE
        ),
        dedupe_key=spanish_core.SPANISH_CORE_PROFILE.dedupe_key,
    )


class TestCardFilter:
    def test_a_clean_card_passes(self) -> None:
        passed, reasons = _filter_cards([_card()])

        assert passed == [0]
        assert reasons == [""]

    def test_f2_rejects_a_card_outside_the_window(self) -> None:
        passed, reasons = _filter_cards([_card("El perro.")])

        assert passed == []
        assert reasons[0] == "length_window"

    def test_f3_rejects_a_digit(self) -> None:
        passed, reasons = _filter_cards([_card("El perro tiene 2 años.")])

        assert passed == []
        assert reasons[0] == "digit_marks"

    def test_script_rejects_a_foreign_letter(self) -> None:
        passed, reasons = _filter_cards([_card("El perro come хлеб.")])

        assert passed == []
        assert reasons[0] == "script"

    def test_f4_rejects_a_missing_final_mark(self) -> None:
        passed, reasons = _filter_cards([_card("El perro come pan")])

        assert passed == []
        assert reasons[0] == "marks"

    def test_f5_rejects_two_final_marks(self) -> None:
        passed, reasons = _filter_cards([_card("El perro come. Pan.")])

        assert passed == []
        assert reasons[0] == "marks"

    def test_f6_rejects_a_quote_mark(self) -> None:
        passed, reasons = _filter_cards([_card('El perro dice "hola".')])

        assert passed == []
        assert reasons[0] == "marks"

    def test_f7_rejects_two_internal_marks(self) -> None:
        passed, reasons = _filter_cards([_card("Sí, no, tal vez.")])

        assert passed == []
        assert reasons[0] == "marks"

    def test_f9_rejects_a_duplicate_sentence(self) -> None:
        cards = [_card(), _card(index=1)]

        passed, reasons = _filter_cards(cards)

        assert passed == [0]
        assert reasons[1] == "unique"
