"""Builders that the cull tests share."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from anki_tools.apkg import deck_writer
from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.shape import OutputShape, write_package
from anki_tools.gate_files import GateFileReader
from anki_tools.languages.french import core as french_core
from anki_tools.languages.french import cull as french_cull
from anki_tools.languages.spanish import cull as spanish_cull
from anki_tools.ranking import assemble, coreflow, corelist, cullflow
from anki_tools.ranking.core import CoreOptions
from anki_tools.ranking.identity import DeckIdentity
from anki_tools.ranking.scan import CoreProfile
from anki_tools.sentence_pack.english_sense import SenseRow
from tests.deck_fixture import write_mp3s

ENGLISH = "A short English line."
CONTENT_POS = corelist.CONTENT_POS


def gate_reader(profile: Any, **values: Any) -> Any:
    """Give a gate reader with ``profile`` that returns each given value by name.

    The type is ``Any``: the gates name ``CoreFiles`` or ``CullFiles`` in their
    signatures, and the reader is a stand-in for both.
    """
    reader = GateFileReader({name: lambda value=value: value for name, value in values.items()})
    reader.profile = profile
    return reader


def card(
    sentence: str,
    index: int = 0,
    english: str = ENGLISH,
    audio: str = "a.mp3",
) -> coreflow.InputCard:
    return coreflow.InputCard(
        note_id=1000 + index,
        sentence=sentence,
        english=english,
        audio=audio,
        index=index,
    )


def target(lemma: str, rank: int, deck_count: int = 3) -> corelist.TargetWord:
    return corelist.TargetWord(
        lemma=lemma,
        rank=rank,
        deck_count=deck_count,
        content_share=1.0,
        weight=1 / math.log2(rank + 1),
    )


def freq_row(
    word: str,
    count: int = 10,
    pm: float = 5.0,
    written: float = 5.0,
) -> cullflow.FrequencyRow:
    return cullflow.FrequencyRow(
        word=word,
        tatoeba_count=count,
        tatoeba_pm=pm,
        written_pm=written,
        zipf=4.0,
    )


def select(
    cards: list[coreflow.InputCard],
    tokens: list[list[tuple[str, str]]],
    targets: list[corelist.TargetWord],
    ranks: dict[str, int],
    **options: Any,
) -> cullflow.SelectionResult:
    """Select cards with the test defaults: no exempt word, a cap of 10 cards."""
    options = {"max_cards": 10, "exempt_top": 0, "content_pos": CONTENT_POS} | options
    return cullflow.select_cards(cards, tokens, targets, ranks, **options)


def tiny_deck(
    tmp_path: Path,
    fields: list[str],
    rows: list[dict[str, str]],
    model_name: str,
    audio: list[str],
    filename: str = "input.apkg",
) -> Path:
    """Write a source deck of ``rows`` with one stub MP3 for each name in ``audio``."""
    media = [str(path) for path in write_mp3s(tmp_path, audio)]
    source = tmp_path / filename
    AnkiPackage.create(
        source,
        "Source Deck",
        fields=fields,
        cards=rows,
        media_files=media,
        model_name=model_name,
    )
    return source


def core_target(
    key: str,
    tier: str = "S",
    position: int = 1,
    senses: frozenset[str] = frozenset(),
) -> corelist.CoreTarget:
    return corelist.CoreTarget(
        key=key,
        tier=tier,
        position=position,
        form=key.split(":")[-1],
        gloss="",
        sense=SenseRow(senses=senses),
    )


def core_deck(tmp_path: Path) -> Path:
    """Write a French source deck of three cards with one stub MP3 each."""
    sentences = ["Le chat dort.", "Le chat mange.", "Le lapin court."]
    english = ["The cat sleeps.", "The cat eats.", "The rabbit runs."]
    rows = [
        {
            "French": sentence,
            "sound": f"[sound:a{index}.mp3]",
            "Back": line,
            "freq ranks": "[ 1 2 ]",
        }
        for index, (sentence, line) in enumerate(zip(sentences, english))
    ]
    return tiny_deck(
        tmp_path,
        ["French", "sound", "Back", "freq ranks"],
        rows,
        french_cull.MODEL_NAME,
        [f"a{index}.mp3" for index in range(len(rows))],
    )


def _verb_tokens(
    cards: Sequence[coreflow.InputCard],
) -> list[list[tuple[str, str]]]:
    return [
        [(word.strip(".").casefold(), "VERB") for word in card.sentence.split()]
        for card in cards
    ]


def run_core(
    tmp_path: Path, core_words: int = 0, **overrides: Any
) -> tuple[deck_writer.RunFiles, Path, CoreProfile]:
    """Run the French core mode on ``core_deck``, with no spaCy and no data file."""
    profile = replace(
        french_core.FRENCH_CORE_PROFILE,
        tokenize_deck=_verb_tokens,
        load_frequency=list,
        read_names=set,
        word_table=lambda: [(1, "chat", "cat")],
        sense_path=tmp_path / "absent_senses.tsv",
        min_units=1,
        max_units=9,
        default_options=CoreOptions(
            max_cards=10, per_word=1, completion_cards=0, tail_cards=1
        ),
    )
    profile = replace(profile, **overrides)
    output = tmp_path / "core.apkg"
    run = coreflow.run_core(
        coreflow.CoreRequest(
            input_path=core_deck(tmp_path),
            output_path=output,
            core_words=core_words,
            max_cards=profile.default_options.max_cards,
        ),
        profile,
    )
    return run, output, profile


CULL_FIELDS = ["French", "sound", "Back", "freq ranks"]
CULL_SENTENCES = [
    "Le chat aime la fille.",
    "Le chat voit le chien.",
    "Le chat aime la maison.",
    "Le chat mange la soupe.",
    "Le chat dort sur le lit.",
]
CULL_ENGLISH = [
    "The cat likes the girl.",
    "The cat sees the dog.",
    "The cat likes the house.",
    "The cat eats the soup.",
    "The cat sleeps on the bed.",
]


def cull_deck(tmp_path: Path) -> Path:
    """Write a French source deck of five cat sentences with one stub MP3 each."""
    rows = [
        {
            "French": sentence,
            "sound": f"[sound:a{index}.mp3]",
            "Back": english,
            "freq ranks": "",
        }
        for index, (sentence, english) in enumerate(zip(CULL_SENTENCES, CULL_ENGLISH))
    ]
    audio = [f"a{index}.mp3" for index in range(len(rows))]
    return tiny_deck(tmp_path, CULL_FIELDS, rows, french_cull.MODEL_NAME, audio)


SPANISH_FIELDS = ["SpanishSentence", "Audio", "EnglishSentence"]
SPANISH_SENTENCES = [
    "El gato come el pescado.",
    "El gato mira por la ventana.",
    "El gato duerme en la cama.",
    "El gato bebe la leche fría.",
    "El gato corre por el jardín.",
]
SPANISH_ENGLISH = [
    "The cat eats the fish.",
    "The cat looks out of the window.",
    "The cat sleeps in the bed.",
    "The cat drinks the cold milk.",
    "The cat runs through the garden.",
]


def spanish_cull_deck(tmp_path: Path) -> Path:
    """Write a Spanish source deck of five cat sentences with one stub MP3 each."""
    rows = [
        {
            "SpanishSentence": sentence,
            "Audio": f"[sound:a{index}.mp3]",
            "EnglishSentence": english,
        }
        for index, (sentence, english) in enumerate(zip(SPANISH_SENTENCES, SPANISH_ENGLISH))
    ]
    audio = [f"a{index}.mp3" for index in range(len(rows))]
    model_name = spanish_cull.SPANISH_CULL_PROFILE.input_model_name
    return tiny_deck(tmp_path, SPANISH_FIELDS, rows, model_name, audio)


def one_word_profile(
    tmp_path: Path, profile: cullflow.CullProfile, word: str
) -> cullflow.CullProfile:
    """Give ``profile`` with ``word`` as the one frequency row and its sense row."""
    senses = tmp_path / "senses.tsv"
    senses.write_text(f"word\tsenses\tfalse_friends\n{word}\tcat\t\n", encoding="utf-8")
    return replace(
        profile, load_frequency=lambda: [freq_row(word)], sense_path=senses
    )


def run_cull(tmp_path: Path) -> tuple[Path, cullflow.CullProfile]:
    """Run the French cull on ``cull_deck`` with the one target word ``chat``."""
    profile = one_word_profile(tmp_path, french_cull.FRENCH_CULL_PROFILE, "chat")
    output = tmp_path / "gated.apkg"
    cullflow.run_cull(
        cullflow.CullRequest(
            input_path=cull_deck(tmp_path),
            output_path=output,
            max_cards=5,
            target_words=1,
            no_reserve=True,
        ),
        profile,
    )
    return output, profile


def write_core_package(
    source: Path,
    output: Path,
    cards: list[coreflow.InputCard],
    kept: list[int],
    profile: CoreProfile,
) -> dict[str, int]:
    identity = DeckIdentity(output)
    values, _bolded, _unbolded = coreflow.note_values(
        source, cards, kept, profile, {}
    )
    return write_package(
        source,
        output,
        OutputShape(
            deck_name=identity.title,
            model_name=identity.model_name,
            input_model_name=profile.input_model_name,
            fields=assemble.output_fields(profile),
            templates=assemble.deck_templates(profile),
            css=profile.card_css(),
        ),
        values,
    )
