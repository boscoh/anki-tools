"""Shared types and gate contract of the French core deck.

The module holds the constants, the filter rules, and the gate contract that
:mod:`cull`, :mod:`core`, and :mod:`core_gates` share. Every gate of the core
audit is a :class:`CoreGate` value that implements the :class:`Gate` protocol,
and the gates run against one :class:`CoreFiles` reader.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple, Protocol, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.apkg.templates import TEMPLATES_DIR, load_text
from anki_tools.languages.french import cull
from anki_tools.languages.french.profile import FRENCH

LOAN_PATH = Path("vocab/french_loanwords.txt")

MIN_UNITS = 5
MAX_UNITS = 9

CORE_FIELD = "Core"
CORE_GLOSS_FIELD = "CoreGloss"

TEMPLATE_DIR = TEMPLATES_DIR / "french_core"
CORE_FRONT_TEMPLATE = load_text(TEMPLATE_DIR / "front.html")
CORE_BACK_TEMPLATE = load_text(TEMPLATE_DIR / "back.html")
CORE_TEMPLATES = (
    (CORE_FRONT_TEMPLATE, CORE_BACK_TEMPLATE),
)

CONTENT_POS = cull.CONTENT_POS
STATUSES = ("covered", "partial", "absent", "absent_sense", "unchecked")
TARGET_SEPARATOR = "|"


class GateResult(NamedTuple):
    """Result of one gate.

    :param name: Gate label, for example ``K1``.
    :param passed: True when the gate accepts the files.
    :param message: Detail line of the gate table.
    """

    name: str
    passed: bool
    message: str


class Gate(Protocol):
    """Contract of one core gate.

    :ivar name: Gate label, for example ``K1``.
    :ivar check: Run the gate on the output files and give the result.
    """

    name: str

    def check(self, files: CoreFiles) -> GateResult:
        """Run the gate on the output files and give the result."""
        ...


@dataclass
class CoreGate:
    """One gate value built from a ``(passed, message)`` check function.

    :param name: Gate label, for example ``K1``.
    :param check: Check that gives ``(passed, message)`` for the files.
    """

    name: str
    _check: Callable[[CoreFiles], tuple[bool, str]]

    def check(self, files: CoreFiles) -> GateResult:
        """Run the check and wrap the result with the gate name."""
        passed, message = self._check(files)
        return GateResult(self.name, passed, message)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row mappings."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object file."""
    try:
        text = Path(path).read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}") from error


@dataclass
class CoreFiles:
    """Lazy reader for the four files of a core run.

    :param apkg_path: Output APKG file.
    """

    apkg_path: Path
    _cache: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set the paths and start an empty cache."""
        self.apkg_path = Path(self.apkg_path)
        self.core_csv, self.coverage_csv, self.manifest_path, self.rank_csv = (
            side_paths(self.apkg_path)
        )

    def _load(self, name: str, loader: Callable[[], Any]) -> Any:
        """Load one value once and cache the value or the error."""
        if name not in self._cache:
            try:
                self._cache[name] = (True, loader())
            except Exception as error:  # noqa: BLE001 - cached for every gate
                self._cache[name] = (False, error)
        ok, value = self._cache[name]
        if not ok:
            raise value
        return value

    def rows(self) -> list[dict[str, str]]:
        """Read the card CSV rows."""
        return self._load("rows", lambda: _read_csv(self.core_csv))

    def coverage(self) -> list[dict[str, str]]:
        """Read the coverage CSV rows."""
        return self._load("coverage", lambda: _read_csv(self.coverage_csv))

    def manifest(self) -> dict[str, Any]:
        """Read the manifest JSON."""
        return self._load("manifest", lambda: _read_json(self.manifest_path))

    def rank_rows(self) -> list[dict[str, str]]:
        """Read the rank CSV rows."""
        return self._load("rank", lambda: _read_csv(self.rank_csv))

    def output(self) -> dict[str, Any]:
        """Read the output APKG and probe its audio."""
        return self._load("output", lambda: cull._read_output(self.apkg_path))

    def tokens(self) -> list[list[tuple[str, str]]]:
        """Tokenize the sentences of the output deck."""
        return self._load(
            "tokens", lambda: cull.tokenize_deck(self.output()["cards"])
        )

    def notes(self) -> dict[int, list[str]]:
        """Read the note fields of the output deck, keyed by note id."""

        def load() -> dict[int, list[str]]:
            with AnkiPackage(self.apkg_path) as package:
                return {
                    int(row["id"]): str(row["flds"]).split("\x1f")
                    for row in package.get_notes()
                }

        return self._load("notes", load)

    def fields(self) -> list[str]:
        """Read the field names of the French sentence model."""

        def load() -> list[str]:
            with AnkiPackage(self.apkg_path) as package:
                models = package.get_models()
                model = models[cull._find_model_id(models)]
                return [entry["name"] for entry in model["flds"]]

        return self._load("fields", load)

    def media(self) -> set[str]:
        """Read the media file names of the output deck."""
        return self._load("media", lambda: set(self.output()["media"]))


def read_loanwords(path: Path = LOAN_PATH) -> frozenset[str]:
    """Read the loan word list of rule C4.

    :param path: Text file, one word per line. A ``#`` line is a comment.
    :returns: Casefolded words, empty when the file is missing.
    """
    source = Path(path)
    if not source.exists():
        return frozenset()
    words = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            words.add(text.casefold())
    return frozenset(words)


def _has_predicate(tokens: Sequence[tuple[str, str]]) -> bool:
    """Give rule C1: the sentence has a verb, auxiliary, or adjective."""
    return any(pos in {"VERB", "AUX", "ADJ"} for _lemma, pos in tokens)


def _holds_name(sentence: str, names: Iterable[str]) -> bool:
    """Give rule C3: a whole word of the name list occurs in the sentence.

    French uses a word boundary. A substring test would reject a common word
    that holds a short name, for example ``tomber`` with the name ``tom``.
    Chinese and Cantonese use the substring test of the yue selector instead.
    """
    words = {word.casefold() for word in cull._WORD_PATTERN.findall(sentence)}
    return any(name.casefold() in words for name in names)


def _holds_loanword(sentence: str, loanwords: Iterable[str]) -> bool:
    """Give rule C4: a loan word occurs in the sentence.

    The rule splits the loan word on non-letters, so ``week-end`` also
    matches the two words ``week`` and ``end`` in the sentence.
    """
    words = {word.casefold() for word in cull._WORD_PATTERN.findall(sentence)}
    for loan in loanwords:
        parts = [
            part.casefold()
            for part in cull._WORD_PATTERN.findall(loan)
        ]
        if parts and all(part in words for part in parts):
            return True
    return False


def filter_core_cards(
    cards: Sequence[cull.InputCard],
    tokens: Sequence[Sequence[tuple[str, str]]],
    media: set[str],
    names: Iterable[str],
    loanwords: Iterable[str],
    *,
    min_units: int = MIN_UNITS,
    max_units: int = MAX_UNITS,
) -> tuple[list[int], list[str]]:
    """Apply rules F1 to F9 and C1 to C5 to the cards.

    :param cards: Input cards.
    :param tokens: Lemma and part of speech per card.
    :param media: File names of the input media map.
    :param names: Name list of rule C3.
    :param loanwords: Loan word list of rule C4.
    :param min_units: First unit count of the window.
    :param max_units: Last unit count of the window.
    :raises ValueError: The card list and the token list differ in size.
    :returns: The input indices that pass, and the reason per input index. A
        passed card has the empty string as its reason.
    """
    if len(cards) != len(tokens):
        raise ValueError("The card list and the token list differ in size.")
    passed, cull_reasons = cull.filter_cards(
        cards, tokens, min_words=min_units, max_words=max_units
    )
    reasons = list(cull_reasons)
    kept: list[int] = []
    for index in passed:
        card = cards[index]
        if not _has_predicate(tokens[index]):
            reasons[index] = "C1"
        elif FRENCH.has_orphan(card.sentence):
            reasons[index] = "C2"
        elif _holds_name(card.sentence, names):
            reasons[index] = "C3"
        elif _holds_loanword(card.sentence, loanwords):
            reasons[index] = "C4"
        elif not card.audio or card.audio not in media:
            reasons[index] = "C5"
        else:
            kept.append(index)
    return kept, reasons


def side_paths(output_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build the four side file paths of a core run.

    :param output_path: Output APKG file.
    :returns: The core CSV, the coverage CSV, the manifest, and the rank CSV.
    """
    path = Path(output_path)
    stem = path.with_suffix("") if path.suffix else path
    return (
        Path(f"{stem}.core.csv"),
        Path(f"{stem}.coverage.csv"),
        Path(f"{stem}.manifest.json"),
        Path(f"{stem}.rank.csv"),
    )
