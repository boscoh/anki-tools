"""The one route behind the four ``fix`` deck commands.

Each command declares its parameters, its language, and its profile values, and
this module owns the shared path check, the ``--check`` guard, the mode branch,
and the request build. A new language adds a command and a
:class:`FixLanguage` value, not a route. ADR 0006 keeps one command per deck.

The seam sits between the CLI and the flows. The four commands and the two
modes cross it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from anki_tools.ranking import coreflow, cullflow, identity
from anki_tools.ranking.coregates import check_core
from anki_tools.ranking.cullgates import check_cull
from anki_tools.ranking.scan import CoreProfile


@dataclass(frozen=True)
class FixLanguage:
    """The values of one deck command that the fix route reads.

    :param name: Canonical language name, for example ``french``.
    :param label: Display name of the deck commands, for example ``French``.
    :param core: Core mode view of the language.
    :param cull: Cull mode view of the language, or None when it has none.
    """

    name: str
    label: str
    core: CoreProfile
    cull: cullflow.CullProfile | None = None

    @property
    def modes(self) -> tuple[str, ...]:
        """Give the mode flags of the deck, in message order."""
        if self.cull is None:
            return ("--core",)
        return ("--select", "--core")


def run_fix(
    language: FixLanguage,
    *,
    apkg_path: Path,
    output: Path | None = None,
    core: bool = False,
    select: bool = False,
    cards: int | None = None,
    reserve: bool = True,
    words: int = 0,
    check: bool = False,
) -> None:
    """Run the mode of a fix command, or its gates when ``check`` is set.

    :param language: Values of the deck command.
    :param apkg_path: Input APKG file.
    :param output: Output APKG file, or None for the default name of the mode.
    :param core: Build the core deck.
    :param select: Run the cull.
    :param cards: Core and select mode: card cap.
    :param reserve: Select mode: add the Swadesh reserve cards.
    :param words: Core mode: tier F words after the Swadesh set.
    :param check: Run the gates on the output and do not run the mode again.
    :raises SystemExit: No mode is set, two modes are set, or a gate fails.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return
    if check and not (core or select):
        raise SystemExit(f"Error: --check needs {' or '.join(language.modes)}.")
    if not any((core, select)):
        raise SystemExit(f"Error: choose a mode: {' '.join(language.modes)}")
    if core and select:
        raise SystemExit(f"Error: choose one mode: {' '.join(language.modes)}")
    if core:
        _run_core(language, apkg_path, output, cards, words, check)
        return
    if select and language.cull is not None:
        _run_cull(language, apkg_path, output, cards, reserve, check)
        return
    raise SystemExit(f"Error: {language.label} has no --select mode.")


def _run_core(
    language: FixLanguage,
    apkg_path: Path,
    output: Path | None,
    cards: int | None,
    words: int,
    check: bool,
) -> None:
    """Build the core deck of a deck command, or run its gates."""
    profile = language.core
    output_path = output or identity.default_output(language.name, "core")
    if check:
        code = check_core(output_path, profile)
        if code:
            raise SystemExit(code)
        return
    print("=" * 50)
    print(f"Processing {language.label} Anki Deck: core mode")
    print("=" * 50)
    options = profile.default_options
    if cards is not None:
        options = replace(options, max_cards=cards)
    request = coreflow.CoreRequest(
        input_path=apkg_path,
        output_path=output_path,
        core_words=words,
        options=options,
    )
    if profile.spoken_weight is not None:
        request = replace(request, spoken_weight=profile.spoken_weight)
    coreflow.run_core(request, profile)


def _run_cull(
    language: FixLanguage,
    apkg_path: Path,
    output: Path | None,
    cards: int | None,
    reserve: bool,
    check: bool,
) -> None:
    """Build the cull deck of a deck command, or run its gates."""
    profile = language.cull
    if profile is None:
        return
    output_path = output or identity.default_output(language.name, "cull")
    if check:
        code = check_cull(output_path, profile)
        if code:
            raise SystemExit(code)
        return
    request = cullflow.CullRequest(
        input_path=apkg_path,
        output_path=output_path,
        no_reserve=not reserve,
    )
    if cards is not None:
        request = replace(request, max_cards=cards)
    cullflow.run_cull(request, profile)
