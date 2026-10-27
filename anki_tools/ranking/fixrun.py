"""The one route behind the four ``fix`` deck commands.

Each command declares its parameters, its language, and its profile values, and
this module owns the shared path check, the ``--check`` guard, the mode branch,
the request build, and the rank report. A new language adds a command and a
:class:`FixLanguage` value, not a route. ADR 0006 keeps one command per deck.

The seam sits between the CLI and the flows. The four commands and the three
modes cross it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from anki_tools.ranking import coreflow, cullflow, identity
from anki_tools.ranking.coregates import check_core
from anki_tools.ranking.cullgates import check_cull
from anki_tools.ranking.pipeline import (
    RankedDeck,
    RankPaths,
    rank_paths,
    run_ranked_deck,
)
from anki_tools.ranking.rank import Sentence
from anki_tools.ranking.scan import CoreProfile


@dataclass(frozen=True)
class FixLanguage:
    """The values of one deck command that the fix route reads.

    :param name: Canonical language name, for example ``french``.
    :param label: Display name of the deck commands, for example ``French``.
    :param core: Core mode view of the language.
    :param cull: Cull mode view of the language, or None when it has none.
    :param rank: Ranker of the language, or None when it has no rank mode.
    :param deck_kwargs: Extra keyword arguments of the rank run, for example the
        sentence field names of the language.
    :param after: Hook that runs the language steps and returns the extra DONE
        lines, or None for no extra lines.
    :param count: Ranked count hook, or None to use the ranked length.
    """

    name: str
    label: str
    core: CoreProfile
    cull: cullflow.CullProfile | None = None
    rank: Callable[[list[Sentence]], list[Sentence]] | None = None
    deck_kwargs: Mapping[str, Any] = field(default_factory=dict)
    after: Callable[[RankPaths, RankedDeck], Sequence[str]] | None = None
    count: Callable[[int], int] | None = None

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
    text_field: str | None = None,
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
    :param text_field: Rank mode: preferred sentence field, or None to infer.
    :param core: Build the core deck.
    :param select: Run the cull.
    :param cards: Core and select mode: card cap.
    :param reserve: Select mode: add the Swadesh reserve cards.
    :param words: Core mode: tier F words after the Swadesh set.
    :param check: Run the gates on the output and do not run the mode again.
    :raises SystemExit: A gate of the ``--check`` run fails.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return
    if check and not (core or select):
        print(f"Error: --check needs {' or '.join(language.modes)}.")
        return
    if core and select:
        print("Error: --core and --select cannot be used together.")
        return
    if core:
        _run_core(language, apkg_path, output, cards, words, check)
        return
    if select and language.cull is not None:
        _run_cull(language, apkg_path, output, cards, reserve, check)
        return
    _run_rank(language, apkg_path, output, text_field)


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


def _run_rank(
    language: FixLanguage,
    apkg_path: Path,
    output: Path | None,
    text_field: str | None,
) -> None:
    """Rank and reorder a deck with the ranker of the language."""
    if language.rank is None:
        return
    paths = rank_paths(apkg_path, output or identity.default_output(language.name, "rank"))
    _print_rank_banner(paths, language.label)
    print("Step 1: Ranking sentences...")
    deck = run_ranked_deck(
        paths, rank=language.rank, text_field=text_field, **language.deck_kwargs
    )
    if not deck.sentences:
        print("No sentences found!")
        return
    ranked = (
        language.count(len(deck.ranked))
        if language.count is not None
        else len(deck.ranked)
    )
    extra = language.after(paths, deck) if language.after is not None else ()
    _print_rank_done(paths, ranked, deck.stats, extra_lines=extra)


def _print_rank_banner(paths: RankPaths, label: str) -> None:
    """Print the banner that starts a fix rank run.

    :param paths: Input, output, and ranking file of the run.
    :param label: Deck language, for example ``Chinese``.
    """
    print("=" * 50)
    print(f"Processing {label} Anki Deck")
    print("=" * 50)
    print(f"Input:   {paths.input_path}")
    print(f"Output:  {paths.output_path}")
    print(f"Ranking: {paths.ranking_path}")
    print()


def _print_rank_done(
    paths: RankPaths,
    ranked_count: int,
    stats: dict[str, Any],
    *,
    extra_lines: Sequence[str] = (),
) -> None:
    """Print the DONE table of a rank run.

    :param paths: Input, output, and ranking file of the run.
    :param ranked_count: Sentences that the ranker kept.
    :param stats: Card counts of the reorder step.
    :param extra_lines: Preformatted lines to print after the card counts.
    """
    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {ranked_count}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    for line in extra_lines:
        print(line)
    print(f"  Output:              {paths.output_path}")
    print(f"  Ranking CSV:         {paths.ranking_path}")
