"""French adapter of the shared core gates K1 to K20.

The module binds the shared :mod:`anki_tools.ranking.coregates` to the French
:class:`anki_tools.ranking.scan.CoreProfile`. Usage: ``anki fix fr --core
--check`` calls :func:`check_core`.
"""

from __future__ import annotations

from pathlib import Path

from anki_tools.languages.french.core_types import FRENCH_CORE_PROFILE
from anki_tools.ranking import coregates as _shared
from anki_tools.ranking.coregates import (
    CoreGate,  # noqa: F401
    Gate,  # noqa: F401
    GateResult,  # noqa: F401
    GATES,  # noqa: F401
    gate_audio,  # noqa: F401
    gate_bold,  # noqa: F401
    gate_card_count,  # noqa: F401
    gate_counts,  # noqa: F401
    gate_coverage,  # noqa: F401
    gate_covered1,  # noqa: F401
    gate_covered2,  # noqa: F401
    gate_filter,  # noqa: F401
    gate_gloss,  # noqa: F401
    gate_hashes,  # noqa: F401
    gate_hit,  # noqa: F401
    gate_identity,  # noqa: F401
    gate_names,  # noqa: F401
    gate_order,  # noqa: F401
    gate_outside,  # noqa: F401
    gate_sense,  # noqa: F401
    gate_similarity,  # noqa: F401
    gate_tail,  # noqa: F401
    gate_templates,  # noqa: F401
    gate_unranked,  # noqa: F401
)
from anki_tools.sentence_pack.audit import print_audit


class CoreFiles(_shared.CoreFiles):
    """French-bound lazy reader of the core output files."""

    def __init__(self, apkg_path: Path) -> None:
        super().__init__(Path(apkg_path), FRENCH_CORE_PROFILE)


def run_gates(files: CoreFiles) -> list[GateResult]:
    """Run every French core gate and catch read errors."""
    return _shared.run_gates(files)


def check_core(output_path: Path) -> int:
    """Run gates K1 to K20 on the files of a French core output.

    :param output_path: Output APKG file.
    :returns: The exit code of the gate table, 0 on a pass.
    """
    return print_audit(_shared.run_gates(CoreFiles(Path(output_path))))
