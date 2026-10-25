"""French adapter of the shared core gates K1 to K20.

The module binds the shared :mod:`anki_tools.ranking.coregates` to the French
:class:`anki_tools.ranking.scan.CoreProfile`. Usage: ``anki fix fr --core
--check`` calls :func:`check_core`.
"""

from __future__ import annotations

from pathlib import Path

from anki_tools.languages.french.core_types import FRENCH_CORE_PROFILE
from anki_tools.ranking import coregates as _shared
from anki_tools.ranking.coregates import GateResult
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
