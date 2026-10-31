"""One gate contract, below ``ranking`` and ``sentence_pack``.

The module holds the gate value, the result row, the runner, and the file
reader contract. The cull and core flows use it from
:mod:`anki_tools.ranking.gates`. The pack flow uses it from
:mod:`anki_tools.sentence_pack.audit` through the adapter that supplies the
profile.

The module imports neither package, so either package can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Protocol, Sequence


class GateResult(NamedTuple):
    """Result of one gate.

    :param name: Gate label, for example ``K1``.
    :param passed: True when the gate accepts the files.
    :param message: Detail line of the gate table.
    """

    name: str
    passed: bool
    message: str


class GateFiles(Protocol):
    """Contract of the file reader that the gates read.

    A read error is cached and re-raised for each gate that needs the file.
    """

    @property
    def profile(self) -> Any:
        """Give the language hooks of the run."""
        ...

    def rows(self) -> list[dict[str, str]]:
        """Read the card CSV rows."""
        ...

    def coverage(self) -> list[dict[str, str]]:
        """Read the coverage CSV rows."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Read the manifest object."""
        ...

    def params(self) -> dict[str, Any]:
        """Read the run options of the manifest."""
        ...

    def output(self) -> dict[str, Any]:
        """Read the output APKG."""
        ...

    def notes(self) -> dict[int, list[str]]:
        """Read the note fields of the output deck, keyed by note id."""
        ...

    def fields(self) -> list[str]:
        """Read the field names of the only note type."""
        ...


@dataclass(frozen=True)
class Gate:
    """One gate value built from a ``(passed, message)`` check function.

    :param name: Gate label, for example ``K1``.
    :param check: Check that gives ``(passed, message)`` for the files.
    """

    name: str
    check: Callable[[Any], tuple[bool, str]]

    def result(self, files: GateFiles) -> GateResult:
        """Run the check and wrap the result with the gate name.

        :param files: Lazy reader of the output files.
        :returns: The result of the gate.
        """
        passed, message = self.check(files)
        return GateResult(self.name, passed, message)


def run_gates(gates: Sequence[Gate], files: GateFiles) -> list[GateResult]:
    """Run every gate and catch the error of a gate that cannot read a file.

    :param gates: Gate table of the mode.
    :param files: Lazy reader of the output files.
    :returns: One result per gate.
    """
    results: list[GateResult] = []
    for gate in gates:
        try:
            results.append(gate.result(files))
        except Exception as error:  # noqa: BLE001 - a gate reports the error
            results.append(
                GateResult(gate.name, False, f"{type(error).__name__}: {error}")
            )
    return results
