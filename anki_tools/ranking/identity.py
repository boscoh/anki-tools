"""One identity for every artifact name of an output package.

The output file stem of a run is the source of truth. The deck title, the note
type name, the manifest ``source.model`` value, and the side-file stems equal
that stem, so a rename is one edit. This module is the only place that builds
these names.

The core flow and the cull flow cross the seam. The mode suffix table is the
one default output stem per language and mode. ADR 0015 records the decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Directory of the default output files.
OUTPUT_DIR = Path("apkg")

#: Suffix of the default output stem, per mode.
MODE_SUFFIX: dict[str, str] = {
    "core": "core",
    "cull": "1000",
}


def default_output(language: str, mode: str) -> Path:
    """Give the default output file of one language and mode.

    :param language: Canonical language name, for example ``french``.
    :param mode: ``core`` or ``cull``.
    :raises KeyError: The mode holds no default stem.
    :returns: ``apkg/<language>_<mode suffix>.apkg``.
    """
    return OUTPUT_DIR / f"{language}_{MODE_SUFFIX[mode]}.apkg"


@dataclass(frozen=True)
class DeckIdentity:
    """The one name that the written deck, its note type, and its side files share.

    :param output_path: Output APKG file of the run.
    """

    output_path: Path

    @property
    def stem(self) -> str:
        """Give the output file stem, the source of truth of every name."""
        return Path(self.output_path).stem

    @property
    def title(self) -> str:
        """Give the deck title, which equals the output file stem."""
        return self.stem

    @property
    def model_name(self) -> str:
        """Give the note type name, which equals the deck title."""
        return self.stem

    def side(self, suffix: str) -> Path:
        """Give one side file of the run, beside the output file.

        :param suffix: File suffix after the stem, for example ``core.csv``.
        :returns: The side file path.
        """
        return Path(self.output_path).with_name(f"{self.stem}.{suffix}")

    def core_sides(self) -> tuple[Path, Path, Path, Path]:
        """Give the card CSV, the coverage CSV, the manifest, and the rank CSV."""
        return (
            self.side("core.csv"),
            self.side("coverage.csv"),
            self.side("manifest.json"),
            self.side("rank.csv"),
        )

    def cull_sides(self) -> tuple[Path, Path, Path]:
        """Give the cull CSV, the coverage CSV, and the manifest."""
        return (
            self.side("cull.csv"),
            self.side("coverage.csv"),
            self.side("manifest.json"),
        )
