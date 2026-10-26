"""Steps of ``anki fix zh`` that follow the reorder.

The rank pipeline writes the reordered deck. This module then fixes the pinyin
field and writes the repository card templates into every model.

The steps belong to the Mandarin package, not to the command: the command
knows only that the pipeline has a tail hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from anki_tools.apkg.templates import apply_templates
from anki_tools.languages.mandarin.pinyin import fix_pinyin
from anki_tools.ranking.pipeline import RankPaths, RankedDeck

DEV_NULL = Path("/dev/null")


def after_rank(paths: RankPaths, _deck: RankedDeck) -> Sequence[str]:
    """Fix the pinyin field, then apply the card templates.

    :param paths: Input, output, and ranking file of the run.
    :param _deck: Ranked deck of the run. The steps do not read it.
    :returns: The extra DONE lines of the steps.
    """
    print("\nFixing pinyin...")
    fix_pinyin(
        paths.output_path,
        output=paths.output_path,
        csv_output=DEV_NULL,
        verbose=False,
    )
    print("\nApplying card templates...")
    models, applied = apply_templates(Path(paths.output_path), verbose=False)
    if applied:
        print(f"  Applied: {', '.join(applied)} to {models} model(s)")
    return (f"  Templates applied:   {models} model(s)",)
