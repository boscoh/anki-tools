"""Card template partials shared by the deck builders."""

from __future__ import annotations

from pathlib import Path

CHAR_LINKS_PATH = Path("templates/char_links.html")


def load_char_links(path: Path = CHAR_LINKS_PATH) -> str:
    """Read the script that links each Han character to the CC-Canto dictionary.

    :param path: HTML partial that holds the ``<script>`` block.
    :returns: The script text, or an empty string when the file is absent.
    """
    path = Path(path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()
