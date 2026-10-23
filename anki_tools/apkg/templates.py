"""Card template and CSS files shared by the deck builders."""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
DEFAULT_TEMPLATE_DIR = TEMPLATES_DIR / "default"
CHAR_LINKS_PATH = TEMPLATES_DIR / "char_links.html"


def load_text(path: Path) -> str:
    """Read a template or CSS file.

    :param path: File to read.
    :raises FileNotFoundError: The file does not exist.
    :returns: The text of the file, without the final newline.
    """
    return Path(path).read_text(encoding="utf-8").strip()


def load_optional_text(path: Path) -> str:
    """Read a template or CSS file, or give an empty string when it is absent.

    :param path: File to read.
    :returns: The text of the file, or an empty string.
    """
    path = Path(path)
    if not path.exists():
        return ""
    return load_text(path)


def load_char_links(path: Path = CHAR_LINKS_PATH) -> str:
    """Read the script that links each Han character to the CC-Canto dictionary.

    :param path: HTML partial that holds the ``<script>`` block.
    :returns: The script text, or an empty string when the file is absent.
    """
    return load_optional_text(path)
