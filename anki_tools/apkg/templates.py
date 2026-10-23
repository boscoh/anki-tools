"""Card template and CSS files shared by the deck builders."""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
SHARED_TEMPLATE_DIR = TEMPLATES_DIR / "shared"
DEFAULT_TEMPLATE_DIR = TEMPLATES_DIR / "default"
SHARED_CSS_PATH = SHARED_TEMPLATE_DIR / "card.css"
CHAR_LINKS_PATH = SHARED_TEMPLATE_DIR / "char_links.html"


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


def load_css_layers(*paths: Path) -> str:
    """Read the CSS files in order. A later rule overrides an earlier rule.

    :param paths: CSS files, the shared sheet first.
    :returns: The joined CSS text, empty when every file is absent.
    """
    parts = [text for text in (load_optional_text(path) for path in paths) if text]
    return "\n".join(parts)


def load_char_links(path: Path = CHAR_LINKS_PATH) -> str:
    """Read the script that links each Han character to the CC-Canto dictionary.

    :param path: HTML partial that holds the ``<script>`` block.
    :returns: The script text, or an empty string when the file is absent.
    """
    return load_optional_text(path)
