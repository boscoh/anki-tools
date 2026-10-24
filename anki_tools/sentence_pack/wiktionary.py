"""Wiktionary Swadesh module fetch and parse helpers.

Several language profiles read the ``Module:Swadesh/data/<lang>`` wikitext to
check their reviewed word forms against the canonical module terms. The fetch
and the parse are identical apart from the module page, so they live here.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

LINK_PATTERN = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
NOTE_PATTERN = re.compile(r"\s*\([^)]*\)\s*")
MODULE_ROW_START = re.compile(r"m\[(\d+)\]\s*=\s*")
MODULE_ROW = re.compile(r"^m\[(\d+)\]\s*=\s*\{(.*)\}\s*$")
MODULE_TERM = re.compile(r'term\s*=\s*"([^"]+)"')


def module_terms_url(page: str) -> str:
    """Build the MediaWiki API URL of a Swadesh module page.

    :param page: Module page name, for example ``Module:Swadesh/data/fr``.
    :returns: The MediaWiki API URL.
    """
    query = urllib.parse.urlencode(
        {
            "action": "parse",
            "page": page,
            "prop": "wikitext",
            "format": "json",
            "formatversion": "2",
        }
    )
    return f"https://en.wiktionary.org/w/api.php?{query}"


def fetch_module_terms(
    page: str, cache_path: Path, *, refresh: bool = False
) -> Path:
    """Download one Swadesh module wikitext into a cache file.

    :param page: Module page name.
    :param cache_path: Cache file of the wikitext.
    :param refresh: Ignore an existing cache file.
    :raises RuntimeError: The download or the JSON parse fails.
    :returns: The cache path.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        return cache_path
    request = urllib.request.Request(
        module_terms_url(page), headers={"User-Agent": "anki-tools"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.load(response)
        text = data["parse"]["wikitext"]
    except (OSError, ValueError, KeyError) as error:
        raise RuntimeError(f"cannot read {module_terms_url(page)}: {error}") from error
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return cache_path


def parse_module_terms(text: str) -> dict[int, tuple[str, ...]]:
    """Parse one Swadesh module wikitext into terms per index.

    A module row can span several lines, so the parser reads each row up to its
    closing brace.

    :param text: Module wikitext.
    :returns: Mapping from index to its raw terms.
    """
    terms: dict[int, tuple[str, ...]] = {}
    for match in MODULE_ROW_START.finditer(text):
        index = int(match.group(1))
        start = match.end()
        depth = 0
        end = 0
        for position in range(start, len(text)):
            char = text[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
        if not end:
            continue
        terms[index] = tuple(MODULE_TERM.findall(text[start:end]))
    return terms


def parse_module_terms_lines(text: str) -> dict[int, tuple[str, ...]]:
    """Parse one-line module rows into terms per index.

    :param text: Module wikitext where every row stays on one line.
    :returns: Mapping from index to its raw terms.
    """
    terms: dict[int, tuple[str, ...]] = {}
    for line in text.splitlines():
        match = MODULE_ROW.match(line.strip())
        if not match:
            continue
        terms[int(match.group(1))] = tuple(MODULE_TERM.findall(match.group(2)))
    return terms


def read_module_terms(
    page: str,
    cache_path: Path,
    parse: Callable[[str], dict[int, tuple[str, ...]]] = parse_module_terms,
) -> dict[int, tuple[str, ...]]:
    """Read cached module terms, with a download on the first run.

    :param page: Module page name.
    :param cache_path: Cache file of the wikitext.
    :param parse: Parser of the wikitext.
    :returns: Mapping from index to its raw terms.
    """
    path = Path(cache_path)
    if not path.exists():
        path = fetch_module_terms(page, path)
    return parse(path.read_text(encoding="utf-8"))
