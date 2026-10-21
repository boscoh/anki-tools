"""Tests for the layout rule: shared code holds no language code."""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import anki_tools

PACKAGE_DIR = Path(anki_tools.__file__).parent
SHARED_DIRS = (PACKAGE_DIR / "ranking", PACKAGE_DIR / "sentence_pack")
LANGUAGE_PREFIX = "anki_tools.languages"
#: Name tokens that mark a symbol as the code of one language. A script name
#: such as ``latin`` or ``han`` is not in the set, because a script is not a
#: language.
LANGUAGE_TOKENS = frozenset({
    "zh",
    "cmn",
    "yue",
    "fr",
    "el",
    "es",
    "chinese",
    "mandarin",
    "cantonese",
    "french",
    "greek",
    "spanish",
    "jyutping",
    "pinyin",
    "hsk",
    "appledaily",
})

def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def _symbol_names(path: Path) -> list[str]:
    """Give the top-level function, class, and constant names of a module.

    :param path: Path to the Python module.
    :returns: The names, in file order.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names.extend(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def test_shared_modules_import_no_language_package() -> None:
    offenders = []
    for directory in SHARED_DIRS:
        for path in sorted(directory.glob("*.py")):
            for module in _imported_modules(path):
                if module.startswith(LANGUAGE_PREFIX):
                    offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_shared_modules_hold_no_language_symbol() -> None:
    """A shared module names no language in a function, class, or constant."""
    offenders = []
    for directory in SHARED_DIRS:
        for path in sorted(directory.glob("*.py")):
            for name in _symbol_names(path):
                tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
                if tokens & LANGUAGE_TOKENS:
                    offenders.append(f"{path.name}: {name}")
    assert offenders == []


def _loaded_language_modules(import_line: str) -> str:
    code = (
        "import sys\n"
        f"{import_line}\n"
        "print(sorted(name for name in sys.modules "
        "if name.startswith('anki_tools.languages')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def test_ranking_rank_pulls_in_no_language_package() -> None:
    assert _loaded_language_modules("import anki_tools.ranking.rank") == "[]"


def test_package_import_pulls_in_no_language_package() -> None:
    assert _loaded_language_modules("import anki_tools") == "[]"
