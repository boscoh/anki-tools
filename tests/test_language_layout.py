"""Tests for the layout rule: shared code imports no language package."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import anki_tools

PACKAGE_DIR = Path(anki_tools.__file__).parent
SHARED_DIRS = (PACKAGE_DIR / "ranking", PACKAGE_DIR / "sentence_pack")
LANGUAGE_PREFIX = "anki_tools.languages"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_shared_modules_import_no_language_package() -> None:
    offenders = []
    for directory in SHARED_DIRS:
        for path in sorted(directory.glob("*.py")):
            for module in _imported_modules(path):
                if module.startswith(LANGUAGE_PREFIX):
                    offenders.append(f"{path.name}: {module}")
    assert offenders == []


def test_ranking_rank_pulls_in_no_language_package() -> None:
    code = (
        "import sys\n"
        "import anki_tools.ranking.rank\n"
        "print(sorted(name for name in sys.modules "
        "if name.startswith('anki_tools.languages')))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert "anki_tools.languages.cantonese" not in result.stdout
