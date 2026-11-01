"""Import reader that the layout and import graph tests share."""

from __future__ import annotations

import ast
from pathlib import Path


def imported_names(path: Path) -> set[str]:
    """Give the absolute names that a module imports.

    A ``from`` import adds the module and each ``module.name``, because the
    name can be a submodule.

    :param path: Path to the Python module.
    :returns: The imported module names.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names
