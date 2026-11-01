"""The package import graph must stay acyclic.

A cycle between two modules forces a function-level import, and it hides which
module owns a name. The test peels the module graph from the leaves: a module
may stay only when every module it imports has already gone. Leftover modules
sit on a cycle.
"""

from __future__ import annotations

from pathlib import Path

from tests.ast_fixture import imported_names

ROOT = Path(__file__).resolve().parents[1]


def _modules() -> dict[str, Path]:
    found = {}
    for path in (ROOT / "anki_tools").rglob("*.py"):
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return found


def test_the_import_graph_has_no_cycle() -> None:
    modules = _modules()
    known = set(modules)
    graph = {name: (imported_names(path) & known) - {name} for name, path in modules.items()}

    resolved: set[str] = set()
    while True:
        ready = {name for name, deps in graph.items() if name not in resolved}
        ready = {name for name in ready if graph[name] <= resolved}
        if not ready:
            break
        resolved |= ready

    cycles = sorted(known - resolved)
    assert cycles == [], f"import cycle among {cycles}"
