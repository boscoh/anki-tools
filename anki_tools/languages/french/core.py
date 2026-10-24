"""French core deck: the shared core flow bound to the French profile.

The module re-exports the French core adapter of :mod:`core_types` and the
French gate check of :mod:`core_gates`. It exists so the old import surface
``from anki_tools.languages.french import core`` keeps working.
"""

from __future__ import annotations

from anki_tools.languages.french.core_gates import check_core  # noqa: F401
from anki_tools.languages.french.core_types import *  # noqa: F401,F403
from anki_tools.languages.french.core_types import _write_core_model  # noqa: F401
