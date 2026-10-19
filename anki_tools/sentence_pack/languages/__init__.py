"""Language profiles of the sentence pipeline.

Importing this package registers the Mandarin, French, Greek, and Cantonese
profiles.
"""

from __future__ import annotations

from . import cantonese, french, greek, mandarin
from anki_tools.sentence_pack.profile import PROFILES, get_profile, profile_names

__all__ = [
    "PROFILES",
    "cantonese",
    "french",
    "get_profile",
    "greek",
    "mandarin",
    "profile_names",
]
