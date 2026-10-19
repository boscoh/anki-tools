"""Shared Swadesh sentence-pack pipeline for every language.

The package holds the language-neutral stages of a sentence pack: the Tatoeba
cache (S0), the candidate pool (S2), the greedy selection (S3), the audio cache
(S4), the APKG build (S5), and the audit gates. A language profile supplies
every language-specific value and function.
"""

from __future__ import annotations
