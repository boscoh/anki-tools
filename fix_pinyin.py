#!/usr/bin/env python3
"""Backwards compatibility - imports from anki_tools package."""

import cyclopts

from anki_tools.pinyin import (
    fix_pinyin,
    check_pinyin,
    get_pypinyin,
    normalize_pinyin,
    format_pinyin,
    remove_tones,
    POLYPHONIC_SKIP,
    NEUTRAL_TONE_PATTERNS,
    PREFERRED_PRONUNCIATION,
)

__all__ = [
    "fix_pinyin",
    "check_pinyin",
    "get_pypinyin",
    "normalize_pinyin",
    "format_pinyin",
    "remove_tones",
    "POLYPHONIC_SKIP",
    "NEUTRAL_TONE_PATTERNS",
    "PREFERRED_PRONUNCIATION",
]

app = cyclopts.App()
app.default(fix_pinyin)

if __name__ == "__main__":
    app()
