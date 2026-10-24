"""Tests of the Cantonese jyutping helper.

An OpenCC converter takes about 15 ms to build, and the deck loop calls the
helper once per note, so the converter must be built one time.
"""

from __future__ import annotations

import opencc
import pytest

from anki_tools.languages.cantonese import jyutping


class _FakeConverter:
    """A converter that returns the text unchanged."""

    def convert(self, text: str) -> str:
        return text


def test_the_converter_is_built_one_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three notes give one OpenCC build."""
    calls: list[str] = []

    def fake_opencc(name: str) -> _FakeConverter:
        calls.append(name)
        return _FakeConverter()

    monkeypatch.setattr(opencc, "OpenCC", fake_opencc)
    jyutping._traditional_converter.cache_clear()

    for _ in range(3):
        assert jyutping.simplified_to_traditional("点解") == "点解"
    assert calls == ["s2t"]

    jyutping._traditional_converter.cache_clear()


def test_a_missing_converter_keeps_the_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """A converter that cannot load leaves the text alone."""
    jyutping._traditional_converter.cache_clear()
    monkeypatch.setattr(jyutping, "_traditional_converter", lambda: None)

    assert jyutping.simplified_to_traditional("点解") == "点解"
