from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from anki_tools.languages.cantonese import jyutping
from tests.deck_fixture import build_contract_deck


@pytest.fixture
def fake_jyutping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the ToJyutping library with a tool that prefixes the text."""
    fake = SimpleNamespace(get_jyutping_text=lambda text: f"jyutping:{text}")
    monkeypatch.setattr(jyutping, "ToJyutping", fake)


@pytest.fixture(scope="module")
def deck_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the contract deck once for each test module. Copy it before a change."""
    return build_contract_deck(tmp_path_factory.mktemp("deck"))
