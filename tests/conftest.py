from __future__ import annotations

from pathlib import Path

import pytest

from tests.deck_fixture import build_contract_deck


@pytest.fixture(scope="module")
def deck_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the contract deck once for each test module. Copy it before a change."""
    return build_contract_deck(tmp_path_factory.mktemp("deck"))
