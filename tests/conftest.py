"""
Shared pytest fixtures for all tests.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
from anki_package import AnkiPackage


@pytest.fixture
def apkg_path():
    """Find the .apkg file in tests/ directory or current directory."""
    # Look in tests/ directory first, then current directory
    for base in [Path('tests'), Path('.')]:
        apkg_files = list(base.glob('*.apkg'))
        if apkg_files:
            return str(apkg_files[0])
    pytest.skip("No .apkg file found in tests/ or current directory")


@pytest.fixture
def pkg(apkg_path):
    """Create an AnkiPackage instance."""
    with AnkiPackage(apkg_path) as p:
        yield p


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for audio extraction."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
