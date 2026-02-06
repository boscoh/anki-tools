#!/usr/bin/env python3
"""Backwards compatibility - imports from anki_tools.cli."""

import cyclopts

from anki_tools.cli import process_app

app = process_app

if __name__ == "__main__":
    app()
