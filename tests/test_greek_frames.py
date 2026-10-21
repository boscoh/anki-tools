"""Tests for the manual frame sentences of the Greek pack."""

from __future__ import annotations

from pathlib import Path

from anki_tools.languages.greek import profile as greek
from anki_tools.sentence_pack.pool import default_frames_path, read_frames

FRAMES_PATH = Path("vocab/greek_frames.csv")


def test_read_frames_reads_the_rows(tmp_path: Path) -> None:
    path = tmp_path / "greek_frames.csv"
    path.write_text(
        "target,sentence,english\nαυτί,Το αυτί μου πονάει.,My ear hurts.\n",
        encoding="utf-8",
    )

    rows = read_frames(path)

    assert rows == [
        {
            "target": "αυτί",
            "sentence": "Το αυτί μου πονάει.",
            "english": "My ear hurts.",
        }
    ]


def test_read_frames_skips_an_empty_sentence(tmp_path: Path) -> None:
    path = tmp_path / "greek_frames.csv"
    path.write_text(
        "target,sentence,english\nαυτί,,My ear hurts.\n",
        encoding="utf-8",
    )

    assert read_frames(path) == []


def test_an_absent_file_gives_no_frames(tmp_path: Path) -> None:
    assert read_frames(tmp_path / "absent.csv") == []


def test_default_frames_path_names_the_language() -> None:
    assert default_frames_path(greek.GREEK) == Path("vocab/greek_frames.csv")


def test_every_frame_hits_its_target() -> None:
    """A frame is useless when the card target does not match its sentence."""
    table = greek.load_word_table()
    forms = {greek.GREEK.form_text(form) for form in table}
    rows = read_frames(FRAMES_PATH)

    assert len(rows) == 47
    missing = [row["target"] for row in rows if row["target"] not in forms]
    assert missing == []
    for row in rows:
        assert row["target"] in greek.hits(row["sentence"], forms), row["sentence"]


def test_one_token_matches_a_form_once() -> None:
    """The surface key and the lemma key of one token name the same form."""
    assert greek.hits("Βρήκα ένα πούπουλο στο πάτωμα.", {"πούπουλο"}) == {
        "πούπουλο": 1
    }
