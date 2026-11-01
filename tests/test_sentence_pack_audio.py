"""Tests for the fetch, validate, and probe path of the sentence pack audio stage."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anki_tools.sentence_pack import audio
from tests.deck_fixture import MP3_STUB

VOICE = "xx-XX-TestNeural"
GTTS_VOICE = "xx-tld"
RATE = audio.RATE_NORMAL
ROWS = [
    {
        "target": target,
        "sentence": sentence,
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "audio_missing": "true",
    }
    for target, sentence in (("alpha", "alpha beta."), ("gamma", "gamma delta."))
]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_profile(tmp_path: Path, native: bool = False) -> Any:
    return SimpleNamespace(
        name="testlang",
        voice=VOICE,
        gtts_voice=GTTS_VOICE,
        tts_lang="xx",
        native_audio=native,
        audio_dir=tmp_path / "audio",
    )


def write_synthesizer(calls: list[str], data: bytes = MP3_STUB) -> Any:
    def synthesize(text: str, dest: Path, **_: object) -> None:
        calls.append(text)
        dest.write_bytes(data)

    return synthesize


def failing_synthesizer(text: str, dest: Path, **_: object) -> None:
    raise RuntimeError("engine down")


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(audio.time, "sleep", slept.append)
    return slept


@pytest.mark.parametrize(
    ("data", "seconds", "valid"),
    [
        pytest.param(MP3_STUB, 1.2, True, id="good file"),
        pytest.param(MP3_STUB, 30.0, False, id="too long"),
        pytest.param(MP3_STUB, 0.1, False, id="too short"),
        pytest.param(b"not an mp3", 1.2, False, id="no mp3 header"),
    ],
)
def test_validate_audio_checks_the_header_and_the_duration(
    tmp_path: Path, data: bytes, seconds: float, valid: bool
) -> None:
    path = tmp_path / "one.mp3"
    path.write_bytes(data)

    ok, duration = audio.validate_audio(path, probe=lambda target: seconds)

    assert ok is valid
    assert duration == (pytest.approx(seconds) if valid else 0.0)


def test_probe_duration_reads_the_first_line_of_the_ffprobe_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audio.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout="0.5\n")
    )

    assert audio.probe_duration(Path("one.mp3")) == 0.5


def test_generate_audio_retries_a_failed_native_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    attempts: list[str] = []

    def flaky_urlopen(request: Any, timeout: float = 0) -> FakeResponse:
        attempts.append(request.full_url)
        if len(attempts) == 1:
            raise OSError("network down")
        return FakeResponse(MP3_STUB)

    monkeypatch.setattr(audio.urllib.request, "urlopen", flaky_urlopen)

    rows = [dict(ROWS[0], audio_id="42", audio_license="CC BY 2.0 FR")]
    result = audio.generate_audio(
        rows,
        make_profile(tmp_path, native=True),
        audio_dir=tmp_path / "audio",
        native_audio="auto",
        rate_limit=0.0,
        probe=lambda path: 1.5,
        synthesize=write_synthesizer([]),
    )

    assert result["native"] == 1
    assert result["failed"] == 0
    assert attempts == [audio.TATOEBA_AUDIO_URL.format(audio_id=42)] * 2
    assert sleeps == [audio.RETRY_DELAYS[0]]
    assert (tmp_path / "audio" / "ta_42.mp3").read_bytes() == MP3_STUB


@pytest.mark.parametrize(
    ("synthesize", "error"),
    [
        pytest.param(write_synthesizer([], b"not an mp3"), "validation failed", id="bad file"),
        pytest.param(failing_synthesizer, "RuntimeError: engine down", id="engine error"),
    ],
)
def test_generate_audio_marks_a_card_missing_after_the_retries(
    tmp_path: Path, sleeps: list[float], synthesize: Any, error: str
) -> None:
    csv_path = tmp_path / "sentences.csv"
    audio_dir = tmp_path / "audio"
    write_rows(csv_path, ROWS[:1])

    result = audio.generate_audio(
        audio.load_sentence_rows(csv_path),
        make_profile(tmp_path),
        audio_dir=audio_dir,
        rate_limit=0.0,
        csv_path=csv_path,
        synthesize=synthesize,
        probe=lambda path: 1.0,
    )

    assert result["failed"] == 2
    assert [failure["error"] for failure in result["failures"]] == [error, error]
    assert sleeps == list(audio.RETRY_DELAYS) * 2
    assert list(audio_dir.glob("*.mp3")) == []
    assert audio.load_sentence_rows(csv_path)[0]["audio_missing"] == "true"


def test_generate_audio_writes_files_index_and_csv(tmp_path: Path) -> None:
    calls: list[str] = []
    audio_dir = tmp_path / "audio"
    csv_path = tmp_path / "sentences.csv"
    write_rows(csv_path, ROWS)

    result = audio.generate_audio(
        audio.load_sentence_rows(csv_path),
        make_profile(tmp_path),
        audio_dir=audio_dir,
        native_audio="none",
        rate_limit=0.0,
        csv_path=csv_path,
        synthesize=write_synthesizer(calls),
        probe=lambda path: 2.5,
    )

    assert result["tts"] == 2
    assert result["words"] == 2
    assert result["failed"] == 0
    assert calls == ["alpha beta.", "gamma delta.", "alpha", "gamma"]
    sentence_files = [
        audio.sentence_audio_name(row["sentence"], GTTS_VOICE, RATE, "tts", 0)
        for row in ROWS
    ]
    word_files = [
        audio.word_audio_name(row["target"], GTTS_VOICE, RATE) for row in ROWS
    ]
    for name in sentence_files + word_files:
        assert (audio_dir / name).is_file()

    with (audio_dir / audio.AUDIO_INDEX_NAME).open(encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    assert [entry["filename"] for entry in index] == sentence_files + word_files
    assert index[0]["duration_s"] == "2.500"
    assert len(index[0]["sha1"]) == 40

    with csv_path.open(encoding="utf-8") as handle:
        rewritten = list(csv.DictReader(handle))
    assert [row["audio_missing"] for row in rewritten] == ["false", "false"]


def test_synthesize_tts_writes_engine_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved: list[tuple[str, str]] = []

    class FakeGtts:
        def __init__(self, text: str, lang: str) -> None:
            saved.append((text, lang))

        def save(self, dest: str) -> None:
            Path(dest).write_bytes(MP3_STUB)

    monkeypatch.setitem(sys.modules, "edge_tts", None)
    monkeypatch.setattr("gtts.gTTS", FakeGtts)
    dest = tmp_path / "one.mp3"

    audio.synthesize_tts(
        "alpha", dest, voice=VOICE, rate=RATE, profile=make_profile(tmp_path)
    )

    assert dest.read_bytes() == MP3_STUB
    assert saved == [("alpha", "xx")]
