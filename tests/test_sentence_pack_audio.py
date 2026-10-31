"""Tests for the fetch, validate, and probe path of the sentence pack audio stage."""

import csv
import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from anki_tools.sentence_pack import audio

VOICE = "xx-XX-TestNeural"
GTTS_VOICE = "xx-tld"
RATE = audio.RATE_NORMAL
MP3_BYTES = b"ID3\x03\x00\x00\x00" + b"\x00" * 64
ROWS = [
    {
        "target": "alpha",
        "sentence": "alpha beta.",
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "audio_missing": "true",
    },
    {
        "target": "gamma",
        "sentence": "gamma delta.",
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "audio_missing": "true",
    },
]


def make_profile(tmp_path: Path, native: bool = False) -> Any:
    return SimpleNamespace(
        name="testlang",
        voice=VOICE,
        gtts_voice=GTTS_VOICE,
        tts_lang="xx",
        native_audio=native,
        audio_dir=tmp_path / "audio",
    )


def write_synthesizer(calls: list[str]) -> Any:
    def synthesize(text: str, dest: Path, **_: object) -> None:
        calls.append(text)
        dest.write_bytes(MP3_BYTES)

    return synthesize


class FakeResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_validate_audio_accepts_good_header(tmp_path: Path) -> None:
    path = tmp_path / "good.mp3"
    path.write_bytes(MP3_BYTES)

    ok, duration = audio.validate_audio(path, probe=lambda target: 1.2)

    assert ok
    assert duration == pytest.approx(1.2)


def test_validate_audio_rejects_bad_duration(tmp_path: Path) -> None:
    path = tmp_path / "long.mp3"
    path.write_bytes(MP3_BYTES)

    ok, duration = audio.validate_audio(path, probe=lambda target: 30.0)

    assert not ok
    assert duration == 0.0


def test_probe_duration_reads_a_real_file(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 4000)

    assert audio.probe_duration(path) == pytest.approx(0.5, abs=0.01)


def test_generate_audio_retries_a_failed_native_fetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts: list[str] = []
    sleeps: list[float] = []

    def flaky_urlopen(request: Any, timeout: float = 0) -> FakeResponse:
        attempts.append(request.full_url)
        if len(attempts) == 1:
            raise OSError("network down")
        return FakeResponse(MP3_BYTES)

    monkeypatch.setattr(audio.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(audio.time, "sleep", sleeps.append)

    rows = [dict(ROWS[0], audio_id="42", audio_license="CC BY 2.0 FR")]
    result = audio.generate_audio(
        rows,
        make_profile(tmp_path, native=True),
        audio_dir=tmp_path / "audio",
        native_audio="auto",
        pack_license="public",
        rate_limit=0.0,
        probe=lambda path: 1.5,
        synthesize=write_synthesizer([]),
    )

    assert result["native"] == 1
    assert result["failed"] == 0
    assert len(attempts) == 2
    assert attempts[0] == audio.TATOEBA_AUDIO_URL.format(audio_id=42)
    assert sleeps == [audio.RETRY_DELAYS[0]]
    assert (tmp_path / "audio" / "ta_42.mp3").read_bytes() == MP3_BYTES


def test_generate_audio_writes_files_index_and_csv(tmp_path: Path) -> None:
    calls: list[str] = []
    audio_dir = tmp_path / "audio"
    csv_path = tmp_path / "sentences.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)

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
    saved: list[tuple[str, str, bool]] = []

    class FakeGtts:
        def __init__(self, text: str, lang: str, slow: bool) -> None:
            saved.append((text, lang, slow))

        def save(self, dest: str) -> None:
            Path(dest).write_bytes(MP3_BYTES)

    monkeypatch.setattr(audio, "tts_engine", lambda: "gtts")
    monkeypatch.setattr("gtts.gTTS", FakeGtts)
    dest = tmp_path / "one.mp3"

    audio.synthesize_tts(
        "alpha", dest, voice=VOICE, rate=RATE, slow=True, profile=make_profile(tmp_path)
    )

    assert dest.read_bytes() == MP3_BYTES
    assert saved == [("alpha", "xx", True)]
