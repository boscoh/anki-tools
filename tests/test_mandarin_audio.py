"""Tests for the Mandarin sentence pack audio stage."""

import csv
from pathlib import Path

import pytest

from anki_tools import mandarin_audio as audio

VOICE = "zh-CN-XiaoxiaoNeural"
RATE = "-10%"
FAKE_MP3_HEAD = b"ID3\x03\x00\x00\x00" + b"\x00" * 64
FAKE_MP3_SYNC = b"\xff\xfb\x90\x00" + b"\x00" * 64

BASE_ROWS = [
    {
        "target": "水",
        "sentence": "我要喝水。",
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "audio_missing": "true",
    },
    {
        "target": "火",
        "sentence": "火很大。",
        "audio_id": "0",
        "audio_source": "tts",
        "audio_license": "",
        "audio_missing": "true",
    },
]


@pytest.fixture
def patched_audio(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def synthesize(text: str, dest: Path, **_: object) -> None:
        calls.append(text)
        dest.write_bytes(FAKE_MP3_HEAD)

    monkeypatch.setattr(audio, "tts_engine", lambda: "gtts")
    monkeypatch.setattr(audio, "synthesize_tts", synthesize)
    monkeypatch.setattr(audio, "probe_duration", lambda path: 2.5)
    monkeypatch.setattr(audio.time, "sleep", lambda seconds: None)
    return calls


def test_audio_hash_is_deterministic() -> None:
    first = audio.audio_hash("我要喝水。", VOICE, RATE)
    second = audio.audio_hash("我要喝水。", VOICE, RATE)

    assert first == second
    assert len(first) == 12
    assert first != audio.audio_hash("我要喝水。", VOICE, "-25%")


def test_sentence_audio_name_native() -> None:
    name = audio.sentence_audio_name("我要喝水。", VOICE, RATE, "native", 1276691)

    assert name == "ta_1276691.mp3"


def test_sentence_audio_name_tts() -> None:
    name = audio.sentence_audio_name("我要喝水。", VOICE, RATE, "tts", 0)

    expected = audio.audio_hash("我要喝水。", VOICE, RATE)
    assert name == f"tts_{expected}.mp3"
    assert name.isascii()


def test_word_audio_name() -> None:
    name = audio.word_audio_name("水", VOICE, RATE)

    assert name == f"word_{audio.audio_hash('水', VOICE, RATE)}.mp3"


def test_native_license_public_blocks_empty_and_nc() -> None:
    assert not audio.license_allows("", "public")
    assert not audio.license_allows("CC BY-NC 4.0", "public")
    assert audio.license_allows("CC BY 2.0 FR", "public")


def test_native_license_private_allows_all() -> None:
    assert audio.license_allows("", "private")
    assert audio.license_allows("CC BY-NC 4.0", "private")


def test_choose_source_public_prefers_tts_for_nc_row() -> None:
    row = {"audio_id": "26082", "audio_license": "CC BY-NC 4.0"}

    assert audio.choose_source(row, "auto", "public") == "tts"
    assert audio.choose_source(row, "auto", "private") == "native"
    assert audio.choose_source(row, "none", "private") == "tts"


def test_has_mp3_header_accepts_id3_and_frame_sync(
    tmp_path: Path,
) -> None:
    id3 = tmp_path / "id3.mp3"
    id3.write_bytes(FAKE_MP3_HEAD)
    sync = tmp_path / "sync.mp3"
    sync.write_bytes(FAKE_MP3_SYNC)
    plain = tmp_path / "plain.mp3"
    plain.write_bytes(b"\x00\x01\x02\x03")

    assert audio.has_mp3_header(id3)
    assert audio.has_mp3_header(sync)
    assert not audio.has_mp3_header(plain)


def test_validate_audio_accepts_good_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "good.mp3"
    path.write_bytes(FAKE_MP3_HEAD)
    monkeypatch.setattr(audio, "probe_duration", lambda target: 1.2)

    ok, duration = audio.validate_audio(path)

    assert ok
    assert duration == pytest.approx(1.2)


def test_validate_audio_rejects_long_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "long.mp3"
    path.write_bytes(FAKE_MP3_HEAD)
    monkeypatch.setattr(audio, "probe_duration", lambda target: 30.0)

    ok, duration = audio.validate_audio(path)

    assert not ok
    assert duration == 0.0


def test_generate_audio_writes_files_index_and_csv(
    tmp_path: Path, patched_audio: list[str]
) -> None:
    audio_dir = tmp_path / "audio"
    csv_path = tmp_path / "sentences.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(BASE_ROWS[0].keys()))
        writer.writeheader()
        writer.writerows(BASE_ROWS)

    result = audio.generate_audio(
        audio.load_sentence_rows(csv_path),
        audio_dir=audio_dir,
        voice=VOICE,
        native_audio="none",
        pack_license="public",
        rate_limit=0.0,
        csv_path=csv_path,
    )

    voice = audio.GTTS_VOICE
    sentence_files = [
        audio.sentence_audio_name(row["sentence"], voice, RATE, "tts", 0)
        for row in BASE_ROWS
    ]
    word_files = [audio.word_audio_name("水", voice, RATE)]

    assert result["sentences"] == 2
    assert result["tts"] == 2
    assert result["words"] == 2
    assert result["failed"] == 0
    assert result["voice"] == voice
    assert patched_audio == ["我要喝水。", "火很大。", "水", "火"]
    for name in sentence_files + word_files:
        assert (audio_dir / name).is_file()

    with (audio_dir / audio.AUDIO_INDEX_NAME).open(encoding="utf-8") as handle:
        index = list(csv.DictReader(handle))
    assert [entry["filename"] for entry in index] == sentence_files + [
        audio.word_audio_name(word, voice, RATE) for word in ("水", "火")
    ]
    assert index[0]["source"] == "tts"
    assert index[0]["audio_id"] == "0"
    assert index[0]["duration_s"] == "2.500"
    assert len(index[0]["sha1"]) == 40

    with csv_path.open(encoding="utf-8") as handle:
        rewritten = list(csv.DictReader(handle))
    assert [row["audio_missing"] for row in rewritten] == ["false", "false"]
    assert [row["audio_license"] for row in rewritten] == ["", ""]
    assert [row["audio_id"] for row in rewritten] == ["0", "0"]
    assert [row["target"] for row in rewritten] == ["水", "火"]


def test_generate_audio_skips_existing_files(
    tmp_path: Path, patched_audio: list[str]
) -> None:
    audio_dir = tmp_path / "audio"
    first = audio.generate_audio(
        BASE_ROWS,
        audio_dir=audio_dir,
        native_audio="none",
        rate_limit=0.0,
        csv_path=None,
    )
    assert first["skipped"] == 0
    second = audio.generate_audio(
        BASE_ROWS,
        audio_dir=audio_dir,
        native_audio="none",
        rate_limit=0.0,
        csv_path=None,
    )

    assert second["skipped"] == 4
    assert second["failed"] == 0


def test_generate_audio_marks_failure_and_removes_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audio_dir = tmp_path / "audio"

    def synthesize(text: str, dest: Path, **_: object) -> None:
        dest.write_bytes(FAKE_MP3_HEAD)

    monkeypatch.setattr(audio, "synthesize_tts", synthesize)
    monkeypatch.setattr(audio, "probe_duration", lambda path: 20.0)
    monkeypatch.setattr(audio.time, "sleep", lambda seconds: None)

    result = audio.generate_audio(
        BASE_ROWS[:1],
        audio_dir=audio_dir,
        native_audio="none",
        rate_limit=0.0,
        csv_path=None,
    )

    assert result["failed"] == 2
    assert not list(audio_dir.glob("*.mp3"))
    assert result["failures"][0]["error"] == "validation failed"
