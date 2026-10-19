"""Audio cache of a Swadesh sentence pack (stage S4).

The module writes one sentence file per card and one word file per unique target
into the audio directory, validates every file with ``ffprobe``, and writes
``index.csv``. A native Tatoeba recording is used only when the language has a
native tier, the row has an audio id, and the recording license fits the pack
license. Every other card uses TTS.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import importlib
import importlib.util
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Sequence

from anki_tools.sentence_pack.profile import LanguageProfile

AUDIO_INDEX_NAME = "index.csv"
TATOEBA_AUDIO_URL = "https://tatoeba.org/audio/download/{audio_id}"

RATE_NORMAL = "-10%"
RATE_SLOW = "-25%"
MIN_DURATION_S = 0.4
MAX_DURATION_S = 15.0
MAX_ATTEMPTS = 3
RETRY_DELAYS = (2.0, 4.0)
DOWNLOAD_TIMEOUT = 120
PUBLIC_PACK = "public"
PRIVATE_PACK = "private"
NATIVE_LICENSE_BLOCK = "CC BY-NC 4.0"

INDEX_FIELDS = [
    "filename",
    "text",
    "voice",
    "source",
    "audio_id",
    "license",
    "bytes",
    "sha1",
    "duration_s",
]


def audio_hash(text: str, voice: str, rate: str) -> str:
    """Build the cache hash for one TTS text.

    :param text: Text that goes to the engine.
    :param voice: Engine voice name.
    :param rate: Engine rate string.
    :returns: The first 12 hex characters of the sha1 digest.
    """
    digest = hashlib.sha1(f"{text}|{voice}|{rate}".encode("utf-8"))
    return digest.hexdigest()[:12]


def sentence_audio_name(
    sentence: str,
    voice: str,
    rate: str,
    source: str,
    audio_id: int | str,
) -> str:
    """Build the file name of a sentence file.

    :param sentence: Sentence text.
    :param voice: Engine voice name.
    :param rate: Engine rate string.
    :param source: ``native`` or ``tts``.
    :param audio_id: Tatoeba audio id for a native file.
    :returns: An ASCII file name.
    """
    if source == "native":
        return f"ta_{int(audio_id)}.mp3"
    return f"tts_{audio_hash(sentence, voice, rate)}.mp3"


def word_audio_name(word: str, voice: str, rate: str) -> str:
    """Build the file name of a word file.

    :param word: Target word text.
    :param voice: Engine voice name.
    :param rate: Engine rate string.
    :returns: An ASCII file name.
    """
    return f"word_{audio_hash(word, voice, rate)}.mp3"


def tts_engine() -> str:
    """Select the TTS engine.

    :returns: ``edge`` when ``edge_tts`` imports, else ``gtts``.
    """
    if importlib.util.find_spec("edge_tts") is None:
        return "gtts"
    return "edge"


def effective_voice(voice: str, profile: LanguageProfile, engine: str) -> str:
    """Select the voice name that goes into the hash and the index.

    :param voice: Requested edge-tts voice.
    :param profile: Language profile.
    :param engine: TTS engine name, ``gtts`` or ``edge``.
    :returns: The requested voice, or the gTTS voice for the gTTS engine.
    """
    return voice if engine == "edge" else profile.gtts_voice


def audio_rate(slow_audio: bool) -> str:
    """Select the rate string.

    :param slow_audio: Use the slow variant.
    :returns: ``-25%`` for a slow file, else ``-10%``.
    """
    return RATE_SLOW if slow_audio else RATE_NORMAL


def license_allows(audio_license: str, pack_license: str) -> bool:
    """Check whether a recording license fits the pack license.

    :param audio_license: Recording license text from the CSV or empty.
    :param pack_license: ``public`` or ``private``.
    :returns: True when the recording may enter the pack.
    """
    if pack_license == PRIVATE_PACK:
        return True
    text = audio_license.strip()
    return bool(text) and text != NATIVE_LICENSE_BLOCK


def choose_source(
    row: dict[str, str],
    native_audio: str,
    pack_license: str,
    profile: LanguageProfile | None = None,
) -> str:
    """Choose the audio tier for one card row.

    :param row: One sentence CSV row.
    :param native_audio: ``none``, ``auto``, or ``only``.
    :param pack_license: ``public`` or ``private``.
    :param profile: Language profile. A profile without a native tier forces TTS.
    :raises ValueError: The language has no native tier but ``auto`` or ``only``
        is set.
    :returns: ``native`` or ``tts``.
    """
    if profile is not None and not profile.native_audio:
        if native_audio in ("auto", "only"):
            raise ValueError(
                f"{profile.name}: no native Tatoeba audio exists; "
                "use --native-audio none"
            )
        return "tts"
    if native_audio not in ("auto", "only"):
        return "tts"
    audio_id = str(row.get("audio_id", "")).strip()
    if not audio_id or audio_id == "0":
        return "tts"
    if not license_allows(row.get("audio_license", ""), pack_license):
        return "tts"
    return "native"


def has_mp3_header(path: Path) -> bool:
    """Check the first bytes for an ID3 tag or an MPEG frame sync.

    :param path: File to read.
    :returns: True when the head is an MP3 header.
    """
    try:
        with Path(path).open("rb") as handle:
            head = handle.read(3)
    except OSError:
        return False
    if len(head) < 2:
        return False
    if head[:3] == b"ID3":
        return True
    return head[0] == 0xFF and (head[1] & 0xE0) == 0xE0


def probe_duration(path: Path) -> float:
    """Read the file duration with ffprobe.

    :param path: Media file.
    :returns: Duration in seconds.
    :raises RuntimeError: ffprobe fails or prints no number.
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"ffprobe failed: {path}") from error
    text = result.stdout.strip().splitlines()
    if not text:
        raise RuntimeError(f"ffprobe printed no duration: {path}")
    try:
        return float(text[0])
    except ValueError as error:
        raise RuntimeError(f"ffprobe duration is not a number: {path}") from error


def validate_audio(
    path: Path, probe: Callable[[Path], float] = probe_duration
) -> tuple[bool, float]:
    """Validate one audio file.

    :param path: Media file.
    :param probe: Function that reads the duration of a file.
    :returns: ``(True, duration)`` when the file is valid, else ``(False, 0.0)``.
    """
    if not Path(path).is_file():
        return False, 0.0
    if not has_mp3_header(path):
        return False, 0.0
    try:
        duration = probe(path)
    except RuntimeError:
        return False, 0.0
    if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        return False, 0.0
    return True, duration


def synthesize_tts(
    text: str,
    dest: Path,
    voice: str,
    rate: str,
    slow: bool,
    profile: LanguageProfile | None = None,
) -> None:
    """Write one TTS file.

    :param text: Text that goes to the engine.
    :param dest: Destination MP3 file.
    :param voice: Requested edge-tts voice.
    :param rate: Engine rate string.
    :param slow: Slow flag for the gTTS engine.
    :param profile: Language profile, which supplies the gTTS language.
    """
    if tts_engine() == "edge":
        import asyncio

        edge_tts = importlib.import_module("edge_tts")

        async def run() -> None:
            communicate = edge_tts.Communicate(text, voice, rate=rate)
            await communicate.save(str(dest))

        asyncio.run(run())
        return

    from gtts import gTTS

    lang = profile.tts_lang if profile is not None else ""
    gTTS(text=text, lang=lang, slow=slow).save(str(dest))


def download_native(audio_id: int | str, dest: Path) -> None:
    """Download one native Tatoeba recording.

    :param audio_id: Tatoeba audio id.
    :param dest: Destination MP3 file.
    """
    url = TATOEBA_AUDIO_URL.format(audio_id=int(audio_id))
    request = urllib.request.Request(url, headers={"User-Agent": "anki-tools"})
    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
        data = response.read()
    if not data:
        raise RuntimeError(f"empty native download: {url}")
    Path(dest).write_bytes(data)


def _throttle(state: dict[str, float], rate_limit: float) -> None:
    if rate_limit > 0:
        wait = state.get("last_request", 0.0) + rate_limit - time.monotonic()
        if wait > 0:
            time.sleep(wait)
    state["last_request"] = time.monotonic()


def _produce_file(
    path: Path,
    produce: Callable[[Path], None],
    *,
    probe: Callable[[Path], float],
    rate_limit: float,
    refresh: bool,
    state: dict[str, float],
) -> tuple[bool, float, str | None, bool]:
    """Create and validate one file, with the retry loop.

    :param path: Destination file.
    :param produce: Callable that writes the file.
    :param probe: Function that reads the duration of a file.
    :param rate_limit: Minimum seconds between two requests.
    :param refresh: Ignore an existing file.
    :param state: Mutable throttle state.
    :returns: ``(ok, duration, error, generated)``.
    """
    if path.exists() and not refresh:
        ok, duration = validate_audio(path, probe)
        if ok:
            return True, duration, None, False
        path.unlink(missing_ok=True)

    last_error = ""
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(RETRY_DELAYS[attempt - 1])
        _throttle(state, rate_limit)
        try:
            produce(path)
        except Exception as error:  # noqa: BLE001 - engine and network errors
            last_error = f"{type(error).__name__}: {error}"
            path.unlink(missing_ok=True)
            continue
        ok, duration = validate_audio(path, probe)
        if ok:
            return True, duration, None, True
        last_error = "validation failed"
        path.unlink(missing_ok=True)
    return False, 0.0, last_error or "unknown error", False


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _index_entry(
    path: Path,
    text: str,
    voice: str,
    source: str,
    audio_id: int,
    license_text: str,
    duration: float,
) -> dict[str, Any]:
    return {
        "filename": Path(path).name,
        "text": text,
        "voice": voice,
        "source": source,
        "audio_id": audio_id,
        "license": license_text,
        "bytes": Path(path).stat().st_size,
        "sha1": _sha1_file(path),
        "duration_s": f"{duration:.3f}",
    }


def write_audio_index(entries: Sequence[dict[str, Any]], path: Path) -> Path:
    """Write the audio index CSV.

    :param entries: Index rows from :func:`generate_audio`.
    :param path: Output file.
    :returns: The written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry)
    return target


def load_sentence_rows(path: Path) -> list[dict[str, str]]:
    """Read the sentence CSV.

    :param path: Sentence CSV file.
    :returns: One dict per card row.
    """
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_sentence_rows(rows: Sequence[dict[str, str]], path: Path) -> Path:
    """Rewrite the sentence CSV with the audio columns updated.

    The column order and every other value stay unchanged.

    :param rows: Rows from :func:`generate_audio`.
    :param path: Sentence CSV file.
    :returns: The written path.
    """
    path = Path(path)
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_audio(
    rows: Sequence[dict[str, str]],
    profile: LanguageProfile,
    *,
    audio_dir: Path | None = None,
    voice: str | None = None,
    native_audio: str = "none",
    slow_audio: bool = False,
    pack_license: str = PUBLIC_PACK,
    rate_limit: float = 1.0,
    refresh: bool = False,
    csv_path: Path | None = None,
    engine: str = "gtts",
    synthesize: Callable[..., None] = synthesize_tts,
    probe: Callable[[Path], float] = probe_duration,
) -> dict[str, Any]:
    """Generate and validate the sentence and word audio files.

    :param rows: Sentence CSV rows.
    :param profile: Language profile.
    :param audio_dir: Output directory for the MP3 files and the index.
    :param voice: Requested edge-tts voice.
    :param native_audio: ``none``, ``auto``, or ``only``.
    :param slow_audio: Generate the slow variant.
    :param pack_license: ``public`` or ``private``.
    :param rate_limit: Minimum seconds between two requests.
    :param refresh: Ignore the existing files and generate them again.
    :param csv_path: Sentence CSV to rewrite, or None to skip the rewrite.
    :param engine: TTS engine name, ``gtts`` or ``edge``.
    :param synthesize: Callable that writes one TTS file.
    :param probe: Function that reads the duration of a file.
    :returns: Counts, failures, total duration, and the written paths.
    """
    audio_dir = Path(audio_dir) if audio_dir is not None else profile.audio_dir
    audio_dir.mkdir(parents=True, exist_ok=True)
    engine_voice = effective_voice(voice or profile.voice, profile, engine)
    rate = audio_rate(slow_audio)
    slow = slow_audio

    state: dict[str, float] = {}
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    updated: list[dict[str, str]] = []
    counts = {
        "native": 0,
        "tts": 0,
        "words": 0,
        "skipped": 0,
        "failed": 0,
        "duration_s": 0.0,
    }

    for row in rows:
        source = choose_source(row, native_audio, pack_license, profile)
        audio_id = int(row["audio_id"]) if source == "native" else 0
        license_text = (
            row.get("audio_license", "").strip() if source == "native" else ""
        )
        sentence = row["sentence"]
        name = sentence_audio_name(sentence, engine_voice, rate, source, audio_id)
        path = audio_dir / name
        if source == "native":
            produce = functools.partial(download_native, audio_id)
        else:
            produce = functools.partial(
                synthesize,
                sentence,
                voice=engine_voice,
                rate=rate,
                slow=slow,
                profile=profile,
            )
        ok, duration, error, generated = _produce_file(
            path,
            produce,
            probe=probe,
            rate_limit=rate_limit,
            refresh=refresh,
            state=state,
        )

        new_row = dict(row)
        new_row["audio_id"] = str(audio_id)
        new_row["audio_source"] = source
        new_row["audio_license"] = license_text
        new_row["audio_missing"] = "false" if ok else "true"
        updated.append(new_row)

        if not ok:
            counts["failed"] += 1
            failures.append({"filename": name, "text": sentence, "error": error or ""})
            continue
        counts[source] += 1
        counts["duration_s"] += duration
        counts["skipped"] += 0 if generated else 1
        entries.append(
            _index_entry(
                path, sentence, engine_voice, source, audio_id, license_text, duration
            )
        )

    seen_words: set[str] = set()
    for row in rows:
        word = row["target"]
        if word in seen_words:
            continue
        seen_words.add(word)
        name = word_audio_name(word, engine_voice, rate)
        path = audio_dir / name
        produce = functools.partial(
            synthesize,
            word,
            voice=engine_voice,
            rate=rate,
            slow=slow,
            profile=profile,
        )
        ok, duration, error, generated = _produce_file(
            path,
            produce,
            probe=probe,
            rate_limit=rate_limit,
            refresh=refresh,
            state=state,
        )
        if not ok:
            counts["failed"] += 1
            failures.append({"filename": name, "text": word, "error": error or ""})
            continue
        counts["words"] += 1
        counts["duration_s"] += duration
        counts["skipped"] += 0 if generated else 1
        entries.append(_index_entry(path, word, engine_voice, "tts", 0, "", duration))

    index_path = write_audio_index(entries, audio_dir / AUDIO_INDEX_NAME)
    written_csv: Path | None = None
    if csv_path is not None:
        written_csv = write_sentence_rows(updated, Path(csv_path))

    return {
        "sentences": len(rows),
        "native": counts["native"],
        "tts": counts["tts"],
        "words": counts["words"],
        "skipped": counts["skipped"],
        "failed": counts["failed"],
        "failures": failures,
        "duration_s": round(counts["duration_s"], 3),
        "engine": engine,
        "voice": engine_voice,
        "index_path": str(index_path),
        "csv_path": str(written_csv) if written_csv is not None else None,
    }
