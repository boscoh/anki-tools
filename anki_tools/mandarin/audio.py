"""Mandarin caller of the shared audio cache (stage S4).

The shared module holds the TTS, native download, probe, and index logic. This
module keeps the Mandarin voice, the module-level patch points of the tests, and
the Mandarin audio directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from anki_tools.sentence_pack import audio as shared
from anki_tools.sentence_pack.languages.mandarin import AUDIO_DIR, MANDARIN

AUDIO_INDEX_NAME = shared.AUDIO_INDEX_NAME
SENTENCES_PATH = MANDARIN.sentence_csv
TATOEBA_AUDIO_URL = shared.TATOEBA_AUDIO_URL
DEFAULT_VOICE = MANDARIN.voice
GTTS_VOICE = MANDARIN.gtts_voice
GTTS_LANG = MANDARIN.tts_lang
RATE_NORMAL = shared.RATE_NORMAL
RATE_SLOW = shared.RATE_SLOW
SLOW_RATE = shared.RATE_SLOW
MIN_DURATION_S = shared.MIN_DURATION_S
MAX_DURATION_S = shared.MAX_DURATION_S
MAX_ATTEMPTS = shared.MAX_ATTEMPTS
RETRY_DELAYS = shared.RETRY_DELAYS
DOWNLOAD_TIMEOUT = shared.DOWNLOAD_TIMEOUT
PUBLIC_PACK = shared.PUBLIC_PACK
PRIVATE_PACK = shared.PRIVATE_PACK
NATIVE_LICENSE_BLOCK = shared.NATIVE_LICENSE_BLOCK
INDEX_FIELDS = shared.INDEX_FIELDS

audio_hash = shared.audio_hash
sentence_audio_name = shared.sentence_audio_name
word_audio_name = shared.word_audio_name
tts_engine = shared.tts_engine
audio_rate = shared.audio_rate
license_allows = shared.license_allows
has_mp3_header = shared.has_mp3_header
probe_duration = shared.probe_duration
download_native = shared.download_native
write_audio_index = shared.write_audio_index
write_sentence_rows = shared.write_sentence_rows
time = shared.time


def load_sentence_rows(path: Path = SENTENCES_PATH) -> list[dict[str, str]]:
    """Read the Mandarin sentence CSV.

    :param path: Sentence CSV file.
    :returns: One dict per card row.
    """
    return shared.load_sentence_rows(path)


def validate_audio(path: Path) -> tuple[bool, float]:
    """Validate one audio file with the patchable ffprobe reader.

    :param path: Media file.
    :returns: ``(True, duration)`` when the file is valid, else ``(False, 0.0)``.
    """
    return shared.validate_audio(path, probe_duration)


def effective_voice(voice: str) -> str:
    """Select the voice name that goes into the hash and the index.

    :param voice: Requested edge-tts voice.
    :returns: The requested voice, or ``gtts:zh-CN`` for the gTTS engine.
    """
    return shared.effective_voice(voice, MANDARIN, tts_engine())


def choose_source(
    row: dict[str, str], native_audio: str, pack_license: str
) -> str:
    """Choose the audio tier for one card row.

    :param row: One sentence CSV row.
    :param native_audio: ``none``, ``auto``, or ``only``.
    :param pack_license: ``public`` or ``private``.
    :returns: ``native`` or ``tts``.
    """
    return shared.choose_source(row, native_audio, pack_license, MANDARIN)


def synthesize_tts(
    text: str,
    dest: Path,
    voice: str,
    rate: str,
    slow: bool,
    profile: Any = None,
) -> None:
    """Write one Mandarin TTS file.

    :param text: Text that goes to the engine.
    :param dest: Destination MP3 file.
    :param voice: Requested edge-tts voice.
    :param rate: Engine rate string.
    :param slow: Slow flag for the gTTS engine.
    :param profile: Language profile from the shared caller, MANDARIN by default.
    """
    shared.synthesize_tts(text, dest, voice, rate, slow, profile or MANDARIN)


def generate_audio(
    rows: Sequence[dict[str, str]],
    audio_dir: Path = AUDIO_DIR,
    voice: str = DEFAULT_VOICE,
    native_audio: str = "none",
    slow_audio: bool = False,
    pack_license: str = PUBLIC_PACK,
    rate_limit: float = 1.0,
    refresh: bool = False,
    csv_path: Path | None = SENTENCES_PATH,
) -> dict[str, Any]:
    """Generate and validate the Mandarin sentence and word audio files.

    :param rows: Sentence CSV rows from :func:`load_sentence_rows`.
    :param audio_dir: Output directory for the MP3 files and the index.
    :param voice: Requested edge-tts voice.
    :param native_audio: ``none``, ``auto``, or ``only``.
    :param slow_audio: Generate the slow variant.
    :param pack_license: ``public`` or ``private``.
    :param rate_limit: Minimum seconds between two requests.
    :param refresh: Ignore the existing files and generate them again.
    :param csv_path: Sentence CSV to rewrite, or None to skip the rewrite.
    :returns: Counts, failures, total duration, and the written paths.
    """
    return shared.generate_audio(
        rows,
        MANDARIN,
        audio_dir=audio_dir,
        voice=voice,
        native_audio=native_audio,
        slow_audio=slow_audio,
        pack_license=pack_license,
        rate_limit=rate_limit,
        refresh=refresh,
        csv_path=csv_path,
        engine=tts_engine(),
        synthesize=synthesize_tts,
        probe=probe_duration,
    )


AudioSynthesizer = Callable[..., None]
