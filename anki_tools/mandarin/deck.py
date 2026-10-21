"""Mandarin caller of the shared APKG build (stage S5).

The shared module holds the note model, the templates, the media list, and the
manifest. This module keeps the Mandarin note names and the attribution text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from anki_tools.sentence_pack import pack as shared
from anki_tools.sentence_pack.languages.mandarin import (
    AUDIO_DIR,
    AUDIO_TSV_PATH,
    BACK_TEMPLATE,
    DECK_DESCRIPTION,
    DECK_NAME,
    EXTRA_CSS,
    FIELDS as _PROFILE_FIELDS,
    FRONT_TEMPLATE,
    MANDARIN,
    MODEL_NAME,
    SENTENCES_PATH,
    SORT_FIELD_INDEX,
    DEFAULT_CSV_PATH,
    build_attribution,
    load_word_table,
    write_attribution,
)
from anki_tools.sentence_pack.pack import (
    AudioIndex,
    audio_author_map as _audio_author_map,
    audio_license_source as _audio_license_source,
    load_audio_index,
    media_paths,
    skipped_license_counts,
    slug,
    tts_engine_name,
)

OUTPUT_PATH = MANDARIN.output_apkg
MANIFEST_PATH = MANDARIN.manifest_path
ATTRIBUTION_PATH = shared.ATTRIBUTION_PATH
CSS_PATH = shared.CSS_PATH
CHECKSUMS_PATH = shared.CHECKSUMS_PATH
WORD_PREFIX = shared.WORD_PREFIX
SENTENCE_PREFIXES = shared.SENTENCE_PREFIXES
LICENSE_KEYS = shared.LICENSE_KEYS

FIELDS = list(_PROFILE_FIELDS)
DEFAULT_AUDIO_LICENSE_SOURCE = {key: 0 for key in LICENSE_KEYS}


def audio_license_source(tsv_path: Path = AUDIO_TSV_PATH) -> dict[str, int]:
    """Count the license values of the Tatoeba Mandarin audio export.

    :param tsv_path: ``cmn_sentences_with_audio.tsv`` file.
    :returns: Empty-license and CC BY-NC 4.0 counts of the whole export.
    """
    return _audio_license_source(tsv_path)


def audio_author_map(tsv_path: Path = AUDIO_TSV_PATH) -> dict[int, str]:
    """Read the author of each Tatoeba audio id.

    :param tsv_path: ``cmn_sentences_with_audio.tsv`` file.
    :returns: Mapping from audio id to author.
    """
    return _audio_author_map(tsv_path)


def build_card(
    row: dict[str, str],
    word: Any,
    sentence_audio: str,
    word_audio: str,
) -> dict[str, Any]:
    """Build the note fields and tags for one Mandarin card.

    :param row: One row of the sentence CSV.
    :param word: Reviewed word form of the card target.
    :param sentence_audio: File name of the sentence MP3.
    :param word_audio: File name of the word MP3.
    :returns: Mapping from field name to value, plus a ``tags`` key.
    """
    return shared.build_card(row, word, sentence_audio, word_audio, MANDARIN)


def card_sort_key(word: Any, row: dict[str, str]) -> tuple:
    """Build the note order key of one card.

    :param word: Reviewed word form of the card target.
    :param row: One row of the sentence CSV.
    :returns: Swadesh index, form variant rank, then Tatoeba id.
    """
    return shared.card_sort_key(word, row, MANDARIN)


def load_css(css_path: Path = CSS_PATH) -> str:
    """Build the card CSS from the shared sheet and the Mandarin classes.

    :param css_path: Base CSS file.
    :returns: Base CSS followed by the pack classes.
    """
    return shared.load_css(css_path, EXTRA_CSS)


def build_pack(
    csv_path: Path = SENTENCES_PATH,
    word_csv: Path = DEFAULT_CSV_PATH,
    audio_dir: Path = AUDIO_DIR,
    output: Path = OUTPUT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    attribution_path: Path = ATTRIBUTION_PATH,
    pack_license: str = shared.PUBLIC_PACK,
) -> dict[str, Any]:
    """Build the APKG file, the manifest, and the attribution file.

    :param csv_path: Sentence CSV with one row per card.
    :param word_csv: Reviewed Swadesh word CSV.
    :param audio_dir: Directory with the MP3 files and ``index.csv``.
    :param output: Path of the APKG file to write.
    :param manifest_path: Path of the manifest JSON file to write.
    :param attribution_path: Path of the attribution Markdown file to write.
    :param pack_license: ``public`` or ``private``.
    :raises FileNotFoundError: The audio index or a media file is missing.
    :returns: The written paths, the media count, the skipped card count, and
        the manifest mapping.
    """
    result = shared.build_pack(
        MANDARIN,
        load_word_table,
        csv_path=csv_path,
        word_csv=word_csv,
        audio_dir=audio_dir,
        output=output,
        manifest_path=manifest_path,
        attribution_path=attribution_path,
        pack_license=pack_license,
    )
    result["attribution_path"] = str(attribution_path)
    return result


__all__ = [
    "AudioIndex",
    "BACK_TEMPLATE",
    "DECK_DESCRIPTION",
    "DECK_NAME",
    "FIELDS",
    "FRONT_TEMPLATE",
    "MODEL_NAME",
    "SORT_FIELD_INDEX",
    "build_attribution",
    "build_card",
    "build_pack",
    "card_sort_key",
    "audio_author_map",
    "audio_license_source",
    "load_audio_index",
    "media_paths",
    "skipped_license_counts",
    "slug",
    "tts_engine_name",
    "write_attribution",
]

MediaFiles = Iterable[str]
Composer = Callable[[], dict[str, Any]]
Cards = Sequence[dict[str, Any]]
