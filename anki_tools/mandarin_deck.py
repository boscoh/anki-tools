"""Build the Mandarin Swadesh sentence pack.

Stage S5 reads the reviewed word table, ``vocab/mandarin_sentences.csv``, and
the audio index in ``audio/mandarin_sentences``. It writes one APKG file, a
manifest, and ``ATTRIBUTION.md``.

The audio index is the only source of file names. The module never recomputes
a hash. A card enters the pack only when its sentence file and its word file
exist in the index. Cards are sorted by Swadesh index, then primary form, then
Tatoeba id, so Anki shows new cards in word order.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from anki_tools.mandarin_audio import (
    AUDIO_DIR,
    AUDIO_INDEX_NAME,
    NATIVE_LICENSE_BLOCK,
    PRIVATE_PACK,
    PUBLIC_PACK,
)
from anki_tools.mandarin_swadesh import WordForm, load_word_table
from anki_tools.package import AnkiPackage

SENTENCES_PATH = Path("vocab/mandarin_sentences.csv")
WORD_CSV_PATH = Path("vocab/english-mandarin.csv")
OUTPUT_PATH = Path("apkg/swadesh_mandarin_sentences.apkg")
MANIFEST_PATH = Path("apkg/swadesh_mandarin_sentences.manifest.json")
ATTRIBUTION_PATH = Path("ATTRIBUTION.md")
CSS_PATH = Path("templates/card.css")
CHECKSUMS_PATH = Path("data/tatoeba/checksums.json")
AUDIO_TSV_PATH = Path("data/tatoeba/cmn_sentences_with_audio.tsv")

MODEL_NAME = "Mandarin Swadesh Sentence"
DECK_NAME = "Mandarin Swadesh Sentences"
FIELDS = [
    "Audio",
    "Sentence",
    "Pinyin",
    "English",
    "Target",
    "Gloss",
    "WordAudio",
    "Source",
]
SORT_FIELD_INDEX = 4

DECK_DESCRIPTION = "Word data CC BY-SA 4.0. Sentence text CC BY 2.0 FR. Audio: gTTS."

SWADESH_URL = "https://en.wiktionary.org/wiki/Module:Swadesh/data/cmn"
TATOEBA_URL = "https://tatoeba.org"
TATOEBA_LICENSE = "CC BY 2.0 FR"
SWADESH_LICENSE = "CC BY-SA 4.0"
CORE_CONCEPTS = 207

SENTENCE_PREFIXES = ("tts_", "ta_")
WORD_PREFIX = "word_"
LICENSE_KEYS = ("empty", NATIVE_LICENSE_BLOCK)
DEFAULT_AUDIO_LICENSE_SOURCE = {"empty": 5742, NATIVE_LICENSE_BLOCK: 84}

FRONT_TEMPLATE = (
    '<div class="front-audio">{{Audio}}</div>\n'
    '<div class="hint">Tap to show the sentence</div>'
)

BACK_TEMPLATE = (
    "{{Audio}}\n"
    "<hr id=answer>\n"
    '<div class="sentence">{{Sentence}}</div>\n'
    '<div class="pinyin">{{Pinyin}}</div>\n'
    '<div class="english">{{English}}</div>\n'
    '<div class="target">{{Target}} {{Gloss}} {{WordAudio}}</div>\n'
    '<div class="source">{{Source}}</div>'
)

EXTRA_CSS = """
.front-audio {
    text-align: center;
}
.sentence {
    font-size: 2.5rem;
    text-align: center;
    margin: 0.5rem 0;
}
.pinyin {
    font-size: 1.2rem;
    text-align: center;
    color: #444;
    margin-bottom: 0.5rem;
}
.english {
    font-size: 1.1rem;
    text-align: center;
    color: #333;
    margin-bottom: 0.75rem;
}
.target {
    font-size: 1.6rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.source {
    font-size: 0.8rem;
    text-align: center;
    color: gray;
}
.hint {
    font-size: 0.9rem;
    text-align: center;
    color: gray;
}
"""


@dataclass
class AudioIndex:
    """Audio file names read from the cache index.

    :param sentences: Mapping from sentence text to its MP3 file name.
    :param words: Mapping from target text to its MP3 file name.
    :param voice: Most common voice value in the index.
    """

    sentences: dict[str, str]
    words: dict[str, str]
    voice: str


def slug(text: str) -> str:
    """Turn a gloss into an Anki tag component.

    :param text: Text to convert, for example ``water``.
    :returns: Lower-case text with each non-alphanumeric run as ``_``.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def load_audio_index(index_path: Path) -> AudioIndex:
    """Read the audio index into sentence and word maps.

    :param index_path: The ``index.csv`` file of the audio cache.
    :raises FileNotFoundError: The index file does not exist.
    :returns: The two name maps and the most common voice.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"audio index not found: {index_path}")
    sentences: dict[str, str] = {}
    words: dict[str, str] = {}
    voices: Counter[str] = Counter()
    with index_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = (row.get("filename") or "").strip()
            text = (row.get("text") or "").strip()
            if not filename or not text:
                continue
            if filename.startswith(WORD_PREFIX):
                words[text] = filename
            elif filename.startswith(SENTENCE_PREFIXES):
                sentences[text] = filename
            else:
                continue
            voice = (row.get("voice") or "").strip()
            if voice:
                voices[voice] += 1
    voice = voices.most_common(1)[0][0] if voices else ""
    return AudioIndex(sentences=sentences, words=words, voice=voice)


def build_card(
    row: dict[str, str],
    word: WordForm,
    sentence_audio: str,
    word_audio: str,
) -> dict[str, Any]:
    """Build the note fields and tags for one card.

    :param row: One row of the sentence CSV.
    :param word: Reviewed word form of the card target.
    :param sentence_audio: File name of the sentence MP3.
    :param word_audio: File name of the word MP3.
    :returns: Mapping from field name to value, plus a ``tags`` key.
    """
    target = row["target"]
    sentence = row["sentence"].replace(target, f"<b>{target}</b>", 1)
    tags = [
        f"swadesh::{slug(word.english)}",
        f"audio::{row.get('audio_source', '')}",
        f"tier::{row.get('tier', '')}",
    ]
    return {
        "Audio": f"[sound:{sentence_audio}]",
        "Sentence": sentence,
        "Pinyin": row["pinyin"],
        "English": row["english"],
        "Target": target,
        "Gloss": f"{word.english} · {word.pinyin}",
        "WordAudio": f"[sound:{word_audio}]",
        "Source": f"Tatoeba #{row['tatoeba_id']} · {row['tatoeba_author']}",
        "tags": tags,
    }


def card_sort_key(word: WordForm, row: dict[str, str]) -> tuple[int, int, int]:
    """Build the note order key of one card.

    :param word: Reviewed word form of the card target.
    :param row: One row of the sentence CSV.
    :returns: Swadesh index, form variant rank, then Tatoeba id.
    """
    variant_rank = 0 if word.variant == "primary" else 1
    return (word.swadesh_index, variant_rank, int(row["tatoeba_id"]))


def media_paths(filenames: Iterable[str], audio_dir: Path = AUDIO_DIR) -> list[str]:
    """Build the absolute media paths of the file names.

    :param filenames: MP3 file names from the audio index.
    :param audio_dir: Directory of the MP3 files.
    :returns: Sorted absolute paths with no duplicate.
    """
    return [str((audio_dir / name).resolve()) for name in sorted(set(filenames))]


def load_css(css_path: Path = CSS_PATH) -> str:
    """Build the card CSS from the shared sheet and the pack classes.

    :param css_path: Base CSS file. A missing file gives only the pack classes.
    :returns: Base CSS followed by the pack classes.
    """
    base = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    return f"{base.rstrip()}\n{EXTRA_CSS.lstrip()}"


def audio_license_source(tsv_path: Path = AUDIO_TSV_PATH) -> dict[str, int]:
    """Count the license values of the Tatoeba Mandarin audio export.

    :param tsv_path: ``cmn_sentences_with_audio.tsv`` file.
    :returns: Empty-license and CC BY-NC 4.0 counts of the whole export.
    """
    if not tsv_path.exists():
        return dict(DEFAULT_AUDIO_LICENSE_SOURCE)
    counts = {"empty": 0, NATIVE_LICENSE_BLOCK: 0}
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            license_text = row[3].strip() if len(row) > 3 else ""
            if not license_text:
                counts["empty"] += 1
            elif license_text == NATIVE_LICENSE_BLOCK:
                counts[NATIVE_LICENSE_BLOCK] += 1
    return counts


def skipped_license_counts(
    rows: Iterable[dict[str, str]], pack_license: str = PUBLIC_PACK
) -> dict[str, int]:
    """Count the native recordings that this build leaves out.

    :param rows: Sentence CSV rows.
    :param pack_license: ``public`` or ``private``.
    :returns: Empty-license and CC BY-NC 4.0 counts of the skipped files.
    """
    counts = {key: 0 for key in LICENSE_KEYS}
    for row in rows:
        if row.get("audio_source") == "native":
            continue
        audio_id = str(row.get("audio_id", "")).strip()
        if not audio_id or audio_id == "0":
            continue
        license_text = row.get("audio_license", "").strip()
        if not license_text:
            counts["empty"] += 1
        elif license_text == NATIVE_LICENSE_BLOCK and pack_license != PRIVATE_PACK:
            counts[NATIVE_LICENSE_BLOCK] += 1
    return counts


def audio_author_map(tsv_path: Path = AUDIO_TSV_PATH) -> dict[int, str]:
    """Read the author of each Tatoeba audio id.

    :param tsv_path: ``cmn_sentences_with_audio.tsv`` file.
    :returns: Mapping from audio id to author. An empty mapping without a file.
    """
    if not tsv_path.exists():
        return {}
    authors: dict[int, str] = {}
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            try:
                authors[int(row[1])] = row[2].strip()
            except ValueError:
                continue
    return authors


def tts_engine_name(voice: str) -> str:
    """Name the engine that produced the voice.

    :param voice: Voice value from the audio index.
    :returns: ``gTTS`` for a gTTS voice, else ``edge-tts``.
    """
    return "gTTS" if voice.startswith("gtts") else "edge-tts"


def _source_dump_date(checksums_path: Path) -> str:
    if not checksums_path.exists():
        return ""
    try:
        data = json.loads(checksums_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    stamps = [
        str(info.get("downloaded_at", ""))[:10]
        for info in data.values()
        if isinstance(info, dict) and info.get("downloaded_at")
    ]
    return max(stamps) if stamps else ""


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _native_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("audio_source") == "native"]


def build_attribution(
    rows: list[dict[str, str]],
    voice: str,
    audio_authors: dict[int, str] | None = None,
) -> str:
    """Build the ``ATTRIBUTION.md`` text.

    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :param audio_authors: Mapping from audio id to author.
    :returns: Markdown text of the attribution file.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        author = row.get("tatoeba_author", "").strip() or "unknown"
        grouped[author].append(int(row["tatoeba_id"]))

    lines = [
        "# Attribution",
        "",
        "Licenses: word data CC BY-SA 4.0, sentence text CC BY 2.0 FR, and",
        "audio per recording. This pack uses synthesized audio only.",
        "",
        f"Sentences come from Tatoeba (<{TATOEBA_URL}>) and use the "
        f"{TATOEBA_LICENSE} license.",
        "",
        "## Word data",
        "",
        f"The Swadesh Mandarin list comes from <{SWADESH_URL}>.",
        f"It uses the {SWADESH_LICENSE} license.",
        "The reviewed forms are in `vocab/english-mandarin.csv`.",
        "",
        "## Sentence text",
        "",
        f"The Tatoeba Mandarin sentences below use the {TATOEBA_LICENSE}",
        "license. Each number is a Tatoeba sentence id.",
    ]
    for author in sorted(grouped):
        ids = ", ".join(str(value) for value in sorted(grouped[author]))
        lines.extend(["", f"### {author}", "", ids])

    lines.extend(
        [
            "",
            "## Audio",
            "",
            f"Engine: {tts_engine_name(voice)}. Voice: {voice}.",
            "Each file is synthesized, so the engine and the voice give",
            "the license.",
        ]
    )

    native = _native_rows(rows)
    if native:
        authors = audio_authors or {}
        lines.extend(
            [
                "",
                "## Native recordings",
                "",
                "Tatoeba recordings keep their own license.",
                "",
                "| Audio id | Author | License |",
                "| -------- | ------ | ------- |",
            ]
        )
        for row in sorted(native, key=lambda item: int(item["audio_id"])):
            audio_id = int(row["audio_id"])
            author = authors.get(audio_id, "") or row.get("tatoeba_author", "")
            license_text = row.get("audio_license", "") or "per recording"
            lines.append(f"| {audio_id} | {author} | {license_text} |")

    return "\n".join(lines) + "\n"


def build_pack(
    csv_path: Path = SENTENCES_PATH,
    word_csv: Path = WORD_CSV_PATH,
    audio_dir: Path = AUDIO_DIR,
    output: Path = OUTPUT_PATH,
    manifest_path: Path = MANIFEST_PATH,
    attribution_path: Path = ATTRIBUTION_PATH,
    pack_license: str = PUBLIC_PACK,
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
    table = load_word_table(csv_path=word_csv)
    by_target = {form.mandarin: form for form in table}
    rows = _read_rows(csv_path)
    index = load_audio_index(Path(audio_dir) / AUDIO_INDEX_NAME)

    used: list[tuple[WordForm, dict[str, str], dict[str, Any]]] = []
    skipped = 0
    for row in rows:
        word = by_target.get(row["target"])
        if word is None:
            skipped += 1
            continue
        if row.get("audio_missing", "").strip().lower() == "true":
            skipped += 1
            continue
        sentence_audio = index.sentences.get(row["sentence"])
        word_audio = index.words.get(row["target"])
        if not sentence_audio or not word_audio:
            skipped += 1
            continue
        card = build_card(row, word, sentence_audio, word_audio)
        used.append((word, row, card))
    used.sort(key=lambda item: card_sort_key(item[0], item[1]))

    filenames = [
        name
        for _, _, card in used
        for name in (
            card["Audio"].removeprefix("[sound:").removesuffix("]"),
            card["WordAudio"].removeprefix("[sound:").removesuffix("]"),
        )
    ]
    media = media_paths(filenames, Path(audio_dir))
    for path in media:
        if not Path(path).exists():
            raise FileNotFoundError(f"media file not found: {path}")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    AnkiPackage.create(
        output,
        DECK_NAME,
        fields=FIELDS,
        cards=[card for _, _, card in used],
        media_files=media,
        model_name=MODEL_NAME,
        question_format=FRONT_TEMPLATE,
        answer_format=BACK_TEMPLATE,
        css=load_css(),
        sort_field_index=SORT_FIELD_INDEX,
        deck_description=DECK_DESCRIPTION,
    )

    packed_rows = [row for _, row, _ in used]
    targets = {row["target"] for row in packed_rows}
    concepts = {
        by_target[target].swadesh_index for target in targets if target in by_target
    }
    manifest = {
        "build_date": date.today().isoformat(),
        "source_dump_date": _source_dump_date(CHECKSUMS_PATH),
        "counts": {
            "words": len(table),
            "cards": len(rows),
            "scaffold": sum(1 for row in rows if row.get("tier") == "scaffold"),
            "native_audio": sum(
                1 for row in rows if row.get("audio_source") == "native"
            ),
            "tts_audio": sum(1 for row in rows if row.get("audio_source") == "tts"),
        },
        "coverage": {
            "core_with_card": round(len(concepts) / CORE_CONCEPTS, 4),
        },
        "voice": index.voice,
        "pack_license": pack_license,
        "audio_license_skipped": skipped_license_counts(rows, pack_license),
        "audio_license_source": audio_license_source(),
        "files": {
            "source_csv": _sha256(Path(word_csv)),
            "sentences_csv": _sha256(Path(csv_path)),
        },
        "licenses": [
            {"data": "Swadesh word data", "license": SWADESH_LICENSE},
            {"data": "Tatoeba sentences", "license": TATOEBA_LICENSE},
            {"data": "Tatoeba audio", "license": "per recording"},
        ],
        "excluded_words": [
            form.mandarin for form in table if form.mandarin not in targets
        ],
    }

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    attribution_path = Path(attribution_path)
    attribution_path.write_text(
        build_attribution(packed_rows, index.voice, audio_author_map()),
        encoding="utf-8",
    )

    return {
        "output": str(output),
        "manifest_path": str(manifest_path),
        "attribution_path": str(attribution_path),
        "media_count": len(media),
        "skipped": skipped,
        "manifest": manifest,
    }
