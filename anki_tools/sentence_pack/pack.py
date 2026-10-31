"""APKG build of a Swadesh sentence pack (stage S5).

The audio index is the only source of file names. The module never recomputes a
hash. A card enters the pack only when its sentence file and its word file exist
in the index. Notes are sorted by the language order key, so Anki shows new cards
in word order.
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
from typing import Any, Callable, Iterable, Sequence

from anki_tools.apkg.package import AnkiPackage
from anki_tools.sentence_pack.audio import (
    NATIVE_LICENSE_BLOCK,
    PRIVATE_PACK,
    PUBLIC_PACK,
)
from anki_tools.sentence_pack.profile import (
    LanguageProfile,
    WordRow,
    excluded_forms,
)
from anki_tools.apkg.templates import (
    SHARED_CSS_PATH,
    TemplateSpec,
    assemble_templates,
    load_optional_text,
)

CSS_PATH = SHARED_CSS_PATH
CHECKSUMS_PATH = Path("data/tatoeba/checksums.json")
ATTRIBUTION_PATH = Path("ATTRIBUTION.md")

SENTENCE_PREFIXES = ("tts_", "ta_")
WORD_PREFIX = "word_"
LICENSE_KEYS = ("empty", NATIVE_LICENSE_BLOCK)
DETAIL_SEPARATOR = " · "


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
    index_path = Path(index_path)
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
    word: WordRow,
    sentence_audio: str,
    word_audio: str,
    profile: LanguageProfile,
) -> dict[str, Any]:
    """Build the note fields and tags for one card.

    :param row: One row of the sentence CSV.
    :param word: Reviewed word form of the card target.
    :param sentence_audio: File name of the sentence MP3.
    :param word_audio: File name of the word MP3.
    :param profile: Language profile.
    :returns: Mapping from field name to value, plus a ``tags`` key.
    """
    target = row["target"]
    if profile.highlight is None:
        sentence = row["sentence"].replace(target, f"<b>{target}</b>", 1)
    else:
        sentence = profile.highlight(row["sentence"], target)
    tags = [
        f"swadesh::{slug(word.english)}",
        f"audio::{row.get('audio_source', '')}",
        f"tier::{row.get('tier', '')}",
    ]
    if profile.lang_tag:
        tags.append(f"lang::{profile.lang_tag}")
    values: dict[str, Any] = {
        "Audio": f"[sound:{sentence_audio}]",
        "Sentence": sentence,
        "English": row["english"],
        "Target": target,
        "Gloss": profile.gloss(word),
        "WordAudio": f"[sound:{word_audio}]",
        "Source": (
            f"Tatoeba #{row['tatoeba_id']}{DETAIL_SEPARATOR}"
            f"{row['tatoeba_author']}"
        ),
    }
    values.update(profile.card_fields(row, word))
    card: dict[str, Any] = {
        name: values[name] for name in profile.fields if name in values
    }
    card["tags"] = tags
    return card


def card_sort_key(
    word: WordRow, row: dict[str, str], profile: LanguageProfile
) -> tuple:
    """Build the note order key of one card.

    :param word: Reviewed word form of the card target.
    :param row: One row of the sentence CSV.
    :param profile: Language profile.
    :returns: Language order key of the note.
    """
    return profile.card_order(word, row)


def media_paths(filenames: Iterable[str], audio_dir: Path) -> list[str]:
    """Build the absolute media paths of the file names.

    :param filenames: MP3 file names from the audio index.
    :param audio_dir: Directory of the MP3 files.
    :returns: Sorted absolute paths with no duplicate.
    """
    return [str((Path(audio_dir) / name).resolve()) for name in sorted(set(filenames))]



def audio_license_source(tsv_path: Path) -> dict[str, int]:
    """Count the license values of a Tatoeba audio export.

    :param tsv_path: ``*_sentences_with_audio.tsv`` file.
    :returns: Empty-license and CC BY-NC 4.0 counts of the whole export.
    """
    if not Path(tsv_path).exists():
        return {key: 0 for key in LICENSE_KEYS}
    counts = {key: 0 for key in LICENSE_KEYS}
    with Path(tsv_path).open(encoding="utf-8", newline="") as handle:
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


def audio_author_map(tsv_path: Path) -> dict[int, str]:
    """Read the author of each Tatoeba audio id.

    :param tsv_path: ``*_sentences_with_audio.tsv`` file.
    :returns: Mapping from audio id to author. An empty mapping without a file.
    """
    if not Path(tsv_path).exists():
        return {}
    authors: dict[int, str] = {}
    with Path(tsv_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            try:
                authors[int(row[1])] = row[2].strip()
            except ValueError:
                continue
    return authors


def used_audio_authors(
    audio_authors: dict[int, str], rows: Iterable[dict[str, str]]
) -> dict[int, str]:
    """Keep the authors of the native recordings that the pack uses.

    :param audio_authors: Mapping from every audio id of the export to its
        author.
    :param rows: Sentence CSV rows of the pack.
    :returns: The entries of the ids in a native row of the pack.
    """
    used: set[int] = set()
    for row in rows:
        if row.get("audio_source") != "native":
            continue
        try:
            audio_id = int(str(row.get("audio_id", "")).strip())
        except ValueError:
            continue
        if audio_id:
            used.add(audio_id)
    return {
        audio_id: author
        for audio_id, author in audio_authors.items()
        if audio_id in used
    }


def tts_engine_name(voice: str) -> str:
    """Name the engine that produced the voice.

    :param voice: Voice value from the audio index.
    :returns: ``gTTS`` for a gTTS voice, else ``edge-tts``.
    """
    return "gTTS" if voice.startswith("gtts") else "edge-tts"


def source_dump_date(checksums_path: Path = CHECKSUMS_PATH) -> str:
    """Read the newest download date of the Tatoeba cache.

    :param checksums_path: Path to ``checksums.json``.
    :returns: ISO date, or an empty string without a cache.
    """
    if not Path(checksums_path).exists():
        return ""
    try:
        data = json.loads(Path(checksums_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    stamps = [
        str(info.get("downloaded_at", ""))[:10]
        for info in data.values()
        if isinstance(info, dict) and info.get("downloaded_at")
    ]
    return max(stamps) if stamps else ""


def sha256_value(path: Path) -> str:
    """Build the manifest hash value of a file.

    :param path: File to hash.
    :returns: ``sha256:`` followed by the hex digest.
    """
    return f"sha256:{hashlib.sha256(Path(path).read_bytes()).hexdigest()}"


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into row mappings.

    :param path: CSV file.
    :returns: One mapping per data row.
    """
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def author_groups(rows: Iterable[dict[str, str]]) -> dict[str, list[int]]:
    """Group the Tatoeba sentence ids by author.

    :param rows: Sentence CSV rows.
    :returns: Mapping from author to sorted sentence ids.
    """
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        author = row.get("tatoeba_author", "").strip() or "unknown"
        grouped[author].append(int(row["tatoeba_id"]))
    return {author: sorted(ids) for author, ids in grouped.items()}


def attribution_lines(
    profile: LanguageProfile, rows: Sequence[dict[str, str]], voice: str
) -> list[str]:
    """Build the numbered attribution lines of the sentence authors.

    :param profile: Language profile.
    :param rows: Sentence CSV rows that entered the pack.
    :param voice: Voice value from the audio index.
    :returns: Markdown lines of the sentence and audio attribution.
    """
    lines = [
        f"Sentences come from Tatoeba (<https://tatoeba.org>) and use the "
        f"{profile.tatoeba_license} license.",
        "",
        "Word data: CC BY-SA 4.0 from "
        f"<{profile.swadesh_url}>. Sentences: {profile.tatoeba_license}.",
        "",
        f"Audio: {tts_engine_name(voice)}, voice {voice}.",
        "",
        "| Author | Tatoeba sentence ids |",
        "| ------ | -------------------- |",
    ]
    for author, ids in sorted(author_groups(rows).items()):
        joined = ", ".join(str(value) for value in ids)
        lines.append(f"| {author} | {joined} |")
    return lines


def write_attribution_section(
    path: Path, title: str, lines: Sequence[str], header: Sequence[str] = ()
) -> Path:
    """Replace or append one ``## <title> pack`` section of the shared file.

    :param path: ``ATTRIBUTION.md`` file.
    :param title: Section title, for example ``Greek``.
    :param lines: Markdown lines of the section body.
    :param header: Lines written when the file does not exist yet.
    :returns: The written path.
    """
    path = Path(path)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    heading = f"## {title} pack"
    section = [heading, ""] + list(lines)
    parts: list[str] = []
    if existing.strip():
        kept: list[str] = []
        skipping = False
        for line in existing.splitlines():
            if line.startswith("## "):
                skipping = line.strip() == heading
            if not skipping:
                kept.append(line)
        parts = kept
    else:
        parts = list(header)
    while parts and not parts[-1].strip():
        parts.pop()
    if parts:
        parts.append("")
    parts.extend(section)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    return path


def build_manifest(
    profile: LanguageProfile,
    table: Sequence[WordRow],
    rows: Sequence[dict[str, str]],
    index: AudioIndex,
    *,
    word_csv: Path,
    csv_path: Path,
    pack_license: str = PUBLIC_PACK,
) -> dict[str, Any]:
    """Build the manifest mapping of one pack.

    :param profile: Language profile.
    :param table: Reviewed word table.
    :param rows: Sentence CSV rows.
    :param index: Audio index of the pack.
    :param word_csv: Reviewed word CSV.
    :param csv_path: Sentence CSV.
    :param pack_license: ``public`` or ``private``.
    :returns: The manifest mapping.
    """
    targets = {row["target"] for row in rows}
    positions = profile.form_positions(table)
    concepts = {
        positions[target][0] for target in targets if target in positions
    }
    manifest: dict[str, Any] = {
        "deck": profile.deck_name,
        "model": profile.model_name,
        "build_date": date.today().isoformat(),
        "source_dump_date": source_dump_date(),
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
            "core_with_card": round(len(concepts) / profile.core_concepts, 4),
        },
        "voice": index.voice,
        "pack_license": pack_license,
        "audio_license_skipped": skipped_license_counts(rows, pack_license),
    }
    manifest.update(profile.extra_manifest(rows))
    manifest["files"] = {
        "source_csv": sha256_value(word_csv),
        "sentences_csv": sha256_value(csv_path),
    }
    manifest["licenses"] = [
        {"data": name, "license": license_text}
        for name, license_text in profile.licenses
    ]
    manifest["excluded_words"] = excluded_forms(profile, table, targets)
    return manifest


def build_pack(
    profile: LanguageProfile,
    load_table: Callable[..., Sequence[WordRow]],
    *,
    csv_path: Path | None = None,
    word_csv: Path | None = None,
    audio_dir: Path | None = None,
    output: Path | None = None,
    manifest_path: Path | None = None,
    attribution_path: Path = ATTRIBUTION_PATH,
    pack_license: str = PUBLIC_PACK,
    css_path: Path = CSS_PATH,
) -> dict[str, Any]:
    """Build the APKG file, the manifest, and the attribution file.

    :param profile: Language profile.
    :param load_table: Callable that reads the reviewed word table.
    :param csv_path: Sentence CSV with one row per card.
    :param word_csv: Reviewed Swadesh word CSV.
    :param audio_dir: Directory with the MP3 files and ``index.csv``.
    :param output: Path of the APKG file to write.
    :param manifest_path: Path of the manifest JSON file to write.
    :param attribution_path: Path of the attribution Markdown file to write.
    :param pack_license: ``public`` or ``private``.
    :param css_path: Base card CSS file.
    :raises FileNotFoundError: The audio index or a media file is missing.
    :returns: The written paths, the media count, the skipped card count, and
        the manifest mapping.
    """
    csv_path = Path(csv_path) if csv_path is not None else profile.sentence_csv
    word_csv = Path(word_csv) if word_csv is not None else profile.word_csv
    audio_dir = Path(audio_dir) if audio_dir is not None else profile.audio_dir
    output = Path(output) if output is not None else profile.output_apkg
    manifest_path = (
        Path(manifest_path) if manifest_path is not None else profile.manifest_path
    )
    table = list(load_table(csv_path=word_csv))
    by_target = {profile.form_text(form): form for form in table}
    rows = read_rows(csv_path)
    index = load_audio_index(audio_dir / "index.csv")
    templates = assemble_templates(
        TemplateSpec(
            front=profile.front_template,
            back=profile.back_template,
            css_layers=(load_optional_text(css_path), profile.extra_css),
            fields=profile.fields,
            char_links=profile.char_links,
        )
    )

    used: list[tuple[WordRow, dict[str, str], dict[str, Any]]] = []
    skipped = 0
    no_target = 0
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
        card = build_card(row, word, sentence_audio, word_audio, profile)
        if "<b>" not in card["Sentence"]:
            skipped += 1
            no_target += 1
            continue
        used.append((word, row, card))
    used.sort(key=lambda item: card_sort_key(item[0], item[1], profile))

    filenames = [
        name
        for _, _, card in used
        for name in (
            card["Audio"].removeprefix("[sound:").removesuffix("]"),
            card["WordAudio"].removeprefix("[sound:").removesuffix("]"),
        )
    ]
    media = media_paths(filenames, audio_dir)
    for path in media:
        if not Path(path).exists():
            raise FileNotFoundError(f"media file not found: {path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    AnkiPackage.create(
        output,
        profile.deck_name,
        fields=list(profile.fields),
        cards=[card for _, _, card in used],
        media_files=media,
        model_name=profile.model_name,
        question_format=templates.front,
        answer_format=templates.back,
        css=templates.css,
        sort_field_index=profile.sort_field_index,
        deck_description=profile.deck_description,
    )

    packed_rows = [row for _, row, _ in used]
    manifest = build_manifest(
        profile,
        table,
        packed_rows,
        index,
        word_csv=word_csv,
        csv_path=csv_path,
        pack_license=pack_license,
    )
    manifest["counts"]["no_target"] = no_target
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    profile.attribution(
        packed_rows,
        index.voice,
        used_audio_authors(audio_author_map_for(profile, audio_dir), packed_rows),
        attribution_path,
    )

    return {
        "output": str(output),
        "manifest_path": str(manifest_path),
        "media_count": len(media),
        "skipped": skipped,
        "no_target": no_target,
        "manifest": manifest,
    }


def audio_author_map_for(
    profile: LanguageProfile, audio_dir: Path  # noqa: ARG001 - profile of the pack
) -> dict[int, str]:
    """Read the audio authors of a pack.

    :param profile: Language profile.
    :param audio_dir: Directory of the MP3 files.
    :returns: Mapping from audio id to author, empty without a native export.
    """
    path = Path("data/tatoeba") / f"{profile.tatoeba_lang}_sentences_with_audio.tsv"
    return audio_author_map(path)
