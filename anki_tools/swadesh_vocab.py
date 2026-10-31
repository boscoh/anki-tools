"""Build the Swadesh vocabulary deck, one card per concept.

The deck is one note type with the target form, its transliteration, its
English gloss, and the audio of the form. The audio comes from the cache
directory, and gTTS writes the missing files. The deck writer writes the
package. ADR 0024 records the seam.
"""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

from gtts import gTTS

from anki_tools.apkg import deck_writer
from anki_tools.apkg.templates import (
    SHARED_CSS_PATH,
    TEMPLATES_DIR,
    TemplateSpec,
    assemble_templates,
    load_optional_text,
    load_text,
)
from anki_tools.ranking.identity import DeckIdentity, default_output
from anki_tools.languages import LANGUAGES
from anki_tools.languages.registry import vocab_languages

ROOT = Path(__file__).parent.parent


@dataclass(frozen=True)
class _BuildLayout:
    """Note fields, templates, and field order of one vocabulary deck."""

    fields: list[str]
    content_order: list[str]
    front_template: str
    answer_format: str
    css_content: str
    css_path: Path


def clean_text_for_audio(text: str) -> str:
    """Remove parenthetical notes and variants from text for audio processing.

    :param text: Raw text (e.g. from CSV).
    :returns: Cleaned text for TTS.
    """
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.split("/")[0].strip()


def _deck_config(language: str) -> dict:
    """Give the paths and the output identity of the vocabulary deck.

    :param language: Language name, for example ``french``.
    :returns: Config dict of the deck.
    """
    lang_lower = language.lower()
    return {
        "language": language,
        "vocab_file": f"vocab/english-{lang_lower}.csv",
        "audio_dir": f"audio/audio_{lang_lower}",
        "identity": DeckIdentity(default_output(lang_lower, "vocab")),
        "audio_field_name": lang_lower,
    }


def _parse_vocab_entries(config: dict) -> list[dict]:
    """Parse vocabulary CSV file to extract entries.

    :param config: Config dict with vocab_file and audio_field_name.
    :returns: List of entry dicts with audio_filename set.
    """
    csv_path = Path(config["vocab_file"])
    if not csv_path.is_absolute():
        csv_path = ROOT / config["vocab_file"]
    entries = []
    try:
        handle = open(csv_path, "r", encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read the vocabulary file {csv_path}.") from exc
    with handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append({k: v.strip() if v else v for k, v in row.items()})

    for entry in entries:
        text = entry[config["audio_field_name"]]
        cleaned_text = clean_text_for_audio(text).replace(" ", "_")
        entry["audio_filename"] = f"{cleaned_text}.mp3"

    return entries


def _prune_outdated_audio(audio_dir: Path, keep: set[str]) -> list[str]:
    """Delete audio files that no list entry refers to.

    :param audio_dir: Directory with generated audio files.
    :param keep: File names used by the current list entries.
    :returns: File names of the deleted audio files.
    """
    if not audio_dir.is_dir():
        return []

    removed = []
    for path in sorted(audio_dir.iterdir()):
        if not path.is_file() or path.name in keep:
            continue
        path.unlink()
        removed.append(path.name)

    return removed


def _generate_audio_files(entries: list[dict], config: dict) -> None:
    """Generate audio files for entries using TTS if they don't exist.

    :param entries: List of entry dicts from _parse_vocab_entries.
    :param config: Config dict with language and audio_dir.
    """
    language = config.get("language")
    if not language:
        return

    entry = LANGUAGES.get(language.lower())
    lang = entry.tts if entry is not None else None
    if not lang:
        print(f"Unknown language for gTTS: {language}")
        return

    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = ROOT / config["audio_dir"]
    audio_dir.mkdir(parents=True, exist_ok=True)

    delay_seconds = 0.5
    total_entries = len(entries)
    existing_count = 0
    generated_count = 0

    for i, entry in enumerate(entries, 1):
        audio_path = audio_dir / entry["audio_filename"]

        if audio_path.exists():
            existing_count += 1
            continue

        try:
            text = entry[config["audio_field_name"]]
            text = clean_text_for_audio(text)

            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(audio_path))
            generated_count += 1
            print(f"[{i}/{total_entries}] Generated: {entry['audio_filename']}")

            if delay_seconds > 0 and i < total_entries:
                time.sleep(delay_seconds)
        except Exception as e:
            print(f"Error generating {entry['audio_filename']}: {e}")
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    if existing_count > 0 or generated_count > 0:
        print(
            f"\nAudio generation complete: {generated_count} generated, "
            f"{existing_count} already existed"
        )


def _content_field_order(csv_fields: list[str], audio_field_name: str) -> list[str]:
    """Order the content fields: target language, transliteration, english.

    Review columns of a sentence-pack word table are not card content. The
    function keeps only the target form, the transliteration, and the English
    gloss.

    :param csv_fields: All CSV column names.
    :param audio_field_name: Name of the audio/target language field.
    :returns: Ordered content field names.
    """
    target = next(
        (f for f in csv_fields if f.lower() == audio_field_name.lower()),
        audio_field_name,
    )
    rest = [f for f in csv_fields if f != target]
    result = [target]
    for name in ("transliteration", "english"):
        found = next((f for f in rest if f.lower() == name), None)
        if found:
            result.append(found)
    return result


CJK_FIELDS = {"characters", "hanzi", "cantonese", "mandarin", "chinese"}
FIELD_CLASSES = {"transliteration": "romanization", "english": "english"}


def _field_classes(field: str) -> str:
    """Give the CSS classes of one content field.

    :param field: CSV column name.
    :returns: The class value of the field div.
    """
    name = field.lower()
    if name in CJK_FIELDS:
        return "target char-links"
    return FIELD_CLASSES.get(name, "target")


def _build_layout(config: dict, entries: list[dict]) -> _BuildLayout:
    """Compute the note fields, templates, and field order from the config.

    :param config: Config dict with the audio field name and language.
    :param entries: List of entry dicts from _parse_vocab_entries.
    :returns: The layout of the deck.
    """
    front_template = load_text(TEMPLATES_DIR / "vocab" / "front.html")
    back_template = load_text(TEMPLATES_DIR / "vocab" / "back.html")

    language_css_path = TEMPLATES_DIR / "vocab" / config["language"].lower() / "card.css"
    css_layers = (
        load_optional_text(SHARED_CSS_PATH),
        load_optional_text(language_css_path),
    )

    csv_fields = (
        [k for k in entries[0].keys() if k != "audio_filename"] if entries else []
    )
    content_order = _content_field_order(csv_fields, config["audio_field_name"])
    fields = content_order + ["audio"]

    field_parts = [
        f"<div class='{_field_classes(field)}'>{{{{{field}}}}}</div>"
        for field in content_order
    ]
    answer_format = back_template.replace("{{fields}}", "\n".join(field_parts))
    templates = assemble_templates(
        TemplateSpec(
            front=front_template,
            back=answer_format,
            css_layers=css_layers,
            fields=fields,
            char_links=True,
        )
    )

    return _BuildLayout(
        fields=fields,
        content_order=content_order,
        front_template=templates.front,
        answer_format=templates.back,
        css_content=templates.css,
        css_path=language_css_path,
    )


def _pack_apkg(
    config: dict, entries: list[dict], layout: _BuildLayout, prune: bool
) -> deck_writer.RunFiles | None:
    """Write the APKG file and prune the outdated audio files.

    :param config: Config dict with the deck and audio settings.
    :param entries: List of entry dicts from _parse_vocab_entries.
    :param layout: Deck layout from _build_layout.
    :param prune: Delete audio files that the current list does not use.
    :returns: The written paths, or None when a form has no audio.
    """
    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = ROOT / config["audio_dir"]
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return None

    keep = {entry["audio_filename"] for entry in entries}
    if prune:
        removed = _prune_outdated_audio(audio_dir, keep)
        if removed:
            print(f"Removed {len(removed)} outdated audio files")

    cards: list[deck_writer.Card] = []
    media: list[deck_writer.MediaFile] = []
    missing: list[str] = []

    for entry in entries:
        audio_path = audio_dir / entry["audio_filename"]

        if not audio_path.exists():
            missing.append(entry.get("english", entry["audio_filename"]))
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        values = {"audio": f"[sound:{entry['audio_filename']}]"}
        for field in layout.content_order:
            values[field] = entry.get(field, "")
        cards.append(deck_writer.Card(values=values))
        media.append(deck_writer.MediaFile(audio_path.absolute(), audio_path.name))

    if missing:
        print(f"Error: {len(missing)} forms have no audio: " + ", ".join(missing))
        return None

    identity = config["identity"]
    identity.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Creating APKG: {identity.output_path}")
    print(f"  Deck: {identity.title}")
    print(f"  Cards: {len(cards)}")
    print(f"  Audio files: {len(media)}")
    if layout.css_content:
        print(f"  CSS: {layout.css_path}")

    files = deck_writer.write_deck(
        deck_writer.DeckWrite(
            output_path=identity.output_path,
            shape=deck_writer.NoteTypeShape(
                deck_name=identity.title,
                model_name=identity.model_name,
                fields=tuple(layout.fields),
                templates=((layout.front_template, layout.answer_format),),
                css=layout.css_content,
            ),
            cards=cards,
            media=media,
        )
    )

    print(f"\nSuccessfully created: {files.output_path}")
    return files


def build_vocab_deck(language: str, prune: bool = True) -> deck_writer.RunFiles | None:
    """Build the Swadesh vocabulary deck of one language.

    :param language: Language name, for example ``french``.
    :param prune: Delete audio files that the current list does not use.
    :returns: The written paths, or None when a form has no audio.
    """
    config = _deck_config(language)
    print(f"Building {language.title()} vocabulary APKG...")

    entries = _parse_vocab_entries(config)
    _generate_audio_files(entries, config)
    layout = _build_layout(config, entries)
    return _pack_apkg(config, entries, layout, prune)


def vocab_language_names() -> tuple[str, ...]:
    """Give the languages that have a vocabulary deck."""
    return tuple(vocab_languages())
