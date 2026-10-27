"""
CLI tools for Anki package (.apkg) files.

Commands:
    inspect - Diagnostic tools to inspect .apkg files
    swadesh - Build Swadesh decks and run their gates
    fix - Fix and optimize decks (zh, fr, yue)
    style - Apply CSS and card templates to any deck
    preview - Build an HTML preview of the templates or of a package
"""

import csv
import importlib.util
import os
import re
import shutil
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Sequence

import cyclopts
from cyclopts import Parameter
from gtts import gTTS

from anki_tools.languages.cantonese import core as cantonese_core
from anki_tools.languages.french import core as french_core
from anki_tools.languages.french import cull as french_cull
from anki_tools.languages.spanish import core as spanish_core
from anki_tools.languages.spanish import cull as spanish_cull
from anki_tools.languages.mandarin import core as mandarin_core
from anki_tools.languages.cantonese.jyutping import fix_jyutping
from anki_tools.languages.mandarin import fix_steps as mandarin_fix_steps
from anki_tools.apkg.package import AnkiPackage
from anki_tools.preview import write_preview
from anki_tools.preview_apkg import write_apkg_preview
from anki_tools.sentence_pack import audio as pack_audio
from anki_tools.sentence_pack import pack as sentence_pack
from anki_tools.sentence_pack import pool as sentence_pool
from anki_tools.sentence_pack import select as sentence_select
from anki_tools.sentence_pack import tatoeba as sentence_tatoeba
from anki_tools.sentence_pack.audit import print_audit as print_gate_table
from anki_tools.sentence_pack.profile import LanguageProfile
from anki_tools.languages import (
    LANGUAGES,
    profile_languages,
    resolve,
    vocab_languages,
)
from anki_tools.languages.cantonese.rank import rank_sentences_yue
from anki_tools.languages.french.rank import rank_sentences_fr
from anki_tools.languages.mandarin.rank import rank_sentences_zh
from anki_tools.languages.spanish.rank import rank_sentences_es
from anki_tools.ranking import fixrun
from anki_tools.ranking.rank import Sentence
from anki_tools.ranking.pipeline import (
    RankPaths,
    RankedDeck,
)
from anki_tools.apkg.templates import (
    DEFAULT_BACK_PATH,
    DEFAULT_CSS_PATH,
    DEFAULT_FRONT_PATH,
    SHARED_CSS_PATH,
    TEMPLATES_DIR,
    apply_templates,
    load_char_links,
    load_css_layers,
    load_text,
)

app = cyclopts.App(help="CLI tools for Anki package (.apkg) files")


# =============================================================================
# Inspect commands - diagnostic tools for .apkg files
# =============================================================================

inspect_app = cyclopts.App(name="inspect", help="Inspect Anki package (.apkg) files")
app.command(inspect_app)


@inspect_app.command
def cards(
    apkg_path: Path,
    *,
    limit: int = 10,
    field: str | None = None,
    search: str | None = None,
):
    """List cards in the deck.

    :param apkg_path: Path to .apkg file.
    :param limit: Maximum cards to show (0 for all).
    :param field: Only show this field name.
    :param search: Filter cards containing this text.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        models = pkg.get_models()
        decks = pkg.get_decks()

        print(f"Total cards: {len(cards)}\n")

        shown = 0
        for i, card in enumerate(cards):
            parsed = pkg.parse_card(card, models, decks)
            fields = parsed["fields"]

            if search:
                text = " ".join(str(v) for v in fields.values())
                if search.lower() not in text.lower():
                    continue

            print(f"Card {i}:")
            print(f"  Deck: {parsed['deck']}")
            print(f"  Type: {parsed['model']}")

            if field:
                if field in fields:
                    value = re.sub(r"<[^>]+>", "", str(fields[field]))
                    print(f"  {field}: {value[:200]}")
            else:
                for name, value in fields.items():
                    value = re.sub(r"<[^>]+>", "", str(value))
                    print(f"  {name}: {value[:100]}")

            if parsed["tags"]:
                print(f"  Tags: {parsed['tags']}")
            print()

            shown += 1
            if limit and shown >= limit:
                remaining = len(cards) - shown
                if remaining > 0:
                    print(f"... and {remaining} more cards")
                break


@inspect_app.command
def decks(apkg_path: Path):
    """List all decks in the package.

    :param apkg_path: Path to .apkg file.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        decks = pkg.get_decks()
        cards = pkg.get_cards()

        print(f"Decks ({len(decks)}):\n")

        for deck_id, deck_info in decks.items():
            name = deck_info.get("name", "Unknown")
            card_count = len([c for c in cards if str(c["did"]) == deck_id])
            print(f"  {name}")
            print(f"    ID: {deck_id}")
            print(f"    Cards: {card_count}")
            print()


@inspect_app.command
def models(apkg_path: Path, *, verbose: bool = False):
    """List note types (models) and their fields.

    :param apkg_path: Path to .apkg file.
    :param verbose: If True, show all model details.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        models = pkg.get_models()

        print(f"Note Types ({len(models)}):\n")

        for model_id, model_info in models.items():
            name = model_info.get("name", "Unknown")
            fields = [f["name"] for f in model_info.get("flds", [])]
            templates = [t["name"] for t in model_info.get("tmpls", [])]

            print(f"  {name}")
            print(f"    ID: {model_id}")
            print(f"    Fields: {', '.join(fields)}")
            print(f"    Templates: {', '.join(templates)}")

            if verbose:
                print(f"    CSS length: {len(model_info.get('css', ''))}")
                for tmpl in model_info.get("tmpls", []):
                    print(f"    Template '{tmpl['name']}':")
                    print(f"      Front: {len(tmpl.get('qfmt', ''))} chars")
                    print(f"      Back: {len(tmpl.get('afmt', ''))} chars")
            print()


@inspect_app.command
def templates(
    apkg_path: Path,
    *,
    model_name: str | None = None,
):
    """Show card templates (question and answer) for each note type.

    :param apkg_path: Path to .apkg file.
    :param model_name: Filter by model name (substring match).
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        models = pkg.get_models(include_templates=True)

        for model_id, model_info in models.items():
            name = model_info.get("name", "Unknown")
            if model_name and model_name.lower() not in name.lower():
                continue

            try:
                width = os.get_terminal_size().columns
            except OSError:
                width = 80
            sep = "=" * width
            rule = "-" * width

            print()
            print(sep)
            print(f"#  {name} (ID: {model_id})")
            print(sep)
            for tmpl in model_info.get("tmpls", []):
                print()
                print(f"##  Template: {tmpl.get('name', 'Card 1')}")
                print(rule)
                print("###  Question (front)")
                print(rule)
                print(tmpl.get("qfmt", "(none)"))
                print()
                print(rule)
                print("###  Answer (back)")
                print(rule)
                print(tmpl.get("afmt", "(none)"))
                print()
            print()


@inspect_app.command
def fields(apkg_path: Path, model_name: str | None = None):
    """List all field names across models.

    :param apkg_path: Path to .apkg file.
    :param model_name: Filter by model name.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        models = pkg.get_models()

        for _model_id, model_info in models.items():
            name = model_info.get("name", "Unknown")

            if model_name and model_name.lower() not in name.lower():
                continue

            fields = model_info.get("flds", [])

            print(f"{name}:")
            for i, field in enumerate(fields):
                print(f"  {i}: {field['name']}")
            print()


@inspect_app.command
def audio(apkg_path: Path, *, extract: Path | None = None):
    """Show audio/media statistics and optionally extract files.

    :param apkg_path: Path to .apkg file.
    :param extract: Directory to extract audio files to.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        stats = pkg.get_audio_statistics()

        print("Media Statistics:")
        print(f"  Total files: {stats['total_media_files']}")
        print(f"  Audio files: {stats['audio_files']}")
        print(f"  Image files: {stats['image_files']}")
        print(f"  Audio formats: {stats['audio_formats']}")
        print()

        if extract:
            extracted = pkg.extract_audio_files(str(extract))
            print(f"Extracted {len(extracted)} audio files to: {extract}")
            for filename in list(extracted.keys())[:10]:
                print(f"  {filename}")
            if len(extracted) > 10:
                print(f"  ... and {len(extracted) - 10} more")


@inspect_app.command
def stats(apkg_path: Path):
    """Show overall statistics for the package.

    :param apkg_path: Path to .apkg file.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        decks = pkg.get_decks()
        models = pkg.get_models()
        audio_stats = pkg.get_audio_statistics()

        print(f"File: {apkg_path.name}")
        print(f"Size: {apkg_path.stat().st_size / 1024:.1f} KB")
        print()
        print("Counts:")
        print(f"  Cards: {len(cards)}")
        print(f"  Decks: {len(decks)}")
        print(f"  Note Types: {len(models)}")
        print(f"  Media Files: {audio_stats['total_media_files']}")
        print(f"  Audio Files: {audio_stats['audio_files']}")
        print()

        print("Decks:")
        for deck_id, deck_info in decks.items():
            count = len([c for c in cards if str(c["did"]) == deck_id])
            print(f"  {deck_info.get('name', 'Unknown')}: {count} cards")
        print()

        print("Note Types:")
        for model_id, model_info in models.items():
            count = len([c for c in cards if str(c["mid"]) == model_id])
            fields = [f["name"] for f in model_info.get("flds", [])]
            print(f"  {model_info.get('name', 'Unknown')}: {count} cards")
            print(f"    Fields: {', '.join(fields)}")


@inspect_app.command
def sample(apkg_path: Path, *, count: int = 5):
    """Show sample field values from cards.

    :param apkg_path: Path to .apkg file.
    :param count: Number of sample values per field.
    """
    with AnkiPackage(str(apkg_path)) as pkg:
        cards = pkg.get_cards()
        models = pkg.get_models()
        decks = pkg.get_decks()

        for model_id, model_info in models.items():
            model_name = model_info.get("name", "Unknown")
            field_names = [f["name"] for f in model_info.get("flds", [])]

            model_cards = [c for c in cards if str(c["mid"]) == model_id]
            if not model_cards:
                continue

            print(f"{model_name} ({len(model_cards)} cards):")
            print("-" * 40)

            for field_name in field_names:
                print(f"\n  {field_name}:")
                samples = []
                for card in model_cards[: count * 2]:
                    parsed = pkg.parse_card(card, models, decks)
                    value = parsed["fields"].get(field_name, "")
                    value = re.sub(r"<[^>]+>", "", str(value)).strip()
                    if value and value not in samples:
                        samples.append(value[:80])
                    if len(samples) >= count:
                        break

                for sample in samples:
                    print(f"    - {sample}")
            print()


# =============================================================================
# Style - apply CSS and card templates (any deck)
# =============================================================================


style_app = cyclopts.App(
    name="style", help="Apply CSS and card templates to an Anki deck"
)
app.command(style_app)


@style_app.command
def apply(
    apkg_path: Path,
    *,
    output: Path | None = None,
    css: Path | None = None,
    front: Path | None = None,
    back: Path | None = None,
):
    """Apply CSS and card templates to a deck.

    The CSS starts with the shared sheet `templates/shared/card.css` and then
    the deck `card.css`, so a deck rule overrides a shared rule. The templates
    and the deck `card.css` default to the repository `templates/default/`
    directory.
    Auto-maps mismatched field names to closest deck fields.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_styled.apkg).
    :param css: Deck CSS file (default: `card.css` in `templates/default/`).
    :param front: Front template file (default: `front.html` in
        `templates/default/`).
    :param back: Back template file (default: `back.html` in
        `templates/default/`).
    """
    apkg_path = Path(apkg_path)
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_styled.apkg"
    output = Path(output)

    if apkg_path.resolve() != output.resolve():
        try:
            shutil.copy(apkg_path, output)
        except OSError as exc:
            raise RuntimeError(f"Cannot copy {apkg_path} to {output}.") from exc

    _n, applied = apply_templates(
        output,
        css_path=css or DEFAULT_CSS_PATH,
        front_path=front or DEFAULT_FRONT_PATH,
        back_path=back or DEFAULT_BACK_PATH,
        verbose=True,
    )
    if not applied:
        print(
            f"\nNothing to apply. Check {DEFAULT_CSS_PATH}, "
            f"{DEFAULT_FRONT_PATH}, and {DEFAULT_BACK_PATH} first."
        )
        return
    print(f"\nSaved: {output}")


# =============================================================================
# Fix commands - optimize decks by language
# =============================================================================

fix_app = cyclopts.App(name="fix", help="Fix and optimize Anki decks")
app.command(fix_app)


DEFAULT_MANDARIN_APKG = Path("apkg/Neri's Chinese Sents Part 2.apkg")
DEFAULT_FRENCH_APKG = Path("apkg/6000_French_Sentences_w_Audio.apkg")
DEFAULT_CANTONESE_APKG = Path("apkg/LTL Cantonese Deck Level 1 - Short.apkg")
DEFAULT_SPANISH_APKG = Path("apkg/Spanish 9000 sentences with native audio.apkg")


@fix_app.command
def zh(
    apkg_path: Path = DEFAULT_MANDARIN_APKG,
    *,
    output: Path | None = None,
    text_field: str | None = None,
    core: bool = False,
    cards: int | None = None,
    words: int = mandarin_core.DEFAULT_CORE_WORDS,
    check: bool = False,
):
    """Fix Chinese deck: rank, reorder, fix pinyin, and apply styling.

    Rank mode writes ``apkg/mandarin_reordered.apkg`` and its rank CSV, then
    fixes the pinyin and writes the card templates. ``--core`` builds
    ``apkg/mandarin_core.apkg`` with its CSV files and manifest.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the language and mode).
    :param text_field: Rank mode: preferred sentence field (default: infer).
    :param core: Build the core deck instead of ranking.
    :param cards: Core mode: card cap (300).
    :param words: Core mode: tier F words after the Swadesh set.
    :param check: Run the gates on the output and do not run the mode again.
    """
    fixrun.run_fix(
        fixrun.FixLanguage(
            name="mandarin",
            label=LANGUAGES["mandarin"].label,
            core=mandarin_core.MANDARIN_CORE_PROFILE,
            rank=rank_sentences_zh,
            deck_kwargs={"romanization_candidates": ("Pinyin",)},
            after=mandarin_fix_steps.after_rank,
        ),
        apkg_path=apkg_path,
        output=output,
        text_field=text_field,
        core=core,
        cards=cards,
        words=words,
        check=check,
    )


@fix_app.command
def fr(
    apkg_path: Path = DEFAULT_FRENCH_APKG,
    *,
    output: Path | None = None,
    text_field: str = "French",
    select: bool = False,
    core: bool = False,
    cards: int | None = None,
    reserve: bool = True,
    words: int = french_core.DEFAULT_CORE_WORDS,
    check: bool = False,
):
    """Fix French deck: rank and reorder it, or select a smaller deck.

    Rank mode writes ``apkg/french_reordered.apkg`` and its rank CSV. Select
    mode, with ``--select``, runs the 1000-card cull and writes
    ``apkg/french_1000.apkg``. ``--core`` builds ``apkg/french_core.apkg``
    with its CSV files and manifest.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the language and mode).
    :param text_field: Rank mode: preferred sentence field (default: French).
    :param select: Run the card cull instead of the rank and reorder.
    :param core: Build the core deck instead of ranking.
    :param cards: Select and core mode: card cap (select 1000, core 300).
    :param reserve: Select mode: add the Swadesh reserve cards.
    :param words: Core mode: tier F words after the Swadesh set.
    :param check: Run the gates on the output and do not run the mode again.
    """
    fixrun.run_fix(
        fixrun.FixLanguage(
            name="french",
            label=LANGUAGES["french"].label,
            core=french_core.FRENCH_CORE_PROFILE,
            cull=french_cull.FRENCH_CULL_PROFILE,
            rank=rank_sentences_fr,
            deck_kwargs={"text_candidates": ("French",)},
        ),
        apkg_path=apkg_path,
        output=output,
        text_field=text_field,
        core=core,
        select=select,
        cards=cards,
        reserve=reserve,
        words=words,
        check=check,
    )


@fix_app.command
def yue(
    apkg_path: Path = DEFAULT_CANTONESE_APKG,
    *,
    output: Path | None = None,
    text_field: str | None = None,
    core: bool = False,
    cards: int | None = None,
    words: int = cantonese_core.DEFAULT_CORE_WORDS,
    check: bool = False,
):
    """Fix Cantonese deck: rank and reorder it, or build a core deck.

    Rank mode writes ``apkg/cantonese_reordered.apkg`` and its rank CSV, with
    the jyutping fixes. ``--core`` builds ``apkg/cantonese_core.apkg`` with its
    CSV files and manifest.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the language and mode).
    :param text_field: Rank mode: preferred sentence field (default: infer).
    :param core: Build the core deck instead of ranking.
    :param cards: Core mode: card cap (300).
    :param words: Core mode: tier F words after the Swadesh set.
    :param check: Run the gates on the output and do not run the mode again.
    """
    counts: dict[str, int] = {}

    def rank_and_capture(sentences: list[Sentence]) -> list[Sentence]:
        ranked = rank_sentences_yue(sentences)
        counts["ranked"] = len(ranked)
        return ranked

    def reorder(ranked: list[Sentence]) -> list[Sentence]:
        print("\nStep 2: Reordering deck...")
        return ranked

    def after_rank(paths: RankPaths, deck: RankedDeck) -> Sequence[str]:
        print("\nFixing jyutping...")
        fix_jyutping(paths.output_path, output=paths.output_path, verbose=False)
        return ()

    fixrun.run_fix(
        fixrun.FixLanguage(
            name="cantonese",
            label=LANGUAGES["cantonese"].label,
            core=cantonese_core.CANTONESE_CORE_PROFILE,
            rank=rank_and_capture,
            deck_kwargs={
                "romanization_candidates": ("Jyutping",),
                "reduce": reorder,
                "remove_unranked": False,
            },
            after=after_rank,
            count=lambda total: counts.get("ranked", total),
        ),
        apkg_path=apkg_path,
        output=output,
        text_field=text_field,
        core=core,
        cards=cards,
        words=words,
        check=check,
    )


@fix_app.command
def es(
    apkg_path: Path = DEFAULT_SPANISH_APKG,
    *,
    output: Path | None = None,
    text_field: str = "Spanish",
    select: bool = False,
    core: bool = False,
    cards: int | None = None,
    reserve: bool = True,
    words: int = spanish_core.DEFAULT_CORE_WORDS,
    check: bool = False,
):
    """Fix Spanish deck: rank and reorder it, or select a smaller deck.

    Rank mode writes ``apkg/spanish_reordered.apkg`` and its rank CSV. Select
    mode, with ``--select``, runs the cull and writes ``apkg/spanish_1000.apkg``.
    ``--core`` builds ``apkg/spanish_core.apkg`` with its CSV files and
    manifest.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the language and mode).
    :param text_field: Rank mode: preferred sentence field (default: Spanish).
    :param select: Run the cull instead of the rank and reorder.
    :param core: Build the core deck instead of ranking.
    :param cards: Select and core mode: card cap (select 1000, core 300).
    :param reserve: Select mode: add the Swadesh reserve cards.
    :param words: Core mode: tier F words after the Swadesh set.
    :param check: Run the gates on the output and do not run the mode again.
    """
    fixrun.run_fix(
        fixrun.FixLanguage(
            name="spanish",
            label=LANGUAGES["spanish"].label,
            core=spanish_core.SPANISH_CORE_PROFILE,
            cull=spanish_cull.SPANISH_CULL_PROFILE,
            rank=rank_sentences_es,
        ),
        apkg_path=apkg_path,
        output=output,
        text_field=text_field,
        core=core,
        select=select,
        cards=cards,
        reserve=reserve,
        words=words,
        check=check,
    )

for _fix_command, _canonical_name in (
    (zh, "mandarin"),
    (fr, "french"),
    (yue, "cantonese"),
    (es, "spanish"),
):
    fix_app.command(_fix_command, name=_canonical_name)

# =============================================================================
# Swadesh commands - build decks and run their gates
# =============================================================================

def _clean_text_for_audio(text: str) -> str:
    """Remove parenthetical notes and variants from text for audio processing.

    :param text: Raw text (e.g. from CSV).
    :returns: Cleaned text for TTS.
    """
    text = re.sub(r"\s*\([^)]*\)", "", text)
    return text.split("/")[0].strip()


def _parse_vocab_entries(config: dict) -> list[dict]:
    """Parse vocabulary CSV file to extract entries.

    :param config: Config dict with vocab_file and audio_field_name.
    :returns: List of entry dicts with audio_filename set.
    """
    csv_path = Path(config["vocab_file"])
    if not csv_path.is_absolute():
        csv_path = Path(__file__).parent.parent / config["vocab_file"]
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
        cleaned_text = _clean_text_for_audio(text).replace(" ", "_")
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
        audio_dir = Path(__file__).parent.parent / config["audio_dir"]
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
            text = _clean_text_for_audio(text)

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


@dataclass(frozen=True)
class _BuildLayout:
    """Note fields, templates, and field order of one vocabulary deck."""

    fields: list[str]
    content_order: list[str]
    front_template: str
    answer_format: str
    css_content: str
    css_path: Path


def _build_layout(config: dict, entries: list[dict]) -> _BuildLayout:
    """Compute the note fields, templates, and field order from the config.

    :param config: Config dict with the audio field name and language.
    :param entries: List of entry dicts from _parse_vocab_entries.
    :returns: The layout of the deck.
    """
    front_template = load_text(TEMPLATES_DIR / "vocab" / "front.html")
    back_template = load_text(TEMPLATES_DIR / "vocab" / "back.html")

    language_css_path = TEMPLATES_DIR / "vocab" / config["language"].lower() / "card.css"
    css_content = load_css_layers(SHARED_CSS_PATH, language_css_path)

    csv_fields = (
        [k for k in entries[0].keys() if k != "audio_filename"] if entries else []
    )
    content_order = _content_field_order(csv_fields, config["audio_field_name"])
    fields = content_order + ["audio"]

    char_click_script = load_char_links()

    cjk_fields = {"characters", "hanzi", "cantonese", "mandarin", "chinese"}
    field_parts: list[str] = []
    for field in content_order:
        name = field.lower()
        if name in cjk_fields:
            classes = "target char-links"
        elif name == "transliteration":
            classes = "romanization"
        elif name == "english":
            classes = "english"
        else:
            classes = "target"
        field_parts.append(f"<div class='{classes}'>{{{{{field}}}}}</div>")
    if char_click_script:
        field_parts.append(char_click_script)
    answer_format = back_template.replace("{{fields}}", "\n".join(field_parts))

    return _BuildLayout(
        fields=fields,
        content_order=content_order,
        front_template=front_template,
        answer_format=answer_format,
        css_content=css_content,
        css_path=language_css_path,
    )


def _pack_apkg(
    config: dict, entries: list[dict], layout: _BuildLayout, prune: bool
) -> None:
    """Write the APKG file and prune the outdated audio files.

    :param config: Config dict with the deck and audio settings.
    :param entries: List of entry dicts from _parse_vocab_entries.
    :param layout: Deck layout from _build_layout.
    :param prune: Delete audio files that the current list does not use.
    """
    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = Path(__file__).parent.parent / config["audio_dir"]
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return

    keep = {entry["audio_filename"] for entry in entries}
    if prune:
        removed = _prune_outdated_audio(audio_dir, keep)
        if removed:
            print(f"Removed {len(removed)} outdated audio files")

    cards = []
    media_files = []
    missing: list[str] = []

    for entry in entries:
        audio_path = audio_dir / entry["audio_filename"]

        if not audio_path.exists():
            missing.append(entry.get("english", entry["audio_filename"]))
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        card = {"audio": f"[sound:{entry['audio_filename']}]"}
        for field in layout.content_order:
            card[field] = entry.get(field, "")
        cards.append(card)

        media_files.append(str(audio_path.absolute()))

    if missing:
        print(f"Error: {len(missing)} forms have no audio: " + ", ".join(missing))
        return

    apkg_dir = Path("apkg")
    apkg_dir.mkdir(exist_ok=True)

    output_path = apkg_dir / config["output_file"]

    print(f"Creating APKG: {output_path}")
    print(f"  Deck: {config['deck_name']}")
    print(f"  Cards: {len(cards)}")
    print(f"  Audio files: {len(media_files)}")
    if layout.css_content:
        print(f"  CSS: {layout.css_path}")

    tmp_path = output_path.with_name(output_path.name + ".tmp")
    AnkiPackage.create(
        tmp_path,
        config["deck_name"],
        fields=layout.fields,
        cards=cards,
        media_files=media_files,
        model_name=config["model_name"],
        question_format=layout.front_template,
        answer_format=layout.answer_format,
        css=layout.css_content,
        sort_field_index=0,
    )
    tmp_path.replace(output_path)

    print(f"\nSuccessfully created: {output_path}")


def _build_apkg(language: str, prune: bool = True) -> None:
    """Build APKG file with vocabulary and audio for the given language.

    :param language: Language name (e.g. mandarin, spanish).
    :param prune: Delete audio files that the current list does not use.
    """
    lang_lower = language.lower()
    lang_title = language.title()
    print(f"Building {lang_title} vocabulary APKG...")

    config = {
        "language": language,
        "vocab_file": f"vocab/english-{lang_lower}.csv",
        "audio_dir": f"audio/audio_{lang_lower}",
        "deck_name": f"Swadesh {lang_title} Vocabulary",
        "model_name": f"Swadesh {lang_title} Vocabulary",
        "output_file": f"swadesh_{lang_lower}.apkg",
        "audio_field_name": lang_lower,
    }

    entries = _parse_vocab_entries(config)

    _generate_audio_files(entries, config)
    layout = _build_layout(config, entries)
    _pack_apkg(config, entries, layout, prune)


VOCAB_LANGUAGES = vocab_languages()
SENTENCE_PACK_LANGUAGE_NAMES = ", ".join(
    language.name for language in profile_languages()
)

swadesh_app = cyclopts.App(
    name="swadesh", help="Build Swadesh decks and run their gates"
)
app.command(swadesh_app)


@swadesh_app.command
def vocab(
    language: Annotated[
        str, Parameter(help=f"One of {', '.join(VOCAB_LANGUAGES)}, or all.")
    ],
    prune: bool = True,
):
    """Build the Swadesh vocabulary deck for a language.

    :param language: One of the supported languages, or all.
    :param prune: Delete audio files that the current list does not use.
    """
    lang = resolve(language)
    if lang == "all":
        for name in VOCAB_LANGUAGES:
            _build_apkg(name, prune=prune)
        return

    if lang not in VOCAB_LANGUAGES:
        print(f"Unknown language: {language}. Supported: {', '.join(VOCAB_LANGUAGES)}")
        return
    _build_apkg(lang, prune=prune)


def _fetch_language(profile: LanguageProfile, refresh: bool) -> None:
    """Download the Tatoeba exports of one language and English (stage S0).

    :param profile: Profile of the language. Its code selects the export.
    :param refresh: Ignore the cache and download every file again.
    """
    files = sentence_tatoeba.language_files(
        profile.tatoeba_lang, native_audio=profile.native_audio
    )
    files.update(sentence_tatoeba.language_files("eng", native_audio=False))
    results = sentence_tatoeba.fetch_tatoeba(files, refresh=refresh)
    for name, info in results.items():
        print(f"{name}.tsv: {info['rows']} rows ({info['source']})")


@swadesh_app.command
def pack(
    language: Annotated[str, Parameter(help=f"One of {SENTENCE_PACK_LANGUAGE_NAMES}.")],
    *,
    refresh: bool = False,
    stage: Literal["fetch", "sentences", "audio", "all"] = "all",
    per_word: int = 2,
    max_cards: int = 450,
    max_unknown: int = 2,
    min_words: int | None = None,
    max_words: int | None = None,
    native_audio: str = "none",
    voice: str = "",
    slow_audio: bool = False,
    pack_license: str = "public",
    tts: str = "",
) -> None:
    """Build one Swadesh sentence pack from the Tatoeba exports (S0 to S5).

    The command runs the stages in order and stops after the ``--stage``
    value. The default value runs every stage and writes the APKG.

    :param language: Language name, for example mandarin.
    :param refresh: Ignore the Tatoeba cache and download every file again.
    :param stage: Last stage to run: fetch, sentences, audio, or all.
    :param per_word: Card limit per word form.
    :param max_cards: Hard card cap.
    :param max_unknown: Unknown token budget per sentence.
    :param min_words: Minimum word count of a card. The profile default
        applies when empty.
    :param max_words: Maximum word count of a card. The profile default
        applies when empty.
    :param native_audio: Tatoeba tier: none, auto, or only.
    :param voice: TTS voice name. The profile default applies when empty.
    :param slow_audio: Generate the slow variant.
    :param pack_license: public or private.
    :param tts: TTS engine, gtts or edge. The installed engine applies when
        empty.
    """
    entry = LANGUAGES.get(resolve(language))
    if entry is None or entry.module is None:
        print(
            f"Unsupported language: {language}. "
            f"Supported: {SENTENCE_PACK_LANGUAGE_NAMES}."
        )
        return
    _fetch_language(entry.profile, refresh=refresh)
    if stage == "fetch":
        return
    _language_sentences(
        entry.module,
        entry.profile,
        per_word=per_word,
        max_cards=max_cards,
        max_unknown=max_unknown,
        min_words=min_words,
        max_words=max_words,
    )
    if stage == "sentences":
        return
    _language_audio(
        entry.module,
        entry.profile,
        native_audio=native_audio,
        voice=voice,
        slow_audio=slow_audio,
        pack_license=pack_license,
        tts=tts,
    )
    if stage == "audio":
        return
    _language_pack(entry.module, entry.profile, pack_license=pack_license)


@swadesh_app.command
def audit(
    language: Annotated[str, Parameter(help=f"One of {SENTENCE_PACK_LANGUAGE_NAMES}.")],
) -> None:
    """Run the quality gates of a Swadesh sentence pack.

    :param language: Language name, for example mandarin.
    """
    lang = resolve(language)
    entry = LANGUAGES.get(lang)
    if entry is None or entry.audit is None:
        print(
            f"Unsupported language: {language}. "
            f"Supported: {SENTENCE_PACK_LANGUAGE_NAMES}."
        )
        return
    code = print_gate_table(entry.audit())
    if code:
        raise SystemExit(code)


def _language_sentences(
    module,
    profile,
    per_word: int,
    max_cards: int,
    max_unknown: int,
    min_words: int | None,
    max_words: int | None,
) -> None:
    """Build the candidate pool and the card CSV files of one sentence pack.

    :param module: Language module of the pack.
    :param profile: Language profile of the pack.
    :param per_word: Card limit per word form.
    :param max_cards: Hard card cap.
    :param max_unknown: Unknown token budget per sentence.
    :param min_words: Minimum word count of a core card.
    :param max_words: Maximum word count of a card.
    """
    table = module.load_word_table()
    failures = module.check_word_table(table)
    if failures:
        for failure in failures:
            print(f"Error: {failure}")
        raise SystemExit(1)
    known = module.known_words(table)
    pool, pool_report = sentence_pool.build_candidate_pool(
        table,
        profile,
        min_units=profile.min_units if min_words is None else min_words,
        max_units=profile.max_units if max_words is None else max_words,
        max_unknown=max_unknown,
        fallback=True,
        known_words=known,
        make_candidate=module.make_candidate,
        frames_path=sentence_pool.default_frames_path(profile),
    )
    cards, selection = sentence_select.select_sentences(
        pool,
        table,
        profile,
        per_word=per_word,
        max_cards=max_cards,
        finalize=profile.finalize_candidate,
    )
    report = sentence_pool.build_pack_report(pool_report, selection)
    print(f"wrote {module.write_sentence_csv(cards)}")
    print(f"wrote {module.write_sentence_review_csv(pool, table, per_word=per_word)}")
    print(f"wrote {sentence_pool.write_pool_report(report, module.POOL_REPORT_PATH)}")
    sentence_pool.print_sentence_report(report)


def _language_audio(
    module,
    profile,
    native_audio: str,
    voice: str,
    slow_audio: bool,
    pack_license: str,
    tts: str,
) -> None:
    """Generate and validate the audio files of one sentence pack.

    :param module: Language module of the pack.
    :param profile: Language profile of the pack.
    :param native_audio: Tatoeba tier: none, auto, or only.
    :param voice: TTS voice name.
    :param slow_audio: Generate the slow variant.
    :param pack_license: public or private.
    :param tts: TTS engine, gtts or edge.
    """
    if tts and tts not in ("gtts", "edge"):
        print(f"Unsupported tts engine: {tts}. Use gtts or edge.")
        return
    rows = pack_audio.load_sentence_rows(module.SENTENCES_PATH)
    engine = "gtts"
    if tts:
        engine = tts
        if tts == "edge" and importlib.util.find_spec("edge_tts") is None:
            print("edge-tts is not installed. Use --tts gtts.")
            return
    try:
        result = pack_audio.generate_audio(
            rows,
            profile,
            voice=voice or profile.voice,
            native_audio=native_audio,
            slow_audio=slow_audio,
            pack_license=pack_license,
            engine=engine,
            csv_path=module.SENTENCES_PATH,
        )
    except ValueError as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error
    print(
        f"engine {result['engine']} voice {result['voice']}: "
        f"{result['tts']} tts, {result['native']} native, "
        f"{result['words']} word, {result['skipped']} cached, "
        f"{result['failed']} failed, {result['duration_s']:.1f} s"
    )
    for failure in result["failures"]:
        print(f"failed: {failure['filename']} {failure['error']}")
    print(f"wrote {result['index_path']}")
    if result["csv_path"]:
        print(f"wrote {result['csv_path']}")


def _language_pack(module, profile, pack_license: str) -> None:
    """Build the APKG file, the manifest, and the attribution of one pack.

    :param module: Language module of the pack.
    :param profile: Language profile of the pack.
    :param pack_license: public or private.
    """
    result = sentence_pack.build_pack(
        profile,
        module.load_word_table,
        pack_license=pack_license,
    )
    counts = result["manifest"]["counts"]
    print(
        f"{counts['cards']} cards, {counts['scaffold']} scaffold, "
        f"{counts['tts_audio']} tts, {counts['native_audio']} native, "
        f"{result['media_count']} media files, {result['skipped']} skipped"
    )
    print(f"wrote {result['output']}")
    print(f"wrote {result['manifest_path']}")


# =============================================================================
# Preview - build the browser preview of the card templates
# =============================================================================


@app.command
def preview(
    output: Path = Path("preview/cards.html"),
    *,
    apkg: Path | None = None,
    limit: int = 50,
    deck: str | None = None,
    search: str | None = None,
    media: bool = True,
    media_dir: Path | None = None,
    open_browser: bool = False,
):
    """Build a browser preview of the card templates or of a package.

    Without ``--apkg`` the command reads the repository templates. With
    ``--apkg`` the command reads the note fields, the card templates, and the
    media files of that package.

    :param output: HTML file to write (default: ``preview/cards.html``).
    :param apkg: Package to preview. Defaults to the repository templates.
    :param limit: Maximum card count of the package preview. Zero shows all.
    :param deck: Only show package cards of a deck with this text.
    :param search: Only show package cards with a field that holds this text.
    :param media: Extract media and link it in the page.
    :param media_dir: Directory for the extracted media files.
    :param open_browser: Open the written file in the default browser.
    """
    if apkg is None:
        cards, sections = write_preview(output)
        print(f"wrote {output} with {cards} cards in {sections} sections")
    else:
        result = write_apkg_preview(
            apkg,
            output,
            limit=limit,
            deck=deck,
            search=search,
            extract_media=media,
            media_dir=media_dir,
        )
        print(
            f"wrote {result.output_path} with {result.cards} of "
            f"{result.total} cards in {result.sections} sections and "
            f"{result.media} media files"
        )
    if open_browser:
        webbrowser.open(Path(output).resolve().as_uri())


def main() -> None:
    """Main entry point. Invokes the cyclopts app."""
    app()


if __name__ == "__main__":
    main()
