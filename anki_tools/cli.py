"""
CLI tools for Anki package (.apkg) files.

Commands:
    inspect - Diagnostic tools to inspect .apkg files
    swadesh - Build Swadesh vocabulary APKG files with audio
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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

import cyclopts
from gtts import gTTS

from anki_tools.languages.cantonese import core as cantonese_core
from anki_tools.languages.cantonese import select_pipeline as cantonese_select
from anki_tools.languages.french import core as french_core
from anki_tools.languages.french import cull as french_cull
from anki_tools.languages.spanish import core as spanish_core
from anki_tools.languages.mandarin import core as mandarin_core
from anki_tools.languages.mandarin import frequency as mandarin_frequency
from anki_tools.languages.cantonese.jyutping import fix_jyutping
from anki_tools.languages.mandarin.pinyin import fix_pinyin
from anki_tools.languages.mandarin.stages import fetch_tatoeba
from anki_tools.apkg.package import AnkiPackage
from anki_tools.preview import write_preview
from anki_tools.preview_apkg import write_apkg_preview
from anki_tools.sentence_pack import audio as pack_audio
from anki_tools.sentence_pack import pack as sentence_pack
from anki_tools.sentence_pack import pool as sentence_pool
from anki_tools.sentence_pack import select as sentence_select
from anki_tools.sentence_pack import tatoeba as sentence_tatoeba
from anki_tools.sentence_pack.audit import print_audit as print_gate_table
from anki_tools.languages import (
    LANGUAGES,
    get_profile,
    profile_languages,
    resolve,
    swadesh_languages,
)
from anki_tools.languages.cantonese.rank import rank_sentences_yue
from anki_tools.languages.french.rank import rank_sentences_fr
from anki_tools.languages.mandarin.rank import rank_sentences_zh
from anki_tools.languages.spanish.rank import rank_sentences_es
from anki_tools.ranking.latin import latin_word_similarity
from anki_tools.ranking.rank import (
    Sentence,
    reduce_sentences_stratified,
)


from anki_tools.ranking.pipeline import (
    RankPaths,
    RankedDeck,
    rank_paths,
    run_ranked_deck,
)
from anki_tools.apkg.templates import (
    DEFAULT_TEMPLATE_DIR,
    SHARED_CSS_PATH,
    TEMPLATES_DIR,
    load_char_links,
    load_css_layers,
    load_text,
)
from anki_tools.languages.cantonese.frequency import (
    CIFU_ATTRIBUTION,
    YUE_FREQUENCY_ATTRIBUTION,
    FREQUENCY_PATH,
    check_frequency,
    load_yue_frequency,
    rank_change_rows,
    write_freq_change_csv,
    write_frequency_tsv,
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

def _extract_template_fields(template: str) -> set[str]:
    """Extract field names from Anki template ({{FieldName}} syntax).

    :param template: Template string (qfmt or afmt).
    :returns: Set of field names referenced.
    """
    matches = re.findall(r"\{\{([^#/}]+)\}\}", template)
    builtins = {"FrontSide", "Tags", "Type", "Deck", "Subdeck", "Card"}
    return {m.strip() for m in matches if m.strip() not in builtins}


def _find_best_match(field: str, deck_fields: set[str]) -> str | None:
    """Find the best matching deck field for a template field.

    :param field: Template field name.
    :param deck_fields: Set of deck field names.
    :returns: Best matching deck field name or None.
    """
    field_lower = field.lower()

    for deck_field in deck_fields:
        if deck_field.lower() == field_lower:
            return deck_field

    field_words = set(re.split(r"[^a-zA-Z]+", field_lower))
    best_match = None
    best_score = 0

    for deck_field in deck_fields:
        deck_words = set(re.split(r"[^a-zA-Z]+", deck_field.lower()))
        common = field_words & deck_words
        if common and len(common) > best_score:
            best_score = len(common)
            best_match = deck_field

    return best_match


def _apply_field_mapping(template: str, mapping: dict[str, str]) -> str:
    """Replace field references in template using mapping.

    :param template: Template string with {{FieldName}} refs.
    :param mapping: Dict mapping old field names to new.
    :returns: Template with refs replaced.
    """
    result = template
    for old_name, new_name in mapping.items():
        result = result.replace(f"{{{{{old_name}}}}}", f"{{{{{new_name}}}}}")
    return result


CARD_CSS_PATH = DEFAULT_TEMPLATE_DIR / "card.css"
FRONT_TEMPLATE_PATH = DEFAULT_TEMPLATE_DIR / "front.html"
BACK_TEMPLATE_PATH = DEFAULT_TEMPLATE_DIR / "back.html"


def _apply_style(
    target_apkg: Path,
    *,
    css_path: Path = CARD_CSS_PATH,
    front_path: Path = FRONT_TEMPLATE_PATH,
    back_path: Path = BACK_TEMPLATE_PATH,
    verbose: bool = True,
) -> tuple[int, list[str]]:
    """Load CSS/templates, map fields to deck, update target package. Returns (models_updated, applied_names)."""
    with AnkiPackage(str(target_apkg)) as pkg:
        models = pkg.get_models(include_templates=True)
        all_fields = set()
        for model in models.values():
            all_fields.update(f["name"] for f in model.get("flds", []))
        if verbose:
            print("Deck fields:")
            for _mid, model in models.items():
                print(f"  {', '.join(f['name'] for f in model['flds'])}")

    css_content = load_css_layers(SHARED_CSS_PATH, css_path)
    if css_content and verbose:
        print(f"\nLoaded: {SHARED_CSS_PATH} and {css_path}")
    elif verbose:
        print(f"\nNo CSS found in {SHARED_CSS_PATH} or {css_path}, skipping CSS")

    qfmt = None
    if front_path.exists():
        qfmt = front_path.read_text(encoding="utf-8")
        if verbose:
            print(f"Loaded: {front_path}")
        missing = _extract_template_fields(qfmt) - all_fields
        if missing:
            mapping = {f: _find_best_match(f, all_fields) for f in missing}
            mapping = {k: v for k, v in mapping.items() if v}
            if mapping and verbose:
                print(f"  Auto-mapping: {mapping}")
            qfmt = _apply_field_mapping(qfmt, mapping)
    elif verbose:
        print(f"No {front_path} found, skipping front template")

    afmt = None
    if back_path.exists():
        afmt = back_path.read_text(encoding="utf-8")
        if verbose:
            print(f"Loaded: {back_path}")
        missing = _extract_template_fields(afmt) - all_fields
        if missing:
            mapping = {f: _find_best_match(f, all_fields) for f in missing}
            mapping = {k: v for k, v in mapping.items() if v}
            if mapping and verbose:
                print(f"  Auto-mapping: {mapping}")
            afmt = _apply_field_mapping(afmt, mapping)
        script = load_char_links()
        if script and "char-links" in afmt:
            afmt = f"{afmt.rstrip()}\n{script}"
    elif verbose:
        print(f"No {back_path} found, skipping back template")

    if not css_content and not qfmt and not afmt:
        return (0, [])

    with AnkiPackage(str(target_apkg)) as pkg:
        models = pkg.get_models()
        for model_id in models:
            pkg.update_model_template(
                model_id=model_id, css=css_content, qfmt=qfmt, afmt=afmt
            )
        pkg.save(target_apkg)

    applied = []
    if css_content:
        applied.append("CSS")
    if qfmt:
        applied.append("front")
    if afmt:
        applied.append("back")
    if verbose:
        print(f"Updated {len(models)} model(s)")
    return (len(models), applied)


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

    _n, applied = _apply_style(
        output,
        css_path=css or CARD_CSS_PATH,
        front_path=front or FRONT_TEMPLATE_PATH,
        back_path=back or BACK_TEMPLATE_PATH,
        verbose=True,
    )
    if not applied:
        print(
            f"\nNothing to apply. Check {CARD_CSS_PATH}, "
            f"{FRONT_TEMPLATE_PATH}, and {BACK_TEMPLATE_PATH} first."
        )
        return
    print(f"\nSaved: {output}")


# =============================================================================
# Freq commands - build word frequency lists
# =============================================================================

freq_app = cyclopts.App(name="freq", help="Build word frequency lists")
app.command(freq_app)


@freq_app.command(name="yue")
def freq_yue(
    *,
    output: Path = FREQUENCY_PATH,
    spoken_weight: float = 0.7,
    refresh: bool = False,
    check: bool = False,
):
    """Build the Cantonese word frequency list from spoken and written counts.

    Blends the Cifu spoken counts with the Cifu and Apple Daily written counts,
    folds variant characters, and applies the override file. Downloads the two
    source files into ~/.cache/anki-tools when they are absent.

    :param output: Output TSV file, sorted by rank.
    :param spoken_weight: Weight of the spoken term in the blend (0 to 1).
    :param refresh: Download the source files again.
    :param check: Run gates G1, G3, and G4 and exit with code 1 on a failure.
    """
    data = load_yue_frequency(spoken_weight=spoken_weight, refresh=refresh)
    write_frequency_tsv(data, output)
    print(f"Wrote {len(data.rows)} words to {output}")
    print(f"  {CIFU_ATTRIBUTION}")
    print(f"  {YUE_FREQUENCY_ATTRIBUTION}")
    if not check:
        return
    failed = False
    for gate, ok, message in check_frequency(data):
        print(f"{gate} {'PASS' if ok else 'FAIL'}: {message}")
        failed = failed or not ok
    if failed:
        raise SystemExit(1)

@freq_app.command(name="zh")
def freq_zh(
    *,
    output: Path = mandarin_frequency.FREQUENCY_PATH,
    rows: int = mandarin_frequency.MAX_ROWS,
    fallback: bool = False,
    refresh: bool = False,
    check: bool = False,
):
    """Build the Mandarin word frequency list from the wordfreq zh list.

    Writes the most frequent forms with their rank, source, and license. The
    wordfreq list needs no download. ``--fallback`` reads the FrequencyWords
    ``zh_cn`` and ``zh_tw`` lists instead, and converts ``zh_tw`` to Simplified.

    :param output: Output TSV file, sorted by rank.
    :param rows: Number of most frequent forms to keep.
    :param fallback: Read the FrequencyWords files instead of wordfreq.
    :param refresh: Download the fallback files again.
    :param check: Run gates Z1 to Z3 and exit with code 1 on a failure.
    """
    if fallback:
        paths = mandarin_frequency.download_zh_sources(force=refresh)
        data = mandarin_frequency.build_frequency_zh_fallback(paths, rows=rows)
    else:
        data = mandarin_frequency.build_frequency_zh(rows=rows)
    mandarin_frequency.write_frequency_tsv(data, output)
    print(f"Wrote {len(data)} words to {output}")
    print(f"  {mandarin_frequency.SOURCE_NAME} {mandarin_frequency.LICENSE}")
    if not check:
        return
    failed = False
    for gate, ok, message in mandarin_frequency.check_frequency(data):
        print(f"{gate} {'PASS' if ok else 'FAIL'}: {message}")
        failed = failed or not ok
    if failed:
        raise SystemExit(1)



# =============================================================================
# Fix commands - optimize decks by language
# =============================================================================

fix_app = cyclopts.App(name="fix", help="Fix and optimize Anki decks")
app.command(fix_app)


def _run_core_command(
    module: Any,
    label: str,
    apkg_path: Path,
    output: Path | None,
    model_id: int | None,
    cards: int | None,
    words: int,
    examples: int,
    tail: int,
    check: bool,
    spoken_weight: float | None = None,
) -> None:
    """Build the core deck for one language.

    :param module: Language core module that holds the core entry points.
    :param label: Language name that the banner shows.
    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the input name).
    :param model_id: Rank mode filter, rejected in core mode.
    :param cards: Core mode: card cap.
    :param words: Core mode: tier F words after the Swadesh set.
    :param examples: Core mode: examples per content target.
    :param tail: Core mode: cards after the core passes.
    :param check: Run the gates on the output only.
    :param spoken_weight: Optional weight of the spoken frequency term.
    """
    if model_id is not None:
        print("Error: --model-id does not apply with --core.")
        return
    output_path = output or module.default_output(apkg_path)
    if check:
        code = module.check_core(output_path)
        if code:
            raise SystemExit(code)
        return
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return
    print("=" * 50)
    print(f"Processing {label} Anki Deck: core mode")
    print("=" * 50)
    options = replace(
        module.DEFAULT_OPTIONS,
        per_word=examples,
        tail_cards=tail,
    )
    if cards is not None:
        options = replace(options, max_cards=cards)
    request = module.CoreRequest(
        input_path=apkg_path,
        output_path=output_path,
        core_words=words,
        options=options,
    )
    if spoken_weight is not None:
        request = replace(request, spoken_weight=spoken_weight)
    module.run_core(request)


def _check_guard(check: bool, modes: Sequence[tuple[str, bool]]) -> bool:
    """Decide if a fix run may continue with the ``--check`` flag.

    :param check: Value of the ``--check`` flag.
    :param modes: Mode flag names and their enabled state, in message order.
    :returns: True when the run may continue, False after the error print.
    """
    if not check or any(enabled for _, enabled in modes):
        return True
    print(f"Error: --check needs {' or '.join(name for name, _ in modes)}.")
    return False


@fix_app.command
def zh(
    apkg_path: Path,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    no_pinyin_fix: bool = False,
    no_style: bool = False,
    verbose: bool = False,
    core: bool = False,
    cards: int | None = None,
    words: int = mandarin_core.DEFAULT_CORE_WORDS,
    examples: int = mandarin_core.DEFAULT_OPTIONS.per_word,
    tail: int = mandarin_core.DEFAULT_OPTIONS.tail_cards,
    check: bool = False,
):
    """Fix Chinese deck: rank, reorder, fix pinyin, and apply styling.

    Ranks sentences by frequency and complexity, reorders the deck,
    fixes pinyin formatting, and applies card styling.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param keep_filtered: If True, keep filtered cards (names, invalid).
    :param no_pinyin_fix: If True, skip pinyin correction.
    :param no_style: If True, skip card styling (CSS/templates).
    :param verbose: If True, show detailed output.
    :param core: Build the core deck instead of ranking and reordering.
    :param cards: Core mode: card cap.
    :param words: Core mode: tier F words after the Swadesh set.
    :param examples: Core mode: examples per content target.
    :param tail: Core mode: cards after the core passes.
    :param check: Core mode: run the gates on the output only.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if not _check_guard(check, (("--core", core),)):
        return

    if core:
        _run_core_command(
            mandarin_core,
            "Chinese",
            apkg_path,
            output,
            model_id,
            cards,
            words,
            examples,
            tail,
            check,
        )
        return

    paths = rank_paths(apkg_path, output)

    def after_rank(paths: RankPaths, deck: RankedDeck) -> Sequence[str]:
        if not no_pinyin_fix:
            print("\nFixing pinyin...")
            fix_pinyin(
                paths.output_path,
                output=paths.output_path,
                csv_output=Path("/dev/null"),
                verbose=verbose,
            )
        styles_applied = 0
        if not no_style:
            print("\nApplying card styles...")
            styles_applied, applied = _apply_style(
                Path(paths.output_path),
                css_path=CARD_CSS_PATH,
                front_path=FRONT_TEMPLATE_PATH,
                back_path=BACK_TEMPLATE_PATH,
                verbose=False,
            )
            if applied:
                print(f"  Applied: {', '.join(applied)} to {styles_applied} model(s)")
        return (f"  Styles applied:      {styles_applied} model(s)",)

    _run_rank_mode(
        paths,
        "Chinese",
        rank_sentences_zh,
        {
            "model_id": model_id,
            "romanization_candidates": ("Pinyin",),
            "keep_filtered": keep_filtered,
        },
        after=after_rank,
    )


DEFAULT_FRENCH_APKG = Path("apkg/6000_French_Sentences_w_Audio.apkg")


@fix_app.command
def fr(
    apkg_path: Path = DEFAULT_FRENCH_APKG,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    text_field: str = "French",
    keep_filtered: bool = False,
    select: bool = False,
    core: bool = False,
    cards: int | None = None,
    targets: int = french_cull.DEFAULT_TARGET_WORDS,
    max_uses: int = french_cull.DEFAULT_MAX_USES,
    reserve: bool = True,
    words: int = french_core.DEFAULT_CORE_WORDS,
    examples: int = french_core.DEFAULT_OPTIONS.per_word,
    tail: int = french_core.DEFAULT_OPTIONS.tail_cards,
    check: bool = False,
):
    """Fix French deck: rank and reorder it, or select a smaller deck.

    Rank mode is the default. It ranks the sentences and writes
    ``<input>_reordered.apkg`` and ``.rank.csv``. Select mode, with
    ``--select``, runs the 1000-card cull and writes ``<input>_culled.apkg``
    with its CSV files and manifest.

    ``--model-id``, ``--text-field``, and ``--keep-filtered`` apply to rank
    mode only. ``--select`` adds the cull options, and ``--core`` adds the
    core options. The tuning values live in ``CullRequest`` and
    ``CoreOptions``, and the specs record them.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the input name).
    :param model_id: Rank mode: filter by specific model ID.
    :param text_field: Rank mode: preferred sentence field (default: French).
    :param keep_filtered: Rank mode: keep filtered cards instead of removing.
    :param select: Run the card cull instead of the rank and reorder.
    :param core: Run the core-word selection instead of the rank and reorder.
    :param cards: Card cap of the chosen mode (select 1000, core 300).
    :param targets: Select mode: target count, 200 to 500.
    :param max_uses: Select mode: most cards that hold one non-exempt word.
    :param reserve: Select mode: add the Swadesh reserve cards.
    :param words: Core mode: tier F words after the Swadesh set.
    :param examples: Core mode: examples per content target, 1 to 3.
    :param tail: Core mode: cards after the core passes.
    :param check: Run the gates on the output and do not run the mode again.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if not _check_guard(check, (("--select", select), ("--core", core))):
        return

    if core and select:
        print("Error: --core and --select cannot be used together.")
        return


    if core:
        _run_core_command(
            french_core,
            "French",
            apkg_path,
            output,
            model_id,
            cards,
            words,
            examples,
            tail,
            check,
        )
        return
    if select:
        if model_id is not None:
            print("Error: --model-id does not apply with --select.")
            return
        output_path = output or french_cull.default_output(apkg_path)
        if check:
            code = french_cull.check_cull(output_path)
            if code:
                raise SystemExit(code)
            return
        request = french_cull.CullRequest(
            input_path=apkg_path,
            output_path=output_path,
            target_words=targets,
            max_uses=max_uses,
            no_reserve=not reserve,
        )
        if cards is not None:
            request = replace(request, max_cards=cards)
        french_cull.run_cull(request)
        return

    paths = rank_paths(apkg_path, output)

    _run_rank_mode(
        paths,
        "French",
        rank_sentences_fr,
        {
            "model_id": model_id,
            "text_field": text_field,
            "text_candidates": ("French",),
            "keep_filtered": keep_filtered,
        },
    )


def _print_rank_banner(paths: RankPaths, label: str, target: str = "") -> None:
    """Print the banner that starts a fix rank run.

    :param paths: Input, output, and ranking file of the run.
    :param label: Deck language, for example ``Chinese``.
    :param target: Preformatted extra line, or an empty string.
    """
    print("=" * 50)
    print(f"Processing {label} Anki Deck")
    print("=" * 50)
    print(f"Input:   {paths.input_path}")
    print(f"Output:  {paths.output_path}")
    print(f"Ranking: {paths.ranking_path}")
    if target:
        print(target)
    print()


def _print_rank_done(
    paths: RankPaths,
    ranked_count: int,
    stats: dict[str, Any],
    select_csv: Path | None = None,
    coverage_csv: Path | None = None,
    *,
    extra_lines: Sequence[str] = (),
) -> None:
    """Print the DONE table of a rank or a select run.

    :param paths: Input, output, and ranking file of the run.
    :param ranked_count: Sentences that the ranker kept.
    :param stats: Card counts of the reorder step.
    :param select_csv: Select CSV of a select run, or None in rank mode.
    :param coverage_csv: Coverage CSV of a select run, or None in rank mode.
    :param extra_lines: Preformatted lines to print after the card counts.
    """
    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {ranked_count}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    for line in extra_lines:
        print(line)
    print(f"  Output:              {paths.output_path}")
    print(f"  Ranking CSV:         {paths.ranking_path}")
    if select_csv is not None:
        print(f"  Select CSV:          {select_csv}")
    if coverage_csv is not None:
        print(f"  Coverage CSV:        {coverage_csv}")


def _run_rank_mode(
    paths: RankPaths,
    label: str,
    rank: Callable[[list[Sentence]], list[Sentence]],
    deck_kwargs: dict[str, Any],
    *,
    target: str = "",
    count: Callable[[int], int] | None = None,
    after: Callable[[RankPaths, RankedDeck], Sequence[str]] | None = None,
) -> None:
    """Rank, reorder, and report one deck, with the language parts as hooks.

    :param paths: Input, output, and ranking file of the run.
    :param label: Deck language, for example ``Chinese``.
    :param rank: Ranker of the language, sentences to ranked sentences.
    :param deck_kwargs: Extra keyword arguments for ``run_ranked_deck``.
    :param target: Preformatted extra banner line, or an empty string.
    :param count: Ranked count hook, or None to use the ranked length.
    :param after: Hook that prints the language steps and returns the extra
        DONE lines, or None for no extra lines.
    """
    _print_rank_banner(paths, label, target=target)
    print("Step 1: Ranking sentences...")
    deck = run_ranked_deck(paths, rank=rank, **deck_kwargs)
    if not deck.sentences:
        print("No sentences found!")
        return
    ranked_count = count(len(deck.ranked)) if count is not None else len(deck.ranked)
    extra_lines = after(paths, deck) if after is not None else ()
    _print_rank_done(paths, ranked_count, deck.stats, extra_lines=extra_lines)


DEFAULT_CANTONESE_APKG = Path("apkg/LTL Cantonese Deck Level 1 - Short.apkg")


@fix_app.command
def yue(
    apkg_path: Path = DEFAULT_CANTONESE_APKG,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    no_select: bool = False,
    max_uses: int = cantonese_select.DEFAULT_MAX_USES,
    freq: Literal["blend", "spoken", "written"] = cantonese_select.DEFAULT_FREQ,
    freq_report: bool = False,
    core: bool = False,
    cards: int | None = None,
    words: int = cantonese_core.DEFAULT_CORE_WORDS,
    examples: int = cantonese_core.DEFAULT_OPTIONS.per_word,
    tail: int = cantonese_core.DEFAULT_OPTIONS.tail_cards,
    check: bool = False,
):
    """Fix Cantonese deck: rank and reorder it, or select a smaller deck.

    Rank mode, with ``--no-select``, ranks the sentences and writes
    ``<input>_reordered.apkg`` and ``.rank.csv``. Select mode is the default.
    It ranks the sentences, then keeps the fewest sentences that show the most
    frequent spoken words, easiest first, and writes ``<input>_reordered.apkg``
    with its rank, select, and coverage CSV files.

    ``--keep-filtered`` applies to rank mode only. ``--core`` builds the core
    deck instead. The select tuning values live in ``SelectRequest`` and the
    core tuning values in ``CoreOptions``, and the specs record them.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the input name).
    :param model_id: Filter by specific model ID.
    :param keep_filtered: Rank mode: keep filtered cards instead of removing.
    :param no_select: Only rank and reorder, as before.
    :param max_uses: Select mode: most cards that hold one non-exempt word.
    :param freq: Frequency term: blend, spoken, or written.
    :param freq_report: Write the sentences that the frequency change moves.
    :param core: Build the core deck instead of ranking and selecting.
    :param cards: Card cap of the chosen mode (select: stop at zero gain).
    :param words: Core mode: tier F words after the Swadesh set.
    :param examples: Core mode: examples per content target.
    :param tail: Core mode: cards after the core passes.
    :param check: Run the gates on the output and do not run the mode again.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if not _check_guard(check, (("--core", core),)):
        return

    if core:
        _run_core_command(
            cantonese_core,
            "Cantonese",
            apkg_path,
            output,
            model_id,
            cards,
            words,
            examples,
            tail,
            check,
            spoken_weight=cantonese_core.SPOKEN_WEIGHT,
        )
        return

    if not no_select:
        output_path = output or cantonese_select.default_output(apkg_path)
        run = cantonese_select.run_select(
            cantonese_select.SelectRequest(
                input_path=apkg_path,
                output_path=output_path,
                model_id=model_id,
                max_sentences=cards,
                max_uses=max_uses,
                freq=freq,
                freq_report=freq_report,
            )
        )
        if run.report is None:
            print("No sentences found!")
            return
        select_csv, coverage_csv = cantonese_select.side_paths(run.paths.output_path)
        print("\nFixing jyutping...")
        fix_jyutping(
            run.paths.output_path,
            output=run.paths.output_path,
            verbose=False,
        )
        _print_rank_done(
            run.paths,
            run.ranked_count,
            run.stats,
            select_csv=select_csv,
            coverage_csv=coverage_csv,
        )
        return

    paths = rank_paths(apkg_path, output)

    extracted: list[Sentence] = []
    counts: dict[str, int] = {}

    def rank_and_capture(sentences: list[Sentence]) -> list[Sentence]:
        extracted.extend(sentences)
        ranked = rank_sentences_yue(sentences, freq=freq)
        counts["ranked"] = len(ranked)
        return ranked

    def reorder(ranked: list[Sentence]) -> list[Sentence]:
        if freq_report and freq != "written":
            comparison = rank_sentences_yue(
                [
                    Sentence(
                        note_id=sentence.note_id,
                        text=sentence.text,
                        original_order=sentence.original_order,
                    )
                    for sentence in extracted
                ],
                freq="written",
            )
            rows = rank_change_rows(comparison, ranked)
            change_csv = (
                paths.input_path.parent / f"{paths.input_path.stem}.freq_change.csv"
            )
            write_freq_change_csv(rows, change_csv)
            print(f"  {len(rows)} sentences move by more than 100 places: {change_csv}")
        print("\nStep 2: Reordering deck...")
        return ranked

    def after_rank(paths: RankPaths, deck: RankedDeck) -> Sequence[str]:
        print("\nFixing jyutping...")
        fix_jyutping(
            paths.output_path,
            output=paths.output_path,
            verbose=False,
        )
        return ()

    _run_rank_mode(
        paths,
        "Cantonese",
        rank_and_capture,
        {
            "model_id": model_id,
            "romanization_candidates": ("Jyutping",),
            "keep_filtered": keep_filtered,
            "reduce": reorder,
            "remove_unranked": False,
        },
        count=lambda ranked_total: counts.get("ranked", ranked_total),
        after=after_rank,
    )


DEFAULT_SPANISH_APKG = Path("apkg/Spanish 9000 sentences with native audio.apkg")

@fix_app.command
def es(
    apkg_path: Path = DEFAULT_SPANISH_APKG,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    text_field: str = "Spanish",
    keep_filtered: bool = False,
    max_sentences: int | None = None,
    similarity_threshold: float = 0.5,
    core: bool = False,
    cards: int | None = None,
    words: int = spanish_core.DEFAULT_CORE_WORDS,
    examples: int = spanish_core.DEFAULT_OPTIONS.per_word,
    tail: int = spanish_core.DEFAULT_OPTIONS.tail_cards,
    check: bool = False,
):
    """Fix Spanish deck: rank and reorder it, or build a core deck.

    Rank mode is the default. ``--core`` builds the core deck of the Swadesh
    rows and the tier F words, and writes ``<input>_core.apkg`` with its CSV
    files and manifest.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: from the input name).
    :param model_id: Rank mode: filter by specific model ID.
    :param text_field: Rank mode: preferred sentence field (default: Spanish).
    :param keep_filtered: Rank mode: keep filtered cards instead of removing.
    :param max_sentences: Rank mode: optional max sentence count (e.g., 3000).
    :param similarity_threshold: Jaccard threshold for deduplication (default 0.5).
    :param core: Build the core deck instead of ranking.
    :param cards: Core mode: card cap.
    :param words: Core mode: tier F words after the Swadesh set (default 50).
    :param examples: Core mode: examples per content target.
    :param tail: Core mode: cards after the core passes.
    :param check: Run the gates on the output and do not run the mode again.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if not _check_guard(check, (("--core", core),)):
        return

    if core:
        _run_core_command(
            spanish_core,
            "Spanish",
            apkg_path,
            output,
            model_id,
            cards,
            words,
            examples,
            tail,
            check,
        )
        return

    paths = rank_paths(apkg_path, output)
    def after_rank(paths: RankPaths, deck: RankedDeck) -> Sequence[str]:
        print(f"Extracted {len(deck.sentences)} sentences")
        return ()

    def reduce_ranked(ranked: list[Sentence]) -> list[Sentence]:
        if not max_sentences or len(ranked) <= max_sentences:
            return ranked
        print(f"\nStep 2: Reducing to {max_sentences} sentences...")
        return reduce_sentences_stratified(
            ranked,
            target_count=max_sentences,
            similarity_threshold=similarity_threshold,
            similarity_fn=latin_word_similarity,
        )

    _run_rank_mode(
        paths,
        "Spanish",
        rank_sentences_es,
        {
            "model_id": model_id,
            "text_field": text_field,
            "keep_filtered": keep_filtered,
            "reduce": reduce_ranked,
            "remove_unranked": bool(max_sentences),
        },
        target=(
            f"Target:  {max_sentences} sentences (stratified reduction)"
            if max_sentences
            else ""
        ),
        after=after_rank,
    )


# =============================================================================
# Swadesh commands - build vocabulary APKG files with audio
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

    css_path = TEMPLATES_DIR / "vocab" / "card.css"
    language_css_path = TEMPLATES_DIR / "vocab" / config["language"].lower() / "card.css"
    css_content = load_css_layers(SHARED_CSS_PATH, css_path, language_css_path)

    csv_fields = (
        [k for k in entries[0].keys() if k != "audio_filename"] if entries else []
    )
    content_order = _content_field_order(csv_fields, config["audio_field_name"])
    fields = content_order + ["audio"]

    char_click_script = load_char_links()

    cjk_fields = {"characters", "hanzi", "cantonese", "mandarin", "chinese"}
    script_15em_fields = {"arabic", "hindi"}
    field_parts: list[str] = []
    for field in content_order:
        if field.lower() in cjk_fields:
            field_parts.append(
                f"<div class='char-links' style='font-size: 3rem'>{{{{{field}}}}}</div>"
            )
        elif field.lower() in script_15em_fields:
            field_parts.append(f"<div style='font-size: 1.5em'>{{{{{field}}}}}</div>")
        else:
            field_parts.append(f"<div style='font-size: 2rem'>{{{{{field}}}}}</div>")
    if char_click_script:
        field_parts.append(char_click_script)
    answer_format = back_template.replace("{{fields}}", "\n".join(field_parts))

    return _BuildLayout(
        fields=fields,
        content_order=content_order,
        front_template=front_template,
        answer_format=answer_format,
        css_content=css_content,
        css_path=css_path,
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


SWADESH_LANGUAGES = swadesh_languages()
SENTENCE_PACK_LANGUAGE_NAMES = ", ".join(
    language.name for language in profile_languages()
)

swadesh_app = cyclopts.App(
    name="swadesh", help="Build Swadesh vocabulary APKG files with audio"
)
app.command(swadesh_app)


@swadesh_app.command
def build(language: str, prune: bool = True):
    """Build Swadesh vocabulary APKG for a language.

    :param language: One of the supported languages, or all.
    :param prune: Delete audio files that the current list does not use.
    """
    lang = resolve(language)
    if lang == "all":
        for name in SWADESH_LANGUAGES:
            _build_apkg(name, prune=prune)
        return

    if lang not in SWADESH_LANGUAGES:
        print(
            f"Unknown language: {language}. Supported: {', '.join(SWADESH_LANGUAGES)}"
        )
        return
    _build_apkg(lang, prune=prune)


@swadesh_app.command
def fetch(*, refresh: bool = False, lang: list[str] | None = None) -> None:
    """Fetch the Tatoeba exports of a language pack (stage S0).

    With no ``--lang`` the command keeps the Mandarin behaviour and downloads the
    Mandarin and English exports. A language with no native Tatoeba audio skips
    the ``sentences_with_audio`` export.

    :param refresh: Ignore the cache and download every file again.
    :param lang: Tatoeba language codes, for example ``ell --lang eng``.
    """
    if not lang:
        results = fetch_tatoeba(refresh=refresh)
        for name, info in results.items():
            print(f"{name}.tsv: {info['rows']} rows ({info['source']})")
        return
    files: dict[str, str] = {}
    for code in lang:
        files.update(
            sentence_tatoeba.language_files(
                code,
                native_audio=code != (get_profile("greek").tatoeba_lang),
            )
        )
    if "eng" not in lang:
        files.update(sentence_tatoeba.language_files("eng", native_audio=False))
    results = sentence_tatoeba.fetch_tatoeba(files, refresh=refresh)
    for name, info in results.items():
        print(f"{name}.tsv: {info['rows']} rows ({info['source']})")

@swadesh_app.command
def sentences(
    language: str,
    per_word: int = 2,
    max_cards: int = 450,
    max_unknown: int = 2,
    min_words: int | None = None,
    max_words: int | None = None,
) -> None:
    """Build the sentence candidates and write the card CSV files (S1 to S3).

    :param language: Language name: mandarin, french, greek, or cantonese.
    :param per_word: Card limit per word form.
    :param max_cards: Hard card cap.
    :param max_unknown: Unknown token budget per sentence.
    :param min_words: Minimum word count for a sentence-pack core card. The
        profile default applies when empty.
    :param max_words: Maximum word count for a sentence-pack card. The profile
        default applies when empty.
    """
    lang = resolve(language)
    entry = LANGUAGES.get(lang)
    if entry is None or entry.module is None:
        print(
            f"Unsupported language: {language}. "
            f"Supported: {SENTENCE_PACK_LANGUAGE_NAMES}."
        )
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


@swadesh_app.command(name="audio")
def sentence_audio(
    language: str,
    *,
    native_audio: str = "none",
    voice: str = "",
    slow_audio: bool = False,
    pack_license: str = "public",
    tts: str = "",
) -> None:
    """Generate and validate the audio files of a sentence pack (stage S4).

    :param language: Language name: mandarin, french, greek, or cantonese.
    :param native_audio: Tatoeba tier: none, auto, or only.
    :param voice: TTS voice name. The profile default applies when empty.
    :param slow_audio: Generate the slow variant.
    :param pack_license: public or private.
    :param tts: TTS engine, gtts or edge. The installed engine applies when empty.
    """
    lang = resolve(language)
    entry = LANGUAGES.get(lang)
    if entry is None or entry.module is None:
        print(
            f"Unsupported language: {language}. "
            f"Supported: {SENTENCE_PACK_LANGUAGE_NAMES}."
        )
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


@swadesh_app.command
def pack(
    language: str,
    *,
    pack_license: str = "public",
) -> None:
    """Build one Swadesh sentence pack (stage S5).

    :param language: Language name: mandarin, french, greek, or cantonese.
    :param pack_license: public or private.
    """
    lang = resolve(language)
    entry = LANGUAGES.get(lang)
    if entry is None or entry.module is None:
        print(
            f"Unsupported language: {language}. "
            f"Supported: {SENTENCE_PACK_LANGUAGE_NAMES}."
        )
        return
    _language_pack(entry.module, entry.profile, pack_license=pack_license)


@swadesh_app.command
def audit(language: str) -> None:
    """Run the quality gates of a Swadesh sentence pack.

    :param language: Language name: mandarin, french, greek, or cantonese.
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
        return
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


@swadesh_app.command
def list_languages():
    """List supported Swadesh languages."""
    print("Supported languages:", ", ".join(SWADESH_LANGUAGES))


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
