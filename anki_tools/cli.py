"""
CLI tools for Anki package (.apkg) files.

Commands:
    inspect - Diagnostic tools to inspect .apkg files
    swadesh - Build Swadesh vocabulary APKG files with audio
    zh - Process Chinese Anki decks (rank, reorder, fix pinyin)
    fr - Analyze French Anki decks (e.g. 6000 French Sentences)
    yue - Analyze Cantonese Anki decks (e.g. LTL Cantonese)
    style - Apply CSS and card templates to any deck
"""

import csv
import os
import re
import tempfile
import time
from pathlib import Path
from textwrap import dedent

import cyclopts
from gtts import gTTS

from anki_tools.jyutping import fix_jyutping
from anki_tools.package import AnkiPackage
from anki_tools.pinyin import fix_pinyin
from anki_tools.rank import (
    SIMILARITY_CONSIDER_DELETE_PENALTY,
    extract_sentences,
    rank_sentences_fr,
    rank_sentences_yue,
    rank_sentences_zh,
    write_ranking_csv,
)
from anki_tools.reorder import load_ranking, reorder_deck

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

        for model_id, model_info in models.items():
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


def _apply_style(
    target_apkg: Path,
    *,
    css_path: Path = Path("templates/card.css"),
    front_path: Path = Path("templates/front.html"),
    back_path: Path = Path("templates/back.html"),
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
            for mid, model in models.items():
                print(f"  {', '.join(f['name'] for f in model['flds'])}")

    css_content = None
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")
        if verbose:
            print(f"\nLoaded: {css_path}")
    elif verbose:
        print(f"\nNo {css_path} found, skipping CSS")

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

    Looks for templates/card.css, templates/front.html, templates/back.html by default.
    Auto-maps mismatched field names to closest deck fields.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_styled.apkg).
    :param css: CSS file (default: templates/card.css).
    :param front: Front template file (default: templates/front.html).
    :param back: Back template file (default: templates/back.html).
    """
    apkg_path = Path(apkg_path)
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_styled.apkg"
    output = Path(output)

    if apkg_path.resolve() != output.resolve():
        import shutil
        shutil.copy(apkg_path, output)

    n, applied = _apply_style(
        output,
        css_path=css or Path("templates/card.css"),
        front_path=front or Path("templates/front.html"),
        back_path=back or Path("templates/back.html"),
        verbose=True,
    )
    if not applied:
        print("\nNothing to apply. Create templates/card.css, templates/front.html, or templates/back.html first.")
        return
    print(f"\nSaved: {output}")


# =============================================================================
# Process commands - Chinese deck processing pipeline
# =============================================================================

process_zh_app = cyclopts.App(name="zh", help="Process Chinese Anki decks")
app.command(process_zh_app)


@process_zh_app.command
def fix(
    apkg_path: Path,
    *,
    output: Path | None = None,
    verbose: bool = False,
):
    """Fix pinyin formatting in deck.

    Corrects: lowercase, syllable separation, tone marks, common errors.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_fixed.apkg).
    :param verbose: If True, show all corrections.
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_fixed.apkg"

    print(f"Fixing pinyin in: {apkg_path}")

    fix_pinyin(
        apkg_path,
        output=output,
        csv_output=None,
        verbose=verbose,
    )

    print(f"\nOutput: {output}")


@process_zh_app.command
def rank(
    apkg_path: Path,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
    no_pinyin_fix: bool = False,
    no_style: bool = False,
    verbose: bool = False,
):
    """Rank and reorder Chinese deck with pinyin fixes and styling.

    Main command for processing a Chinese Anki deck. Ranks sentences by frequency
    and complexity, reorders the deck, fixes pinyin, and applies styling.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param keep_filtered: If True, keep filtered cards (names, invalid).
    :param no_pinyin_fix: If True, skip pinyin correction.
    :param no_style: If True, skip card styling (CSS/templates).
    :param verbose: If True, show detailed output.
    """
    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"

    print("=" * 50)
    print("Processing Chinese Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Ranking: {ranking_csv}")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id)

    if not sentences:
        print("No sentences found!")
        return

    ranked = rank_sentences_zh(sentences)
    write_ranking_csv(ranked, str(ranking_csv))

    print("\nStep 2: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
    )

    if not no_pinyin_fix:
        print("\nStep 3: Fixing pinyin...")
        fix_pinyin(
            output,
            output=output,
            csv_output=Path("/dev/null"),
            verbose=verbose,
        )

    styles_applied = 0
    if not no_style:
        print("\nStep 4: Applying card styles...")
        styles_applied, applied = _apply_style(
            Path(output),
            css_path=Path("templates/card.css"),
            front_path=Path("templates/front.html"),
            back_path=Path("templates/back.html"),
            verbose=False,
        )
        if applied:
            print(f"  Applied: {', '.join(applied)} to {styles_applied} model(s)")

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Styles applied:      {styles_applied} model(s)")
    print(f"  Output:              {output}")
    print(f"  Ranking CSV:         {ranking_csv}")


# =============================================================================
# FR commands - analyze French decks
# =============================================================================

process_fr_app = cyclopts.App(
    name="fr", help="Analyze French Anki decks (e.g. 6000 French Sentences)"
)
app.command(process_fr_app)

DEFAULT_FRENCH_APKG = Path("apkg/6000_French_Sentences_w_Audio.apkg")


@process_fr_app.command
def rank(
    apkg_path: Path = DEFAULT_FRENCH_APKG,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    text_field: str = "French",
    keep_filtered: bool = False,
):
    """Rank and reorder French deck by complexity, frequency, and uniqueness.

    Uses word-based scoring for French. Ranks sentences and reorders the deck.
    Ranking uses only the sentence text; translation is not used.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param text_field: Preferred sentence field (default: French); inferred from deck if missing.
    :param keep_filtered: If True, keep filtered cards instead of removing.
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"

    print("=" * 50)
    print("Processing French Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Ranking: {ranking_csv}")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id, text_field=text_field)
    if not sentences:
        print("No sentences found!")
        return

    ranked = rank_sentences_fr(sentences)
    write_ranking_csv(ranked, str(ranking_csv))

    print("\nStep 2: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
    )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Output:              {output}")
    print(f"  Ranking CSV:         {ranking_csv}")


@process_fr_app.command(name="similar")
def similar_fr(
    apkg_path: Path = DEFAULT_FRENCH_APKG,
    *,
    ranking_csv: Path | None = None,
):
    """Show sentence pairs with high similarity (from .rank.csv).

    :param apkg_path: Input .apkg file (used to default ranking CSV path).
    :param ranking_csv: Ranking CSV (default: same stem as apkg with .rank.csv).
    """
    apkg_path = Path(apkg_path)
    if ranking_csv is None:
        ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"
    if not ranking_csv.exists():
        print(f"Error: File not found: {ranking_csv}")
        return

    threshold = SIMILARITY_CONSIDER_DELETE_PENALTY
    with open(ranking_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if not (r.get("similar_to") or "").strip():
                continue
            raw = r.get("similarity") or r.get("similarity_penalty") or "0"
            try:
                if float(raw) > threshold:
                    rows.append(r)
            except ValueError:
                pass

    if not rows:
        print(f"No pairs with similarity > {threshold} (consider deleting above this).")
        return

    print(f"Pairs worth reviewing for deletion (similarity > {threshold}, {len(rows)}):\n")
    for i, r in enumerate(rows, 1):
        sent = r.get("sentence", "")
        similar_to = r.get("similar_to", "")
        sim = r.get("similarity") or r.get("similarity_penalty") or ""
        print(f"{i}. [similarity {sim}]")
        print(f"   SENT:   {sent}")
        print(f"   CLOSEST: {similar_to}")
        print()


# =============================================================================
# YUE commands - analyze Cantonese decks
# =============================================================================

process_yue_app = cyclopts.App(
    name="yue", help="Analyze Cantonese Anki decks (e.g. LTL Cantonese)"
)
app.command(process_yue_app)

DEFAULT_CANTONESE_APKG = Path("apkg/LTL Cantonese Deck Level 1 - Short.apkg")


@process_yue_app.command(name="similar")
def similar_yue(
    apkg_path: Path = DEFAULT_CANTONESE_APKG,
    *,
    ranking_csv: Path | None = None,
):
    """Show sentence pairs with high similarity (from .rank.csv).

    :param apkg_path: Input .apkg file (used to default ranking CSV path).
    :param ranking_csv: Ranking CSV (default: same stem as apkg with .rank.csv).
    """
    apkg_path = Path(apkg_path)
    if ranking_csv is None:
        ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"
    if not ranking_csv.exists():
        print(f"Error: File not found: {ranking_csv}")
        return

    threshold = SIMILARITY_CONSIDER_DELETE_PENALTY
    with open(ranking_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            if not (r.get("similar_to") or "").strip():
                continue
            raw = r.get("similarity") or r.get("similarity_penalty") or "0"
            try:
                if float(raw) > threshold:
                    rows.append(r)
            except ValueError:
                pass

    if not rows:
        print(f"No pairs with similarity > {threshold} (consider deleting above this).")
        return

    print(f"Pairs worth reviewing for deletion (similarity > {threshold}, {len(rows)}):\n")
    for i, r in enumerate(rows, 1):
        sent = r.get("sentence", "")
        similar_to = r.get("similar_to", "")
        sim = r.get("similarity") or r.get("similarity_penalty") or ""
        print(f"{i}. [similarity {sim}]")
        print(f"   SENT:   {sent}")
        print(f"   CLOSEST: {similar_to}")
        print()


@process_yue_app.command
def rank(
    apkg_path: Path = DEFAULT_CANTONESE_APKG,
    *,
    output: Path | None = None,
    model_id: int | None = None,
    keep_filtered: bool = False,
):
    """Rank and reorder Cantonese deck with traditional Chinese and jyutping.

    Main command for processing a Cantonese Anki deck.
    Converts to traditional Chinese, generates jyutping, ranks, and reorders.

    :param apkg_path: Input .apkg file.
    :param output: Output .apkg file (default: input_reordered.apkg).
    :param model_id: Filter by specific model ID.
    :param keep_filtered: If True, keep filtered cards (names, invalid).
    """
    apkg_path = Path(apkg_path)
    if not apkg_path.exists():
        print(f"Error: File not found: {apkg_path}")
        return

    if output is None:
        output = apkg_path.parent / f"{apkg_path.stem}_reordered.apkg"

    ranking_csv = apkg_path.parent / f"{apkg_path.stem}.rank.csv"

    print("=" * 50)
    print("Processing Cantonese Anki Deck")
    print("=" * 50)
    print(f"Input:   {apkg_path}")
    print(f"Output:  {output}")
    print(f"Ranking: {ranking_csv}")
    print()

    print("Step 1: Ranking sentences...")
    sentences = extract_sentences(str(apkg_path), model_id)

    if not sentences:
        print("No sentences found!")
        return

    ranked = rank_sentences_yue(sentences)
    write_ranking_csv(ranked, str(ranking_csv))

    print(f"\nStep 2: Reordering deck...")
    stats = reorder_deck(
        str(apkg_path),
        str(output),
        str(ranking_csv),
        remove_filtered=not keep_filtered,
    )

    print("\nStep 3: Fixing jyutping...")
    fix_jyutping(
        output,
        output=output,
        verbose=False,
    )

    print()
    print("=" * 50)
    print("DONE!")
    print("=" * 50)
    print(f"  Sentences ranked:    {len(ranked)}")
    print(f"  Cards reordered:     {stats['cards_reordered']}")
    print(f"  Cards removed:       {stats['cards_removed']}")
    print(f"  Output:              {output}")
    print(f"  Ranking CSV:         {ranking_csv}")


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
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({k: v.strip() if v else v for k, v in row.items()})

    for entry in entries:
        text = entry[config["audio_field_name"]]
        cleaned_text = _clean_text_for_audio(text).replace(" ", "_")
        entry["audio_filename"] = f"{cleaned_text}.mp3"

    return entries


def _generate_audio_files(entries: list[dict], config: dict) -> None:
    """Generate audio files for entries using TTS if they don't exist.

    :param entries: List of entry dicts from _parse_vocab_entries.
    :param config: Config dict with language and audio_dir.
    """
    language = config.get("language")
    if not language:
        return

    GTTS_LANG = {
        "arabic": "ar",
        "cantonese": "yue",
        "german": "de",
        "greek": "el",
        "hindi": "hi",
        "mandarin": "zh-cn",
        "spanish": "es",
    }

    lang = GTTS_LANG.get(language.lower())
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
    """Order fields: target language, transliteration (if any), english, then rest.

    :param csv_fields: All CSV column names.
    :param audio_field_name: Name of the audio/target language field.
    :returns: Ordered list of field names for card content.
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
    for f in rest:
        if f not in result:
            result.append(f)
    return result


def _build_apkg(language: str) -> None:
    """Build APKG file with vocabulary and audio for the given language.

    :param language: Language name (e.g. mandarin, spanish).
    """
    lang_lower = language.lower()
    lang_title = language.title()
    print(f"Building {lang_title} vocabulary APKG...")

    config = {
        "language": language,
        "vocab_file": f"vocab/english-{lang_lower}.csv",
        "audio_dir": f"audio/audio_{lang_lower}",
        "deck_name": f"Swadesh {lang_title} 207",
        "model_name": f"Swadesh {lang_title}",
        "output_file": f"swadesh_{lang_lower}.apkg",
        "audio_field_name": lang_lower,
    }

    entries = _parse_vocab_entries(config)

    _generate_audio_files(entries, config)

    audio_dir = Path(config["audio_dir"])
    if not audio_dir.is_absolute():
        audio_dir = Path(__file__).parent.parent / config["audio_dir"]
    if not audio_dir.exists():
        print(f"Error: Audio directory not found: {audio_dir}")
        return

    # Load CSS from templates/card.css if it exists
    css_path = Path(__file__).parent.parent / "templates" / "card.css"
    css_content = None
    if css_path.exists():
        css_content = css_path.read_text(encoding="utf-8")

    csv_fields = (
        [k for k in entries[0].keys() if k != "audio_filename"] if entries else []
    )
    content_order = _content_field_order(csv_fields, config["audio_field_name"])
    fields = content_order + ["audio"]

    char_click_script = dedent(r"""
    <script>
    function expand(elemId, url) {
        var elem = document.getElementById(elemId);
        if (elem) {
        var isCJK = ch => /[\u4e00-\u9fff]/.test(ch);
        elem.innerHTML = elem.innerHTML
            .split("")
            .filter(ch => !/\s/.test(ch))
            .map(ch => isCJK(ch) ? `<a href="${url}${ch}">${ch}</a>` : ch)
            .join("");
        }
    }
    expand('chars', `https://cantonese.org/search.php?q=`)
    </script>
    """)

    cjk_fields = {"characters", "hanzi", "cantonese", "mandarin", "chinese"}
    script_15em_fields = {"arabic", "hindi"}
    answer_parts = ["{{audio}}", "<hr id=answer>"]
    for field in content_order:
        if field.lower() in cjk_fields:
            answer_parts.append(
                f"<div id='chars' style='font-size: 3rem'>{{{{{field}}}}}</div>"
            )
        elif field.lower() in script_15em_fields:
            answer_parts.append(f"<div style='font-size: 1.5em'>{{{{{field}}}}}</div>")
        else:
            answer_parts.append(f"<div style='font-size: 2rem'>{{{{{field}}}}}</div>")
    answer_parts.append(char_click_script)
    answer_format = "\n".join(answer_parts)

    cards = []
    media_files = []

    for entry in entries:
        audio_path = audio_dir / entry["audio_filename"]

        if not audio_path.exists():
            print(f"Warning: Audio file not found: {audio_path}")
            continue

        card = {"audio": f"[sound:{entry['audio_filename']}]"}
        for field in content_order:
            card[field] = entry.get(field, "")
        cards.append(card)

        media_files.append(str(audio_path.absolute()))

    if not media_files:
        print("Error: No audio files found. Check audio files exist.")
        return

    apkg_dir = Path("apkg")
    apkg_dir.mkdir(exist_ok=True)

    output_path = apkg_dir / config["output_file"]

    print(f"Creating APKG: {output_path}")
    print(f"  Deck: {config['deck_name']}")
    print(f"  Cards: {len(cards)}")
    print(f"  Audio files: {len(media_files)}")
    if css_content:
        print("  CSS: templates/card.css")

    AnkiPackage.create(
        output_path,
        config["deck_name"],
        fields=fields,
        cards=cards,
        media_files=media_files,
        model_name=config["model_name"],
        question_format="{{audio}}",
        answer_format=answer_format,
        css=css_content,
        sort_field_index=0,
    )

    print(f"\nSuccessfully created: {output_path}")


SWADESH_LANGUAGES = (
    "mandarin",
    "spanish",
    "greek",
    "german",
    "cantonese",
    "arabic",
    "hindi",
)

swadesh_app = cyclopts.App(
    name="swadesh", help="Build Swadesh vocabulary APKG files with audio"
)
app.command(swadesh_app)


@swadesh_app.command
def build(language: str):
    """Build Swadesh vocabulary APKG for a language.

    :param language: One of mandarin, spanish, greek, german, cantonese, arabic, hindi.
    """
    lang = language.lower().strip()
    if lang not in SWADESH_LANGUAGES:
        print(
            f"Unknown language: {language}. Supported: {', '.join(SWADESH_LANGUAGES)}"
        )
        return
    _build_apkg(lang)


@swadesh_app.command
def list_languages():
    """List supported Swadesh languages."""
    print("Supported languages:", ", ".join(SWADESH_LANGUAGES))


def main() -> None:
    """Main entry point. Invokes the cyclopts app."""
    app()


if __name__ == "__main__":
    main()
